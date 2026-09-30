/* mmx play: plays .mmx files in the terminal while they decode.

   Three threads. The decode thread owns the decoder (every mmx_decoder_* call happens there): it steps it, places
   seeks with mmx_decoder_seek_point + mmx_decoder_focus, and publishes how far the audio from the play position is
   final. The audio thread (miniaudio's callback) copies final samples to the device and plays silence while a seek is
   being placed or the decoder has fallen behind; after such a gap it waits for a lead of LEAD_S seconds before it
   plays again, so a slow decode stutters once instead of every buffer. The main thread reads keys and draws one status
   line from a snapshot; it never prints while it holds the lock, so a stalled terminal cannot stall the audio.
   A file with entry points (patchwork landscape decoding) decodes in segments side by side; the bar shows them
   filling in.

   Keys: space pause, left/right 10 s, down/up 60 s, 0-9 jump to 0-90 %, n next file, q quit. Without a terminal on
   stdin the files just play (and on POSIX systems keys can be piped in, which the tests use). MMX_PLAY_NULL=1 plays
   into miniaudio's null device: the same timing, no sound. */
#if !defined(_WIN32)
#if !defined(_POSIX_C_SOURCE)
#define _POSIX_C_SOURCE 200809L
#endif
#define _DEFAULT_SOURCE                 /* TIOCGWINSZ on glibc */
#define _DARWIN_C_SOURCE                /* and on macOS */
#endif
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include <signal.h>
#ifdef _WIN32
#include <windows.h>
#include <conio.h>
#include <io.h>
#else
#include <time.h>
#include <unistd.h>
#include <termios.h>
#include <sys/ioctl.h>
#include <sys/select.h>
#endif
#include "play_miniaudio.h"
#include "minimix/decoder.h"
#include "minimix/mmx_format.h"
#include "minimix/mmx_reader.h"
#include "log.h"
#include "play.h"

#define STEP_HOPS 32            /* decoder frames per step: about 0.7 s of audio, a few hundredths of a second of work */
#define LEAD_S 2.0              /* after a gap, play again once this much is final (or the rest of the file) */
#define SEEK_BUDGET_S 12.0      /* a seek decodes at most this much audio before it plays (about 1.5 s at 8x) */
#define BAR_MAX 60

typedef struct
{
    MMXFile file;
    MMXAudioBuffer pcm;
    MMXDecoder *dec;
    unsigned long long frames, lead;
    unsigned int channels;
    unsigned long rate;
    pthread_t thread;
    int thread_started;
    pthread_mutex_t lock;
    /* shared, under lock */
    unsigned long long pos;             /* next frame for the device */
    unsigned long long final_end;       /* end of the final run that starts at pos */
    long long seek_to;                  /* a seek the decode thread has not placed yet, -1 = none */
    long long placed;                   /* where the last seek landed, until the main thread has reported it; -1 */
    int paused, buffering, stop, complete, failed;
    unsigned long underruns;
    unsigned long nseg;                 /* segments (1 without entry points) and how far each is final */
    unsigned long long *seg_lo, *seg_final;
} Player;

/* what the main thread draws: copied under the lock, printed after it */
typedef struct
{
    unsigned long long pos;
    long long seek, placed;
    int paused, failed;
    unsigned long long *seg_lo, *seg_final;
} Snapshot;

static volatile sig_atomic_t interrupted = 0;
static void on_signal(int sig) { (void)sig; interrupted = 1; }

static void sleep_ms(unsigned int ms)
{
#ifdef _WIN32
    Sleep(ms);
#else
    struct timespec t;
    t.tv_sec = ms / 1000;
    t.tv_nsec = (long)(ms % 1000) * 1000000L;
    nanosleep(&t, NULL);
#endif
}

/* ------------------------------------------------------------------ decode thread --- */

static void publish(Player *p)   /* decode thread only: the final run at pos and the segments */
{
    pthread_mutex_lock(&p->lock);
    p->final_end = p->complete ? p->frames : mmx_decoder_final_from(p->dec, p->pos);
    mmx_decoder_segments(p->dec, p->seg_lo, p->seg_final, p->nseg);
    pthread_mutex_unlock(&p->lock);
}

static void *decode_main(void *arg)
{
    Player *p = (Player *)arg;
    mmx_decoder_focus(p->dec, 0);
    for (;;)
    {
        long long seek;
        int stop, complete;
        pthread_mutex_lock(&p->lock);
        seek = p->seek_to;
        stop = p->stop;
        complete = p->complete;
        pthread_mutex_unlock(&p->lock);
        if (stop)
            break;
        if (seek >= 0)
        {
            /* the exact spot unless reaching it costs more than SEEK_BUDGET_S of audio, then an easier neighbouring
               entry point (a song whose segments read many earlier ones) */
            unsigned long long at = mmx_decoder_seek_point(p->dec, (unsigned long long)seek,
                                                           (unsigned long long)(SEEK_BUDGET_S * (double)p->rate));
            mmx_decoder_focus(p->dec, at);
            pthread_mutex_lock(&p->lock);
            if (p->seek_to == seek)             /* not overtaken by a newer key press */
            {
                p->seek_to = -1;
                p->pos = at;
                p->final_end = at;              /* nothing counts as final there until publish() has looked */
                p->placed = (long long)at;
                p->buffering = 1;
            }
            pthread_mutex_unlock(&p->lock);
            publish(p);
            continue;
        }
        if (!complete)
        {
            int rc = mmx_decoder_step(p->dec, STEP_HOPS);
            pthread_mutex_lock(&p->lock);
            if (rc < 0) p->failed = 1;
            if (rc == 0) p->complete = 1;
            pthread_mutex_unlock(&p->lock);
            if (rc < 0)
                break;
            publish(p);
        }
        else
        {
            publish(p);
            sleep_ms(20);
        }
    }
    return NULL;
}

/* ------------------------------------------------------------------ audio thread --- */

static void on_audio(ma_device *dev, void *out, const void *in, ma_uint32 n)
{
    Player *p = (Player *)dev->pUserData;
    float *o = (float *)out;
    unsigned long long done = 0;
    (void)in;
    pthread_mutex_lock(&p->lock);
    if (!p->paused && p->seek_to < 0 && !p->failed && p->pos < p->frames)
    {
        unsigned long long avail = p->final_end > p->pos ? p->final_end - p->pos : 0;
        if (p->buffering && (avail >= p->lead || p->final_end >= p->frames))
            p->buffering = 0;
        if (!p->buffering)
        {
            unsigned long long k = avail < n ? avail : n;
            memcpy(o, p->pcm.samples + p->pos * p->channels, (size_t)(k * p->channels) * sizeof(float));
            p->pos += k;
            done = k;
            if (k < n && p->pos < p->frames)
            {
                p->buffering = 1;       /* the decoder fell behind: a gap, then LEAD_S before playing on */
                p->underruns++;
            }
        }
    }
    pthread_mutex_unlock(&p->lock);
    if (done < n)
        memset(o + done * p->channels, 0, (size_t)((n - done) * p->channels) * sizeof(float));
}

/* ------------------------------------------------------------------ keys --- */

enum { KEY_NONE, KEY_PAUSE, KEY_BACK10, KEY_FWD10, KEY_BACK60, KEY_FWD60, KEY_NEXT, KEY_QUIT, KEY_PERCENT0 };

typedef struct
{
    int tty;                            /* keys come from a terminal (raw mode) */
    int pipe;                           /* POSIX: keys piped in (tests) */
#ifndef _WIN32
    struct termios saved;
#endif
} Keys;

static void keys_open(Keys *k)
{
    memset(k, 0, sizeof(*k));
#ifdef _WIN32
    k->tty = _isatty(_fileno(stdin));
#else
    if (isatty(0) && tcgetattr(0, &k->saved) == 0)
    {
        struct termios raw = k->saved;
        raw.c_lflag &= ~(tcflag_t)(ICANON | ECHO);
        raw.c_cc[VMIN] = 0;
        raw.c_cc[VTIME] = 0;
        k->tty = tcsetattr(0, TCSANOW, &raw) == 0;
    }
    else
        k->pipe = 1;
#endif
}

static void keys_close(Keys *k)
{
#ifndef _WIN32
    if (k->tty)
        tcsetattr(0, TCSANOW, &k->saved);
#endif
    k->tty = 0;
}

static int key_of_char(int c)
{
    return c == ' ' || c == 'p' ? KEY_PAUSE : c == 'n' ? KEY_NEXT : c == 'q' ? KEY_QUIT
         : c >= '0' && c <= '9' ? KEY_PERCENT0 + (c - '0') : KEY_NONE;
}

#ifndef _WIN32
static int byte_within(unsigned char *b, unsigned int ms)   /* one more byte of a sequence, if it comes in time */
{
    fd_set set;
    struct timeval tv;
    FD_ZERO(&set);
    FD_SET(0, &set);
    tv.tv_sec = 0;
    tv.tv_usec = (long)ms * 1000L;
    return select(1, &set, NULL, NULL, &tv) > 0 && read(0, b, 1) == 1;
}
#endif

/* waits up to ms for a key */
static int key_read(Keys *k, unsigned int ms)
{
#ifdef _WIN32
    unsigned int waited = 0;
    if (!k->tty) { sleep_ms(ms); return KEY_NONE; }
    while (!_kbhit())
    {
        if (waited >= ms) return KEY_NONE;
        sleep_ms(10);
        waited += 10;
    }
    {
        int c = _getch();
        if (c == 0 || c == 224)         /* a function or arrow key: its code follows */
        {
            c = _getch();
            return c == 75 ? KEY_BACK10 : c == 77 ? KEY_FWD10 : c == 80 ? KEY_BACK60 : c == 72 ? KEY_FWD60 : KEY_NONE;
        }
        return c == 27 ? KEY_QUIT : key_of_char(c);
    }
#else
    fd_set set;
    struct timeval tv;
    unsigned char b;
    ssize_t r;
    if (!k->tty && !k->pipe) { sleep_ms(ms); return KEY_NONE; }
    FD_ZERO(&set);
    FD_SET(0, &set);
    tv.tv_sec = ms / 1000;
    tv.tv_usec = (long)(ms % 1000) * 1000L;
    if (select(1, &set, NULL, NULL, &tv) <= 0)
        return KEY_NONE;
    r = read(0, &b, 1);
    if (r <= 0)
    {
        if (k->pipe) { k->pipe = 0; sleep_ms(ms); }   /* the pipe ended: just play on */
        return KEY_NONE;
    }
    if (b == 27)
    {
        /* a lone ESC quits; ESC [ or ESC O starts a sequence that runs to its final byte (0x40-0x7e): the four plain
           arrows mean something, everything else (Ctrl+arrow, function keys, Alt+key) is swallowed whole */
        unsigned char c, last = 0;
        unsigned int len = 0;
        if (!byte_within(&c, 30))
            return KEY_QUIT;
        if (c != '[' && c != 'O')
            return KEY_NONE;
        while (byte_within(&last, 30))
        {
            len++;
            if (last >= 0x40 && last <= 0x7e)
                break;
            if (len > 16)
                return KEY_NONE;
        }
        if (len != 1)
            return KEY_NONE;
        return last == 'D' ? KEY_BACK10 : last == 'C' ? KEY_FWD10 : last == 'B' ? KEY_BACK60 : last == 'A' ? KEY_FWD60
             : KEY_NONE;
    }
    return key_of_char(b);
#endif
}

/* ------------------------------------------------------------------ display --- */

static void clock_text(unsigned long long frames, unsigned long rate, char *out, size_t size)
{
    unsigned long long s = rate ? frames / rate : 0;
    if (s >= 3600)
        snprintf(out, size, "%llu:%02llu:%02llu", s / 3600, (s / 60) % 60, s % 60);
    else
        snprintf(out, size, "%llu:%02llu", s / 60, s % 60);
}

/* text from a file (tags, the path) with control characters shown as '?': a tag must not drive the terminal */
static void print_clean(const char *s)
{
    for (; *s; s++)
        putchar((unsigned char)*s < 0x20 || (unsigned char)*s == 0x7f ? '?' : *s);
}

static unsigned int terminal_width(void)
{
#ifdef _WIN32
    CONSOLE_SCREEN_BUFFER_INFO info;
    if (GetConsoleScreenBufferInfo(GetStdHandle(STD_OUTPUT_HANDLE), &info))
        return (unsigned int)(info.srWindow.Right - info.srWindow.Left + 1);
#else
    struct winsize w;
    if (ioctl(1, TIOCGWINSZ, &w) == 0 && w.ws_col > 0)
        return w.ws_col;
#endif
    return 80;
}

static void print_header(const char *path, const MMXFile *f, unsigned int index, unsigned int count)
{
    char *title = mmx_file_metadata_get(f, "title"), *artist = mmx_file_metadata_get(f, "artist");
    char *album = mmx_file_metadata_get(f, "album");
    char len[32];
    const char *kind = f->quality != 0 ? "lossy" : f->ll_dropped || f->ll_qdrop > 1 ? "near-lossless" : "lossless";
    clock_text(f->frame_count, f->sample_rate, len, sizeof(len));
    if (count > 1)
        printf("[%u/%u] ", index + 1, count);
    print_clean(title ? title : path);
    if (title && artist)
    {
        printf(" - ");
        print_clean(artist);
    }
    printf("\n");
    if (album)
    {
        printf("      ");
        print_clean(album);
        printf("\n");
    }
    printf("      %.1f kHz, %u bit, %s, %s, %s%s\n", f->sample_rate / 1000.0, f->source_bits ? f->source_bits : 16,
           f->channels == 1 ? "mono" : f->channels == 2 ? "stereo" : "multichannel", kind, len,
           f->ll2_pld ? ", entry points" : "");
    free(title);
    free(artist);
    free(album);
}

/* one line, as wide as the terminal: state, time, and a bar of the file - played '#', decoded '=', not decoded
   yet '.', the play head '|' */
static void draw(const Player *p, const Snapshot *s)
{
    char now[32], len[32], bar[BAR_MAX + 1], line[256];
    unsigned int width = terminal_width(), fixed, nbar, i;
    unsigned long long shown = s->seek >= 0 ? (unsigned long long)s->seek : s->pos;
    unsigned long g;
    clock_text(shown, p->rate, now, sizeof(now));
    clock_text(p->frames, p->rate, len, sizeof(len));
    fixed = 2 + 2 + 1 + (unsigned int)strlen(now) + 3 + (unsigned int)strlen(len) + 2 + 2 + 2 + 11;
    nbar = width > fixed + 8 ? width - fixed - 1 : 0;
    if (nbar > BAR_MAX) nbar = BAR_MAX;
    for (i = 0; i < nbar; i++)
    {
        unsigned long long a = p->frames * i / nbar, b = p->frames * (i + 1) / nbar;
        int final = 1;
        for (g = 0; g < p->nseg; g++)   /* a cell is decoded when every segment it touches is final across it */
        {
            unsigned long long lo = s->seg_lo[g], hi = g + 1 < p->nseg ? s->seg_lo[g + 1] : p->frames;
            unsigned long long x = a > lo ? a : lo, y = b < hi ? b : hi;
            if (x < y && s->seg_final[g] < y)
                final = 0;
        }
        bar[i] = b <= s->pos ? '#' : final ? '=' : '.';
        if (s->pos >= a && s->pos < b)
            bar[i] = '|';
    }
    bar[nbar] = '\0';
    if (nbar)
        snprintf(line, sizeof(line), "  %s %s / %s  [%s]  %s", s->paused ? "||" : " >", now, len, bar,
                 s->seek >= 0 ? "seeking ..." : s->paused ? "paused" : "");
    else
        snprintf(line, sizeof(line), "  %s %s / %s", s->paused ? "||" : " >", now, len);
    printf("\r%-*s", (int)(width > 1 ? width - 1 : 1), line);
    fflush(stdout);
}

/* ------------------------------------------------------------------ one file --- */

static int player_open(Player *p, const char *path)
{
    memset(p, 0, sizeof(*p));
    p->seek_to = -1;
    p->placed = -1;
    p->buffering = 1;
    if (mmx_reader_read(path, &p->file, 1) != 0)
        return -1;
    if (mmx_decoder_open(&p->file, &p->pcm, &p->dec) != 0)
    {
        mmx_file_free(&p->file);
        return -1;
    }
    p->frames = p->file.frame_count;
    p->channels = p->file.channels;
    p->rate = p->file.sample_rate;
    p->lead = (unsigned long long)(LEAD_S * (double)p->rate);
    p->nseg = mmx_decoder_segments(p->dec, NULL, NULL, 0);
    if (p->nseg < 1) p->nseg = 1;
    p->seg_lo = (unsigned long long *)calloc(p->nseg, sizeof(unsigned long long));
    p->seg_final = (unsigned long long *)calloc(p->nseg, sizeof(unsigned long long));
    if (!p->seg_lo || !p->seg_final || pthread_mutex_init(&p->lock, NULL) != 0)
    {
        free(p->seg_lo);
        free(p->seg_final);
        mmx_decoder_close(p->dec);
        mmx_audio_buffer_free(&p->pcm);
        mmx_file_free(&p->file);
        return -1;
    }
    mmx_decoder_segments(p->dec, p->seg_lo, p->seg_final, p->nseg);
    return 0;
}

static void player_close(Player *p)
{
    if (p->thread_started)
    {
        pthread_mutex_lock(&p->lock);
        p->stop = 1;
        pthread_mutex_unlock(&p->lock);
        pthread_join(p->thread, NULL);
    }
    pthread_mutex_destroy(&p->lock);
    free(p->seg_lo);
    free(p->seg_final);
    mmx_decoder_close(p->dec);
    mmx_audio_buffer_free(&p->pcm);
    mmx_file_free(&p->file);
}

static void report(const char *what, unsigned long long frame, unsigned long rate)   /* the line log without a terminal */
{
    char c[32];
    clock_text(frame, rate, c, sizeof(c));
    printf("%s %s\n", what, c);
}

/* plays one file; returns 1 to go on with the next one, 0 to quit, -1 on an error */
static int play_one(ma_context *ctx, const char *path, unsigned int index, unsigned int count, Keys *keys, int tty_out)
{
    Player p;
    Snapshot s;
    ma_device dev;
    ma_device_config cfg;
    int result = 1;
    unsigned long long last_pos = ~0ULL;
    if (player_open(&p, path) != 0)
    {
        mmx_error("%s: not a playable .mmx file", path);
        return -1;
    }
    print_header(path, &p.file, index, count);
    s.seg_lo = (unsigned long long *)calloc(p.nseg, sizeof(unsigned long long));
    s.seg_final = (unsigned long long *)calloc(p.nseg, sizeof(unsigned long long));
    cfg = ma_device_config_init(ma_device_type_playback);
    cfg.playback.format = ma_format_f32;
    cfg.playback.channels = p.channels;
    cfg.sampleRate = (ma_uint32)p.rate;
    cfg.dataCallback = on_audio;
    cfg.pUserData = &p;
    if (!s.seg_lo || !s.seg_final || ma_device_init(ctx, &cfg, &dev) != MA_SUCCESS)
    {
        mmx_error("no audio output device");
        free(s.seg_lo);
        free(s.seg_final);
        player_close(&p);
        return -1;
    }
    if (pthread_create(&p.thread, NULL, decode_main, &p) != 0)
    {
        mmx_error("cannot start the decode thread");
        result = -1;
    }
    else
    {
        p.thread_started = 1;
        if (ma_device_start(&dev) != MA_SUCCESS)
        {
            mmx_error("cannot start the audio output");
            result = -1;
        }
    }
    while (result == 1)
    {
        int key = key_read(keys, 100);
        long long jump = -1;
        if (interrupted)
            key = KEY_QUIT;
        if (key == KEY_NEXT)
            break;
        if (key == KEY_QUIT)
        {
            result = 0;
            break;
        }
        pthread_mutex_lock(&p.lock);
        if (key == KEY_PAUSE)
            p.paused = !p.paused;
        else if (key == KEY_BACK10 || key == KEY_FWD10 || key == KEY_BACK60 || key == KEY_FWD60 || key >= KEY_PERCENT0)
        {
            long long base = p.seek_to >= 0 ? p.seek_to : (long long)p.pos;
            long long d = (long long)p.rate * (key == KEY_BACK10 ? -10 : key == KEY_FWD10 ? 10 : key == KEY_BACK60 ? -60 : 60);
            long long t = key >= KEY_PERCENT0 ? (long long)(p.frames / 10 * (unsigned long long)(key - KEY_PERCENT0))
                                              : base + d;
            if (t < 0) t = 0;
            if (t >= (long long)p.frames) t = p.frames ? (long long)p.frames - 1 : 0;
            p.seek_to = jump = t;
        }
        s.pos = p.pos;
        s.seek = p.seek_to;
        s.placed = p.placed;
        s.paused = p.paused;
        s.failed = p.failed;
        p.placed = -1;
        memcpy(s.seg_lo, p.seg_lo, p.nseg * sizeof(unsigned long long));
        memcpy(s.seg_final, p.seg_final, p.nseg * sizeof(unsigned long long));
        pthread_mutex_unlock(&p.lock);
        if (s.failed)
        {
            mmx_error("%s: bitstream error while decoding", path);
            result = -1;
            break;
        }
        if (tty_out)
            draw(&p, &s);
        else
        {
            if (jump >= 0) report("seek", (unsigned long long)jump, p.rate);
            if (s.placed >= 0) report("playing from", (unsigned long long)s.placed, p.rate);
            if (s.pos / p.rate != last_pos / p.rate && s.pos / p.rate % 30 == 0) report("at", s.pos, p.rate);
        }
        last_pos = s.pos;
        if (s.pos >= p.frames && s.seek < 0)
            break;                      /* played to the end */
    }
    ma_device_uninit(&dev);             /* stops the callback before the buffers go */
    if (tty_out)
        printf("\n");
    if (p.underruns)
        mmx_warning("the decoder fell behind %lu time%s", p.underruns, p.underruns == 1 ? "" : "s");
    free(s.seg_lo);
    free(s.seg_final);
    player_close(&p);
    return result;
}

int mmx_cmd_play(const MMXOptions *opts)
{
    ma_context ctx;
    ma_backend null_backend[1] = { ma_backend_null };
    const char *e = getenv("MMX_PLAY_NULL");
    int silent = e && atoi(e);
    Keys keys;
    unsigned int i;
    int rc = 0, tty_out;
#ifdef _WIN32
    UINT codepage = GetConsoleOutputCP();
    tty_out = _isatty(_fileno(stdout));
    if (tty_out)
        SetConsoleOutputCP(CP_UTF8);    /* tags are UTF-8; put back below */
#else
    tty_out = isatty(1);
#endif
    /* without MMX_PLAY_NULL the null device does not count: miniaudio falls back on it when nothing else works */
    if (ma_context_init(silent ? null_backend : NULL, silent ? 1 : 0, NULL, &ctx) != MA_SUCCESS
        || (!silent && ctx.backend == ma_backend_null))
    {
        mmx_error("no audio output available");
#ifdef _WIN32
        if (tty_out) SetConsoleOutputCP(codepage);
#endif
        return 1;
    }
    signal(SIGINT, on_signal);
    signal(SIGTERM, on_signal);
#ifdef _WIN32
    signal(SIGBREAK, on_signal);
#else
    signal(SIGHUP, on_signal);
    signal(SIGQUIT, on_signal);
#endif
    keys_open(&keys);
    if (keys.tty && tty_out)
        printf("space pause   arrows seek   0-9 jump   n next   q quit\n\n");
    for (i = 0; i < opts->file_count; i++)
    {
        int r = play_one(&ctx, opts->files[i], i, opts->file_count, &keys, tty_out);
        if (r < 0) rc = 1;
        if (r == 0 || interrupted) break;
    }
    keys_close(&keys);
    ma_context_uninit(&ctx);
#ifdef _WIN32
    if (tty_out) SetConsoleOutputCP(codepage);
#endif
    return rc;
}
