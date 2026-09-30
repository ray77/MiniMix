/* pld_seek: a player's seek on a patchwork-landscape file. Opens the file, focuses the decoder on `seconds`, steps
   until `hold` seconds from there are final and reports the wait, then decodes the rest and writes the whole result
   (to compare with the source: the order of the segments must not change a bit).
   Usage: pld_seek file.mmx seconds [hold] [out.wav]
   PLD_SNAP=budget-seconds: land the seek on mmx_decoder_seek_point (a player's snap to an entry point)
   PLD_PRE=seconds: decode that much audio first (lanes busy), then seek; the wait is timed from the seek
   PLD_STRESS=1: after the seek, a new focus after every decoder step (a scrubbing user) - the result must not change */
#if !defined(_POSIX_C_SOURCE)
#define _POSIX_C_SOURCE 199309L   /* clock_gettime, nanosleep */
#endif
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "minimix/decoder.h"
#include "minimix/mmx_reader.h"
#include "wav_writer.h"
static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return (double)t.tv_sec + t.tv_nsec * 1e-9; }
int main(int argc, char **argv)
{
    MMXFile f; MMXAudioBuffer out; MMXDecoder *d = NULL; double t0, t1; int rc = 1;
    unsigned long long target, want;
    if (argc < 3) { fprintf(stderr, "usage: pld_seek file.mmx seconds [hold] [out.wav]\n"); return 2; }
    mmx_file_init(&f);
    if (mmx_reader_read(argv[1], &f, 1) != 0) return 1;
    t0 = now();
    if (mmx_decoder_open(&f, &out, &d) != 0) return 1;
    target = (unsigned long long)(atof(argv[2]) * f.sample_rate);
    if (target >= f.frame_count) target = f.frame_count ? f.frame_count - 1 : 0;
    want = target + (unsigned long long)((argc > 3 ? atof(argv[3]) : 5.0) * f.sample_rate);
    if (want > f.frame_count) want = f.frame_count;
    if (getenv("PLD_PRE"))                      /* a seek while the decoder is busy: first decode this many seconds */
    {
        unsigned long long pre = (unsigned long long)(atof(getenv("PLD_PRE")) * f.sample_rate), tot;
        do
        {
            unsigned long long lo[512], fin[512];
            unsigned long n = mmx_decoder_segments(d, lo, fin, 512), q;
            rc = mmx_decoder_step(d, 16);
            for (tot = 0, q = 0; q < n && q < 512; q++) tot += fin[q] - lo[q];
        } while (rc > 0 && tot < pre);
        t0 = now();
    }
    if (getenv("PLD_SNAP"))
    {
        unsigned long long p = mmx_decoder_seek_point(d, target, (unsigned long long)(atof(getenv("PLD_SNAP")) * f.sample_rate));
        want = p + (want - target);
        target = p;
        printf("snapped to %.1f s | ", (double)target / f.sample_rate);
    }
    mmx_decoder_focus(d, target);
    while (rc > 0 && mmx_decoder_final_from(d, target) < want) rc = mmx_decoder_step(d, 16);
    t1 = now();
    {
        unsigned long long lo[64], fin[64];
        unsigned long n = mmx_decoder_segments(d, lo, fin, 64), s, done = 0;
        for (s = 0; s < n && s < 64; s++) if (fin[s] > lo[s]) done++;
        printf("seek to %.1f s: %.2f s until %.1f s playable (%lu of %lu segments touched)\n", atof(argv[2]), t1 - t0,
               (double)(want - target) / f.sample_rate, done, n);
    }
    if (getenv("PLD_STRESS"))                   /* a scrubbing user: a new focus after every step */
    {
        unsigned long k = 0;
        while (rc > 0)
        {
            if (k % 20 == 0)                    /* a new spot every 20 steps: a user seeking every few seconds */
                mmx_decoder_focus(d, (unsigned long long)((k * 7919u) % 1000u) * f.frame_count / 1000u);
            k++;
            rc = mmx_decoder_step(d, 16);
        }
        printf("stress: %lu focus changes\n", (k + 19) / 20);
    }
    while (rc > 0) rc = mmx_decoder_step(d, 4096);
    printf("whole file after %.2f s\n", now() - t0);
    mmx_decoder_close(d);
    if (rc == 0 && argc > 4 && mmx_wav_write(argv[4], &out, f.source_bits ? f.source_bits : 16) != 0) rc = 1;
    return rc != 0;
}
