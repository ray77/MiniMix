#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <sys/time.h>
#include <stdio.h>
#include "minimix/analyzer.h"
#include "minimix/similarity.h"
#include "reference_graph.h"
#include "spectrum.h"
#include "lossless.h"
#include "tns.h"
#include "log.h"
#include "threads.h"

/* Wall clock for the analysis report: with the frames running on several cores
   the processor time the report used to print (clock()) is the sum over the
   workers, which is not what anybody waits for. Nothing but the report reads
   it - the number never enters the file. */
static double mmx_wall_seconds(void)
{
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (double)tv.tv_sec + (double)tv.tv_usec * 1e-6;
}

#define FP_DIM 40                    /* fingerprint bands used */
#define MAX_CANDS 160
#define CANDS_RESERVE 24             /* slots the continuations leave free for star and pair candidates */
#define SWITCH_BITS_DEFAULT 480.0    /* table entry + context reset when a new block starts (MMX_SWITCH_BITS overrides: experiment) */
static double switch_bits(void)
{
    static double v = -1.0;
    if (v < 0.0)
    {
        const char *e = getenv("MMX_SWITCH_BITS");
        v = e ? atof(e) : SWITCH_BITS_DEFAULT;
    }
    return v;
}
#define SWITCH_BITS switch_bits()
#define CONTINUE_AUDIO_EXTRA 12.0    /* keeping a reference block alive through a frame coded without prediction */
#define DRIFT 4                      /* samples a long reference may drift per frame (tempo drift, live timing) */
#define SECTION_FRAMES 32
#define SECTION_HOP 16
/* Both numbers above are in SAMPLES and FRAMES, not in time - and the drift between two played
   repeats is a property of the music, not of the sample rate. At 88.2 kHz a frame covers 11.6 ms
   instead of 23.2, so the same DRIFT of 4 samples per frame allows only half as much drift per
   second, and a section fingerprint spans 0.37 s instead of 0.74. Measured reference gain on the
   same three tracks: 7.89/8.89/6.23 % at 44.1 kHz against 4.02/3.19/2.42 % at 88.2 kHz.
   MMX_DRIFT overrides the per-frame alignment window (experiment). */
static long long drift_samples(void)
{
    static long long v = -1;
    if (v < 0)
    {
        const char *e = getenv("MMX_DRIFT");
        v = e ? atol(e) : DRIFT;
        if (v < 0) v = 0;
    }
    return v;
}
#define SILENT_DBFS -75.0
#define DEPTH_PENALTY 0.10
#define DP_STRIDE (MAX_CANDS + 1)     /* DP states per frame: AUDIO + candidates */
#define DP_LAG_EXACT 2                /* Viterbi lag at which every chain depth is exact when scored */

typedef struct
{
    unsigned char n_sources;
    long long start[MMX_MAX_SOURCES];
    float cost;
} Candidate;

typedef struct
{
    unsigned int k;               /* fingerprint candidates per frame */
    unsigned int section_k;       /* section-level candidates */
    unsigned int pairs;           /* two-source combinations to try */
    unsigned int onset_k;
    unsigned int continuations;   /* how many candidates of the previous frame are continued */
    unsigned int drift;           /* how many continued lineages get a drift-corrected variant */
    unsigned int dp_passes;
    unsigned int lag;             /* fixed-lag Viterbi: frames between the decision and the commit */
    int coded_estimate;           /* score with the estimate that mirrors the coder (mmx_frame_estimate_bits_coded) */
} Effort;

static void effort_for_level(unsigned int level, Effort *e)
{
    static const unsigned int ks[10] = {0, 2, 4, 6, 8, 12, 16, 24, 32, 48};
    const char *lag_env = getenv("MMX_DP_LAG");   /* experiment */
    if (level < 1) level = 1;
    if (level > MMX_ANALYSIS_MAX) level = MMX_ANALYSIS_MAX;
    e->k = ks[level];
    e->section_k = level >= 2 ? 2 + level / 3 : 0;
    e->pairs = level >= 4 ? (level >= 7 ? 3 : 1) : 0;
    e->onset_k = level >= 3 ? 2 + level / 2 : 0;
    /* lineages carried into the next frame: the best candidates by cost of the
       previous frame. Level 9 used to keep them all: with 48 fingerprint hits
       on top, the list (MAX_CANDS) was full in 99 % of the frames, the pair
       and star candidates were silently dropped and the file was 3.4 % LARGER
       than at level 5 (title A). */
    e->continuations = level >= 9 ? 64 : level >= 7 ? 48 : level >= 5 ? 24 : level >= 3 ? 12 : 4;
    e->drift = level >= 9 ? 24 : level >= 7 ? 16 : level >= 5 ? 6 : level >= 3 ? 2 : 0;
    e->dp_passes = 3;
    e->lag = level >= 7 ? 16 : DP_LAG_EXACT;
    if (lag_env && atoi(lag_env) >= DP_LAG_EXACT) e->lag = (unsigned int)atoi(lag_env);
    /* levels 7-9 score with the estimate that mirrors the coder: half the
       per-frame scatter against the real bits (title A: audio 17 -> 8 %,
       residual 32 -> 20 %), level 7 two-pass 7,474,909 -> 7,431,757 B */
    e->coded_estimate = level >= 7;
    if ((lag_env = getenv("MMX_CODED_ESTIMATE")) != NULL) e->coded_estimate = atoi(lag_env) != 0;   /* experiment */
    if ((lag_env = getenv("MMX_CONTINUATIONS")) != NULL && atoi(lag_env) > 0) e->continuations = (unsigned int)atoi(lag_env);   /* experiment */
}

void mmx_analysis_params_default(MMXAnalysisParams *p)
{
    memset(p, 0, sizeof(*p));
    p->level = 5;
    p->quality = 7;
    p->max_depth = 3;
}

const float *mmx_analysis_coefs(const MMXAnalysis *a, unsigned long f, unsigned int c)
{
    return a->coefs + ((size_t)f * a->channels + c) * MMX_HOP;
}

const MMXFramePsy *mmx_analysis_psy(const MMXAnalysis *a, unsigned long f, unsigned int c)
{
    return a->psy + (size_t)f * a->channels + c;
}

const float *mmx_analysis_model_thr(const MMXAnalysis *a, unsigned long f, unsigned int c)
{
    size_t i = (size_t)f * a->channels + c;
    return a->model_thr ? a->model_thr + i * MMX_MAX_BANDS : a->psy[i].thr;
}

/* Snapshot of the model's thresholds, taken when the analysis is done and
   before the encoder scales them (bitrate offset, clean samples). */
static void capture_model_thr(MMXAnalysis *a)
{
    size_t i, n = (size_t)a->frame_count * a->channels;
    a->model_thr = (float *)malloc(sizeof(float) * n * MMX_MAX_BANDS);
    if (!a->model_thr)
        return;   /* the guard falls back to the current thresholds */
    for (i = 0; i < n; i++)
        memcpy(a->model_thr + i * MMX_MAX_BANDS, a->psy[i].thr, sizeof(float) * MMX_MAX_BANDS);
}

void mmx_analysis_free(MMXAnalysis *a)
{
    if (!a)
        return;
    free(a->coefs);
    free(a->psy);
    free(a->model_thr);
    free(a->transient);
    free(a->silent);
    free(a->plan);
    free(a->repeats);
    memset(a, 0, sizeof(*a));
}

/* ---------- per frame precomputation ---------- */

/* Sub-window analysis (1024-sample windows every 512 samples): the masking
   threshold of a frame is the minimum over the sub-windows its 2048-sample
   window covers, so noise spread over the frame stays masked in every 23 ms
   part of it (pre-echo control without block switching). The same windows are
   what `mmx compare` evaluates. Per-coefficient power of a 1024-MDCT is half
   that of a 2048-MDCT for the same time-domain noise; the weight of each
   sub-window below also accounts for how much of the frame's noise it sees. */
#define SUB_WIN 1024
#define SUB_STEP 512

/* ---------- debug: what each rule of the model costs (MMX_DEBUG_PSY) ---------- */

/* Per EQ region and frame class (stationary / transient), over the bands that
   carry signal: by how many dB each rule lowers the threshold below the plain
   masking threshold of the frame (frame_thr), how often a band rests on the
   absolute threshold of hearing, the mean tonality offset, and the tonality
   histogram. Printed to stderr after the analysis. */
static struct
{
    int on;
    double alpha_sum;
    unsigned long alpha_n, alpha_hist[10];
    double offset_db[MMX_EQ_BANDS], bark[MMX_EQ_BANDS];
    double sub_db[2][MMX_EQ_BANDS], temporal_db[2][MMX_EQ_BANDS], cliff_db[2][MMX_EQ_BANDS];
    unsigned long n[2][MMX_EQ_BANDS], abs_n[2][MMX_EQ_BANDS], sub_n[2][MMX_EQ_BANDS];
} psy_debug;

static void psy_debug_frame(const MMXBandLayout *L, const MMXFramePsy *p, unsigned int cutoff, int transient)
{
    unsigned int b, t = transient ? 1 : 0;
    double alpha = p->tonality;
    psy_debug.alpha_sum += alpha;
    psy_debug.alpha_n++;
    psy_debug.alpha_hist[(unsigned int)(alpha * 9.999)]++;
    for (b = 0; b < cutoff; b++)
    {
        unsigned int e = L->eq_band[b];
        double n = (double)(L->band_start[b + 1] - L->band_start[b]);
        if (p->energy[b] <= p->frame_thr[b] * n * 0.5)
            continue; /* no audible signal in the band: nothing to spend bits on */
        psy_debug.n[t][e]++;
        psy_debug.offset_db[e] += p->tonality_eq[e] * (14.5 + L->band_bark[b]) + (1.0 - p->tonality_eq[e]) * 5.5;
        psy_debug.bark[e] += L->band_bark[b];
        psy_debug.sub_db[t][e] += 10.0 * log10(p->frame_thr[b] / p->thr[b]);
        if (p->thr[b] < p->frame_thr[b]) psy_debug.sub_n[t][e]++;
        if (p->frame_thr[b] <= L->abs_thr[b] * 1.0001) psy_debug.abs_n[t][e]++;
    }
}

/* Accumulates the tightening of one rule: thresholds `before` vs. `after` of a frame. */
static void psy_debug_rule(const MMXBandLayout *L, const MMXFramePsy *p, const float *before, unsigned int cutoff, double *acc)
{
    unsigned int b;
    for (b = 0; b < cutoff; b++)
    {
        double n = (double)(L->band_start[b + 1] - L->band_start[b]);
        if (p->energy[b] <= p->frame_thr[b] * n * 0.5 || before[b] >= 1e29f)
            continue;
        acc[L->eq_band[b]] += 10.0 * log10(before[b] / p->thr[b]);
    }
}

static void psy_debug_print(void)
{
    static const char *region[MMX_EQ_BANDS] = { "<200", "200-500", "500-1k", "1k-2k", "2k-4k", "4k-8k", "8k-12k", ">12k" };
    unsigned int e, t, h;
    fprintf(stderr, "psychoacoustic model: tonality alpha mean %.3f, histogram (0.0-0.1 ... 0.9-1.0):", psy_debug.alpha_n ? psy_debug.alpha_sum / psy_debug.alpha_n : 0.0);
    for (h = 0; h < 10; h++) fprintf(stderr, " %.1f%%", psy_debug.alpha_n ? 100.0 * psy_debug.alpha_hist[h] / psy_debug.alpha_n : 0.0);
    fprintf(stderr, "\n  region            ");
    for (e = 0; e < MMX_EQ_BANDS; e++) fprintf(stderr, " %8s", region[e]);
    fprintf(stderr, "\n  mean bark         ");
    for (e = 0; e < MMX_EQ_BANDS; e++) fprintf(stderr, " %8.1f", (psy_debug.n[0][e] + psy_debug.n[1][e]) ? psy_debug.bark[e] / (psy_debug.n[0][e] + psy_debug.n[1][e]) : 0.0);
    fprintf(stderr, "\n  mean offset dB    ");
    for (e = 0; e < MMX_EQ_BANDS; e++) fprintf(stderr, " %8.1f", (psy_debug.n[0][e] + psy_debug.n[1][e]) ? psy_debug.offset_db[e] / (psy_debug.n[0][e] + psy_debug.n[1][e]) : 0.0);
    for (t = 0; t < 2; t++)
    {
        const char *cls = t ? "transient " : "stationary";
        fprintf(stderr, "\n  %s bands  ", cls);
        for (e = 0; e < MMX_EQ_BANDS; e++) fprintf(stderr, " %8lu", psy_debug.n[t][e]);
        fprintf(stderr, "\n    on abs. thr.    ");
        for (e = 0; e < MMX_EQ_BANDS; e++) fprintf(stderr, " %7.1f%%", psy_debug.n[t][e] ? 100.0 * psy_debug.abs_n[t][e] / psy_debug.n[t][e] : 0.0);
        fprintf(stderr, "\n    sub-window hits ");
        for (e = 0; e < MMX_EQ_BANDS; e++) fprintf(stderr, " %7.1f%%", psy_debug.n[t][e] ? 100.0 * psy_debug.sub_n[t][e] / psy_debug.n[t][e] : 0.0);
        fprintf(stderr, "\n    sub-window dB   ");
        for (e = 0; e < MMX_EQ_BANDS; e++) fprintf(stderr, " %8.2f", psy_debug.n[t][e] ? psy_debug.sub_db[t][e] / psy_debug.n[t][e] : 0.0);
        fprintf(stderr, "\n    temporal dB     ");
        for (e = 0; e < MMX_EQ_BANDS; e++) fprintf(stderr, " %8.2f", psy_debug.n[t][e] ? psy_debug.temporal_db[t][e] / psy_debug.n[t][e] : 0.0);
        fprintf(stderr, "\n    cliff dB        ");
        for (e = 0; e < MMX_EQ_BANDS; e++) fprintf(stderr, " %8.2f", psy_debug.n[t][e] ? psy_debug.cliff_db[t][e] / psy_debug.n[t][e] : 0.0);
    }
    fprintf(stderr, "\n  (dB = mean lowering of the threshold by that rule over the bands with signal; the rules apply in this order)\n");
}

/* ---------- per frame precomputation (one worker per frame) ----------
   The frames are independent: each one writes only its own MDCT coefficients,
   its own psychoacoustic rows, its own transient/silence flag and its own
   fingerprint. What the work runs *through* - the codec's window buffer and
   MDCT work area, the three FFT work areas - is per worker, so nothing is
   shared but the read-only input, the band layouts and the constants below.
   The two counters the loop used to accumulate are summed afterwards in frame
   order, so they do not depend on the schedule either. */
typedef struct
{
    MMXCodec *codec;          /* worker 0: the caller's codec; the others: a private copy */
    MMXCodec own;
    int own_live;
    MMXSpectrum spec, sub_spec, pre_spec;
    int spec_live;
    float sub_win[SUB_WIN], sub_coefs[SUB_WIN / 2], frame_amp[MMX_HOP];
    float sub_peak[MMX_MAX_BANDS], sub_sum[MMX_MAX_BANDS];
    float pre_win[MMX_PRE_ATTACK];
    MMXFramePsy sub_psy[5];
} PreWorker;

typedef struct
{
    const MMXAudioBuffer *audio;
    MMXAnalysis *a;
    float *fp;
    PreWorker *w;
    const MMXBandLayout *sub_bands;
    const MMXBandLayout *pre_bands;
    const unsigned int *sub_map;
    const float *weight;
    double pre_attack_db, silent_power;
    unsigned int quality, nb, dim;
    int no_subwin, transient_only, debug;
    double relax_db;
} PreJob;

static void precompute_frame(void *ctx, unsigned long f, unsigned int worker)
{
    PreJob *J = (PreJob *)ctx;
    PreWorker *W = &J->w[worker];
    MMXCodec *codec = W->codec;
    MMXAnalysis *a = J->a;
    const MMXAudioBuffer *audio = J->audio;
    unsigned int c, b, nb = J->nb, dim = J->dim;
    long long start = (long long)f * MMX_HOP - MMX_HOP;
    float *v = J->fp + (size_t)f * FP_DIM;
    double mean = 0.0, total = 0.0;
    int transient = 0;

    for (c = 0; c < a->channels; c++)
    {
        float *coefs = a->coefs + ((size_t)f * a->channels + c) * MMX_HOP;
        MMXFramePsy *p = a->psy + (size_t)f * a->channels + c;
        unsigned int w;
        mmx_codec_window_mdct(codec, audio, start, c, coefs);
        /* masking analysis on a leak-free spectrum of the same window (codec->win holds it) */
        mmx_spectrum_analyze(&W->spec, codec->win, W->frame_amp);
        mmx_psy_analyze(&codec->bands, W->frame_amp, NULL, J->quality, p);
        for (b = 0; b < MMX_MAX_BANDS; b++) { W->sub_peak[b] = 0.0f; W->sub_sum[b] = 0.0f; }
        memcpy(p->frame_thr, p->thr, sizeof(p->frame_thr));
        /* every sub-window that overlaps this frame's window, also the two
           that overlap only partially (their noise still lands there) */
        for (w = 0; w < 5; w++)
        {
            MMXFramePsy *sp = &W->sub_psy[w];
            long long ws = start - SUB_STEP + (long long)w * SUB_STEP;
            long long i;
            for (i = 0; i < SUB_WIN; i++)
            {
                long long pos = ws + i;
                W->sub_win[i] = (pos >= 0 && pos < (long long)audio->frame_count) ? audio->samples[(size_t)pos * audio->channels + c] : 0.0f;
            }
            mmx_spectrum_analyze(&W->sub_spec, W->sub_win, W->sub_coefs);
            mmx_psy_analyze(J->sub_bands, W->sub_coefs, W->sub_win, J->quality, sp);
            if (sp->transient && J->pre_attack_db >= 0.0)
            {
                /* the pre-echo rule from the signal right before the attack */
                long long end = ws + sp->attack_pos;
                for (i = 0; i < MMX_PRE_ATTACK; i++)
                {
                    long long pos = end - MMX_PRE_ATTACK + i;
                    W->pre_win[i] = (pos >= 0 && pos < (long long)audio->frame_count) ? audio->samples[(size_t)pos * audio->channels + c] : 0.0f;
                }
                mmx_psy_pre_attack(J->sub_bands, J->pre_bands, &W->pre_spec, W->pre_win, J->quality, J->pre_attack_db, sp);
            }
            transient |= sp->transient;
            p->transient |= sp->transient;
            if (w >= 1 && w <= 3 && sp->attack_db > p->attack_db)
                p->attack_db = sp->attack_db;
            for (b = 0; b < nb; b++)
            {
                float t = sp->thr[J->sub_map[b]], e = sp->energy[J->sub_map[b]];
                p->sub_thr[w][b] = t < 1e29f ? J->weight[w] * t : t;
                if (w >= 1 && w <= 3)
                {
                    W->sub_sum[b] += e;
                    if (e > W->sub_peak[b]) W->sub_peak[b] = e;
                }
            }
        }
        if (!J->no_subwin && (!J->transient_only || p->transient) && !(J->relax_db > 0.0 && p->attack_db < J->relax_db))
            for (w = 0; w < 5; w++)
                for (b = 0; b < nb; b++)
                    if (p->sub_thr[w][b] < p->thr[b])
                        p->thr[b] = p->sub_thr[w][b];
        for (b = 0; b < nb; b++)
            p->crest[b] = W->sub_sum[b] > 0.0f ? W->sub_peak[b] / (W->sub_sum[b] / 3.0f) : 1.0f;
        for (b = 0; b < nb; b++)
            total += p->energy[b];
        if (J->debug)
            psy_debug_frame(&codec->bands, p, codec->cutoff_band, p->transient);
    }
    a->transient[f] = (unsigned char)transient;
    a->silent[f] = total < J->silent_power;

    /* fingerprint: mean log band energy over channels, mean removed (gain invariant) */
    for (b = 0; b < dim; b++)
    {
        double e = 0.0;
        for (c = 0; c < a->channels; c++)
            e += a->psy[(size_t)f * a->channels + c].energy[b];
        v[b] = (float)log10(e + 1e-6);
        mean += v[b];
    }
    mean /= dim;
    for (b = 0; b < dim; b++)
        v[b] -= (float)mean;
    for (; b < FP_DIM; b++)
        v[b] = 0.0f;
}

static void pre_workers_free(PreWorker *w, unsigned int n)
{
    unsigned int i;
    if (!w)
        return;
    for (i = 0; i < n; i++)
    {
        if (w[i].spec_live)
        {
            mmx_spectrum_free(&w[i].spec);
            mmx_spectrum_free(&w[i].sub_spec);
            mmx_spectrum_free(&w[i].pre_spec);
        }
        if (w[i].own_live)
            mmx_codec_free(&w[i].own);
    }
    free(w);
}

static int precompute(const MMXAudioBuffer *audio, MMXCodec *codec, const MMXAnalysisParams *params, MMXAnalysis *a, float *fp)
{
    unsigned long f, nf = a->frame_count;
    unsigned int b, nb = codec->bands.band_count, dim = nb < FP_DIM ? nb : FP_DIM;
    unsigned int want = psy_debug.on ? 1 : mmx_threads(), live = 0, i;
    MMXBandLayout sub_bands, pre_bands;
    unsigned int sub_map[MMX_MAX_BANDS];
    float weight[5];
    PreWorker *w = NULL;
    PreJob job;
    int plain_weights = getenv("MMX_SUBWIN_EXACT") == NULL;   /* MMX_SUBWIN_EXACT: experiment with the derived weights (see below); the verified derivation needs the 1024->2048 unit factor first, so 2/8 stays the default */
    int rc = -1;

    if (mmx_bands_init(&sub_bands, audio->sample_rate, SUB_WIN / 2) != 0 ||
        mmx_bands_init(&pre_bands, audio->sample_rate, MMX_PRE_ATTACK / 2) != 0)
        return -1;
    w = (PreWorker *)calloc(want, sizeof(PreWorker));
    if (!w)
        return -1;
    for (i = 0; i < want; i++)
    {
        if (i == 0)
            w[i].codec = codec;
        else
        {
            w[i].own_live = 1;   /* mmx_codec_init zeroes first: freeing a partial one is safe */
            if (mmx_codec_init(&w[i].own, codec->sample_rate, codec->channels, codec->quality) != 0)
                break;
            w[i].own.bands = codec->bands;                  /* plain data: the layout this run was set up with */
            w[i].own.short_bands = codec->short_bands;
            w[i].own.cutoff_band = codec->cutoff_band;
            w[i].own.short_cutoff_band = codec->short_cutoff_band;
            w[i].own.tns_k0 = codec->tns_k0;
            w[i].own.tns_k1 = codec->tns_k1;
            w[i].codec = &w[i].own;
        }
        w[i].spec_live = 1;
        if (mmx_spectrum_init(&w[i].spec, MMX_WIN) != 0 ||
            mmx_spectrum_init(&w[i].sub_spec, SUB_WIN) != 0 ||
            mmx_spectrum_init(&w[i].pre_spec, MMX_PRE_ATTACK) != 0)
            break;
        live = i + 1;
    }
    if (live == 0)
        goto done;
    /* Share of the frame's quantization noise that each sub-window sees: the
       noise follows the squared synthesis (KBD) window, the analysis weights
       with its Blackman-Harris window. The centre window sees all of it, the
       two inner halves one half each (the neighbour frame adds the other), the
       outer windows next to nothing (-29 dB with KBD alpha 4). The threshold
       of a sub-window scales with the inverse share (1 / 2 / 887). The former
       weights 2 (inner) / 8 (outer) treated the noise as spread evenly over the
       frame: 3 dB too generous in the centre, 20 dB too strict at the edges. */
    for (b = 0; b < 5; b++)
    {
        long long ws = -SUB_STEP + (long long)b * SUB_STEP, i2;
        double num = 0.0, den = 0.0;
        for (i2 = 0; i2 < SUB_WIN; i2++)
        {
            long long pos = ws + i2;
            double bh = w[0].sub_spec.window[i2] * w[0].sub_spec.window[i2];
            den += bh;
            if (pos >= 0 && pos < (long long)MMX_WIN) num += bh * codec->mdct.window[pos] * codec->mdct.window[pos];
        }
        weight[b] = plain_weights ? MMX_SUB_WEIGHT(b) : (float)(den / num);
    }
    for (b = 0; b < nb; b++)
    {
        unsigned int j;
        sub_map[b] = sub_bands.band_count - 1;
        for (j = 0; j < sub_bands.band_count; j++)
        {
            double lo = sub_bands.band_start[j] * (double)audio->sample_rate / SUB_WIN;
            double hi = sub_bands.band_start[j + 1] * (double)audio->sample_rate / SUB_WIN;
            if (codec->bands.band_hz[b] >= lo && codec->bands.band_hz[b] < hi) { sub_map[b] = j; break; }
        }
    }

    mmx_psy_prewarm();   /* every environment knob of the model resolved before the frames run in parallel */
    memset(&job, 0, sizeof(job));
    job.audio = audio;
    job.a = a;
    job.fp = fp;
    job.w = w;
    job.sub_bands = &sub_bands;
    job.pre_bands = &pre_bands;
    job.sub_map = sub_map;
    job.weight = weight;
    job.pre_attack_db = mmx_psy_pre_attack_db();
    job.silent_power = pow(10.0, SILENT_DBFS / 10.0) * codec->bands.fs_peak_power * 2.0;
    job.quality = params->quality;
    job.nb = nb;
    job.dim = dim;
    job.debug = psy_debug.on;
    job.no_subwin = getenv("MMX_NO_SUBWIN") != NULL;                  /* experiment: cost of the sub-window rule */
    job.transient_only = getenv("MMX_SUBWIN_TRANSIENT_ONLY") != NULL; /* experiment */
    { const char *e = getenv("MMX_SUBWIN_RELAX_DB"); job.relax_db = e ? atof(e) : 0.0; } /* experiment: frames with an attack below this keep the frame threshold */

    /* the pool hands out slot numbers up to mmx_threads() - 1, so it is only
       used when every one of those slots got its scratch */
    if (live == want && want > 1)
        mmx_parallel_for(nf, precompute_frame, &job);
    else
        for (f = 0; f < nf; f++)
            precompute_frame(&job, f, 0);

    /* counted in frame order, never by the worker that happened to do the frame */
    for (f = 0; f < nf; f++)
    {
        if (a->transient[f]) a->transient_frames++;
        if (a->silent[f]) a->silent_frames++;
    }
    /* experiment: frames whose neighbours carry no attack either keep the frame
       threshold (the sub-window rule only where an attack is within a frame) */
    if (getenv("MMX_SUBWIN_GUARDED"))
        for (f = 0; f < nf; f++)
        {
            unsigned int c;
            if (a->transient[f] || (f > 0 && a->transient[f - 1]) || (f + 1 < nf && a->transient[f + 1]))
                continue;
            for (c = 0; c < a->channels; c++)
            {
                MMXFramePsy *p = a->psy + (size_t)f * a->channels + c;
                memcpy(p->thr, p->frame_thr, sizeof(p->thr));
            }
        }
    rc = 0;
done:
    pre_workers_free(w, want);
    return rc;
}

static float fp_distance(const float *a, const float *b)
{
    float d = 0.0f;
    unsigned int i;
    for (i = 0; i < FP_DIM; i++)
    {
        float x = a[i] - b[i];
        d += x * x;
    }
    return d;
}

/* keeps the k smallest distances */
typedef struct { unsigned long frame; float dist; } Hit;

static void hits_insert(Hit *hits, unsigned int k, unsigned long frame, float dist)
{
    unsigned int i;
    if (dist >= hits[k - 1].dist)
        return;
    i = k - 1;
    while (i > 0 && hits[i - 1].dist > dist)
    {
        hits[i] = hits[i - 1];
        i--;
    }
    hits[i].frame = frame;
    hits[i].dist = dist;
}

/* ---------- candidate scoring ---------- */

typedef struct
{
    MMXCodec *codec;
    const MMXAnalysis *a;
    const MMXAudioBuffer *audio;
    const MMXAudioBuffer *sources;   /* what the references read: the input, or a previous pass's decoded output */
    MMXFrameSyntax syn;
    float *target[MMX_MAX_CH];
    float *resid[MMX_MAX_CH];
    int lossless;
    int coded_estimate;
    unsigned int source_bits;
    double cal_audio[2], cal_ref[2]; /* estimate -> real bits per [stationary, transient] (1 = uncalibrated) */
} Scorer;

/* The lossy bit estimate of the analyzer: the fast one, or the one that mirrors the coder. */
static double estimate_bits(const Scorer *sc, const float *x, unsigned long f, unsigned int c)
{
    const MMXFramePsy *p = mmx_analysis_psy(sc->a, f, c);
    if (sc->coded_estimate)
        return mmx_frame_estimate_bits_coded(&sc->codec->bands, x, p->thr, sc->codec->cutoff_band, p->crest);
    return mmx_frame_estimate_bits(&sc->codec->bands, x, p->thr, sc->codec->cutoff_band);
}

/* lossless coder: one broadband gain per channel and source (time-domain predictor). With two
   sources the second gain is fitted on what the first leaves, sequentially - exactly the order the
   lossless coder uses - so a pair is never scored with a prediction the coder cannot make. */
static void broadband_gains(MMXCodec *codec, MMXFrameSyntax *syn, unsigned int n_sources, float *const *target)
{
    unsigned int c, e, s;
    for (c = 0; c < codec->channels; c++)
    {
        double resid[MMX_HOP];
        unsigned long k;
        for (k = 0; k < MMX_HOP; k++)
            resid[k] = target[c][k];
        for (s = 0; s < n_sources; s++)
        {
            double xy = 0.0, yy = 0.0, g;
            unsigned char pol;
            signed char idx;
            for (k = 0; k < MMX_HOP; k++)
            {
                xy += resid[k] * codec->src[s][c][k];
                yy += (double)codec->src[s][c][k] * codec->src[s][c][k];
            }
            g = yy > 1e-20 ? xy / yy : 0.0;
            for (k = 0; k < MMX_HOP; k++)
                resid[k] -= g * codec->src[s][c][k];
            idx = mmx_gain_index(g, &pol);
            syn->polarity[s][c] = pol;
            for (e = 0; e < MMX_EQ_BANDS; e++)
                syn->gain[s][c][e] = idx;
        }
    }
}

static double score_candidate(Scorer *sc, unsigned long f, unsigned int n_sources, const long long *start)
{
    MMXCodec *codec = sc->codec;
    const MMXAnalysis *a = sc->a;
    unsigned int c, s;
    unsigned long k;
    double bits = 0.0;

    for (s = 0; s < n_sources; s++)
        for (c = 0; c < a->channels; c++)
            mmx_codec_window_mdct(codec, sc->sources, start[s], c, codec->src[s][c]);
    for (c = 0; c < a->channels; c++)
        sc->target[c] = (float *)mmx_analysis_coefs(a, f, c);
    mmx_codec_fit_gains(codec, &sc->syn, n_sources, sc->target);
    if (sc->lossless)
        broadband_gains(codec, &sc->syn, n_sources, sc->target);
    mmx_codec_predict(codec, &sc->syn, n_sources);
    for (c = 0; c < a->channels; c++)
    {
        const float *t = sc->target[c];
        for (k = 0; k < MMX_HOP; k++)
            sc->resid[c][k] = t[k] - codec->pred[c][k];
        if (sc->lossless)
            bits += mmx_ll_estimate_bits(sc->resid[c], MMX_HOP, sc->source_bits);
        else
            bits += estimate_bits(sc, sc->resid[c], f, c);
    }
    /* side information: the fast estimate keeps its 8 bits per source (level 5 unchanged), the coded one the measured cost */
    return (sc->coded_estimate ? MMX_GAIN_SIDE_BITS * n_sources * a->channels : 8.0 * n_sources) + bits * sc->cal_ref[a->transient[f] ? 1 : 0];
}

/* Prediction gain per EQ band (dB) of a candidate: target energy / residual energy. */
static void band_gains_of(Scorer *sc, unsigned long f, unsigned int n_sources, const long long *start, float *gain_db)
{
    MMXCodec *codec = sc->codec;
    const MMXAnalysis *a = sc->a;
    double te[MMX_EQ_BANDS], re[MMX_EQ_BANDS];
    unsigned int c, s, b, e;
    unsigned long k;
    for (e = 0; e < MMX_EQ_BANDS; e++) { te[e] = 0.0; re[e] = 0.0; }
    for (s = 0; s < n_sources; s++)
        for (c = 0; c < a->channels; c++)
            mmx_codec_window_mdct(codec, sc->sources, start[s], c, codec->src[s][c]);
    for (c = 0; c < a->channels; c++)
        sc->target[c] = (float *)mmx_analysis_coefs(a, f, c);
    mmx_codec_fit_gains(codec, &sc->syn, n_sources, sc->target);
    if (sc->lossless)
        broadband_gains(codec, &sc->syn, n_sources, sc->target);
    mmx_codec_predict(codec, &sc->syn, n_sources);
    for (c = 0; c < a->channels; c++)
        for (b = 0; b < codec->bands.band_count; b++)
        {
            e = codec->bands.eq_band[b];
            for (k = codec->bands.band_start[b]; k < codec->bands.band_start[b + 1]; k++)
            {
                double t = sc->target[c][k], r = t - codec->pred[c][k];
                te[e] += t * t;
                re[e] += r * r;
            }
        }
    for (e = 0; e < MMX_EQ_BANDS; e++)
        gain_db[e] = (float)(te[e] > 1e-12 && re[e] > 1e-12 ? 10.0 * log10(te[e] / re[e]) : 0.0);
}

static double audio_bits_of(const Scorer *sc, unsigned long f)
{
    double bits = 0.0;
    unsigned int c;
    for (c = 0; c < sc->a->channels; c++)
    {
        if (sc->lossless)
            bits += mmx_ll_estimate_bits(mmx_analysis_coefs(sc->a, f, c), MMX_HOP, sc->source_bits);
        else
            bits += estimate_bits(sc, mmx_analysis_coefs(sc->a, f, c), f, c);
    }
    return bits * sc->cal_audio[sc->a->transient[f] ? 1 : 0];
}

/* Local re-alignment of a continued lineage: tests offsets guess-DRIFT..guess+DRIFT
   at full rate and returns the best (highest normalized correlation). */
static long long refine_offset(const MMXAudioBuffer *audio, long long target_start, long long guess, unsigned int radius)
{
    long long d, best = guess;
    double best_ncc = -2.0;
    unsigned long n = MMX_WIN;
    const float *target;
    if (target_start < 0 || target_start + (long long)n > (long long)audio->frame_count)
        return guess;
    target = audio->samples + (size_t)target_start * audio->channels;
    for (d = -(long long)radius; d <= (long long)radius; d++)
    {
        long long s = guess + d;
        double ncc, gain;
        if (s < 0 || s + (long long)n > target_start)
            continue;
        mmx_similarity_measure(target, audio->samples + (size_t)s * audio->channels, n, audio->channels, &ncc, &gain);
        if (ncc > best_ncc)
        {
            best_ncc = ncc;
            best = s;
        }
    }
    return best;
}

/* sample-exact alignment of the target window against the file around `guess` */
static int align(const MMXAudioBuffer *audio, long long target_start, long long guess, long long *aligned)
{
    MMXMatch m;
    long long limit = target_start;            /* source window must end before the target window */
    const float *target;
    float *padded = NULL;
    int rc;

    if (target_start < 0)
        return -1;                              /* first frame never references */
    if (guess < 0) guess = 0;
    target = audio->samples + (size_t)target_start * audio->channels;
    if (target_start + MMX_WIN > (long long)audio->frame_count)
    {
        /* last frame: pad with zeros */
        unsigned long avail = (unsigned long)((long long)audio->frame_count - target_start);
        padded = (float *)calloc((size_t)MMX_WIN * audio->channels, sizeof(float));
        if (!padded)
            return -1;
        memcpy(padded, target, sizeof(float) * avail * audio->channels);
        target = padded;
    }
    rc = mmx_similarity_best_match(target, MMX_WIN, audio->channels, audio->samples, audio->frame_count,
                                   (unsigned long long)guess, MMX_HOP / 2, (unsigned long long)limit, &m);
    free(padded);
    if (rc != 0)
        return -1;
    *aligned = (long long)m.source_frame;
    return 0;
}

#define COST_UNKNOWN (-1.0f)   /* scored later; never replaces a known cost */

static void add_candidate(Candidate *list, unsigned int *count, unsigned int n_sources, const long long *start, float cost)
{
    unsigned int i;
    for (i = 0; i < *count; i++)
    {
        if (list[i].n_sources == n_sources && list[i].start[0] == start[0] &&
            (n_sources < 2 || list[i].start[1] == start[1]))
        {
            if (cost >= 0.0f && (list[i].cost < 0.0f || cost < list[i].cost)) list[i].cost = cost;
            return;
        }
    }
    if (*count < MAX_CANDS)
    {
        list[*count].n_sources = (unsigned char)n_sources;
        list[*count].start[0] = start[0];
        list[*count].start[1] = n_sources > 1 ? start[1] : 0;
        list[*count].cost = cost;
        (*count)++;
    }
}


/* ---------- worker copies of the scratch the search runs through ----------
   score_candidate() is a pure function of (frame, source starts) given the
   analysis and the source signal, and align() is a pure function of (target
   window, guess) on the read-only input. Neither reads anything an earlier
   candidate wrote. What they compute *through* is scratch: the codec's window
   buffer, its MDCT work area and its source and prediction arrays, and the Scorer's frame
   syntax and residual buffers. Each worker slot therefore gets its own copy of
   exactly that scratch, with the band layout and the transform tables taken
   from the run's own codec, so every slot computes the same numbers as the
   single-threaded path. Slot 0 is the caller's own Scorer, so MMX_THREADS=1
   allocates nothing and runs the original code. */
typedef struct
{
    MMXCodec codec;
    Scorer sc;
    int codec_live;
} SearchWorker;

typedef struct
{
    unsigned int n;                  /* usable slots; 1 = the single-threaded path */
    unsigned int slots;              /* entries allocated in `extra` */
    int parallel;                    /* every slot the pool can hand out exists */
    Scorer *sc[MMX_THREADS_MAX];
    SearchWorker *extra;             /* [slots] */
} Workers;

static void workers_free(Workers *W)
{
    unsigned int i, c;
    if (W->extra)
    {
        for (i = 0; i < W->slots; i++)     /* calloc'd: an untouched slot frees as nothing */
        {
            SearchWorker *w = &W->extra[i];
            mmx_frame_syntax_free(&w->sc.syn);
            for (c = 0; c < MMX_MAX_CH; c++)
                free(w->sc.resid[c]);
            if (w->codec_live)
                mmx_codec_free(&w->codec);
        }
        free(W->extra);
    }
    memset(W, 0, sizeof(*W));
    W->n = 1;
}

static void workers_init(Workers *W, Scorer *sc0)
{
    unsigned int want = mmx_threads(), i, c;
    const MMXCodec *s = sc0->codec;

    memset(W, 0, sizeof(*W));
    W->n = 1;
    W->sc[0] = sc0;
    if (want < 2 || want > MMX_THREADS_MAX)
        return;
    W->extra = (SearchWorker *)calloc(want - 1, sizeof(SearchWorker));
    if (!W->extra)
        return;
    W->slots = want - 1;
    for (i = 0; i + 1 < want; i++)
    {
        SearchWorker *w = &W->extra[i];
        w->codec_live = 1;   /* mmx_codec_init zeroes first: freeing a partial one is safe */
        if (mmx_codec_init(&w->codec, s->sample_rate, s->channels, s->quality) != 0)
            break;
        w->codec.bands = s->bands;                 /* plain data: the layout this run was set up with */
        w->codec.short_bands = s->short_bands;
        w->codec.cutoff_band = s->cutoff_band;
        w->codec.short_cutoff_band = s->short_cutoff_band;
        w->codec.tns_k0 = s->tns_k0;
        w->codec.tns_k1 = s->tns_k1;
        w->sc = *sc0;                              /* the read-only settings */
        w->sc.codec = &w->codec;
        memset(&w->sc.syn, 0, sizeof(w->sc.syn));
        memset(w->sc.resid, 0, sizeof(w->sc.resid));
        memset(w->sc.target, 0, sizeof(w->sc.target));
        if (mmx_frame_syntax_init(&w->sc.syn, s->channels, MMX_HOP) != 0)
            break;
        for (c = 0; c < s->channels; c++)
            if ((w->sc.resid[c] = (float *)malloc(sizeof(float) * MMX_HOP)) == NULL)
                break;
        if (c < s->channels)
            break;
        W->sc[i + 1] = &w->sc;
        W->n = i + 2;
    }
    /* the pool hands out slot numbers up to mmx_threads() - 1: use it only when
       every one of them has its scratch, otherwise stay single-threaded */
    W->parallel = W->n == want;
}

/* ---------- the two parallel steps of one frame ---------- */

typedef struct
{
    Workers *W;
    Candidate *list;
    unsigned long f;
} ScoreJob;

static void score_task(void *ctx, unsigned long i, unsigned int worker)
{
    ScoreJob *J = (ScoreJob *)ctx;
    Candidate *c = &J->list[i];
    if (c->cost < 0.0f)
        c->cost = (float)score_candidate(J->W->sc[worker], J->f, c->n_sources, c->start);
}

/* Scores every candidate whose cost is not known yet. Each task writes only its
   own candidate's cost, so the list is the same whatever order they finish in. */
static void score_pending(Workers *W, Candidate *list, unsigned int n, unsigned long f)
{
    ScoreJob J;
    unsigned int i;
    if (!W->parallel || n < 2)
    {
        for (i = 0; i < n; i++)
            if (list[i].cost < 0.0f)
                list[i].cost = (float)score_candidate(W->sc[0], f, list[i].n_sources, list[i].start);
        return;
    }
    J.W = W;
    J.list = list;
    J.f = f;
    mmx_parallel_for(n, score_task, &J);
}

typedef struct
{
    const MMXAudioBuffer *audio;
    long long target_start;
    const long long *guess;
    long long *out;
} RefineJob;

static void refine_task(void *ctx, unsigned long i, unsigned int worker)
{
    RefineJob *J = (RefineJob *)ctx;
    (void)worker;
    J->out[i] = refine_offset(J->audio, J->target_start, J->guess[i], drift_samples());
}

/* The drift-corrected variants of the continued lineages. Each one only reads
   the input signal around its own guess, so they are computed together; the
   loop below then uses them in its own order. Guesses the loop will not reach
   are computed too (the list may fill up first) - the result is unused, never
   different. */
static void refine_batch(const Workers *W, const MMXAudioBuffer *audio, long long target_start,
                         const long long *guess, long long *out, unsigned int m)
{
    RefineJob J;
    unsigned int i;
    if (!W->parallel || m < 2)
    {
        for (i = 0; i < m; i++)
            out[i] = refine_offset(audio, target_start, guess[i], drift_samples());
        return;
    }
    J.audio = audio;
    J.target_start = target_start;
    J.guess = guess;
    J.out = out;
    mmx_parallel_for(m, refine_task, &J);
}

/* ---------- prediction gains of the finished plan ---------- */

typedef struct
{
    Workers *W;
    MMXAnalysis *a;
} GainJob;

static void gain_task(void *ctx, unsigned long f, unsigned int worker)
{
    GainJob *J = (GainJob *)ctx;
    MMXFramePlan *p = &J->a->plan[f];
    if (p->n_sources)
        band_gains_of(J->W->sc[worker], f, p->n_sources, p->src_start, p->band_gain_db);
    else
        memset(p->band_gain_db, 0, sizeof(p->band_gain_db));
}

/* ---------- the alignment probes of the whole file, in one pass ----------
   The three candidate sources that open a frame - the k nearest fingerprints,
   the onset pool and the section-level repeats - read nothing the timeline
   loop decides: the fingerprints, the silence and transient flags and the
   section fingerprints are all finished before the loop starts, and align()
   only reads the input signal. So every frame's probes are computed once, over
   all frames at once, and the loop just walks the finished list in the same
   order it used to produce it. This is where most of the analysis time sits
   (alignment is a full-rate correlation per probe), and it is the one part of
   the search that is not chained to the frame before it. */
/* The list must hold every probe the three groups can produce at the highest
   level, because a probe that did not fit would silently be a candidate the
   single-threaded coder had and this one has not - a different file, not a
   slower one. The bound is therefore derived from the same expressions
   effort_for_level uses, the compile-time check below ties it to the k table,
   and mmx_analyzer_run refuses to run if the numbers ever grow past it. */
#define EF_K_MAX 48                                  /* ks[MMX_ANALYSIS_MAX] in effort_for_level */
#define EF_ONSET_K_MAX (2 + MMX_ANALYSIS_MAX / 2)
#define EF_SECTION_K_MAX (2 + MMX_ANALYSIS_MAX / 3)
#define PROBE_MAX (EF_K_MAX + EF_ONSET_K_MAX + EF_SECTION_K_MAX)
typedef char probe_max_covers_top_level[EF_K_MAX + EF_ONSET_K_MAX + EF_SECTION_K_MAX <= PROBE_MAX ? 1 : -1];

typedef struct
{
    unsigned char n;                 /* probes of this frame, in group order */
    unsigned char fp_n;              /* how many of them come from the fingerprint group */
    unsigned char ok[PROBE_MAX];     /* the alignment found a source window */
    long long start[PROBE_MAX];
} FrameProbes;

typedef struct
{
    const MMXAudioBuffer *audio;
    const MMXAnalysis *a;
    const float *fp;
    const float *section_fp;
    unsigned long section_windows;
    Effort ef;
    FrameProbes *out;
    const Scorer *sc;                /* read only: audio_bits_of writes nothing through it */
    double *audio_bits;              /* [nf] the frame's cost as plain audio */
} ProbeJob;

static void probe_frame(void *ctx, unsigned long f, unsigned int worker)
{
    ProbeJob *J = (ProbeJob *)ctx;
    const MMXAnalysis *a = J->a;
    const Effort *ef = &J->ef;
    FrameProbes *P = &J->out[f];
    long long target_start = (long long)f * MMX_HOP - MMX_HOP;
    const float *v = J->fp + (size_t)f * FP_DIM;
    Hit hits[MAX_CANDS + 1];
    unsigned long j;
    unsigned int i, n = 0;

    (void)worker;
    /* the cost of coding the frame as plain audio: a pure function of the
       frame's coefficients and thresholds, so it belongs in this pass too */
    J->audio_bits[f] = audio_bits_of(J->sc, f);
    P->n = 0;
    P->fp_n = 0;
    if (a->silent[f] || target_start < MMX_WIN)
        return;

    /* 1. every earlier frame of the file, ranked by fingerprint distance */
    for (i = 0; i <= ef->k; i++) { hits[i].frame = 0; hits[i].dist = 1e30f; }
    for (j = 0; j + 2 < f; j++)
    {
        if (a->silent[j])
            continue;
        if ((long long)j * MMX_HOP - MMX_HOP + MMX_WIN > target_start)
            break;
        hits_insert(hits, ef->k, j, fp_distance(v, J->fp + (size_t)j * FP_DIM));
    }
    for (i = 0; i < ef->k && hits[i].dist < 1e29f && n < PROBE_MAX; i++)
    {
        long long aligned;
        P->ok[n] = align(J->audio, target_start, (long long)hits[i].frame * MMX_HOP - MMX_HOP, &aligned) == 0;
        P->start[n] = P->ok[n] ? aligned : 0;
        n++;
    }
    P->fp_n = (unsigned char)n;

    /* 2. onset pool: transient frames matched against earlier transient frames */
    if (ef->onset_k && a->transient[f])
    {
        for (i = 0; i <= ef->onset_k; i++) { hits[i].frame = 0; hits[i].dist = 1e30f; }
        for (j = 0; j + 2 < f; j++)
        {
            if (!a->transient[j] || a->silent[j])
                continue;
            if ((long long)j * MMX_HOP - MMX_HOP + MMX_WIN > target_start)
                break;
            hits_insert(hits, ef->onset_k, j, fp_distance(v, J->fp + (size_t)j * FP_DIM));
        }
        for (i = 0; i < ef->onset_k && hits[i].dist < 1e29f && n < PROBE_MAX; i++)
        {
            long long aligned;
            P->ok[n] = align(J->audio, target_start, (long long)hits[i].frame * MMX_HOP - MMX_HOP, &aligned) == 0;
            P->start[n] = P->ok[n] ? aligned : 0;
            n++;
        }
    }

    /* 3. section-level repeats: similar earlier windows of ~0.75 s, same relative offset */
    if (ef->section_k && J->section_windows && f >= SECTION_FRAMES / 2)
    {
        unsigned long w = (f - SECTION_FRAMES / 2) / SECTION_HOP, u;
        if (w < J->section_windows)
        {
            for (i = 0; i <= ef->section_k; i++) { hits[i].frame = 0; hits[i].dist = 1e30f; }
            for (u = 0; u < w; u++)
            {
                long long shift = (long long)(w - u) * SECTION_HOP * MMX_HOP;
                if (target_start - shift + MMX_WIN > target_start && shift < MMX_WIN)
                    continue;
                hits_insert(hits, ef->section_k, u, fp_distance(J->section_fp + w * FP_DIM, J->section_fp + u * FP_DIM));
            }
            for (i = 0; i < ef->section_k && hits[i].dist < 1e29f && n < PROBE_MAX; i++)
            {
                long long shift = (long long)(w - hits[i].frame) * SECTION_HOP * MMX_HOP, aligned;
                P->ok[n] = align(J->audio, target_start, target_start - shift, &aligned) == 0;
                P->start[n] = P->ok[n] ? aligned : 0;
                n++;
            }
        }
    }
    P->n = (unsigned char)n;
}

static int cand_cost_cmp(const void *x, const void *y)
{
    float a = ((const Candidate *)x)->cost, b = ((const Candidate *)y)->cost;
    return (a > b) - (a < b);
}

/* ---------- dynamic program over the timeline ---------- */

static int lineage_continues(const Candidate *prev, const Candidate *cur)
{
    unsigned int s;
    if (prev->n_sources != cur->n_sources)
        return 0;
    for (s = 0; s < cur->n_sources; s++)
        if (prev->start[s] + MMX_HOP != cur->start[s])
            return 0;
    return 1;
}

/* ---------- fixed-lag Viterbi over the timeline ----------
   States per frame: 0 = plain audio, 1..n = candidates. A frame is committed
   `lag` frames later (traceback from the best current state); the committed
   plan feeds the reference graph. With the lag of 2 (levels 1-6) every
   candidate's chain depth is exact when it is evaluated (sources end at least
   two frames before the target). A longer lag (levels 7-9) lets the path
   change its mind about a block for longer before it is final; the frames
   between the commit point and the current frame then carry the depth they
   have on the current best path (tentative overlay, refreshed before every
   frame), the commit re-checks against the final graph and falls back to
   AUDIO when the overlay was wrong (counted, rare). After a commit the other
   states of that frame are pruned and the frames in between are recomputed,
   which keeps the path consistent; the continuation of a candidate is looked
   up in a table (cont) so a step costs O(candidates), not O(candidates^2). */

typedef struct
{
    const MMXAudioBuffer *audio;
    MMXAnalysis *a;
    Candidate *cands;
    unsigned int *cand_count;
    int *cont;                    /* [f * MAX_CANDS + i]: candidate of f-1 that continues into i, or -1 */
    double *best;                 /* [f * DP_STRIDE + state] */
    int *from;
    unsigned char *sdepth;        /* [f * DP_STRIDE + state]: chain depth of the state */
    unsigned char *tent;          /* [f]: depth on the current best path (uncommitted frames) */
    int *tent_state;              /* [f]: state on the current best path (0 = AUDIO) */
    unsigned int lag;
    unsigned char max_depth;
    MMXReferenceGraph *graph;
    unsigned long seg_frames;     /* > 0: sources only inside the frame's own segment (MMXAnalysisParams) */
    unsigned long committed;      /* frames < committed are final */
    unsigned long fallbacks;      /* commits whose depth failed against the final graph */
    double total;
} Dp;

static int dp_best_state(const Dp *dp, unsigned long f)
{
    const double *bf = dp->best + (size_t)f * DP_STRIDE;
    unsigned int i, n = dp->cand_count[f];
    int state = 0;
    double m = bf[0];
    for (i = 1; i <= n; i++)
        if (bf[i] < m) { m = bf[i]; state = (int)i; }
    return state;
}

/* Depth and state of every uncommitted frame up to f along the best path ending at f. */
static void dp_tentative(Dp *dp, unsigned long f)
{
    int state;
    unsigned long k = f;
    if (dp->lag <= DP_LAG_EXACT || f < dp->committed)
        return;
    state = dp_best_state(dp, f);
    for (;;)
    {
        dp->tent[k] = dp->sdepth[(size_t)k * DP_STRIDE + state];
        dp->tent_state[k] = state;
        if (k == dp->committed || k == 0)
            break;
        state = dp->from[(size_t)k * DP_STRIDE + state];
        k--;
    }
}

/* Source window of frame g when it is committed, or lies on the tentative best
   path (long lag), with exactly one source. */
static int planned_source(const Dp *dp, unsigned long g, long long *src)
{
    if (g < dp->committed)
    {
        if (dp->a->plan[g].n_sources != 1)
            return 0;
        *src = dp->a->plan[g].src_start[0];
        return 1;
    }
    if (dp->lag > DP_LAG_EXACT && dp->tent_state[g] > 0)
    {
        const Candidate *cd = dp->cands + (size_t)g * MAX_CANDS + (dp->tent_state[g] - 1);
        if (cd->n_sources != 1)
            return 0;
        *src = cd->start[0];
        return 1;
    }
    return 0;
}

static void dp_step(Dp *dp, unsigned long f)
{
    unsigned int n = dp->cand_count[f], i, prev_n = f ? dp->cand_count[f - 1] : 0;
    double *bf = dp->best + (size_t)f * DP_STRIDE;
    int *ff = dp->from + (size_t)f * DP_STRIDE;
    unsigned char *sd = dp->sdepth + (size_t)f * DP_STRIDE;
    const int *cont = dp->cont + (size_t)f * MAX_CANDS;
    const Candidate *list = dp->cands + (size_t)f * MAX_CANDS;
    const double *pb = f ? dp->best + (size_t)(f - 1) * DP_STRIDE : NULL;
    const unsigned char *tent = dp->lag > DP_LAG_EXACT ? dp->tent : NULL;
    double audio_bits = dp->a->plan[f].audio_bits, prev_min = 0.0;
    int prev_min_i = 0;
    long long target_start = (long long)f * MMX_HOP - MMX_HOP;

    if (f)
    {
        dp_tentative(dp, f - 1);
        prev_min = pb[0];
        for (i = 1; i <= prev_n; i++)
            if (pb[i] < prev_min) { prev_min = pb[i]; prev_min_i = (int)i; }
    }
    bf[0] = audio_bits + (f ? (pb[0] <= prev_min + SWITCH_BITS ? pb[0] : prev_min + SWITCH_BITS) : 0.0);
    ff[0] = f ? (pb[0] <= prev_min + SWITCH_BITS ? 0 : prev_min_i) : 0;
    sd[0] = 0;

    for (i = 0; i < n; i++)
    {
        double cost = list[i].cost, enter = prev_min + SWITCH_BITS;
        int enter_from = prev_min_i;
        int d = mmx_refgraph_check_tentative(dp->graph, target_start, list[i].n_sources, list[i].start, tent, dp->committed);
        if (d >= 0 && dp->seg_frames)
        {   /* --pld: every segment decodes on its own - no source before the segment's first sample */
            const long long seg0 = (long long)(f / dp->seg_frames * dp->seg_frames) * MMX_HOP - MMX_HOP;
            unsigned int k2;
            for (k2 = 0; k2 < list[i].n_sources && k2 < 2; k2++)
                if (seg0 > 0 && list[i].start[k2] - 256 < seg0) d = -1;
        }
        if (d < 0)
        {
            bf[i + 1] = 1e30;
            ff[i + 1] = 0;
            sd[i + 1] = 0;
            continue;
        }
        sd[i + 1] = (unsigned char)d;
        if (cost > audio_bits + CONTINUE_AUDIO_EXTRA)
            cost = audio_bits + CONTINUE_AUDIO_EXTRA;   /* frame coded without prediction inside the block */
        cost *= 1.0 + DEPTH_PENALTY * (double)(d - 1);
        if (f && cont[i] >= 0 && pb[cont[i] + 1] < enter)
        {
            enter = pb[cont[i] + 1];
            enter_from = cont[i] + 1;
        }
        bf[i + 1] = cost + (f ? enter : SWITCH_BITS);
        ff[i + 1] = f ? enter_from : 0;
    }
}

/* Writes the plan entry of frame g from a state and records it in the graph. */
static void dp_commit_frame(Dp *dp, unsigned long g, int state)
{
    MMXFramePlan *p = &dp->a->plan[g];
    long long target_start = (long long)g * MMX_HOP - MMX_HOP;
    int d = 0;
    if (state == 0)
    {
        p->n_sources = 0;
        p->ref_bits = p->audio_bits;
    }
    else
    {
        const Candidate *cd = dp->cands + (size_t)g * MAX_CANDS + (state - 1);
        p->n_sources = cd->n_sources;
        p->src_start[0] = cd->start[0];
        p->src_start[1] = cd->start[1];
        p->ref_bits = cd->cost;
        d = mmx_refgraph_check(dp->graph, target_start, p->n_sources, p->src_start);
        if (d < 0) { p->n_sources = 0; p->ref_bits = p->audio_bits; d = 0; dp->fallbacks++; }
    }
    p->depth = (unsigned char)d;
    mmx_refgraph_set(dp->graph, g, (unsigned char)d, p->n_sources, p->src_start);
    /* the runner-up (another lineage with the next-cheapest path through this
       frame) is kept for the encoder's closed loop, then the other states of
       the committed frame are pruned */
    {
        double *bg = dp->best + (size_t)g * DP_STRIDE, m = 1e30;
        unsigned int i, n = dp->cand_count[g];
        int alt = -1;
        p->alt_n_sources = 0;
        for (i = 1; i <= n; i++)
            if ((int)i != state && bg[i] < m) { m = bg[i]; alt = (int)i; }
        if (state != 0 && alt > 0 && m < 1e30)
        {
            const Candidate *cd = dp->cands + (size_t)g * MAX_CANDS + (alt - 1);
            p->alt_n_sources = cd->n_sources;
            p->alt_src_start[0] = cd->start[0];
            p->alt_src_start[1] = cd->start[1];
            p->alt_bits = (float)m;
        }
        for (i = 0; i <= n; i++)
            if ((int)i != state) bg[i] = 1e30;
    }
    dp->committed = g + 1;
}

/* Called after dp_step(f): commits frame f - lag and recomputes the frames after it. */
static void dp_commit_lagged(Dp *dp, unsigned long f)
{
    unsigned long g, k;
    int state;
    if (f < dp->lag)
        return;
    g = f - dp->lag;
    if (g < dp->committed)
        return;
    state = dp_best_state(dp, f);
    for (k = f; k > g; k--)
        state = dp->from[(size_t)k * DP_STRIDE + state];
    dp_commit_frame(dp, g, state);
    for (k = g + 1; k <= f; k++)
        dp_step(dp, k);
}

static void dp_commit_tail(Dp *dp, unsigned long nf)
{
    unsigned long g;
    int state;
    if (nf == 0)
        return;
    state = dp_best_state(dp, nf - 1);
    dp->total = dp->best[(size_t)(nf - 1) * DP_STRIDE + state];
    /* commit the remaining uncommitted frames along the best path, oldest first */
    {
        int *path = (int *)malloc(sizeof(int) * (nf - dp->committed + 1));
        unsigned long i = 0, k;
        if (!path)
            return;
        for (k = nf - 1; ; k--)
        {
            path[i++] = state;
            if (k == dp->committed) break;
            state = dp->from[(size_t)k * DP_STRIDE + state];
        }
        for (g = dp->committed; g < nf; g++)
            dp_commit_frame(dp, g, path[nf - 1 - g]);
        free(path);
    }
}

int mmx_analyzer_run(const MMXAudioBuffer *audio, MMXCodec *codec, const MMXAnalysisParams *params, MMXAnalysis *a)
{
    Effort ef;
    unsigned long nf, f;
    unsigned int c;
    float *fp = NULL, *section_fp = NULL;
    Candidate *cands = NULL;          /* [f * MAX_CANDS] */
    unsigned int *cand_count = NULL;  /* [f] */
    int *cont = NULL;                 /* [f * MAX_CANDS] continuation table of the DP */
    double *best = NULL;              /* DP cost [f * MAX_CANDS] */
    int *from = NULL;                 /* backpointer */
    unsigned char *sdepth = NULL;     /* [f * DP_STRIDE] chain depth per DP state */
    unsigned char *tent = NULL;       /* [f] tentative depth (long lag) */
    int *tent_state = NULL;           /* [f] tentative state (long lag) */
    Scorer sc;
    Workers W;
    FrameProbes *probes = NULL;
    double *abits = NULL;
    MMXReferenceGraph graph;
    Dp dp;
    double t0 = mmx_wall_seconds();
    unsigned long sections_count = 0, section_windows = 0, full_frames = 0;
    int rc = -1;

    memset(a, 0, sizeof(*a));
    a->thr_scale = 1.0;
    memset(&sc, 0, sizeof(sc));
    memset(&W, 0, sizeof(W));
    W.n = 1;
    W.sc[0] = &sc;
    memset(&graph, 0, sizeof(graph));
    effort_for_level(params->level, &ef);
    a->coded_estimate = ef.coded_estimate && !params->lossless;
    if (ef.k + ef.onset_k + ef.section_k > PROBE_MAX)
    {   /* cannot happen with the effort table above; if it ever does, stop
           rather than quietly drop a candidate the old coder would have had */
        mmx_error("Analyzer: %u alignment probes per frame do not fit the list (%d)",
                  ef.k + ef.onset_k + ef.section_k, PROBE_MAX);
        goto done;
    }

    nf = mmx_codec_frame_count(audio->frame_count);
    a->frame_count = nf;
    a->channels = audio->channels;
    a->coefs = (float *)malloc(sizeof(float) * (size_t)nf * audio->channels * MMX_HOP);
    a->psy = (MMXFramePsy *)calloc((size_t)nf * audio->channels, sizeof(MMXFramePsy));
    a->transient = (unsigned char *)calloc(nf, 1);
    a->silent = (unsigned char *)calloc(nf, 1);
    a->plan = (MMXFramePlan *)calloc(nf, sizeof(MMXFramePlan));
    fp = (float *)calloc((size_t)nf * FP_DIM, sizeof(float));
    cands = (Candidate *)calloc((size_t)nf * MAX_CANDS, sizeof(Candidate));
    cand_count = (unsigned int *)calloc(nf, sizeof(unsigned int));
    cont = (int *)malloc(sizeof(int) * (size_t)nf * MAX_CANDS);
    best = (double *)malloc(sizeof(double) * (size_t)nf * DP_STRIDE);
    from = (int *)malloc(sizeof(int) * (size_t)nf * DP_STRIDE);
    sdepth = (unsigned char *)calloc((size_t)nf * DP_STRIDE, 1);
    tent = (unsigned char *)calloc(nf, 1);
    tent_state = (int *)calloc(nf, sizeof(int));
    if (!a->coefs || !a->psy || !a->transient || !a->silent || !a->plan || !fp || !cands || !cand_count || !cont ||
        !best || !from || !sdepth || !tent || !tent_state)
    {
        mmx_error("Out of memory in analyzer (%lu frames)", nf);
        goto done;
    }
    if (mmx_frame_syntax_init(&sc.syn, audio->channels, MMX_HOP) != 0)
        goto done;
    for (c = 0; c < audio->channels; c++)
    {
        sc.resid[c] = (float *)malloc(sizeof(float) * MMX_HOP);
        if (!sc.resid[c])
            goto done;
    }
    sc.codec = codec;
    sc.a = a;
    sc.audio = audio;
    sc.sources = params->sources ? params->sources : audio;
    sc.lossless = params->lossless;
    sc.coded_estimate = ef.coded_estimate;
    sc.source_bits = params->source_bits;
    for (c = 0; c < 2; c++)
    {
        sc.cal_audio[c] = params->cal_audio[c] > 0.0 ? params->cal_audio[c] : 1.0;
        sc.cal_ref[c] = params->cal_ref[c] > 0.0 ? params->cal_ref[c] : 1.0;
    }
    /* The lossless coder can take two sources since revision 7 (a second broadband gain fitted on
       what the first leaves, a second side stage), but it is OFF unless MMX_LL_PAIRS=1. Measured
       lossless, analysis 7, against the single-source files: title A -0.116 %, title C -0.038 %,
       title B -0.058 %, title D +0.207 % - a net +0.011 % over the four. The pair is used on thousands of frames and pays exactly its own cost; on
       title D the estimate picks pairs the coder does not redeem. Played material never gets a
       pair and stays byte-identical either way. */
    if (params->lossless)
    {
        const char *e = getenv("MMX_LL_PAIRS");
        if (!(e && atoi(e) != 0))
            ef.pairs = 0;
    }
    if (mmx_refgraph_init(&graph, nf, params->max_depth) != 0)
        goto done;
    workers_init(&W, &sc);

    mmx_info("Analyzing structure (%lu frames, level %u)...", nf, params->level);
    memset(&psy_debug, 0, sizeof(psy_debug));
    psy_debug.on = getenv("MMX_DEBUG_PSY") != NULL;
    precompute(audio, codec, params, a, fp);
    for (c = 0; c < a->channels; c++)
    {
        /* temporal smoothing needs the raw thresholds of the neighbours: work on a copy */
        MMXFramePsy *raw = (MMXFramePsy *)malloc(sizeof(MMXFramePsy) * nf);
        if (!raw)
            goto done;
        for (f = 0; f < nf; f++)
            raw[f] = a->psy[(size_t)f * a->channels + c];
        /* The neighbour-frame rule (mmx_psy_temporal) is off: the sub-windows 1
           and 3 already judge the frame's noise in the overlap with the exact
           signal there, and the neighbour's own threshold also counts its
           sub-windows outside this frame's window. Measured on title A it
           costs 3.5 dB of threshold on stationary frames (13 % of the bits) for
           no gain of the yardstick (experiment: MMX_TEMPORAL=1 switches it on). */
        for (f = 0; f < nf && !getenv("MMX_NO_TEMPORAL"); f++)
        {
            MMXFramePsy *p = &a->psy[(size_t)f * a->channels + c];
            mmx_psy_temporal(f ? &raw[f - 1] : NULL, p, f + 1 < nf ? &raw[f + 1] : NULL);
            if (psy_debug.on)
                psy_debug_rule(&codec->bands, p, raw[f].thr, codec->cutoff_band, psy_debug.temporal_db[p->transient ? 1 : 0]);
        }
        free(raw);
        /* the first and last frame are cut off at the file boundary: their noise
           would end in a hard edge, so they are coded 20 dB cleaner */
        {
            unsigned int b;
            MMXFramePsy *first = &a->psy[c], *last = &a->psy[(size_t)(nf - 1) * a->channels + c];
            for (b = 0; b < MMX_MAX_BANDS; b++)
            {
                if (first->thr[b] < 1e29f) first->thr[b] *= 0.01f;
                if (last->thr[b] < 1e29f) last->thr[b] *= 0.01f;
            }
        }
        if (!getenv("MMX_NO_CLIFF")) /* experiment: cost of the neighbour-band rule */
            for (f = 0; f < nf; f++)
            {
                MMXFramePsy *p = &a->psy[(size_t)f * a->channels + c];
                float before[MMX_MAX_BANDS];
                if (psy_debug.on) memcpy(before, p->thr, sizeof(before));
                mmx_psy_limit_cliffs(p, codec->bands.band_count);
                if (psy_debug.on)
                    psy_debug_rule(&codec->bands, p, before, codec->cutoff_band, psy_debug.cliff_db[p->transient ? 1 : 0]);
            }
    }
    if (psy_debug.on)
        psy_debug_print();

    /* section-level fingerprints: mean over SECTION_FRAMES frames every SECTION_HOP frames */
    section_windows = nf > SECTION_FRAMES ? (nf - SECTION_FRAMES) / SECTION_HOP + 1 : 0;
    if (ef.section_k && section_windows)
    {
        unsigned long w;
        section_fp = (float *)calloc((size_t)section_windows * FP_DIM, sizeof(float));
        if (!section_fp)
            goto done;
        for (w = 0; w < section_windows; w++)
        {
            unsigned int i, d;
            for (i = 0; i < SECTION_FRAMES; i++)
                for (d = 0; d < FP_DIM; d++)
                    section_fp[w * FP_DIM + d] += fp[(w * SECTION_HOP + i) * FP_DIM + d] / SECTION_FRAMES;
        }
        /* novelty-based section count for the report */
        {
            double prev_d = 0.0;
            for (w = 1; w < section_windows; w++)
            {
                double d = fp_distance(section_fp + w * FP_DIM, section_fp + (w - 1) * FP_DIM);
                if (d > 4.0 && d > 2.0 * prev_d)
                    sections_count++;
                prev_d = d;
            }
        }
    }
    a->sections = sections_count + 1;

    memset(&dp, 0, sizeof(dp));
    dp.audio = audio;
    dp.a = a;
    dp.cands = cands;
    dp.cand_count = cand_count;
    dp.cont = cont;
    dp.best = best;
    dp.from = from;
    dp.sdepth = sdepth;
    dp.tent = tent;
    dp.tent_state = tent_state;
    dp.lag = ef.lag;
    dp.seg_frames = params->seg_frames;
    dp.max_depth = params->max_depth;
    dp.graph = &graph;

    /* every frame's alignment probes at once: nothing here depends on the plan */
    probes = (FrameProbes *)calloc(nf, sizeof(FrameProbes));
    abits = (double *)calloc(nf, sizeof(double));
    if (!probes || !abits)
    {
        mmx_error("Out of memory in analyzer (%lu frames)", nf);
        goto done;
    }
    {
        ProbeJob pj;
        memset(&pj, 0, sizeof(pj));
        pj.audio = audio;
        pj.a = a;
        pj.fp = fp;
        pj.section_fp = section_fp;
        pj.section_windows = section_windows;
        pj.ef = ef;
        pj.out = probes;
        pj.sc = &sc;
        pj.audio_bits = abits;
        if (W.parallel)
            mmx_parallel_for(nf, probe_frame, &pj);
        else
            for (f = 0; f < nf; f++)
                probe_frame(&pj, f, 0);
    }

    mmx_info("Searching the whole file and planning references (fixed-lag Viterbi)...");
    for (f = 0; f < nf; f++)
    {
        long long target_start = (long long)f * MMX_HOP - MMX_HOP;
        unsigned int n = 0, i;
        Candidate *list = cands + (size_t)f * MAX_CANDS;
        double audio_bits = abits[f];

        a->plan[f].audio_bits = (float)audio_bits;
        a->est_audio_bits += audio_bits;
        if (a->silent[f] || target_start < MMX_WIN)
        {
            cand_count[f] = 0;
            a->plan[f].best_bits = (float)audio_bits;
            a->est_best_bits += audio_bits;
            dp_step(&dp, f);
            dp_commit_lagged(&dp, f);
            continue;
        }

        /* 1-3. the fingerprint, onset and section probes of this frame, aligned
           in the whole-file pass above, walked in exactly the order they were
           produced in */
        {
            const FrameProbes *P = &probes[f];
            unsigned int m;
            for (m = 0; m < P->n; m++)
            {
                long long start[2];
                a->candidates_tested++;
                if (!P->ok[m])
                    continue;
                if (m < P->fp_n)
                    a->candidates_aligned++;
                start[0] = P->start[m];
                start[1] = 0;
                add_candidate(list, &n, 1, start, COST_UNKNOWN);
            }
        }

        /* 4. continuations of the previous frame's best candidates (long repeats) */
        if (f > 0 && cand_count[f - 1])
        {
            const Candidate *prev = cands + (size_t)(f - 1) * MAX_CANDS;
            unsigned int pn = cand_count[f - 1];
            unsigned int limit = pn < ef.continuations ? pn : ef.continuations;
            Candidate sorted[MAX_CANDS];
            memcpy(sorted, prev, sizeof(Candidate) * pn);
            if (limit < pn)
                qsort(sorted, pn, sizeof(Candidate), cand_cost_cmp);
            unsigned int nd = limit < ef.drift ? limit : ef.drift;
            long long dguess[MAX_CANDS], drifted[MAX_CANDS];
            for (i = 0; i < nd; i++)
                dguess[i] = sorted[i].start[0] + MMX_HOP;
            refine_batch(&W, audio, target_start, dguess, drifted, nd);
            for (i = 0; i < limit && n + CANDS_RESERVE < MAX_CANDS; i++)   /* keep room for the star and pair candidates */
            {
                long long start[2];
                unsigned int s, ok = 1;
                for (s = 0; s < sorted[i].n_sources; s++)
                {
                    start[s] = sorted[i].start[s] + MMX_HOP;
                    if (start[s] + MMX_WIN > target_start)
                        ok = 0;
                }
                if (!ok)
                    continue;
                /* exact continuation (keeps the block lineage) ... */
                add_candidate(list, &n, sorted[i].n_sources, start, COST_UNKNOWN);
                /* ... and a drift-corrected variant for the best lineages (level >= 3) */
                if (ef.drift && i < ef.drift && sorted[i].n_sources == 1)
                {
                    long long refined[2];
                    refined[0] = drifted[i];
                    refined[1] = 0;
                    if (refined[0] != start[0] && refined[0] + MMX_WIN <= target_start)
                        add_candidate(list, &n, 1, refined, COST_UNKNOWN);
                }
            }
        }

        /* 5. star instead of chain: a candidate whose source lies in a
           committed frame (or, with a long lag, a frame on the current best path)
           that references something itself is also tried at the source of that
           source; the redirected candidate is scored like any other */
        {
            unsigned int n0 = n;
            if (f)
                dp_tentative(&dp, f - 1);
            for (i = 0; i < n0 && n < MAX_CANDS; i++)
            {
                long long start[2];
                unsigned int s, changed = 0;
                for (s = 0; s < list[i].n_sources; s++)
                {
                    long long src = list[i].start[s], src_of_src;
                    long long g = (long long)((src + MMX_HOP) / MMX_HOP);
                    start[s] = src;
                    if (g >= 0 && (unsigned long)g < f && planned_source(&dp, (unsigned long)g, &src_of_src))
                    {
                        long long root = src_of_src + (src - ((long long)g * MMX_HOP - MMX_HOP));
                        if (root >= 0 && root + MMX_WIN <= target_start)
                        {
                            start[s] = root;
                            changed = 1;
                        }
                    }
                }
                if (changed)
                    add_candidate(list, &n, list[i].n_sources, start, COST_UNKNOWN);
            }
        }

        /* score every candidate whose cost is not known yet */
        score_pending(&W, list, n, f);
        /* two-source combinations among the best singles */
        if (ef.pairs && n >= 2)
        {
            Candidate singles[4];
            unsigned int ns = 0, x, y;
            Candidate tmp[MAX_CANDS];
            memcpy(tmp, list, sizeof(Candidate) * n);
            qsort(tmp, n, sizeof(Candidate), cand_cost_cmp);
            for (i = 0; i < n && ns < 3; i++)
                if (tmp[i].n_sources == 1)
                    singles[ns++] = tmp[i];
            for (x = 0; x < ns; x++)
                for (y = x + 1; y < ns && (x * ns + y) < ef.pairs * 2 + 1; y++)
                {
                    long long start[2];
                    start[0] = singles[x].start[0];
                    start[1] = singles[y].start[0];
                    if (n < MAX_CANDS)
                        add_candidate(list, &n, 2, start, (float)score_candidate(&sc, f, 2, start));
                }
        }
        score_pending(&W, list, n, f);
        cand_count[f] = n;
        if (n > a->cand_per_frame) a->cand_per_frame = n;
        if (n >= MAX_CANDS) full_frames++;
        {
            float best = (float)audio_bits;
            for (i = 0; i < n; i++)
                if (list[i].cost < best) best = list[i].cost;
            a->plan[f].best_bits = best;
            a->est_best_bits += best;
        }
        /* continuation table for the DP: the candidate of f-1 that each candidate continues (unique) */
        for (i = 0; i < n; i++)
        {
            const Candidate *prev = f ? cands + (size_t)(f - 1) * MAX_CANDS : NULL;
            unsigned int j2, pn = f ? cand_count[f - 1] : 0;
            cont[(size_t)f * MAX_CANDS + i] = -1;
            for (j2 = 0; j2 < pn; j2++)
                if (lineage_continues(&prev[j2], &list[i])) { cont[(size_t)f * MAX_CANDS + i] = (int)j2; break; }
        }
        dp_step(&dp, f);
        dp_commit_lagged(&dp, f);
        {
            static long long dbg = -2;
            if (dbg == -2) { const char *e = getenv("MMX_DEBUG_ANALYZE"); dbg = e ? atol(e) : -1; }
            if ((long long)f == dbg)
            {
                Candidate tmp[MAX_CANDS];
                memcpy(tmp, list, sizeof(Candidate) * n);
                qsort(tmp, n, sizeof(Candidate), cand_cost_cmp);
                fprintf(stderr, "analyzer frame %lu (%.3f s): audio %.0f bits, %u candidates\n", f, (double)target_start / audio->sample_rate, audio_bits, n);
                for (i = 0; i < n && i < 24; i++)
                    fprintf(stderr, "  %s src %.3f s%s cost %.0f bits\n", tmp[i].n_sources == 2 ? "REF2" : "REF ",
                            (double)tmp[i].start[0] / audio->sample_rate, tmp[i].n_sources == 2 ? "" : "        ", tmp[i].cost);
                for (i = 0; i < n && i < 24; i++)
                    if (tmp[i].n_sources == 2)
                        fprintf(stderr, "  REF2 pair %.3f + %.3f s cost %.0f\n", (double)tmp[i].start[0] / audio->sample_rate, (double)tmp[i].start[1] / audio->sample_rate, tmp[i].cost);
            }
        }

        if (params->verbose && (f % 500) == 0)
            mmx_debug("frame %lu/%lu: %u candidates, audio %.0f bits, best ref %.0f bits", f, nf, n, audio_bits,
                      n ? (double)list[0].cost : 0.0);
    }

    /* all frames decided: commit the tail of the path */
    dp_commit_tail(&dp, nf);
    a->est_plan_bits = dp.total;
    mmx_debug("planner: lag %u, %lu frames with a full candidate list (%u), %lu frames fell back to AUDIO at the commit (chain depth on the tentative path was wrong)",
              dp.lag, full_frames, MAX_CANDS, dp.fallbacks);
    {
        const char *e = getenv("MMX_DEBUG_PLAN");
        if (e)
        {
            long long f0 = atol(e), f1 = f0 + 40;
            const char *colon = strchr(e, ':');
            if (colon) f1 = atol(colon + 1);
            for (f = (unsigned long)(f0 < 0 ? 0 : f0); f < nf && (long long)f <= f1; f++)
            {
                const MMXFramePlan *p = &a->plan[f];
                const Candidate *list = cands + (size_t)f * MAX_CANDS;
                float best = 1e30f;
                unsigned int i, cont = 0;
                for (i = 0; i < cand_count[f]; i++) if (list[i].cost < best) best = list[i].cost;
                if (f > 0 && p->n_sources && a->plan[f - 1].n_sources == p->n_sources && a->plan[f - 1].src_start[0] + MMX_HOP == p->src_start[0] &&
                    (p->n_sources < 2 || a->plan[f - 1].src_start[1] + MMX_HOP == p->src_start[1]))
                    cont = 1;
                fprintf(stderr, "plan f %lu (%.3f s): %s src %.3f%s%.3f cost %.0f (audio %.0f, best cand %.0f) %s depth %u\n", f,
                        (double)((long long)f * MMX_HOP - MMX_HOP) / audio->sample_rate,
                        p->n_sources == 0 ? "AUDIO" : p->n_sources == 1 ? "REF  " : "REF2 ",
                        p->n_sources ? (double)p->src_start[0] / audio->sample_rate : 0.0, p->n_sources == 2 ? " + " : " ",
                        p->n_sources == 2 ? (double)p->src_start[1] / audio->sample_rate : 0.0,
                        (double)p->ref_bits, (double)p->audio_bits, (double)best, cont ? "cont" : "NEW ", p->depth);
            }
        }
    }

    /* per-band prediction gains of the plan (what the decomposition explains per band) */
    {
        GainJob gj;
        gj.W = &W;
        gj.a = a;
        if (W.parallel)
            mmx_parallel_for(nf, gain_task, &gj);
        else
            for (f = 0; f < nf; f++)
                gain_task(&gj, f, 0);
    }

    /* section-level repeat map: for every 0.75 s window the closest earlier window;
       consecutive windows with the same lag form one repeat */
    if (section_fp && section_windows)
    {
        unsigned long w, cap = section_windows, cnt = 0;
        long long cur_lag = 0, run_start = -1;
        double dist_sum = 0.0;
        a->repeats = (MMXRepeat *)calloc(cap ? cap : 1, sizeof(MMXRepeat));
        for (w = 0; w <= section_windows && a->repeats; w++)
        {
            long long best_u = -1, lag = 0;
            double best_d = 1e30;
            unsigned long u;
            if (w < section_windows)
                for (u = 0; u < w; u++)
                {
                    double d = fp_distance(section_fp + w * FP_DIM, section_fp + u * FP_DIM);
                    if (d < best_d) { best_d = d; best_u = (long long)u; }
                }
            lag = best_u >= 0 ? (long long)w - best_u : 0;
            if (w < section_windows && best_u >= 0 && best_d < 1.5 && (run_start < 0 || lag == cur_lag))
            {
                if (run_start < 0) { run_start = (long long)w; cur_lag = lag; dist_sum = 0.0; }
                dist_sum += best_d;
                continue;
            }
            if (run_start >= 0 && (long long)w - run_start >= 2 && cnt < cap)
            {
                double fs = (double)audio->sample_rate;
                a->repeats[cnt].start_s = (double)run_start * SECTION_HOP * MMX_HOP / fs;
                a->repeats[cnt].end_s = ((double)w * SECTION_HOP + SECTION_FRAMES) * MMX_HOP / fs;
                a->repeats[cnt].source_s = (double)(run_start - cur_lag) * SECTION_HOP * MMX_HOP / fs;
                a->repeats[cnt].distance = dist_sum / (double)((long long)w - run_start);
                cnt++;
            }
            run_start = -1;
            if (w < section_windows && best_u >= 0 && best_d < 1.5)
            {
                run_start = (long long)w; cur_lag = lag; dist_sum = best_d;
            }
        }
        a->repeat_count = cnt;
    }

    /* plan statistics */
    for (f = 0; f < nf; f++)
    {
        const MMXFramePlan *p = &a->plan[f];
        if (p->n_sources == 1) a->planned_ref_frames++;
        if (p->n_sources == 2) a->planned_ref2_frames++;
        if (f == 0 || p->n_sources != a->plan[f - 1].n_sources ||
            (p->n_sources && p->src_start[0] != a->plan[f - 1].src_start[0] + MMX_HOP) ||
            (p->n_sources > 1 && p->src_start[1] != a->plan[f - 1].src_start[1] + MMX_HOP))
            a->planned_blocks++;
    }
    capture_model_thr(a);
    a->seconds = mmx_wall_seconds() - t0;
    rc = 0;

done:
    workers_free(&W);
    free(probes);
    free(abits);
    free(fp);
    free(section_fp);
    free(cands);
    free(cand_count);
    free(cont);
    free(best);
    free(from);
    free(sdepth);
    free(tent);
    free(tent_state);
    for (c = 0; c < MMX_MAX_CH; c++)
        free(sc.resid[c]);
    mmx_frame_syntax_free(&sc.syn);
    mmx_refgraph_free(&graph);
    if (rc != 0)
        mmx_analysis_free(a);
    return rc;
}
