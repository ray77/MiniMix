#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>
#include <time.h>
#include "percept.h"
#include "minimix/codec.h"
#include "spectrum.h"
#include "log.h"

#define P_SUB_WIN 1024
#define P_SUB_STEP 512
#define P_SILENT_DBFS -75.0
#define P_GAIN_MIN_DB -36.0        /* the syntax's gain range (MMX_GAIN_MIN / MMX_GAIN_MAX, 0.5 dB steps) */
#define P_GAIN_MAX_DB 12.0
#define P_SUB_FLOOR_DB 6.0         /* a 1024 sub-window band carries a quarter of the frame band's power (half the coefficients at half the power) */
#define P_MIN_RUN 16               /* frames: a run shorter than 0.37 s pays two seams for nothing */
#define P_ENV_BLOCK 32             /* samples per block of the onset envelope used for the sample-exact refinement */
#define P_ENV_RADIUS 1024          /* samples searched around the frame-grid alignment */
#define P_RANDOM_PAIRS 4000
#define P_CHROMA_LO_HZ 55.0        /* pitch classes are read from this range of the leak-free spectrum */
#define P_CHROMA_HI_HZ 4000.0
#define P_CHROMA_FLOOR_DB -24.0    /* classes further below the strongest one do not count */
#define P_CHROMA_WEIGHT 2.0        /* weight of a pitch-class cell against a band cell */

static double db_of(double power)
{
    return 10.0 * log10(power > 1e-30 ? power : 1e-30);
}

static double clampd(double v, double lo, double hi)
{
    return v < lo ? lo : v > hi ? hi : v;
}

/* Squared distance between target frame t and source frame s (one channel):
   level fit per EQ band over the audible bands of the target, then every
   band cell and every (sub-window, EQ band) cell of an audible EQ band as
   the dB difference floored at the target's threshold. eq_acc / eq_n
   (optional) collect the per-EQ-band sums. */
static double pair_dist2(const MMXPerceptFrame *t, const MMXPerceptFrame *s, const unsigned char *eq_of, unsigned int dim,
                         double *acc_n, double *eq_acc, double *eq_n)
{
    double gsum[MMX_EQ_BANDS], asum[MMX_EQ_BANDS], g[MMX_EQ_BANDS], acc = 0.0, n = 0.0;
    unsigned int gcnt[MMX_EQ_BANDS], acnt[MMX_EQ_BANDS];
    unsigned int e, b, w;
    for (e = 0; e < MMX_EQ_BANDS; e++) { gsum[e] = 0.0; asum[e] = 0.0; gcnt[e] = 0; acnt[e] = 0; }
    for (b = 0; b < dim; b++)
    {
        double d = t->spec[b] - s->spec[b];
        e = eq_of[b];
        asum[e] += d;
        acnt[e]++;
        if (t->spec[b] > t->thr[b]) { gsum[e] += d; gcnt[e]++; }
    }
    for (e = 0; e < MMX_EQ_BANDS; e++)
        g[e] = clampd(gcnt[e] ? gsum[e] / gcnt[e] : acnt[e] ? asum[e] / acnt[e] : 0.0, P_GAIN_MIN_DB, P_GAIN_MAX_DB);
    for (b = 0; b < dim; b++)
    {
        double lt = t->spec[b] > t->thr[b] ? t->spec[b] : t->thr[b];
        double ls = s->spec[b] + g[eq_of[b]], d;
        if (ls < t->thr[b]) ls = t->thr[b];
        d = lt - ls;
        acc += d * d;
        n += 1.0;
        if (eq_acc) { eq_acc[eq_of[b]] += d * d; eq_n[eq_of[b]] += 1.0; }
    }
    for (e = 0; e < MMX_EQ_BANDS; e++)
    {
        double floor_db;
        if (!t->audible[e])
            continue;
        /* the EQ band's threshold in sub-window units: the loudest band threshold of the region, a quarter of the power */
        floor_db = -1e9;
        for (b = 0; b < dim; b++)
            if (eq_of[b] == e && t->thr[b] > floor_db && t->thr[b] < 1e8) floor_db = t->thr[b];
        floor_db -= P_SUB_FLOOR_DB;
        for (w = 0; w < MMX_PERCEPT_SUBW; w++)
        {
            double dt = t->sub[w][e], ds = s->sub[w][e] + g[e], d;
            if (dt < floor_db) dt = floor_db;
            if (ds < floor_db) ds = floor_db;
            d = dt - ds;
            acc += d * d;
            n += 1.0;
            if (eq_acc) { eq_acc[e] += d * d; eq_n[e] += 1.0; }
        }
    }
    for (b = 0; b < MMX_PERCEPT_CHROMA; b++)
    {
        double ct = t->chroma[b] > P_CHROMA_FLOOR_DB ? t->chroma[b] : P_CHROMA_FLOOR_DB;
        double cs = s->chroma[b] > P_CHROMA_FLOOR_DB ? s->chroma[b] : P_CHROMA_FLOOR_DB;
        double d = ct - cs;
        acc += P_CHROMA_WEIGHT * d * d;
        n += P_CHROMA_WEIGHT;
        if (eq_acc) { eq_acc[MMX_EQ_BANDS] += d * d; eq_n[MMX_EQ_BANDS] += 1.0; }
    }
    *acc_n = n;
    return acc;
}

/* distance^2 of target frame f against the frame lag frames earlier, over the channels */
static double frame_dist2(const MMXPercept *P, unsigned long f, unsigned long lag, double *eq_acc, double *eq_n)
{
    double acc = 0.0, n = 0.0, an;
    unsigned int c;
    for (c = 0; c < P->channels; c++)
    {
        acc += pair_dist2(&P->frames[f * P->channels + c], &P->frames[(f - lag) * P->channels + c], P->eq_of, P->dim, &an, eq_acc, eq_n);
        n += an;
    }
    return n > 0.0 ? acc / n : 0.0;
}

static int cmp_double(const void *a, const void *b)
{
    double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}

static double median(double *v, unsigned long n)
{
    if (!n) return 0.0;
    qsort(v, n, sizeof(double), cmp_double);
    return n & 1 ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
}

/* ---------- descriptors ---------- */

static int describe(const MMXAudioBuffer *audio, unsigned int quality, MMXPercept *P)
{
    MMXBandLayout bands, sub_bands;
    MMXSpectrum spec, sub_spec;
    MMXFramePsy psy;
    float *win = NULL, *amp = NULL, *sub_win = NULL, *sub_amp = NULL;
    unsigned long f, nf = P->frame_count;
    unsigned int c, b, w, j, nch = audio->channels;
    double silent_power;
    int rc = -1;

    memset(&spec, 0, sizeof(spec));
    memset(&sub_spec, 0, sizeof(sub_spec));
    if (mmx_bands_init(&bands, audio->sample_rate, MMX_HOP) != 0 || mmx_bands_init(&sub_bands, audio->sample_rate, P_SUB_WIN / 2) != 0 ||
        mmx_spectrum_init(&spec, MMX_WIN) != 0 || mmx_spectrum_init(&sub_spec, P_SUB_WIN) != 0)
        goto done;
    win = (float *)malloc(sizeof(float) * MMX_WIN);
    amp = (float *)malloc(sizeof(float) * MMX_HOP);
    sub_win = (float *)malloc(sizeof(float) * P_SUB_WIN);
    sub_amp = (float *)malloc(sizeof(float) * (P_SUB_WIN / 2));
    if (!win || !amp || !sub_win || !sub_amp)
        goto done;
    P->dim = bands.cutoff_band[quality > MMX_QUALITY_MAX ? MMX_QUALITY_MAX : quality];
    if (P->dim > bands.band_count) P->dim = bands.band_count;
    if (P->dim > MMX_PERCEPT_DIM) P->dim = MMX_PERCEPT_DIM;
    for (b = 0; b < P->dim; b++) P->eq_of[b] = bands.eq_band[b];
    silent_power = pow(10.0, P_SILENT_DBFS / 10.0) * bands.fs_peak_power * 2.0;

    for (f = 0; f < nf; f++)
    {
        long long start = (long long)f * MMX_HOP - MMX_HOP, i;
        double total = 0.0;
        for (c = 0; c < nch; c++)
        {
            MMXPerceptFrame *d = &P->frames[f * nch + c];
            unsigned int e;
            for (i = 0; i < (long long)MMX_WIN; i++)
            {
                long long pos = start + i;
                win[i] = (pos >= 0 && pos < (long long)audio->frame_count) ? audio->samples[(size_t)pos * nch + c] : 0.0f;
            }
            mmx_spectrum_analyze(&spec, win, amp);
            mmx_psy_analyze(&bands, amp, NULL, quality, &psy);
            for (e = 0; e < MMX_EQ_BANDS; e++) d->audible[e] = 0;
            for (b = 0; b < P->dim; b++)
            {
                double n = (double)(bands.band_start[b + 1] - bands.band_start[b]);
                d->spec[b] = (float)db_of(psy.energy[b]);
                d->thr[b] = psy.thr[b] < 1e29f ? (float)db_of((double)psy.thr[b] * n) : 1e9f;
                if (d->spec[b] > d->thr[b]) d->audible[bands.eq_band[b]] = 1;
                total += psy.energy[b];
            }
            {
                double cp[MMX_PERCEPT_CHROMA], cmax = 1e-30;
                unsigned long k;
                for (b = 0; b < MMX_PERCEPT_CHROMA; b++) cp[b] = 0.0;
                for (k = 0; k < MMX_HOP; k++)
                {
                    double hz = (k + 0.5) * (double)audio->sample_rate / MMX_WIN;
                    long long semi;
                    if (hz < P_CHROMA_LO_HZ || hz > P_CHROMA_HI_HZ)
                        continue;
                    semi = (long long)floor(12.0 * log(hz / 27.5) / log(2.0) + 0.5);   /* A0 = 27.5 Hz */
                    cp[semi % MMX_PERCEPT_CHROMA] += (double)amp[k] * amp[k];
                }
                for (b = 0; b < MMX_PERCEPT_CHROMA; b++) if (cp[b] > cmax) cmax = cp[b];
                for (b = 0; b < MMX_PERCEPT_CHROMA; b++) d->chroma[b] = (float)(db_of(cp[b]) - db_of(cmax));
            }
            for (w = 0; w < MMX_PERCEPT_SUBW; w++)
            {
                double se[MMX_EQ_BANDS];
                long long ws = start + (long long)w * P_SUB_STEP;
                for (e = 0; e < MMX_EQ_BANDS; e++) se[e] = 0.0;
                for (i = 0; i < (long long)P_SUB_WIN; i++)
                {
                    long long pos = ws + i;
                    sub_win[i] = (pos >= 0 && pos < (long long)audio->frame_count) ? audio->samples[(size_t)pos * nch + c] : 0.0f;
                }
                mmx_spectrum_analyze(&sub_spec, sub_win, sub_amp);
                for (j = 0; j < sub_bands.band_count; j++)
                {
                    unsigned long k;
                    for (k = sub_bands.band_start[j]; k < sub_bands.band_start[j + 1]; k++)
                        se[sub_bands.eq_band[j]] += (double)sub_amp[k] * sub_amp[k];
                }
                for (e = 0; e < MMX_EQ_BANDS; e++) d->sub[w][e] = (float)db_of(se[e]);
            }
        }
        P->silent[f] = total < silent_power;
    }
    rc = 0;
done:
    free(win);
    free(amp);
    free(sub_win);
    free(sub_amp);
    if (spec.n) mmx_spectrum_free(&spec);
    if (sub_spec.n) mmx_spectrum_free(&sub_spec);
    return rc;
}

/* ---------- the map ---------- */

static void match_update(MMXPerceptMatch *m, unsigned long lag, double dist)
{
    if (!m->lag || dist < m->dist)
    {
        m->lag = lag;
        m->dist = (float)dist;
    }
}

/* per-EQ-band breakdown of a section's match */
static void section_eq(const MMXPercept *P, unsigned long w, MMXPerceptMatch *m)
{
    double eq_acc[MMX_EQ_BANDS + 1], eq_n[MMX_EQ_BANDS + 1];
    unsigned long f, f0 = w * MMX_PERCEPT_HOP;
    unsigned int e;
    for (e = 0; e <= MMX_EQ_BANDS; e++) { eq_acc[e] = 0.0; eq_n[e] = 0.0; }
    if (!m->lag)
        return;
    for (f = f0; f < f0 + MMX_PERCEPT_SECTION && f < P->frame_count; f++)
        if (!P->silent[f] && f >= m->lag)
            frame_dist2(P, f, m->lag, eq_acc, eq_n);
    for (e = 0; e <= MMX_EQ_BANDS; e++)
        m->eq_dist[e] = (float)(eq_n[e] > 0.0 ? sqrt(eq_acc[e] / eq_n[e]) : 0.0);
}

/* per-EQ-band RMS (plus the chroma cells as slot MMX_EQ_BANDS) of one frame pair */
static void frame_eq(const MMXPercept *P, unsigned long f, unsigned long lag, double *out)
{
    double eq_acc[MMX_EQ_BANDS + 1], eq_n[MMX_EQ_BANDS + 1];
    unsigned int e;
    for (e = 0; e <= MMX_EQ_BANDS; e++) { eq_acc[e] = 0.0; eq_n[e] = 0.0; }
    frame_dist2(P, f, lag, eq_acc, eq_n);
    for (e = 0; e <= MMX_EQ_BANDS; e++) out[e] = eq_n[e] > 0.0 ? sqrt(eq_acc[e] / eq_n[e]) : 0.0;
}

int mmx_percept_analyze(const MMXAudioBuffer *audio, unsigned int quality, int verbose, MMXPercept *P)
{
    unsigned long nf, f, lag, w, ns, i;
    double *psum = NULL, *samples = NULL;
    unsigned long *pcnt = NULL;
    unsigned long lag_1s, lag_4s;
    clock_t t0 = clock();
    int rc = -1;

    memset(P, 0, sizeof(*P));
    nf = mmx_codec_frame_count(audio->frame_count);
    P->frame_count = nf;
    P->channels = audio->channels;
    P->sample_rate = audio->sample_rate;
    P->frames = (MMXPerceptFrame *)calloc((size_t)nf * audio->channels, sizeof(MMXPerceptFrame));
    P->silent = (unsigned char *)calloc(nf ? nf : 1, 1);
    ns = nf >= MMX_PERCEPT_SECTION ? (nf - MMX_PERCEPT_SECTION) / MMX_PERCEPT_HOP + 1 : 0;
    P->sections = ns;
    P->best = (MMXPerceptMatch *)calloc(ns ? ns : 1, sizeof(MMXPerceptMatch));
    P->best_1s = (MMXPerceptMatch *)calloc(ns ? ns : 1, sizeof(MMXPerceptMatch));
    P->best_4s = (MMXPerceptMatch *)calloc(ns ? ns : 1, sizeof(MMXPerceptMatch));
    psum = (double *)malloc(sizeof(double) * (nf + 1));
    pcnt = (unsigned long *)malloc(sizeof(unsigned long) * (nf + 1));
    samples = (double *)malloc(sizeof(double) * (nf > P_RANDOM_PAIRS ? nf : P_RANDOM_PAIRS));
    if (!P->frames || !P->silent || !P->best || !P->best_1s || !P->best_4s || !psum || !pcnt || !samples)
        goto done;
    mmx_info("Perceptual map: descriptors of %lu frames...", nf);
    if (describe(audio, quality, P) != 0)
        goto done;
    lag_1s = (unsigned long)(1.0 * audio->sample_rate / MMX_HOP + 0.5);
    lag_4s = (unsigned long)(4.0 * audio->sample_rate / MMX_HOP + 0.5);

    mmx_info("Perceptual map: every section against every earlier lag (%lu sections)...", ns);
    /* lag by lag: the frame distances, then every section's mean over its frames */
    for (lag = 2; lag < nf; lag++)
    {
        /* prefix sums of the frame distances and of the counted frames along f */
        double sum = 0.0;
        unsigned long cnt = 0, first_w;
        first_w = (lag + MMX_PERCEPT_HOP - 1) / MMX_PERCEPT_HOP;   /* sections whose first frame has a source */
        if (first_w >= ns)
            break;
        psum[lag] = 0.0;
        pcnt[lag] = 0;
        for (f = lag; f < nf; f++)
        {
            if (!P->silent[f] && !P->silent[f - lag])
            {
                sum += frame_dist2(P, f, lag, NULL, NULL);
                cnt++;
            }
            psum[f + 1] = sum;
            pcnt[f + 1] = cnt;
        }
        for (w = first_w; w < ns; w++)
        {
            unsigned long f0 = w * MMX_PERCEPT_HOP, f1 = f0 + MMX_PERCEPT_SECTION, n = pcnt[f1] - pcnt[f0];
            double d;
            if (n < MMX_PERCEPT_SECTION / 2)
                continue;                          /* mostly silent */
            d = sqrt((psum[f1] - psum[f0]) / (double)n);
            match_update(&P->best[w], lag, d);
            if (lag >= lag_1s) match_update(&P->best_1s[w], lag, d);
            if (lag >= lag_4s) match_update(&P->best_4s[w], lag, d);
        }
        if (verbose && (lag % 1000) == 0)
            mmx_debug("percept lag %lu / %lu", lag, nf);
    }
    for (w = 0; w < ns; w++)
    {
        section_eq(P, w, &P->best[w]);
        section_eq(P, w, &P->best_1s[w]);
        section_eq(P, w, &P->best_4s[w]);
    }
    /* references: the predecessor frame (stationarity) and random far pairs */
    i = 0;
    for (f = 1; f < nf; f++)
        if (!P->silent[f] && !P->silent[f - 1])
        {
            double eq[MMX_EQ_BANDS + 1];
            unsigned int e;
            samples[i++] = sqrt(frame_dist2(P, f, 1, NULL, NULL));
            frame_eq(P, f, 1, eq);
            for (e = 0; e <= MMX_EQ_BANDS; e++) P->adjacent_eq[e] += eq[e];
        }
    P->adjacent_db = median(samples, i);
    if (i) { unsigned int e; for (e = 0; e <= MMX_EQ_BANDS; e++) P->adjacent_eq[e] /= (double)i; }
    {
        unsigned long rng = 12345, tries = 0;
        i = 0;
        while (i < P_RANDOM_PAIRS && tries < 20 * P_RANDOM_PAIRS && nf > lag_4s + 2)
        {
            unsigned long a, b_;
            rng = rng * 1103515245UL + 12345UL;
            a = (rng >> 8) % nf;
            rng = rng * 1103515245UL + 12345UL;
            b_ = (rng >> 8) % nf;
            tries++;
            if (a < b_) { unsigned long t = a; a = b_; b_ = t; }
            if (a - b_ < lag_4s || P->silent[a] || P->silent[b_])
                continue;
            {
                double eq[MMX_EQ_BANDS + 1];
                unsigned int e;
                samples[i++] = sqrt(frame_dist2(P, a, a - b_, NULL, NULL));
                frame_eq(P, a, a - b_, eq);
                for (e = 0; e <= MMX_EQ_BANDS; e++) P->random_eq[e] += eq[e];
            }
        }
        P->random_db = median(samples, i);
        if (i) { unsigned int e; for (e = 0; e <= MMX_EQ_BANDS; e++) P->random_eq[e] /= (double)i; }
    }
    P->seconds = (double)(clock() - t0) / CLOCKS_PER_SEC;
    rc = 0;
done:
    free(psum);
    free(pcnt);
    free(samples);
    if (rc != 0)
        mmx_percept_free(P);
    return rc;
}

/* ---------- the plan ---------- */

/* Log RMS envelope in blocks of P_ENV_BLOCK samples (mono sum) from `start`. */
static void envelope(const MMXAudioBuffer *audio, long long start, unsigned long blocks, double *env)
{
    unsigned long i, k;
    unsigned int c, nch = audio->channels;
    for (i = 0; i < blocks; i++)
    {
        double acc = 0.0;
        for (k = 0; k < P_ENV_BLOCK; k++)
        {
            long long pos = start + (long long)(i * P_ENV_BLOCK + k);
            double s = 0.0;
            if (pos >= 0 && pos < (long long)audio->frame_count)
                for (c = 0; c < nch; c++) s += audio->samples[(size_t)pos * nch + c];
            acc += s * s;
        }
        env[i] = db_of(acc / P_ENV_BLOCK + 1e-12);
    }
}

/* Offset (samples) that best aligns the source's onset envelope with the
   target's over the run: least squares of the level-free envelope difference,
   coarse (P_ENV_BLOCK / 2 steps) then fine (every sample). */
static long long refine_offset(const MMXAudioBuffer *audio, long long target_start, long long src_start, unsigned long n_samples)
{
    unsigned long blocks = n_samples / P_ENV_BLOCK, i;
    double *et, *es, best_cost = 1e300;
    long long best = 0, d, step;
    if (blocks < 8)
        return 0;
    et = (double *)malloc(sizeof(double) * blocks);
    es = (double *)malloc(sizeof(double) * blocks);
    if (!et || !es) { free(et); free(es); return 0; }
    envelope(audio, target_start, blocks, et);
    for (step = P_ENV_BLOCK / 2; step >= 1; step /= 4)   /* 16, 4, 1 */
    {
        long long lo = step == P_ENV_BLOCK / 2 ? -(long long)P_ENV_RADIUS : best - step * 4;
        long long hi = step == P_ENV_BLOCK / 2 ? (long long)P_ENV_RADIUS : best + step * 4;
        long long cand_best = best;
        for (d = lo; d <= hi; d += step)
        {
            double mean = 0.0, cost = 0.0;
            if (src_start + d < 0)
                continue;
            envelope(audio, src_start + d, blocks, es);
            for (i = 0; i < blocks; i++) mean += et[i] - es[i];
            mean /= blocks;
            for (i = 0; i < blocks; i++) { double x = et[i] - es[i] - mean; cost += x * x; }
            if (cost < best_cost) { best_cost = cost; cand_best = d; }
        }
        best = cand_best;
    }
    free(et);
    free(es);
    return best;
}

int mmx_percept_plan(const MMXAudioBuffer *audio, MMXPercept *P, double max_db, double min_lag_s)
{
    unsigned long nf = P->frame_count, ns = P->sections, f, w, cap = 0, cnt = 0;
    unsigned long *lag_of = NULL;
    float *dist_of = NULL;
    const MMXPerceptMatch *best;
    unsigned long min_lag = (unsigned long)(min_lag_s * audio->sample_rate / MMX_HOP + 0.5);
    int rc = -1;

    free(P->runs);
    P->runs = NULL;
    P->run_count = 0;
    best = min_lag_s >= 4.0 ? P->best_4s : min_lag_s >= 1.0 ? P->best_1s : P->best;
    lag_of = (unsigned long *)calloc(nf ? nf : 1, sizeof(unsigned long));
    dist_of = (float *)malloc(sizeof(float) * (nf ? nf : 1));
    if (!lag_of || !dist_of)
        goto done;
    for (f = 0; f < nf; f++) dist_of[f] = 1e30f;
    /* every frame takes the best section that covers it */
    for (w = 0; w < ns; w++)
    {
        const MMXPerceptMatch *m = &best[w];
        if (!m->lag || m->dist > max_db || m->lag < min_lag)
            continue;
        for (f = w * MMX_PERCEPT_HOP; f < w * MMX_PERCEPT_HOP + MMX_PERCEPT_SECTION && f < nf; f++)
            if (m->dist < dist_of[f]) { dist_of[f] = m->dist; lag_of[f] = m->lag; }
    }
    /* continuity: keep the previous frame's lag when a covering section supports it */
    for (f = 1; f < nf; f++)
    {
        unsigned long w0, w1;
        if (!lag_of[f] || !lag_of[f - 1] || lag_of[f] == lag_of[f - 1])
            continue;
        w0 = f >= MMX_PERCEPT_SECTION - 1 ? (f - (MMX_PERCEPT_SECTION - 1) + MMX_PERCEPT_HOP - 1) / MMX_PERCEPT_HOP : 0;
        w1 = f / MMX_PERCEPT_HOP;
        for (w = w0; w <= w1 && w < ns; w++)
            if (best[w].lag == lag_of[f - 1] && best[w].dist <= max_db)
            {
                lag_of[f] = lag_of[f - 1];
                dist_of[f] = best[w].dist;
                break;
            }
    }
    /* runs */
    for (f = 0; f < nf; f++)
        if (lag_of[f] && (f == 0 || lag_of[f] != lag_of[f - 1])) cap++;
    P->runs = (MMXPerceptRun *)calloc(cap ? cap : 1, sizeof(MMXPerceptRun));
    if (!P->runs)
        goto done;
    f = 0;
    while (f < nf)
    {
        unsigned long g = f, i;
        MMXPerceptRun *r;
        double dsum = 0.0;
        long long target_start, src;
        if (!lag_of[f]) { f++; continue; }
        while (g < nf && lag_of[g] == lag_of[f]) g++;
        if (g - f < P_MIN_RUN || f < lag_of[f] + 2) { f = g; continue; }
        for (i = f; i < g; i++) dsum += dist_of[i];
        r = &P->runs[cnt];
        r->f0 = f;
        r->f1 = g;
        r->lag = lag_of[f];
        r->dist = dsum / (double)(g - f);
        {
            double eq_acc[MMX_EQ_BANDS + 1], eq_n[MMX_EQ_BANDS + 1];
            unsigned int e;
            for (e = 0; e <= MMX_EQ_BANDS; e++) { eq_acc[e] = 0.0; eq_n[e] = 0.0; }
            for (i = f; i < g; i++)
                if (!P->silent[i] && !P->silent[i - r->lag])
                    frame_dist2(P, i, r->lag, eq_acc, eq_n);
            /* a region without a cell (above the cutoff, or never audible) gives no evidence: it is not played */
            for (e = 0; e <= MMX_EQ_BANDS; e++) r->eq_dist[e] = (float)(eq_n[e] > 0.0 ? sqrt(eq_acc[e] / eq_n[e]) : 99.0);
        }
        target_start = (long long)f * MMX_HOP - MMX_HOP;
        src = target_start - (long long)r->lag * MMX_HOP;
        r->offset = refine_offset(audio, target_start, src, (g - f) * MMX_HOP + MMX_HOP);
        if (src + r->offset < 0) r->offset = -src;
        r->src_start = src + r->offset;
        /* a source inside an earlier run's clean region plays from that run's source (star, not chain) */
        for (;;)
        {
            long long lo = r->src_start, hi = r->src_start + (long long)(g - f) * MMX_HOP + MMX_HOP;
            unsigned long q;
            int moved = 0;
            for (q = 0; q < cnt; q++)
            {
                const MMXPerceptRun *e = &P->runs[q];
                long long clean_lo = (long long)e->f0 * MMX_HOP, clean_hi = (long long)e->f1 * MMX_HOP - MMX_HOP;
                if (lo >= clean_lo && hi <= clean_hi)
                {
                    r->src_start = e->src_start + (lo - ((long long)e->f0 * MMX_HOP - MMX_HOP));
                    moved = 1;
                    break;
                }
            }
            if (!moved) break;
        }
        cnt++;
        f = g;
    }
    P->run_count = cnt;
    rc = 0;
done:
    free(lag_of);
    free(dist_of);
    return rc;
}

/* ---------- report ---------- */

static double share_under(const MMXPerceptMatch *m, unsigned long ns, double x)
{
    unsigned long w, n = 0;
    for (w = 0; w < ns; w++)
        if (m[w].lag && m[w].dist <= x) n++;
    return ns ? 100.0 * n / ns : 0.0;
}

static double median_of(const MMXPerceptMatch *m, unsigned long ns)
{
    double *v = (double *)malloc(sizeof(double) * (ns ? ns : 1)), r;
    unsigned long w, n = 0;
    if (!v) return 0.0;
    for (w = 0; w < ns; w++)
        if (m[w].lag) v[n++] = m[w].dist;
    r = median(v, n);
    free(v);
    return r;
}

void mmx_percept_print(const MMXPercept *P, double max_db, int json)
{
    static const double xs[6] = {2.0, 3.0, 4.0, 5.0, 6.0, 8.0};
    static const char *regions[MMX_EQ_BANDS] = {"<200", "200-500", "500-1k", "1k-2k", "2k-4k", "4k-8k", "8k-12k", ">12k"};
    double fs = (double)P->sample_rate, played = 0.0;
    unsigned long i, w;
    unsigned int e, k;
    for (i = 0; i < P->run_count; i++) played += (double)(P->runs[i].f1 - P->runs[i].f0);
    if (json)
    {
        printf("  \"percept\": { \"sections\": %lu, \"adjacent_db\": %.2f, \"random_db\": %.2f, \"seconds\": %.1f,\n    \"share_under\": [", P->sections, P->adjacent_db, P->random_db, P->seconds);
        for (k = 0; k < 6; k++)
            printf("%s { \"db\": %.0f, \"any\": %.1f, \"lag_1s\": %.1f, \"lag_4s\": %.1f }", k ? "," : "", xs[k],
                   share_under(P->best, P->sections, xs[k]), share_under(P->best_1s, P->sections, xs[k]), share_under(P->best_4s, P->sections, xs[k]));
        printf(" ],\n    \"runs\": [");
        for (i = 0; i < P->run_count; i++)
        {
            const MMXPerceptRun *r = &P->runs[i];
            printf("%s { \"start\": %.3f, \"end\": %.3f, \"source\": %.3f, \"lag\": %.3f, \"dist\": %.2f }", i ? "," : "",
                   (double)r->f0 * MMX_HOP / fs, (double)r->f1 * MMX_HOP / fs, (double)(r->src_start + MMX_HOP) / fs, (double)r->lag * MMX_HOP / fs, r->dist);
        }
        printf(" ] }\n");
        return;
    }
    printf("\nPerceptual similarity map\n");
    printf("  Distance = RMS dB deviation over the half-bark band levels and the 23 ms sub-window levels per EQ band,\n");
    printf("  after a level fit per EQ band (the codec's gain range), cells floored at the masking threshold.\n");
    printf("  Sections of %.2f s every %.2f s, %lu sections, %.1f s of search.\n", MMX_PERCEPT_SECTION * (double)MMX_HOP / fs,
           MMX_PERCEPT_HOP * (double)MMX_HOP / fs, P->sections, P->seconds);
    printf("  References: a frame against its predecessor (stationarity) %.2f dB, random pairs >= 4 s apart %.2f dB.\n", P->adjacent_db, P->random_db);
    printf("  Share of the file with an earlier section under a distance:\n");
    printf("    %-10s %12s %12s %12s\n", "distance", "any lag", "lag >= 1 s", "lag >= 4 s");
    for (k = 0; k < 6; k++)
        printf("    <= %2.0f dB  %10.1f %% %10.1f %% %10.1f %%\n", xs[k], share_under(P->best, P->sections, xs[k]),
               share_under(P->best_1s, P->sections, xs[k]), share_under(P->best_4s, P->sections, xs[k]));
    printf("    median     %10.2f    %10.2f    %10.2f    dB\n", median_of(P->best, P->sections), median_of(P->best_1s, P->sections), median_of(P->best_4s, P->sections));
    {
        /* where the best matches (lag >= 1 s) deviate: mean per EQ region */
        double acc[MMX_EQ_BANDS + 1];
        unsigned long n = 0;
        for (e = 0; e <= MMX_EQ_BANDS; e++) acc[e] = 0.0;
        for (w = 0; w < P->sections; w++)
            if (P->best_1s[w].lag)
            {
                for (e = 0; e <= MMX_EQ_BANDS; e++) acc[e] += P->best_1s[w].eq_dist[e];
                n++;
            }
        printf("  Mean deviation per EQ region (dB): the best match (lag >= 1 s), against the predecessor frame and a random pair\n");
        printf("    %-12s", "");
        for (e = 0; e <= MMX_EQ_BANDS; e++) printf("%9s", e < MMX_EQ_BANDS ? regions[e] : "chroma");
        printf("\n    %-12s", "best match");
        for (e = 0; e <= MMX_EQ_BANDS; e++) printf("%9.2f", n ? acc[e] / n : 0.0);
        printf("\n    %-12s", "predecessor");
        for (e = 0; e <= MMX_EQ_BANDS; e++) printf("%9.2f", P->adjacent_eq[e]);
        printf("\n    %-12s", "random pair");
        for (e = 0; e <= MMX_EQ_BANDS; e++) printf("%9.2f", P->random_eq[e]);
        printf("\n");
    }
    printf("  Runs under %.1f dB (lag >= 1 s, >= %.2f s long): %lu runs, %.1f s = %.1f %% of the file would play from an earlier section\n",
           max_db, P_MIN_RUN * (double)MMX_HOP / fs, P->run_count, played * MMX_HOP / fs, P->frame_count ? 100.0 * played / P->frame_count : 0.0);
    for (i = 0; i < P->run_count && i < 60; i++)
    {
        const MMXPerceptRun *r = &P->runs[i];
        printf("    %7.2f - %7.2f s  <-  %7.2f s  (lag %6.2f s, offset %+5lld samples, distance %.2f dB)\n",
               (double)r->f0 * MMX_HOP / fs, (double)r->f1 * MMX_HOP / fs, (double)(r->src_start + MMX_HOP) / fs,
               (double)r->lag * MMX_HOP / fs, r->offset, r->dist);
    }
    if (P->run_count > 60)
        printf("    ... %lu more\n", P->run_count - 60);
}

void mmx_percept_free(MMXPercept *P)
{
    if (!P) return;
    free(P->frames);
    free(P->silent);
    free(P->best);
    free(P->best_1s);
    free(P->best_4s);
    free(P->runs);
    memset(P, 0, sizeof(*P));
}
