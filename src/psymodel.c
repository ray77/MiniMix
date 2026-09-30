#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "lab.h"
#include "psymodel.h"
#include "mdct.h"
#include "spectrum.h"

#define MMX_PI 3.14159265358979323846
#define SPL_FULL_SCALE 96.0
#define TRANSIENT_DB 10.0
#define TRANSIENT_MIN_DBFS -60.0

static double bark_of_hz(double f)
{
    return 13.0 * atan(0.00076 * f) + 3.5 * atan((f / 7500.0) * (f / 7500.0));
}

/* Terhardt absolute threshold of hearing, dB SPL. */
static double abs_threshold_db(double f)
{
    double k = f / 1000.0, t;
    if (k < 0.02) k = 0.02;
    t = 3.64 * pow(k, -0.8) - 6.5 * exp(-0.6 * (k - 3.3) * (k - 3.3)) + 1e-3 * pow(k, 4.0);
    if (t < -5.0) t = -5.0;
    if (t > SPL_FULL_SCALE) t = SPL_FULL_SCALE;
    return t;
}

/* Schroeder spreading function in dB for a bark distance. */
static double spread_db(double dz)
{
    return 15.81 + 7.5 * (dz + 0.474) - 17.5 * sqrt(1.0 + (dz + 0.474) * (dz + 0.474));
}

double mmx_quality_offset_db(unsigned int quality)
{
    static const double table[MMX_QUALITY_MAX + 1] = { 12.0, 12.0, 9.0, 7.0, 5.0, 3.0, 1.5, 0.0, -2.0, -4.0, -6.0 };
    if (quality > MMX_QUALITY_MAX) quality = MMX_QUALITY_MAX;
    return table[quality];
}

double mmx_quality_bandwidth_hz(unsigned int quality)
{
    /* quality 7 codes up to 19 kHz: nothing audible is cut at the default point (17 kHz before) */
    static const double table[MMX_QUALITY_MAX + 1] = { 11000, 11000, 12000, 13000, 14000, 15000, 16000, 19000, 19500, 20000, 20500 };
    if (quality > MMX_QUALITY_MAX) quality = MMX_QUALITY_MAX;
    return table[quality];
}

int mmx_bands_init(MMXBandLayout *L, unsigned long sample_rate, unsigned long m)
{
    double bin_hz = (double)sample_rate / 2.0 / (double)m;
    unsigned int b, j, q;
    unsigned long start = 0;
    MMXMdct mdct;
    float *sine, *coefs;
    unsigned long i;

    memset(L, 0, sizeof(*L));
    L->sample_rate = sample_rate;
    L->m = m;

    /* band edges: at least 4 bins, at most half a bark, until m */
    while (start < m && L->band_count < MMX_MAX_BANDS)
    {
        double z0 = bark_of_hz(start * bin_hz);
        unsigned long end = start + 4;
        while (end < m && bark_of_hz(end * bin_hz) - z0 < 0.5)
            end++;
        if (end > m || L->band_count == MMX_MAX_BANDS - 1)
            end = m;
        L->band_start[L->band_count] = (unsigned int)start;
        L->band_count++;
        start = end;
    }
    L->band_start[L->band_count] = (unsigned int)m;

    for (b = 0; b < L->band_count; b++)
    {
        double lo = L->band_start[b] * bin_hz, hi = L->band_start[b + 1] * bin_hz;
        L->band_hz[b] = (float)(0.5 * (lo + hi));
        L->band_bark[b] = (float)bark_of_hz(L->band_hz[b]);
        L->eq_band[b] = (unsigned char)(L->band_hz[b] < 200 ? 0 : L->band_hz[b] < 500 ? 1 : L->band_hz[b] < 1000 ? 2 :
                                        L->band_hz[b] < 2000 ? 3 : L->band_hz[b] < 4000 ? 4 : L->band_hz[b] < 8000 ? 5 :
                                        L->band_hz[b] < 12000 ? 6 : 7);
    }

    /* spreading matrix, row-normalized */
    for (b = 0; b < L->band_count; b++)
    {
        double sum = 0.0;
        for (j = 0; j < L->band_count; j++)
        {
            double s = pow(10.0, spread_db((double)L->band_bark[b] - L->band_bark[j]) / 10.0);
            L->spread[b][j] = (float)s;
            sum += s;
        }
        for (j = 0; j < L->band_count; j++)
            L->spread[b][j] = (float)(L->spread[b][j] / sum);
    }

    /* calibration: peak MDCT power of a full-scale sine centered on a bin */
    if (mmx_mdct_init(&mdct, 2 * m) != 0)
        return -1;
    sine = (float *)malloc(sizeof(float) * 2 * m);
    coefs = (float *)malloc(sizeof(float) * m);
    if (!sine || !coefs)
    {
        free(sine); free(coefs);
        mmx_mdct_free(&mdct);
        return -1;
    }
    {
        double f = (m / 8 + 0.5) * bin_hz; /* arbitrary mid band bin center */
        for (i = 0; i < 2 * m; i++)
            sine[i] = (float)sin(2.0 * MMX_PI * f * (double)i / (double)sample_rate);
        mmx_mdct_forward(&mdct, sine, coefs);
        L->fs_peak_power = 0.0;
        for (i = 0; i < m; i++)
            if ((double)coefs[i] * coefs[i] > L->fs_peak_power)
                L->fs_peak_power = (double)coefs[i] * coefs[i];
    }
    free(sine); free(coefs);
    mmx_mdct_free(&mdct);

    for (b = 0; b < L->band_count; b++)
        L->abs_thr[b] = (float)(L->fs_peak_power * pow(10.0, (abs_threshold_db(L->band_hz[b]) - SPL_FULL_SCALE) / 10.0));

    for (q = 0; q <= MMX_QUALITY_MAX; q++)
    {
        double bw = mmx_quality_bandwidth_hz(q);
        L->cutoff_band[q] = L->band_count;
        for (b = 0; b < L->band_count; b++)
            if (L->band_start[b] * bin_hz >= bw)
            {
                L->cutoff_band[q] = b;
                break;
            }
    }
    L->bwe_band = L->band_count;    /* band replication off until the file header asks for it */
    L->bwe_src0 = 0;
    L->bwe_mode = 0;
    return 0;
}

/* Band replication crossover (see psymodel.h). The source region is the octave
   below the crossover bin, clamped to at least 4 bins and never below bin 4:
   an octave transposition maps a harmonic series onto itself, so the copy does
   not beat against the coded part below. */
void mmx_bands_set_epb(MMXBandLayout *L, unsigned int lowrate, unsigned int fold_250hz, unsigned int flags, int fill_db)
{
    L->lowrate = lowrate ? 1u : 0u;
    L->epb_fold_hz = fold_250hz >= 255 ? 1e9 : 250.0 * fold_250hz;
    L->epb_noise = (flags & 1u) != 0;
    L->epb_coded = (flags & 2u) != 0;
    L->epb_ref = (flags & 4u) != 0;
    L->epb_fill_gain = pow(10.0, fill_db / 20.0);
}

unsigned int mmx_bands_set_bwe(MMXBandLayout *L, double hz, unsigned int mode)
{
    unsigned int b;
    L->bwe_mode = mode;
    L->bwe_band = L->band_count;
    L->bwe_src0 = 0;
    if (hz <= 0.0)
        return L->band_count;
    for (b = 1; b < L->band_count; b++)
        if ((double)L->band_hz[b] >= hz) { L->bwe_band = b; break; }
    if (L->bwe_band >= L->band_count)
        return L->band_count;
    {
        unsigned long x = L->band_start[L->bwe_band];
        unsigned long s0 = x / 2;
        if (x < 16) { L->bwe_band = L->band_count; return L->band_count; }   /* crossover too low to patch from */
        if (s0 < 4) s0 = 4;
        L->bwe_src0 = (unsigned int)s0;
    }
    return L->bwe_band;
}

double mmx_psy_spl_db(const MMXBandLayout *L, double power)
{
    if (power <= 0.0)
        return -200.0;
    return SPL_FULL_SCALE + 10.0 * log10(power / L->fs_peak_power);
}

/* Pre-echo control without block switching: the quantization noise of a frame
   spreads over its whole 2048-sample window. If an attack sits inside the
   window, the noise before the attack is only masked by the (quieter) signal
   there. Returns a threshold scale (<= 1) from the energy ratio of the part
   before the loudest sub-block to the attack itself; 1.0 when no transient. */
static double transient_scale(const float *window, unsigned long n, int *is_transient, float *attack_db, unsigned int *attack_pos)
{
    unsigned long parts = 16, len = n / parts, p, i, attack = 0;
    double e[16], floor_energy, e_max = 0.0, pre = 0.0, ratio;

    *is_transient = 0;
    *attack_db = 0.0f;
    *attack_pos = 0;
    floor_energy = pow(10.0, TRANSIENT_MIN_DBFS / 10.0) * (double)len;
    for (p = 0; p < parts; p++)
    {
        double s = 0.0;
        for (i = 0; i < len; i++)
        {
            double v = window[p * len + i];
            s += v * v;
        }
        e[p] = s;
        if (s > e_max) { e_max = s; attack = p; }
    }
    if (e_max <= floor_energy || attack == 0)
        return 1.0;
    for (p = 0; p < attack; p++)
        pre += e[p];
    pre /= (double)attack;
    if (pre > 0.0)
        *attack_db = (float)(10.0 * log10(e_max / pre));
    if (e_max <= pre * pow(10.0, TRANSIENT_DB / 10.0))
        return 1.0;
    *is_transient = 1;
    *attack_pos = (unsigned int)(attack * len);
    if (mmx_psy_pre_attack_db() >= 0.0)
        return 1.0;                    /* the caller applies the pre-attack rule (mmx_psy_pre_attack) */
    ratio = 0.05 * pre / e_max;        /* noise stays ~13 dB below the pre-attack level */
    if (ratio < 1e-4) ratio = 1e-4;    /* never more than 40 dB below the frame's own threshold */
    if (ratio > 1.0) ratio = 1.0;
    return ratio;
}

/* Experiments (environment, encoder and `mmx compare` alike, no bitstream change):
   MMX_PRE_ATTACK_DB=<margin>  pre-echo rule from the signal before the attack instead of the transient scale
   MMX_TONALITY_REGION=<hz>    tonality per EQ region for bands from <hz> upwards instead of the whole spectrum */
double mmx_psy_pre_attack_db(void)
{
    static double margin = -2.0;
    if (margin == -2.0)
    {
        const char *e = mmx_lab_getenv("MMX_PRE_ATTACK_DB");
        margin = e ? atof(e) : -1.0;
    }
    return margin;
}

static double tonality_region_hz(void)
{
    static double hz = -1.0;
    if (hz < 0.0)
    {
        const char *e = mmx_lab_getenv("MMX_TONALITY_REGION");
        hz = e ? atof(e) : 1e30;
    }
    return hz;
}

/* Reads both environment knobs once, before any thread runs the model: the two
   caches above are then never written again and the analysis can run the frames
   in parallel. */
void mmx_psy_prewarm(void)
{
    (void)mmx_psy_pre_attack_db();
    (void)tonality_region_hz();
}

/* Tonality index from the spectral flatness of the powers p[0..n): 0 dB = noise, -60 dB = tone. */
static double tonality_of(double log_sum, double lin_sum, double count)
{
    double sfm_db = count > 0 ? 10.0 * log10(exp(log_sum / count) / (lin_sum / count)) : 0.0;
    double alpha = sfm_db / -60.0;
    if (alpha < 0.0) alpha = 0.0;
    if (alpha > 1.0) alpha = 1.0;
    return alpha;
}

void mmx_psy_analyze(const MMXBandLayout *L, const float *coefs, const float *window,
                     unsigned int quality, MMXFramePsy *out)
{
    unsigned int b, j, e, nb = L->band_count, cutoff = L->cutoff_band[quality > MMX_QUALITY_MAX ? MMX_QUALITY_MAX : quality];
    unsigned long k;
    double log_sum = 0.0, lin_sum = 0.0, count = 0.0, alpha;
    double offset_lin = pow(10.0, mmx_quality_offset_db(quality) / 10.0);
    double tighten = 1.0, region_hz = tonality_region_hz();

    memset(out, 0, sizeof(*out));

    for (b = 0; b < nb; b++)
    {
        double e = 0.0, lg = 0.0, n = (double)(L->band_start[b + 1] - L->band_start[b]);
        for (k = L->band_start[b]; k < L->band_start[b + 1]; k++)
        {
            double p = (double)coefs[k] * coefs[k];
            e += p;
            lg += log(p + 1e-30);
        }
        out->energy[b] = (float)e;
        out->crest[b] = 1.0f;
        /* line prominence: the strongest bin over the median power of its +-6 neighbours (a partial
           among partials still stands out locally, where the band's flatness no longer sees it) */
        {
            double best = 0.0;
            unsigned long k0 = L->band_start[b], k1 = L->band_start[b + 1];
            for (k = k0; k < k1; k++)
            {
                double nb_[13], p = (double)coefs[k] * coefs[k], med;
                unsigned int cnt = 0, i, jj;
                long kk;
                if (p <= 0.0) continue;
                for (kk = (long)k - 6; kk <= (long)k + 6; kk++)
                {
                    if (kk < 0 || kk >= (long)L->m || kk == (long)k) continue;
                    nb_[cnt++] = (double)coefs[kk] * coefs[kk];
                }
                if (cnt < 6) continue;
                for (i = 1; i < cnt; i++)          /* insertion sort, 12 values */
                {
                    double v = nb_[i]; jj = i;
                    while (jj > 0 && nb_[jj - 1] > v) { nb_[jj] = nb_[jj - 1]; jj--; }
                    nb_[jj] = v;
                }
                med = nb_[cnt / 2] + 1e-30;
                if (p / med > best) best = p / med;
            }
            out->prom[b] = best > 1.0 ? (float)(10.0 * log10(best)) : 0.0f;
            if (out->prom[b] > 40.0f) out->prom[b] = 40.0f;
        }
        /* within-band flatness: bin powers of noise are exponentially distributed
           (geometric mean 0.56 of the arithmetic), a tone leaves most bins at the
           window's sidelobe floor */
        out->flat[b] = e > 0.0 ? (float)(exp(lg / n) / (e / n)) : 1.0f;
    }

    /* tonality from the spectral flatness of band-averaged power (0 dB = noise, -60 dB = tone) */
    for (b = 0; b < cutoff; b++)
    {
        double n = (double)(L->band_start[b + 1] - L->band_start[b]);
        double p = out->energy[b] / n + 1e-12;
        log_sum += log(p);
        lin_sum += p;
        count += 1.0;
    }
    alpha = tonality_of(log_sum, lin_sum, count);
    out->tonality = (float)alpha;
    for (e = 0; e < MMX_EQ_BANDS; e++)
        out->tonality_eq[e] = (float)alpha;

    /* region tonality (experiment): the flatness of the bins inside an EQ region
       decides for that region alone, so a tonal bass does not make the noise-like
       top strict and a single tone does not make everything else generous */
    if (region_hz < 1e29)
        for (e = 0; e < MMX_EQ_BANDS; e++)
        {
            double ls = 0.0, ln = 0.0, cnt = 0.0;
            for (b = 0; b < cutoff; b++)
            {
                if (L->eq_band[b] != e || L->band_hz[b] < region_hz)
                    continue;
                for (k = L->band_start[b]; k < L->band_start[b + 1]; k++)
                {
                    double p = (double)coefs[k] * coefs[k] + 1e-12;
                    ls += log(p);
                    ln += p;
                    cnt += 1.0;
                }
            }
            if (cnt > 0.0)
                out->tonality_eq[e] = (float)tonality_of(ls, ln, cnt);
        }

    if (window)
        tighten = transient_scale(window, 2 * L->m, &out->transient, &out->attack_db, &out->attack_pos);

    for (b = 0; b < nb; b++)
    {
        double spread = 0.0, offset_db, thr, n = (double)(L->band_start[b + 1] - L->band_start[b]), a;
        if (b >= cutoff)
        {
            out->thr[b] = 1e30f; /* band is not coded at all */
            continue;
        }
        for (j = 0; j < nb; j++)
            spread += out->energy[j] * L->spread[b][j];
        a = out->tonality_eq[L->eq_band[b]];
        offset_db = a * (14.5 + L->band_bark[b]) + (1.0 - a) * 5.5;
        thr = spread * pow(10.0, -offset_db / 10.0) / n;
        thr *= tighten;
        thr *= offset_lin;
        if (thr < L->abs_thr[b])
            thr = L->abs_thr[b];
        out->thr[b] = (float)thr;
    }
}

void mmx_psy_pre_attack(const MMXBandLayout *L, const MMXBandLayout *pre_bands, MMXSpectrum *pre_spec,
                        const float *pre, unsigned int quality, double margin_db, MMXFramePsy *out)
{
    MMXFramePsy pp;
    float amp[MMX_PRE_ATTACK / 2];
    unsigned int b, j = 0;
    /* per-coefficient power of the same noise grows with the window length */
    double scale = pow(10.0, -margin_db / 10.0) * (double)L->m / (double)pre_bands->m;
    double bin_hz = (double)L->sample_rate / 2.0 / (double)pre_bands->m;

    if (!out->transient)
        return;
    mmx_spectrum_analyze(pre_spec, pre, amp);
    mmx_psy_analyze(pre_bands, amp, NULL, quality, &pp);
    for (b = 0; b < L->band_count; b++)
    {
        double t;
        if (out->thr[b] >= 1e29f)
            continue;
        while (j + 1 < pre_bands->band_count && pre_bands->band_start[j + 1] * bin_hz <= L->band_hz[b])
            j++;
        t = pp.thr[j] >= 1e29f ? 1e30 : (double)pp.thr[j] * scale;
        if (t < L->abs_thr[b]) t = L->abs_thr[b];
        if (t < out->thr[b]) out->thr[b] = (float)t;
    }
}

/* Quantization noise of an MDCT band leaks into the neighbouring bands at
   about -14 dB (main lobe of the window). Thresholds must therefore not drop
   by more than ADJ_LEAK_DB from one band to the next, otherwise the allowed
   noise of a loud band lands above the threshold of a quiet neighbour. */
#define ADJ_LEAK_DB 14.0

void mmx_psy_limit_cliffs(MMXFramePsy *p, unsigned int band_count)
{
    float f = (float)pow(10.0, ADJ_LEAK_DB / 10.0);
    unsigned int b, pass;
    for (pass = 0; pass < 3; pass++)
        for (b = 0; b + 1 < band_count; b++)
        {
            if (p->thr[b] >= 1e29f || p->thr[b + 1] >= 1e29f)
                continue;
            if (p->thr[b] > p->thr[b + 1] * f) p->thr[b] = p->thr[b + 1] * f;
            if (p->thr[b + 1] > p->thr[b] * f) p->thr[b + 1] = p->thr[b] * f;
        }
}

#define POST_MASKING_DB 6.0

void mmx_psy_temporal(const MMXFramePsy *prev, MMXFramePsy *cur, const MMXFramePsy *next)
{
    unsigned int b;
    double post = pow(10.0, POST_MASKING_DB / 10.0);
    for (b = 0; b < MMX_MAX_BANDS; b++)
    {
        float t = cur->thr[b];
        if (t >= 1e29f)
            continue;
        if (prev && prev->thr[b] < t)
            t = prev->thr[b];                       /* pre-echo: must stay under the earlier, quieter frame */
        if (next && next->thr[b] * (float)post < t)
            t = next->thr[b] * (float)post;         /* post-masking of the later frame */
        cur->thr[b] = t;
    }
}

void mmx_psy_stereo_threshold(const MMXBandLayout *L, const MMXFramePsy *left, const MMXFramePsy *right,
                              float *thr_out)
{
    unsigned int b;
    for (b = 0; b < L->band_count; b++)
    {
        float t = left->thr[b] < right->thr[b] ? left->thr[b] : right->thr[b];
        thr_out[b] = t >= 1e29f ? t : t * 0.5f;
    }
}

double mmx_psy_nmr_db(const MMXBandLayout *L, const MMXFramePsy *p, unsigned int band, double noise_energy)
{
    double n = (double)(L->band_start[band + 1] - L->band_start[band]);
    double allowed = (double)p->thr[band] * n;
    if (allowed <= 0.0)
        return 200.0;
    if (noise_energy <= 0.0)
        return -200.0;
    return 10.0 * log10(noise_energy / allowed);
}
