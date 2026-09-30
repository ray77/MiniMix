#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>
#include "minimix/quality.h"
#include "minimix/codec.h"

double mmx_quality_frame_nmr(const MMXCodec *codec, const MMXFramePsy *psy,
                             const float *orig, const float *rec, const unsigned char *skip, unsigned int *bands_over)
{
    const MMXBandLayout *L = &codec->bands;
    unsigned int b, over = 0;
    double worst = -200.0;
    for (b = 0; b < codec->cutoff_band; b++)
    {
        unsigned long k;
        double noise = 0.0, nmr, n = (double)(L->band_start[b + 1] - L->band_start[b]);
        if (skip && skip[b])
            continue;
        for (k = L->band_start[b]; k < L->band_start[b + 1]; k++)
            noise += (double)(rec[k] - orig[k]) * (rec[k] - orig[k]);
        if (psy->energy[b] <= psy->thr[b] * n * 0.5)
            continue; /* band carried no audible signal; noise below threshold by construction */
        nmr = mmx_psy_nmr_db(L, psy, b, noise);
        if (nmr > 0.0)
            over++;
        if (nmr > worst)
            worst = nmr;
    }
    if (bands_over)
        *bands_over = over;
    return worst;
}

/* Sub-frame analysis: 1024-sample windows every 512 samples, Blackman-Harris
   windowed FFT (sidelobes -92 dB) so the allowed low-frequency noise is not
   smeared into the high bands by the analysis itself (a sine-windowed MDCT
   leaks at -24 dB). Power spectra are scaled to the MDCT calibration of the
   band layout so the absolute threshold of hearing stays valid. 11.6 ms
   resolution attributes noise to the region it really lands in. */
#include "spectrum.h"
#define SUB_WIN 1024
#define SUB_HOP 512

/* Amplitude spectrum of a window of `buf` channel ch starting at `start`;
   `diff` (optional) subtracts a second buffer first (the error signal). */
static void sub_spectrum(MMXSpectrum *sp, const MMXAudioBuffer *buf, const MMXAudioBuffer *diff, long long start,
                         unsigned int ch, float *win, float *amp)
{
    long long i;
    unsigned int nch = buf->channels;
    for (i = 0; i < SUB_WIN; i++)
    {
        long long p = start + i;
        double v = (p >= 0 && p < (long long)buf->frame_count) ? buf->samples[(size_t)p * nch + ch] : 0.0;
        if (diff && p >= 0 && p < (long long)diff->frame_count)
            v -= diff->samples[(size_t)p * nch + ch];
        win[i] = (float)v;
    }
    mmx_spectrum_analyze(sp, win, amp);
}

/* --- beyond the model: what the NMR per band does not represent --- */

#define GHOST_BLOCK 256      /* 5.8 ms */
#define GHOST_BEFORE 8       /* blocks before the attack examined (46 ms) */
#define GHOST_AFTER 4        /* ... after it (23 ms) */
#define GHOST_FLOOR_DBFS -70.0
#define GHOST_LIST 64

typedef struct
{
    double seconds, attack_seconds, esr_db, err_dbfs, sig_dbfs;
    int before;
} GhostPlace;

static void ghost_insert(GhostPlace *list, unsigned long *n, const GhostPlace *g)
{
    unsigned long i, j;
    for (i = 0; i < *n && list[i].esr_db >= g->esr_db; i++)
        ;
    if (i >= GHOST_LIST)
        return;
    if (*n < GHOST_LIST) (*n)++;
    for (j = *n - 1; j > i; j--) list[j] = list[j - 1];
    list[i] = *g;
}

/* Time-domain error envelope around an attack at sample `at`: in every 5.8 ms
   block of the 46 ms before and the 23 ms after it, the error energy against
   the signal energy of the block. An error louder than the signal in a block
   before the attack is a ghost note (a source hit that lands early in tracker
   mode) or pre-echo, whatever the band NMR says about the 23 ms window. */
static void attack_envelope(const MMXAudioBuffer *o, const MMXAudioBuffer *d, unsigned long at, MMXQualityReport *r,
                            GhostPlace *list, unsigned long *n_list, int keep_list)
{
    int j, ghost = 0;
    unsigned int nch = o->channels;
    for (j = -GHOST_BEFORE; j < GHOST_AFTER; j++)
    {
        long long b0 = (long long)at + (long long)j * GHOST_BLOCK, b1 = b0 + GHOST_BLOCK, i;
        double es = 0.0, ee = 0.0, esr, err_dbfs, sig_dbfs;
        GhostPlace g;
        if (b0 < 0 || b1 > (long long)o->frame_count)
            continue;
        for (i = b0 * nch; i < b1 * nch; i++)
        {
            double x = o->samples[i], e = (double)d->samples[i] - x;
            es += x * x;
            ee += e * e;
        }
        err_dbfs = 10.0 * log10(ee / (double)(GHOST_BLOCK * nch) + 1e-30);
        sig_dbfs = 10.0 * log10(es / (double)(GHOST_BLOCK * nch) + 1e-30);
        if (err_dbfs < GHOST_FLOOR_DBFS)
            continue;
        esr = err_dbfs - sig_dbfs;
        if (j < 0 && esr > 0.0)
        {
            ghost = 1;
            if (esr > r->worst_ghost_db)
            {
                r->worst_ghost_db = esr;
                r->worst_ghost_seconds = (double)b0 / o->sample_rate;
            }
        }
        if (keep_list)
        {
            g.seconds = (double)b0 / o->sample_rate;
            g.attack_seconds = (double)at / o->sample_rate;
            g.esr_db = esr;
            g.err_dbfs = err_dbfs;
            g.sig_dbfs = sig_dbfs;
            g.before = j < 0;
            ghost_insert(list, n_list, &g);
        }
    }
    if (ghost) r->attacks_ghost++;
}

/* Clipping (flat tops at the full scale of the source bit depth), DC offset of
   the error per second, and border effects: the first-difference energy of the
   error signal next to the 1024 and 128 sample grids against its mean. */
static void beyond_model(const MMXAudioBuffer *o, const MMXAudioBuffer *d, MMXQualityReport *r)
{
    unsigned int nch = o->channels, c, bits = o->source_bits ? o->source_bits : 16;
    unsigned long long i, n = o->frame_count;
    double fs = (pow(2.0, (double)bits - 1.0) - 1.0) / pow(2.0, (double)bits - 1.0) - 1e-7;
    double diff_all = 0.0, diff_1024 = 0.0, diff_128 = 0.0, dc_sum[MMX_MAX_CH], dc_orig[MMX_MAX_CH], dc_worst = 0.0;
    unsigned long long n_all = 0, n_1024 = 0, n_128 = 0, sec_len = o->sample_rate, in_sec = 0, sec = 0;
    int debug_levels = getenv("MMX_DEBUG_LEVELS") != NULL;
    memset(dc_sum, 0, sizeof(dc_sum));
    memset(dc_orig, 0, sizeof(dc_orig));
    for (i = 0; i < n; i++)
    {
        unsigned long p1024 = (unsigned long)(i % 1024), p128 = (unsigned long)(i % 128);
        int near_1024 = p1024 < 8 || p1024 >= 1024 - 8, near_128 = p128 < 8 || p128 >= 128 - 8;
        for (c = 0; c < nch; c++)
        {
            double xo = o->samples[i * nch + c], xd = d->samples[i * nch + c], e = xd - xo;
            int fso = fabs(xo) >= fs, fsd = fabs(xd) >= fs;
            r->clipped_decoded += fsd;
            r->clipped_original += fso;
            r->clipped_new += fsd && !fso;
            dc_sum[c] += e;
            dc_orig[c] += xo;
            if (i > 0)
            {
                double de = e - ((double)d->samples[(i - 1) * nch + c] - o->samples[(i - 1) * nch + c]);
                diff_all += de * de;
                n_all++;
                if (near_1024) { diff_1024 += de * de; n_1024++; }
                if (near_128) { diff_128 += de * de; n_128++; }
            }
        }
        if (++in_sec == sec_len || i + 1 == n)
        {
            for (c = 0; c < nch; c++)
            {
                double dc = fabs(dc_sum[c] / (double)in_sec);
                if (dc > dc_worst) dc_worst = dc;
                if (debug_levels && dc > pow(10.0, -55.0 / 20.0))
                    fprintf(stderr, "DC of the error at %llu s ch %u: %.1f dBFS (the original's DC there %.1f dBFS)\n", sec, c,
                            20.0 * log10(dc), 20.0 * log10(fabs(dc_orig[c] / (double)in_sec) + 1e-30));
                dc_sum[c] = 0.0;
                dc_orig[c] = 0.0;
            }
            in_sec = 0;
            sec++;
        }
    }
    r->dc_worst_db = dc_worst > 0.0 ? 20.0 * log10(dc_worst) : -200.0;
    r->border_1024_db = n_1024 && diff_all > 0.0 ? 10.0 * log10((diff_1024 / (double)n_1024) / (diff_all / (double)n_all)) : 0.0;
    r->border_128_db = n_128 && diff_all > 0.0 ? 10.0 * log10((diff_128 / (double)n_128) / (diff_all / (double)n_all)) : 0.0;
}

/* --- weighted score: a second yardstick next to bands_over_percent ---------

   Why a second one. `bands_over_percent` counts (11.6 ms, band, channel) cells
   whose noise exceeds the masking threshold, each cell worth one count. Two
   things that the ear does are missing from that bookkeeping, and both of them
   show up exactly where the unweighted number stops explaining listening
   impressions (Opus at 128 kbit/s scores 80 % of the cells "over" in the
   8-12 kHz region on dense rock and still sounds cleaner than a decoder that
   scores 41 % there):

   (1) Cells are not equal. A cell is one half-bark band of one 11.6 ms window.
       How much of the cochlea it occupies and how much loudness it carries vary
       by more than an order of magnitude across the spectrum, and a cell that
       sits 1 dB above the absolute threshold of hearing is counted exactly like
       a fortissimo cell at 1 kHz.

   (2) Above roughly 5-8 kHz the auditory system does not resolve temporal fine
       structure: phase locking in the auditory nerve decays from about 1.5 kHz
       and is essentially gone above 4-5 kHz (Palmer & Russell 1986; Moore,
       "The Role of Temporal Fine Structure in Normal and Impaired Hearing",
       2014), so what reaches the percept there is the band envelope. That is
       the explicit basis of perceptual noise substitution (Herre & Schulz,
       "Extending the MPEG-4 AAC Codec by Perceptual Noise Substitution",
       AES 104, 1998), of spectral band replication, and of CELT's per-band
       energy conservation in Opus (Valin, Vos, Terriberry, "High-Quality,
       Low-Delay Music Coding in the Opus Codec", AES 135, 2013). A waveform
       metric counts an energy-matched noise band up there as a total loss; the
       ear counts it as (nearly) nothing: the model cannot judge `--pns`; listen.

   The weighting. Every evaluated cell (the same cells, the same 0 dB
   criterion) carries

       w(cell) = w_erb(band) * N'(original energy of the cell)

   w_erb(band) is the width of the band on the ERB-rate scale of Glasberg &
   Moore ("Derivation of auditory filter shapes from notched-noise data",
   Hearing Research 47, 1990): ERB(f) = 24.7 (4.37 f/kHz + 1) Hz, cumulative
   E(f) = 21.4 log10(4.37 f/kHz + 1). Integrating a distortion measure over the
   ERB-rate scale rather than over a list of bands is what ITU-R BS.1387 (PEAQ)
   does with its ERB-spaced filterbank; it is the "amount of cochlea" this band
   occupies. MiniMix's bands are half-bark, and Bark and ERB do not track each
   other below ~500 Hz (a half-bark band at 100 Hz spans ~1.4 ERB, one at
   10 kHz ~0.46 ERB), so this alone moves weight from the highs to the lows.

   N'(E) is Zwicker's specific loudness (ISO 532 B; Fastl & Zwicker,
   "Psychoacoustics", 3rd ed., ch. 8):

       N'(E) = (0.5 + 0.5 * E / E_TQ)^0.23 - 1,     0 for E <= E_TQ

   with E_TQ the band's threshold in quiet (MMXBandLayout.abs_thr times the
   number of coefficients). The 0.23 exponent is the compressive loudness law.
   The standard form carries a prefactor (E_TQ/E_0)^0.23; it is dropped here on
   purpose: that prefactor exists because E is an absolute intensity, while in
   MiniMix E_TQ *is* the inverse transfer function of outer and middle ear, so
   referring E to E_TQ already performs the transmission correction that a
   loudness model applies before the compression. Keeping the prefactor would
   apply the ear's frequency response a second time. The consequence is the
   intended one: a cell just above the threshold of hearing weighs ~0, a cell
   30 dB above it weighs ~3, one 60 dB above it ~20 - compressive, not linear.

   The crossover. Above MMX_W_CROSSOVER_HZ the cell's error energy is replaced
   by the band *energy* error,

       noise_eff = (sqrt(E_decoded) - sqrt(E_original))^2

   i.e. the error that a correctly-enveloped but differently-shaped band would
   produce, fed through the very same masking threshold and NMR formula. An
   energy-matched noise band scores 0 (the ear cannot tell it apart), a hole
   scores the full band energy (the ear hears it), a band that is 6 dB too loud
   scores accordingly. 8 kHz is chosen as a deliberately *conservative* value
   inside the 4-10 kHz range the literature above supports (phase locking is
   gone by 5 kHz; AAC-PNS and SBR crossovers sit at 4-10 kHz); MMX_W_CROSSOVER
   overrides it for sensitivity checks.

   The numbers reported. w_over_percent (the share of the perceptual weight
   whose cell is above the threshold - the direct analogue of
   bands_over_percent), w_mean_nmr_db (the weighted mean NMR), and the noise
   loudness

       100 * sum w_erb * N'(noise / masked threshold) / sum w_erb * N'(E / E_TQ)

   the loudness of the audible, unmasked noise as a percentage of the loudness
   of the music - a graded number, where the two "over" shares are yes/no.
   Separately, and not folded into any of them, the high band energy error.

   What this deliberately does NOT do: it does not change a single coding
   decision, it does not replace the unweighted number, and none of its
   constants were fitted to make MiniMix win. */

#define MMX_W_CROSSOVER_HZ 8000.0
#define MMX_W_EXPONENT 0.23

/* ERB-rate (Glasberg & Moore 1990) of a frequency in Hz. */
static double erb_rate(double hz)
{
    if (hz < 0.0) hz = 0.0;
    return 21.4 * log10(4.37e-3 * hz + 1.0);
}

/* Zwicker specific loudness of an energy referred to a reference energy
   (threshold in quiet, or the masking threshold for the noise). */
static double zwicker_loudness(double e, double e_ref)
{
    double r;
    if (e_ref <= 0.0 || e <= 0.0)
        return 0.0;
    r = 0.5 + 0.5 * e / e_ref;
    return r > 1.0 ? pow(r, MMX_W_EXPONENT) - 1.0 : 0.0;
}

/* MMX_W_LOUDNESS=0 drops the loudness factor and weights by ERB width alone
   (a sensitivity knob for the weighting, not a second yardstick). */
static int weighted_use_loudness(void)
{
    const char *e = getenv("MMX_W_LOUDNESS");
    return e ? atoi(e) != 0 : 1;
}

static double weighted_crossover_hz(void)
{
    const char *e = getenv("MMX_W_CROSSOVER");
    double v = e ? atof(e) : 0.0;
    return v > 0.0 ? v : MMX_W_CROSSOVER_HZ;
}

int mmx_quality_compare(const MMXAudioBuffer *original, const MMXAudioBuffer *decoded, unsigned int quality,
                        unsigned int max_hz, double tolerance_db, MMXQualityReport *report)
{
    MMXBandLayout L, pre_bands;
    MMXSpectrum sa, pre_spec;
    MMXFramePsy psy;
    float *win, *orig_coefs, *err_coefs, *dec_coefs, pre_win[MMX_PRE_ATTACK];
    double pre_attack_db = mmx_psy_pre_attack_db();
    unsigned long s, ns, i, n, bands_total = 0, bands_over_total = 0, frames_over = 0, worst_frame = 0, cutoff, nsec;
    unsigned int c, worst_channel = 0, worst_band = 0, b, e;
    double sum_nmr = 0.0, worst = -200.0, err2 = 0.0, sig2 = 0.0, maxdiff = 0.0, scale;
    unsigned long eq_total[MMX_EQ_BANDS], eq_over[MMX_EQ_BANDS];
    double eq_sum[MMX_EQ_BANDS], eq_worst[MMX_EQ_BANDS];
    double *lvl;   /* band energy per second and EQ region, [sec][region][original / decoded] */
    /* gurgling: the level difference decoded - original per sub-window and band (see quality.h) */
    double *wdif = NULL;
    unsigned char *wok = NULL;
    double worig[MMX_MAX_BANDS], wdec[MMX_MAX_BANDS];
    int wrow;
    /* stereo image: complex spectra of both channels (original, decoded) and the accumulators per region */
    double *cplx = NULL, *ore[2], *oim[2], *dre[2], *dim[2];
    unsigned long st_cells[MMX_EQ_BANDS], st_ild_over[MMX_EQ_BANDS], st_coh_over[MMX_EQ_BANDS];
    double st_ild[MMX_EQ_BANDS], st_coh[MMX_EQ_BANDS];
    int stereo = original->channels == 2;
    /* weighted score (see "weighted score" above): ERB width per band, the
       crossover band, and the accumulators */
    double w_erb[MMX_MAX_BANDS], w_sum = 0.0, w_over_sum = 0.0, w_nmr_sum = 0.0;
    double nl_noise_sum = 0.0, nl_sig_sum = 0.0, cross_hz = weighted_crossover_hz();
    int w_loudness = weighted_use_loudness();
    double eq_w_sum[MMX_EQ_BANDS], eq_w_over[MMX_EQ_BANDS];
    unsigned int hi_from;
    unsigned long hi_cells = 0, hi_over3 = 0;
    double hi_abs_sum = 0.0, hi_bias_sum = 0.0, hi_worst = 0.0;

    memset(report, 0, sizeof(*report));
    for (e = 0; e < MMX_EQ_BANDS; e++)
    {
        eq_total[e] = eq_over[e] = 0; eq_sum[e] = 0.0; eq_worst[e] = -200.0;
        st_cells[e] = st_ild_over[e] = st_coh_over[e] = 0; st_ild[e] = st_coh[e] = 0.0;
        eq_w_sum[e] = eq_w_over[e] = 0.0;
    }
    if (original->sample_rate != decoded->sample_rate || original->channels != decoded->channels ||
        original->frame_count != decoded->frame_count)
        return -1;
    if (mmx_bands_init(&L, original->sample_rate, SUB_WIN / 2) != 0 || mmx_spectrum_init(&sa, SUB_WIN) != 0 ||
        mmx_bands_init(&pre_bands, original->sample_rate, MMX_PRE_ATTACK / 2) != 0 || mmx_spectrum_init(&pre_spec, MMX_PRE_ATTACK) != 0)
        return -1;
    cutoff = L.cutoff_band[quality > MMX_QUALITY_MAX ? MMX_QUALITY_MAX : quality];
    if (max_hz)
        for (b = 0; b < cutoff; b++)
            if (L.band_start[b] * (double)original->sample_rate / SUB_WIN >= (double)max_hz) { cutoff = b; break; }
    hi_from = (unsigned int)cutoff;
    for (b = 0; b < cutoff; b++)
    {
        double f0 = L.band_start[b] * (double)original->sample_rate / SUB_WIN;
        double f1 = L.band_start[b + 1] * (double)original->sample_rate / SUB_WIN;
        w_erb[b] = erb_rate(f1) - erb_rate(f0);
        if (hi_from == (unsigned int)cutoff && (double)L.band_hz[b] >= cross_hz)
            hi_from = b;
    }
    win = (float *)malloc(sizeof(float) * SUB_WIN);
    orig_coefs = (float *)malloc(sizeof(float) * SUB_WIN / 2);
    err_coefs = (float *)malloc(sizeof(float) * SUB_WIN / 2);
    dec_coefs = (float *)malloc(sizeof(float) * SUB_WIN / 2);
    nsec = (unsigned long)(original->frame_count / original->sample_rate) + 2;
    lvl = (double *)calloc((size_t)nsec * MMX_EQ_BANDS * 2, sizeof(double));
    ns = (unsigned long)(original->frame_count / SUB_HOP) + 3;
    wdif = (double *)calloc((size_t)ns * (cutoff ? cutoff : 1), sizeof(double));
    wok = (unsigned char *)calloc((size_t)ns * (cutoff ? cutoff : 1), 1);
    if (stereo)
        cplx = (double *)calloc((size_t)8 * SUB_WIN / 2, sizeof(double));
    if (!win || !orig_coefs || !err_coefs || !dec_coefs || !lvl || !wdif || !wok || (stereo && !cplx))
    {
        free(win); free(orig_coefs); free(err_coefs); free(dec_coefs); free(lvl); free(cplx); free(wdif); free(wok);
        mmx_spectrum_free(&sa);
        mmx_spectrum_free(&pre_spec);
        return -1;
    }
    for (c = 0; c < 2; c++)
    {
        ore[c] = cplx ? cplx + (size_t)(4 * c) * SUB_WIN / 2 : NULL;
        oim[c] = cplx ? cplx + (size_t)(4 * c + 1) * SUB_WIN / 2 : NULL;
        dre[c] = cplx ? cplx + (size_t)(4 * c + 2) * SUB_WIN / 2 : NULL;
        dim[c] = cplx ? cplx + (size_t)(4 * c + 3) * SUB_WIN / 2 : NULL;
    }

    for (s = 0; s < ns; s++)
    {
        long long start = (long long)s * SUB_HOP - SUB_HOP * 2; /* same grid as the encoder's sub-windows */
        int frame_over = 0;
        int have_dec = start + SUB_WIN / 2 >= 0;   /* dec_coefs is only filled then (weighted score) */
        wrow = 0;
        for (b = 0; b < cutoff; b++) worig[b] = wdec[b] = 0.0;
        static double debug_time = -2.0;
        if (debug_time == -2.0) { const char *e = getenv("MMX_DEBUG_TIME"); debug_time = e ? atof(e) : -1.0; }
        for (c = 0; c < original->channels; c++)
        {
            sub_spectrum(&sa, original, NULL, start, c, win, orig_coefs);
            if (stereo)
            {
                memcpy(ore[c], sa.re, sizeof(double) * SUB_WIN / 2);
                memcpy(oim[c], sa.im, sizeof(double) * SUB_WIN / 2);
            }
            mmx_psy_analyze(&L, orig_coefs, win, quality, &psy);
            if (psy.transient && pre_attack_db >= 0.0)
            {
                long long j, end = start + psy.attack_pos;
                for (j = 0; j < MMX_PRE_ATTACK; j++)
                {
                    long long p = end - MMX_PRE_ATTACK + j;
                    pre_win[j] = (p >= 0 && p < (long long)original->frame_count) ? original->samples[(size_t)p * original->channels + c] : 0.0f;
                }
                mmx_psy_pre_attack(&L, &pre_bands, &pre_spec, pre_win, quality, pre_attack_db, &psy);
            }
            sub_spectrum(&sa, decoded, original, start, c, win, err_coefs);
            /* band levels per second and EQ region, decoded against original (tracker: a repeat
               played from its source at the wrong level; not an error the NMR sees as such) */
            if (start + SUB_WIN / 2 >= 0)
            {
                unsigned long sec = (unsigned long)((start + SUB_WIN / 2) / (long long)original->sample_rate);
                sub_spectrum(&sa, decoded, NULL, start, c, win, dec_coefs);
                if (stereo)
                {
                    memcpy(dre[c], sa.re, sizeof(double) * SUB_WIN / 2);
                    memcpy(dim[c], sa.im, sizeof(double) * SUB_WIN / 2);
                }
                /* stereo image of the cell: both channels are known once the second is analysed */
                if (stereo && c == 1)
                    for (b = 0; b < cutoff; b++)
                    {
                        unsigned long k, nb = L.band_start[b + 1] - L.band_start[b];
                        double el = 0.0, er = 0.0, cr = 0.0, fl = 0.0, fr = 0.0, fc = 0.0, d_ild, d_coh;
                        for (k = L.band_start[b]; k < L.band_start[b + 1]; k++)
                        {
                            el += ore[0][k] * ore[0][k] + oim[0][k] * oim[0][k];
                            er += ore[1][k] * ore[1][k] + oim[1][k] * oim[1][k];
                            cr += ore[0][k] * ore[1][k] + oim[0][k] * oim[1][k];
                            fl += dre[0][k] * dre[0][k] + dim[0][k] * dim[0][k];
                            fr += dre[1][k] * dre[1][k] + dim[1][k] * dim[1][k];
                            fc += dre[0][k] * dre[1][k] + dim[0][k] * dim[1][k];
                        }
                        /* cells with signal in both channels (30 dB SPL, as the level check) */
                        if (el <= 0.0 || er <= 0.0 || mmx_psy_spl_db(&L, el * sa.scale / (double)nb) < 30.0 ||
                            mmx_psy_spl_db(&L, er * sa.scale / (double)nb) < 30.0)
                            continue;
                        d_ild = fabs(10.0 * log10((fl + 1e-30) / (fr + 1e-30)) - 10.0 * log10(el / er));
                        d_coh = fabs((fl > 0.0 && fr > 0.0 ? fc / sqrt(fl * fr) : 0.0) - cr / sqrt(el * er));
                        e = L.eq_band[b];
                        st_cells[e]++;
                        st_ild[e] += d_ild;
                        st_coh[e] += d_coh;
                        if (d_ild > 2.0) st_ild_over[e]++;
                        if (d_coh > 0.3) st_coh_over[e]++;
                    }
                if (sec < nsec)
                    for (b = 0; b < cutoff; b++)
                    {
                        unsigned long k;
                        double eo = 0.0, ed = 0.0;
                        for (k = L.band_start[b]; k < L.band_start[b + 1]; k++)
                        {
                            eo += (double)orig_coefs[k] * orig_coefs[k];
                            ed += (double)dec_coefs[k] * dec_coefs[k];
                        }
                        lvl[(sec * MMX_EQ_BANDS + L.eq_band[b]) * 2] += eo;
                        lvl[(sec * MMX_EQ_BANDS + L.eq_band[b]) * 2 + 1] += ed;
                        worig[b] += eo;
                        wdec[b] += ed;
                        wrow = 1;
                    }
            }
            if (debug_time >= 0.0 && start <= (long long)(debug_time * original->sample_rate) && start + SUB_WIN > (long long)(debug_time * original->sample_rate))
            {
                fprintf(stderr, "compare sub-window start %lld (%.3f s) ch %u transient %d tonality %.2f\n", start, (double)start / original->sample_rate, c, psy.transient, psy.tonality);
                for (b = 0; b < cutoff; b++)
                {
                    unsigned long k;
                    double noise = 0.0, nb = (double)(L.band_start[b + 1] - L.band_start[b]);
                    for (k = L.band_start[b]; k < L.band_start[b + 1]; k++)
                        noise += (double)err_coefs[k] * err_coefs[k];
                    fprintf(stderr, "  band %2u (%5.0f Hz) energy %10.4g thr*n %10.4g noise %10.4g nmr %7.2f\n", b, L.band_hz[b], psy.energy[b],
                            psy.thr[b] * nb, noise, mmx_psy_nmr_db(&L, &psy, b, noise));
                }
            }
            for (b = 0; b < cutoff; b++)
            {
                unsigned long k;
                double noise = 0.0, nmr, nb = (double)(L.band_start[b + 1] - L.band_start[b]);
                if (psy.thr[b] >= 1e29f || psy.energy[b] <= psy.thr[b] * nb * 0.5)
                    continue;
                for (k = L.band_start[b]; k < L.band_start[b + 1]; k++)
                    noise += (double)err_coefs[k] * err_coefs[k];
                nmr = mmx_psy_nmr_db(&L, &psy, b, noise);
                sum_nmr += nmr;
                bands_total++;
                if (nmr > 0.0) { bands_over_total++; frame_over = 1; }
                if (nmr > worst) { worst = nmr; worst_frame = s; worst_channel = c; worst_band = b; }
                e = L.eq_band[b];
                eq_total[e]++;
                eq_sum[e] += nmr;
                if (nmr > 0.0) eq_over[e]++;
                if (nmr > eq_worst[e]) eq_worst[e] = nmr;
                /* the same cell on the weighted yardstick */
                if (have_dec)
                {
                    double e_tq = (double)L.abs_thr[b] * nb, eo = (double)psy.energy[b], ed = 0.0;
                    double noise_w, nmr_w, w;
                    for (k = L.band_start[b]; k < L.band_start[b + 1]; k++)
                        ed += (double)dec_coefs[k] * dec_coefs[k];
                    if (b >= hi_from)
                    {
                        double da = sqrt(ed) - sqrt(eo);   /* band energy error, not waveform error */
                        double d_db = 10.0 * log10((ed + 1e-30) / (eo + 1e-30));
                        noise_w = da * da;
                        hi_cells++;
                        hi_abs_sum += fabs(d_db);
                        hi_bias_sum += d_db;
                        if (fabs(d_db) > 3.0) hi_over3++;
                        if (fabs(d_db) > fabs(hi_worst)) hi_worst = d_db;
                    }
                    else
                        noise_w = noise;
                    nmr_w = mmx_psy_nmr_db(&L, &psy, b, noise_w);
                    w = w_loudness ? w_erb[b] * zwicker_loudness(eo, e_tq) : w_erb[b];
                    w_sum += w;
                    w_nmr_sum += w * nmr_w;
                    eq_w_sum[e] += w;
                    if (nmr_w > 0.0) { w_over_sum += w; eq_w_over[e] += w; }
                    nl_noise_sum += w_erb[b] * zwicker_loudness(noise_w, (double)psy.thr[b] * nb);
                    nl_sig_sum += w_erb[b] * zwicker_loudness(eo, e_tq);
                }
            }
        }
        if (frame_over) frames_over++;
        /* the cell of the gurgling metric: both levels floored at the absolute
           threshold of hearing, the cell counts when the original is above it */
        if (wrow)
            for (b = 0; b < cutoff; b++)
            {
                double fl = (double)L.abs_thr[b] * (double)(L.band_start[b + 1] - L.band_start[b]) * original->channels;
                if (worig[b] <= fl)
                    continue;
                wok[(size_t)s * cutoff + b] = 1;
                wdif[(size_t)s * cutoff + b] = 10.0 * log10((wdec[b] > fl ? wdec[b] : fl) / worig[b]);
            }
    }
    /* per second: the fluctuation of those differences (warble index), the
       holes and how often a band's hole state changes (toggles) */
    {
        double sum[MMX_MAX_BANDS], sum2[MMX_MAX_BANDS], warble_sum = 0.0;
        double worst_w[10], worst_t[10];
        unsigned long cnt[MMX_MAX_BANDS], tg[MMX_MAX_BANDS], worst_ws[10], worst_ts[10];
        signed char state[MMX_MAX_BANDS];
        unsigned long cells = 0, holes = 0, toggles = 0, secs = 0, cur = 0;
        int have = 0, debug = getenv("MMX_DEBUG_WARBLE") != NULL;
        unsigned int j;
        for (j = 0; j < 10; j++) { worst_w[j] = worst_t[j] = 0.0; worst_ws[j] = worst_ts[j] = 0; }
        for (s = 0; s <= ns; s++)
        {
            double t = (double)((long long)s * SUB_HOP - SUB_HOP * 2 + SUB_WIN / 2) / (double)original->sample_rate;
            unsigned long sec = t >= 0.0 ? (unsigned long)t : 0;
            if (s < ns && t < 0.0)
                continue;
            if (have && (s == ns || sec != cur))
            {
                double idx = 0.0;
                unsigned long nb = 0, tgs = 0;
                for (b = 0; b < cutoff; b++)
                {
                    if (cnt[b] >= 2)
                    {
                        double m = sum[b] / (double)cnt[b], v = sum2[b] / (double)cnt[b] - m * m;
                        idx += v > 0.0 ? v : 0.0;
                        nb++;
                    }
                    tgs += tg[b];
                }
                if (nb)
                {
                    double w = sqrt(idx / (double)nb);
                    warble_sum += w;
                    secs++;
                    toggles += tgs;
                    if (w > report->warble_worst_db) { report->warble_worst_db = w; report->warble_worst_seconds = (double)cur; }
                    if ((double)tgs > report->toggles_worst) { report->toggles_worst = (double)tgs; report->toggles_worst_seconds = (double)cur; }
                    for (j = 0; j < 10; j++)
                        if (w > worst_w[j]) { unsigned int q; for (q = 9; q > j; q--) { worst_w[q] = worst_w[q - 1]; worst_ws[q] = worst_ws[q - 1]; } worst_w[j] = w; worst_ws[j] = cur; break; }
                    for (j = 0; j < 10; j++)
                        if ((double)tgs > worst_t[j]) { unsigned int q; for (q = 9; q > j; q--) { worst_t[q] = worst_t[q - 1]; worst_ts[q] = worst_ts[q - 1]; } worst_t[j] = (double)tgs; worst_ts[j] = cur; break; }
                }
                have = 0;
            }
            if (s == ns)
                break;
            if (!have)
            {
                for (b = 0; b < cutoff; b++) { sum[b] = sum2[b] = 0.0; cnt[b] = tg[b] = 0; state[b] = -1; }
                cur = sec;
                have = 1;
            }
            for (b = 0; b < cutoff; b++)
            {
                double d;
                int hole;
                if (!wok[(size_t)s * cutoff + b])
                    continue;
                d = wdif[(size_t)s * cutoff + b];
                sum[b] += d;
                sum2[b] += d * d;
                cnt[b]++;
                cells++;
                hole = d < -MMX_WARBLE_HOLE_DB;
                if (hole) holes++;
                if (state[b] >= 0 && hole != (int)state[b]) tg[b]++;
                state[b] = (signed char)hole;
            }
        }
        report->warble_db = secs ? warble_sum / (double)secs : 0.0;
        report->hole_percent = cells ? 100.0 * (double)holes / (double)cells : 0.0;
        report->toggles_per_second = secs ? (double)toggles / (double)secs : 0.0;
        if (debug)
        {
            fprintf(stderr, "warble: index %.2f dB, holes %.2f %%, %.2f band toggles per second over %lu s\n",
                    report->warble_db, report->hole_percent, report->toggles_per_second, secs);
            fprintf(stderr, "  worst seconds by warble index:");
            for (j = 0; j < 10 && worst_w[j] > 0.0; j++) fprintf(stderr, " %lus(%.1f)", worst_ws[j], worst_w[j]);
            fprintf(stderr, "\n  worst seconds by toggles:");
            for (j = 0; j < 10 && worst_t[j] > 0.0; j++) fprintf(stderr, " %lus(%.0f)", worst_ts[j], worst_t[j]);
            fprintf(stderr, "\n");
        }
    }
    /* band level per second: cells whose original level is above 30 dB SPL (-66 dBFS) count */
    {
        unsigned long sec, cells = 0, over1 = 0, over3 = 0, per_sec = original->sample_rate / SUB_HOP * original->channels;
        int debug_levels = getenv("MMX_DEBUG_LEVELS") != NULL;
        report->level_worst_db = 0.0;
        if (debug_levels) fprintf(stderr, "band levels per second, decoded - original (dB), regions <200 200-500 500-1k 1k-2k 2k-4k 4k-8k 8k-12k >12k Hz:\n");
        for (sec = 0; sec < nsec; sec++)
        {
            int shown = 0;
            for (e = 0; e < MMX_EQ_BANDS; e++)
            {
                double eo = lvl[(sec * MMX_EQ_BANDS + e) * 2], ed = lvl[(sec * MMX_EQ_BANDS + e) * 2 + 1], d;
                if (eo <= 0.0 || mmx_psy_spl_db(&L, eo / (double)(per_sec ? per_sec : 1)) < 30.0)
                    continue;
                d = 10.0 * log10((ed + 1e-30) / eo);
                cells++;
                report->eq_level_cells[e]++;
                report->eq_level_abs_db[e] += fabs(d);
                report->eq_level_bias_db[e] += d;
                if (fabs(d) > 1.0) over1++;
                if (fabs(d) > 3.0) over3++;
                if (fabs(d) > fabs(report->level_worst_db))
                {
                    report->level_worst_db = d;
                    report->level_worst_seconds = (double)sec;
                    report->level_worst_band = e;
                }
                if (debug_levels && fabs(d) > 1.0)
                {
                    if (!shown) fprintf(stderr, "  %4lu s:", sec);
                    fprintf(stderr, " [%u] %+.1f", e, d);
                    shown = 1;
                }
            }
            if (shown) fprintf(stderr, "\n");
        }
        report->level_over_1db_percent = cells ? 100.0 * (double)over1 / (double)cells : 0.0;
        report->level_over_3db_percent = cells ? 100.0 * (double)over3 / (double)cells : 0.0;
        for (e = 0; e < MMX_EQ_BANDS; e++)
            if (report->eq_level_cells[e])
            {
                report->eq_level_abs_db[e] /= (double)report->eq_level_cells[e];
                report->eq_level_bias_db[e] /= (double)report->eq_level_cells[e];
            }
    }
    {
        unsigned long cells_all = 0, cells_hi = 0;
        double ild_all = 0.0, coh_all = 0.0, ild_hi = 0.0, coh_hi = 0.0;
        for (e = 0; e < MMX_EQ_BANDS; e++)
        {
            report->stereo_cells[e] = st_cells[e];
            report->stereo_ild_err_db[e] = st_cells[e] ? st_ild[e] / (double)st_cells[e] : 0.0;
            report->stereo_coh_err[e] = st_cells[e] ? st_coh[e] / (double)st_cells[e] : 0.0;
            report->stereo_ild_over_percent[e] = st_cells[e] ? 100.0 * (double)st_ild_over[e] / (double)st_cells[e] : 0.0;
            report->stereo_coh_over_percent[e] = st_cells[e] ? 100.0 * (double)st_coh_over[e] / (double)st_cells[e] : 0.0;
            cells_all += st_cells[e]; ild_all += st_ild[e]; coh_all += st_coh[e];
            if (e >= 5) { cells_hi += st_cells[e]; ild_hi += st_ild[e]; coh_hi += st_coh[e]; }
        }
        report->stereo_ild_err_db_all = cells_all ? ild_all / (double)cells_all : 0.0;
        report->stereo_coh_err_all = cells_all ? coh_all / (double)cells_all : 0.0;
        report->stereo_ild_err_db_hi = cells_hi ? ild_hi / (double)cells_hi : 0.0;
        report->stereo_coh_err_hi = cells_hi ? coh_hi / (double)cells_hi : 0.0;
    }
    free(win); free(orig_coefs); free(err_coefs); free(dec_coefs); free(lvl); free(cplx); free(wdif); free(wok);
    mmx_spectrum_free(&sa);
    mmx_spectrum_free(&pre_spec);
    report->frames = ns;
    /* pre-echo: attacks = 128-sample blocks at least 10 dB above the mean of the
       previous 8 blocks; the SNR of those 8 blocks (46 ms) before the attack is
       the metric. Mono downmix, all channels summed. */
    {
        unsigned long blk = 128, nb = (unsigned long)(original->frame_count / blk), i2, j, n_ghosts = 0;
        GhostPlace ghosts[GHOST_LIST];
        const char *gl = getenv("MMX_DEBUG_ATTACKS");   /* lists the worst places around the attacks */
        int ghost_list = gl ? (atoi(gl) > 0 && atoi(gl) < GHOST_LIST ? atoi(gl) : GHOST_LIST) : 0;
        double *e_sig = (double *)calloc(nb ? nb : 1, sizeof(double));
        double *e_err = (double *)calloc(nb ? nb : 1, sizeof(double));
        double floor_e = pow(10.0, -60.0 / 10.0) * blk * original->channels;
        report->worst_pre_attack_snr_db = 200.0;
        if (e_sig && e_err)
        {
            for (i2 = 0; i2 < nb; i2++)
            {
                double s2 = 0.0, d2 = 0.0;
                for (j = i2 * blk * original->channels; j < (i2 + 1) * blk * original->channels; j++)
                {
                    double d = (double)original->samples[j] - decoded->samples[j];
                    s2 += (double)original->samples[j] * original->samples[j];
                    d2 += d * d;
                }
                e_sig[i2] = s2;
                e_err[i2] = d2;
            }
            for (i2 = 8; i2 < nb; i2++)
            {
                double pre = 0.0, pre_err = 0.0, snr;
                for (j = i2 - 8; j < i2; j++) { pre += e_sig[j]; pre_err += e_err[j]; }
                if (e_sig[i2] <= floor_e || e_sig[i2] < 10.0 * pre / 8.0)
                    continue;
                report->attacks++;
                attack_envelope(original, decoded, i2 * blk, report, ghosts, &n_ghosts, ghost_list);
                snr = 10.0 * log10((pre + 1e-30) / (pre_err + 1e-30));
                if (pre <= floor_e * 8.0)
                    snr = pre_err <= floor_e * 8.0 ? 200.0 : -200.0; /* digital silence before the attack must stay silent */
                if (snr < 10.0)
                {
                    static int debug = -1;
                    if (debug < 0) debug = getenv("MMX_DEBUG_PREECHO") != NULL; /* lists every smeared attack */
                    if (debug)
                        fprintf(stderr, "pre-echo: attack at %.3f s (sample %lu) %.1f dB over the 46 ms before, pre-attack SNR %.1f dB\n",
                                (double)i2 * blk / original->sample_rate, i2 * blk, 10.0 * log10(e_sig[i2] / (pre / 8.0 + 1e-30)), snr);
                    report->attacks_pre_echo++;
                }
                if (snr < report->worst_pre_attack_snr_db)
                {
                    report->worst_pre_attack_snr_db = snr;
                    report->worst_pre_attack_seconds = (double)i2 * blk / original->sample_rate;
                }
            }
        }
        free(e_sig);
        free(e_err);
        if (ghost_list)
        {
            unsigned long g;
            fprintf(stderr, "worst places around the %lu attacks (5.8 ms blocks, error over signal, error above -70 dBFS):\n", report->attacks);
            for (g = 0; g < n_ghosts && (int)g < ghost_list; g++)
                fprintf(stderr, "  %8.3f s  error %+6.1f dB over the signal  (error %6.1f dBFS, signal %6.1f dBFS, %s the attack at %.3f s by %.1f ms)\n",
                        ghosts[g].seconds, ghosts[g].esr_db, ghosts[g].err_dbfs, ghosts[g].sig_dbfs, ghosts[g].before ? "before" : "after",
                        ghosts[g].attack_seconds, fabs(ghosts[g].seconds - ghosts[g].attack_seconds) * 1000.0);
        }
    }
    beyond_model(original, decoded, report);

    n = (unsigned long)(original->frame_count * original->channels);
    scale = (double)(1UL << ((original->source_bits ? original->source_bits : 16) - 1));
    for (i = 0; i < n; i++)
    {
        double e = (double)original->samples[i] - decoded->samples[i];
        double d = fabs(e) * scale;
        err2 += e * e;
        sig2 += (double)original->samples[i] * original->samples[i];
        if (d > maxdiff) maxdiff = d;
    }

    report->worst_nmr_db = worst;
    report->mean_nmr_db = bands_total ? sum_nmr / bands_total : -200.0;
    for (e = 0; e < MMX_EQ_BANDS; e++)
    {
        report->eq_cells[e] = eq_total[e];
        report->eq_over_percent[e] = eq_total[e] ? 100.0 * (double)eq_over[e] / eq_total[e] : 0.0;
        report->eq_worst_nmr_db[e] = eq_worst[e];
        report->eq_mean_nmr_db[e] = eq_total[e] ? eq_sum[e] / eq_total[e] : -200.0;
    }
    report->bands_over_percent = bands_total ? 100.0 * (double)bands_over_total / bands_total : 0.0;
    report->w_crossover_hz = (unsigned int)(cross_hz + 0.5);
    report->w_over_percent = w_sum > 0.0 ? 100.0 * w_over_sum / w_sum : 0.0;
    report->w_mean_nmr_db = w_sum > 0.0 ? w_nmr_sum / w_sum : -200.0;
    report->w_noise_loudness_percent = nl_sig_sum > 0.0 ? 100.0 * nl_noise_sum / nl_sig_sum : 0.0;
    for (e = 0; e < MMX_EQ_BANDS; e++)
    {
        report->eq_w_over_percent[e] = eq_w_sum[e] > 0.0 ? 100.0 * eq_w_over[e] / eq_w_sum[e] : 0.0;
        report->eq_weight_percent[e] = w_sum > 0.0 ? 100.0 * eq_w_sum[e] / w_sum : 0.0;
    }
    report->hi_energy_cells = hi_cells;
    report->hi_energy_err_db = hi_cells ? hi_abs_sum / (double)hi_cells : 0.0;
    report->hi_energy_bias_db = hi_cells ? hi_bias_sum / (double)hi_cells : 0.0;
    report->hi_energy_over3_percent = hi_cells ? 100.0 * (double)hi_over3 / (double)hi_cells : 0.0;
    report->hi_energy_worst_db = hi_worst;
    report->frames_over_percent = ns ? 100.0 * (double)frames_over / ns : 0.0;
    report->snr_db = err2 > 1e-30 ? 10.0 * log10((sig2 + 1e-30) / err2) : 200.0;
    report->max_diff_lsb = maxdiff;
    report->passed = worst <= tolerance_db;
    report->worst_frame = worst_frame;
    report->worst_channel = worst_channel;
    report->worst_band = worst_band;

    return 0;
}
