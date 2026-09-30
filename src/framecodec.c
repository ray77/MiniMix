#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#else
#include <pthread.h>
#endif
#include "lab.h"
#include "framecodec.h"
/* allocation experiments (encoder side only, the syntax is unchanged):
   MMX_ALLOC_ZERO=0  no band is zeroed by its energy against the threshold; the quantiser decides per coefficient
                     and a band is zero only when every coefficient rounds to zero
   MMX_ALLOC_K=n     at most n coded coefficients per long band (the strongest), 0 = no cap */
static int alloc_zero = -1, alloc_k = 0;
static void alloc_knobs(void)
{
    const char *e;
    if (alloc_zero >= 0) return;
    alloc_zero = (e = mmx_lab_getenv("MMX_ALLOC_ZERO")) ? atoi(e) != 0 : 1;
    alloc_k = (e = mmx_lab_getenv("MMX_ALLOC_K")) ? atoi(e) : 0;
}
/* keep the n strongest coefficients of the band, zero the rest; returns whether any is left */
static int cap_band(int *q, unsigned long k0, unsigned long k1, int n)
{
    unsigned long k, kept[64], j;
    int keepq[64], m = 0, any = 0;
    if (n <= 0) { for (k = k0; k < k1; k++) if (q[k]) return 1; return 0; }
    if (n > 64) n = 64;
    for (k = k0; k < k1; k++)
    {
        int a = q[k] < 0 ? -q[k] : q[k], i;
        if (!a) continue;
        /* insert into the top-n list (descending by |q|, the earlier bin wins ties) */
        for (i = m; i > 0; i--)
        {
            int ai = q[kept[i - 1]] < 0 ? -q[kept[i - 1]] : q[kept[i - 1]];
            if (ai >= a) break;
            if (i < n) kept[i] = kept[i - 1];
        }
        if (i < n) { kept[i] = k; if (m < n) m++; }
    }
    for (j = 0; j < (unsigned long)m; j++) keepq[j] = q[kept[j]];
    for (k = k0; k < k1; k++) q[k] = 0;
    for (j = 0; j < (unsigned long)m; j++) { q[kept[j]] = keepq[j]; any = 1; }
    return any;
}
#define MMX_EPB_EST_BITS 2.0   /* estimated cost of one EPB energy symbol (measured: see est_debug elem SF) */

#define SF_BIAS 48
#define PNS_FLAG_INIT ((1U << MMX_PROB_BITS) - 64)  /* noise bands are rare: start near "no", an unused flag costs 0.05 bit */

/* ---------- debug: where the bits go (MMX_DEBUG_BITS, encoder only) ---------- */

#define BITDUMP_COLS (MMX_EQ_BANDS + 2)     /* 8 EQ regions, frame side info, short frames */
static struct
{
    int armed;                                     /* the next mmx_frame_encode call is booked */
    unsigned int cls;                              /* class of that call */
    double bits[MMX_BITDUMP_CLASSES][BITDUMP_COLS];
    unsigned long frames[MMX_BITDUMP_CLASSES];
} bitdump;

void mmx_frame_bitdump_next(unsigned int cls)
{
    bitdump.armed = 1;
    bitdump.cls = cls < MMX_BITDUMP_CLASSES ? cls : 0;
}

/* per-band cost accounting for the encoder's closed loop: armed for one call */
static double (*band_book)[MMX_MAX_BANDS];

void mmx_frame_encode_book_bands(double (*out)[MMX_MAX_BANDS])
{
    band_book = out;
}

/* M0: where the bits go, per syntax element (armed for one call each) */
const char *const mmx_elem_names[MMX_ELEM_COUNT] = { "frame side", "replication", "intensity", "band zero flags", "noise bands",
                                                   "scalefactors", "coef zero flags", "signs", "coef gt1/gt2", "coef escape", "short frames" };
static double *elem_book, *est_elem_book;
void mmx_frame_encode_book_elems(double *out) { elem_book = out; }
void mmx_frame_estimate_book_elems(double *out) { est_elem_book = out; }
#define ELEM(e) do { if (eb) { double now_ = mmx_rc_enc_bits(rc); eb[e] += now_ - ep; ep = now_; } } while (0)

void mmx_frame_bitdump_print(void)
{
    static const char *cls_name[MMX_BITDUMP_CLASSES] = { "AUDIO stationary", "AUDIO transient", "REF stationary", "REF transient" };
    static const char *col_name[BITDUMP_COLS] = { "<200", "200-500", "500-1k", "1k-2k", "2k-4k", "4k-8k", "8k-12k", ">12k", "side", "short" };
    double total = 0.0, col_sum[BITDUMP_COLS];
    unsigned long frames = 0;
    unsigned int c, k;
    for (k = 0; k < BITDUMP_COLS; k++) col_sum[k] = 0.0;
    for (c = 0; c < MMX_BITDUMP_CLASSES; c++)
    {
        frames += bitdump.frames[c];
        for (k = 0; k < BITDUMP_COLS; k++) { total += bitdump.bits[c][k]; col_sum[k] += bitdump.bits[c][k]; }
    }
    if (total <= 0.0)
        return;
    fprintf(stderr, "bit share by EQ region and frame class (%% of %.0f kbit in %lu frames; kbit/s per class at 44.1 kHz)\n", total / 1000.0, frames);
    fprintf(stderr, "  %-17s %7s %7s", "class", "frames", "kbit/s");
    for (k = 0; k < BITDUMP_COLS; k++) fprintf(stderr, " %8s", col_name[k]);
    fprintf(stderr, "\n");
    for (c = 0; c < MMX_BITDUMP_CLASSES; c++)
    {
        double cls_total = 0.0;
        for (k = 0; k < BITDUMP_COLS; k++) cls_total += bitdump.bits[c][k];
        fprintf(stderr, "  %-17s %7lu %7.1f", cls_name[c], bitdump.frames[c],
                bitdump.frames[c] ? cls_total / ((double)bitdump.frames[c] * MMX_HOP / 44100.0) / 1000.0 : 0.0);
        for (k = 0; k < BITDUMP_COLS; k++) fprintf(stderr, " %7.2f%%", 100.0 * bitdump.bits[c][k] / total);
        fprintf(stderr, "\n");
    }
    fprintf(stderr, "  %-17s %7lu %7.1f", "all", frames, frames ? total / ((double)frames * MMX_HOP / 44100.0) / 1000.0 : 0.0);
    for (k = 0; k < BITDUMP_COLS; k++) fprintf(stderr, " %7.2f%%", 100.0 * col_sum[k] / total);
    fprintf(stderr, "\n");
}

int mmx_frame_syntax_init(MMXFrameSyntax *s, unsigned int channels, unsigned long m)
{
    unsigned int c;
    memset(s, 0, sizeof(*s));
    s->channels = channels;
    s->m = m;
    for (c = 0; c < channels; c++)
    {
        s->q[c] = (int *)calloc(m, sizeof(int));
        if (!s->q[c])
        {
            mmx_frame_syntax_free(s);
            return -1;
        }
    }
    return 0;
}

void mmx_frame_syntax_free(MMXFrameSyntax *s)
{
    unsigned int c;
    if (!s)
        return;
    for (c = 0; c < MMX_MAX_CH; c++)
        free(s->q[c]);
    memset(s, 0, sizeof(*s));
}

void mmx_frame_syntax_copy(MMXFrameSyntax *dst, const MMXFrameSyntax *src)
{
    unsigned int c;
    int *keep[MMX_MAX_CH];
    for (c = 0; c < MMX_MAX_CH; c++)
        keep[c] = dst->q[c];
    memcpy(dst, src, sizeof(*dst));
    for (c = 0; c < MMX_MAX_CH; c++)
    {
        dst->q[c] = keep[c];
        if (c < src->channels && keep[c] && src->q[c])
            memcpy(dst->q[c], src->q[c], sizeof(int) * src->m);
    }
}

void mmx_contexts_init(MMXCodecContexts *c)
{
    MMXProb *p = (MMXProb *)c;
    size_t n = offsetof(MMXCodecContexts, has_prev) / sizeof(MMXProb), i;
    for (i = 0; i < n; i++)
        p[i] = MMX_PROB_INIT;
    c->has_prev = 0;
    memset(c->prev_sf, 0, sizeof(c->prev_sf));
    memset(c->prev_q, 0, sizeof(c->prev_q));
    memset(c->prev_gain, 0, sizeof(c->prev_gain));
    memset(c->prev_polarity, 0, sizeof(c->prev_polarity));
    memset(c->prev_tns, 0, sizeof(c->prev_tns));
    c->prev_block_type = MMX_BT_LONG;
    memset(c->prev_pns, 0, sizeof(c->prev_pns));
    memset(c->prev_pns_level, SF_BIAS, sizeof(c->prev_pns_level));
    memset(c->prev_nrg, 0, sizeof(c->prev_nrg));
    memset(c->prev_nf, 0, sizeof(c->prev_nf));
    for (i = 0; i < sizeof(c->pns_flag) / sizeof(MMXProb); i++)
        (&c->pns_flag[0][0][0])[i] = PNS_FLAG_INIT;
    for (i = 0; i < sizeof(c->is_flag) / sizeof(MMXProb); i++)
        (&c->is_flag[0][0][0])[i] = PNS_FLAG_INIT;   /* intensity bands are rare too: an unused flag costs 0.02 bit */
    memset(c->prev_is, 0, sizeof(c->prev_is));
    memset(c->prev_is_pos, 0, sizeof(c->prev_is_pos));
    for (i = 0; i < sizeof(c->nf_flag) / sizeof(MMXProb); i++)
        (&c->nf_flag[0][0])[i] = PNS_FLAG_INIT;   /* unused (fixed quality): 0.05 bit per region and channel */
    memset(c->prev_bwe, 0, sizeof(c->prev_bwe));
    memset(c->prev_bwe_level, SF_BIAS, sizeof(c->prev_bwe_level));
    memset(c->prev_bwe_mix, 0, sizeof(c->prev_bwe_mix));
    memset(c->prev_bwe_last, SF_BIAS, sizeof(c->prev_bwe_last));
}

/* Noise filling levels of a long frame: per channel and EQ region from the
   first substitutable region on, a flag (context: the region had a level in
   the previous frame) and the level as a delta against that previous level. */
static void encode_nf(MMXRangeEncoder *rc, MMXCodecContexts *ctx, const MMXBandLayout *L, const MMXFrameSyntax *s, unsigned int mode)
{
    unsigned int c, e, e0 = mmx_nf_first_region(L);
    for (c = 0; c < s->channels; c++)
        for (e = e0; e < MMX_EQ_BANDS; e++)
        {
            unsigned int prev = ctx->prev_nf[c][e], on = s->nf_level[c][e] != 0;
            mmx_rc_enc_bit(rc, &ctx->nf_flag[mode][prev != 0], on);
            if (on)
                mmx_rc_enc_seg(rc, ctx->nf_level[mode], (long long)s->nf_level[c][e] - (long long)(prev ? prev : MMX_NF_LEVEL_DEFAULT));
        }
}

static int decode_nf(MMXRangeDecoder *rc, MMXCodecContexts *ctx, const MMXBandLayout *L, MMXFrameSyntax *s, unsigned int mode)
{
    unsigned int c, e, e0 = mmx_nf_first_region(L);
    memset(s->nf_level, 0, sizeof(s->nf_level));
    for (c = 0; c < s->channels; c++)
        for (e = e0; e < MMX_EQ_BANDS; e++)
        {
            unsigned int prev = ctx->prev_nf[c][e];
            if (mmx_rc_dec_bit(rc, &ctx->nf_flag[mode][prev != 0]))
            {
                long long lvl = mmx_rc_dec_seg(rc, ctx->nf_level[mode]) + (long long)(prev ? prev : MMX_NF_LEVEL_DEFAULT);
                if (lvl < 1 || lvl >= MMX_NF_LEVELS || rc->failed)
                    return -1;
                s->nf_level[c][e] = (unsigned char)lvl;
            }
        }
    return 0;
}

/* ------------------------------------------------- intensity stereo */

unsigned int mmx_is_first_band(const MMXBandLayout *L)
{
    unsigned int b;
    for (b = 1; b < L->band_count; b++)
        if (L->band_hz[b] >= (float)MMX_IS_MIN_HZ)
            return b;
    return L->band_count;
}

signed char mmx_is_position(double el, double er)
{
    double v;
    if (el <= 0.0) return (signed char)-MMX_IS_POS_MAX;
    if (er <= 0.0) return (signed char)MMX_IS_POS_MAX;
    v = floor(10.0 * log10(el / er) / MMX_IS_STEP_DB + 0.5);
    if (v < -MMX_IS_POS_MAX) v = -MMX_IS_POS_MAX;
    if (v > MMX_IS_POS_MAX) v = MMX_IS_POS_MAX;
    return (signed char)v;
}

/* The gains are part of the format: encoder and decoder read the same table. */
void mmx_is_gains(signed char pos, float *gl, float *gr)
{
    static float tab[2 * MMX_IS_POS_MAX + 1][2];
    static int ready = 0;
    int i;
    if (!ready)
    {
        for (i = -MMX_IS_POS_MAX; i <= MMX_IS_POS_MAX; i++)
        {
            double r = pow(10.0, (double)i * MMX_IS_STEP_DB / 10.0);
            tab[i + MMX_IS_POS_MAX][0] = (float)sqrt(2.0 * r / (1.0 + r));
            tab[i + MMX_IS_POS_MAX][1] = (float)sqrt(2.0 / (1.0 + r));
        }
        ready = 1;
    }
    i = pos < -MMX_IS_POS_MAX ? -MMX_IS_POS_MAX : pos > MMX_IS_POS_MAX ? MMX_IS_POS_MAX : pos;
    *gl = tab[i + MMX_IS_POS_MAX][0];
    *gr = tab[i + MMX_IS_POS_MAX][1];
}

static void encode_tns(MMXRangeEncoder *rc, MMXCodecContexts *ctx, const MMXFrameSyntax *s)
{
    unsigned int c, i;
    for (c = 0; c < s->channels; c++)
    {
        const MMXTns *t = &s->tns[c];
        int active = t->active && t->order > 0;
        mmx_rc_enc_bit(rc, &ctx->tns_flag[ctx->prev_tns[c]], active);
        if (active)
        {
            mmx_rc_enc_ueg(rc, ctx->tns_order, t->order - 1);
            for (i = 0; i < t->order; i++)
                mmx_rc_enc_bypass(rc, t->coef[i], MMX_TNS_COEF_BITS);
        }
        ctx->prev_tns[c] = (unsigned char)active;
    }
}

static int decode_tns(MMXRangeDecoder *rc, MMXCodecContexts *ctx, MMXFrameSyntax *s)
{
    unsigned int c, i;
    for (c = 0; c < s->channels; c++)
    {
        MMXTns *t = &s->tns[c];
        memset(t, 0, sizeof(*t));
        t->active = (int)mmx_rc_dec_bit(rc, &ctx->tns_flag[ctx->prev_tns[c]]);
        if (t->active)
        {
            unsigned long o = mmx_rc_dec_ueg(rc, ctx->tns_order) + 1;
            if (o > MMX_TNS_MAX_ORDER)
                return -1;
            t->order = (unsigned int)o;
            for (i = 0; i < t->order; i++)
                t->coef[i] = (unsigned char)mmx_rc_dec_bypass(rc, MMX_TNS_COEF_BITS);
        }
        ctx->prev_tns[c] = (unsigned char)t->active;
    }
    return 0;
}

static double mmx_sf_tab[MMX_SF_COUNT];
static void sf_fill_table(void)
{
    unsigned int i;
    for (i = 0; i < MMX_SF_COUNT; i++)
        mmx_sf_tab[i] = pow(2.0, ((double)i - SF_BIAS) / 4.0);
}
#ifdef _WIN32
static BOOL CALLBACK sf_fill_once(PINIT_ONCE once, PVOID param, PVOID *context)
{
    (void)once; (void)param; (void)context;
    sf_fill_table();
    return TRUE;
}
#endif

double mmx_sf_step(unsigned int sf)
{
    /* the same value as pow(2, (sf - SF_BIAS) / 4), from a table: this is called
       once per coefficient in the quantizer loops and pow() is not cheap */
    /* Built once under pthread_once. The encoder has worker threads now, and a plain "ready" flag is
       a data race: a second thread can see ready == 1 while the table is still half filled and read a
       0.0 step, which is silently wrong rather than a crash. mmx_sf_step is on the decode path too. */
#ifdef _WIN32
    static INIT_ONCE sf_once = INIT_ONCE_STATIC_INIT;
    InitOnceExecuteOnce(&sf_once, sf_fill_once, NULL, NULL);
#else
    static pthread_once_t sf_once = PTHREAD_ONCE_INIT;
    pthread_once(&sf_once, sf_fill_table);
#endif
    if (sf < MMX_SF_COUNT)
        return mmx_sf_tab[sf];
    return pow(2.0, ((double)sf - SF_BIAS) / 4.0);
}

unsigned int mmx_sf_for_threshold(double thr)
{
    double step, v;
    if (thr <= 0.0)
        return 0;
    step = sqrt(12.0 * thr);
    v = floor(4.0 * log(step) / log(2.0)) + SF_BIAS;
    if (v < 0.0) v = 0.0;
    if (v > MMX_SF_COUNT - 1) v = MMX_SF_COUNT - 1;
    return (unsigned int)v;
}

double mmx_gain_value(signed char index, unsigned char polarity)
{
    double g;
    if (index == MMX_GAIN_OFF)
        return 0.0;
    g = pow(10.0, (double)index / 40.0);
    return polarity ? -g : g;
}

signed char mmx_gain_index(double gain, unsigned char *polarity)
{
    double a = fabs(gain), idx;
    *polarity = gain < 0.0;
    if (a < pow(10.0, (MMX_GAIN_MIN - 0.5) / 40.0))
    {
        *polarity = 0;
        return MMX_GAIN_OFF;
    }
    idx = floor(40.0 * log10(a) + 0.5);
    if (idx < MMX_GAIN_MIN) idx = MMX_GAIN_MIN;
    if (idx > MMX_GAIN_MAX) idx = MMX_GAIN_MAX;
    return (signed char)idx;
}

/* Quantizes one band with a given step; returns the noise energy. e (optional)
   receives the per-coefficient errors. */
/* Rounding with a small dead zone: magnitudes below 0.62 steps become zero
   (cheaper to code), everything else rounds to nearest. The noise check of the
   caller keeps the band within its threshold either way. */
#define DEADZONE_DEFAULT 0.62
/* EXPERIMENT MMX_DEADZONE=<x>: the rounding threshold below which a coefficient becomes zero (0.5 =
   plain rounding, no dead zone). A partial near the threshold sits just under 0.62 step in one frame
   and above it in the next; the dead zone is what chops it. */
static double deadzone_value(void)
{
    static double dz = -1.0;
    if (dz < 0.0) { const char *e = mmx_lab_getenv("MMX_DEADZONE"); dz = e ? atof(e) : DEADZONE_DEFAULT; }
    return dz;
}
#define DEADZONE deadzone_value()

/* Noise filling (encoder side, MMXNfZeroWeight): in a band the decoder will
   fill, a coefficient that quantizes to zero gets its energy back as noise,
   so its distortion counts alpha times its energy (what remains is the wrong
   waveform, weighted by the encoder's own perceptual factor; 1.0 = plain).
   Applies from first_band on to regions no source predicts (all regions with
   resid). The weight travels with the call, not in a global: the decoder
   never sees it, and two encoders in one process do not share it. */
static const double zw_ones[MMX_MAX_CH] = { 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0 };

static double zero_weight_of(const MMXNfZeroWeight *z, const MMXFrameSyntax *s, const MMXBandLayout *L,
                             unsigned int c, unsigned int b)
{
    unsigned int src, e = L->eq_band[b];
    if (!z || b < z->first_band || s->block_type == MMX_BT_SHORT || z->alpha[e] >= 1.0)
        return 1.0;
    if (!z->resid)
        for (src = 0; src < MMX_MAX_SOURCES; src++)
            if (s->gain[src][c][e] != MMX_GAIN_OFF)
                return 1.0;
    return z->alpha[e];
}

/* zw: weight of the distortion of the zeros (see MMXNfZeroWeight); the errors e of
   zeros are scaled by sqrt(zw) so that the L/R check of an M/S pair sees the
   same discount. */
static double quantize_band(const float *x, unsigned long k0, unsigned long k1, double step, double zw, int *q, int *any, double *e)
{
    double inv = 1.0 / step, noise = 0.0, sz = zw < 1.0 ? sqrt(zw) : 1.0;
    unsigned long k;
    *any = 0;
    for (k = k0; k < k1; k++)
    {
        double v = x[k] * inv, av = v < 0.0 ? -v : v, err;
        int qi = av < DEADZONE ? 0 : (int)floor(av + 0.5);
        if (v < 0.0) qi = -qi;
        q[k] = qi;
        if (qi) *any = 1;
        err = x[k] - qi * step;
        if (!qi) err *= sz;
        if (e) e[k - k0] = err;
        noise += err * err;
    }
    return noise;
}

unsigned int mmx_frame_keep_band(const MMXBandLayout *L, unsigned int b, const float *x, int *q,
                                 double keep_rel, int steps)
{
    unsigned long k0 = L->band_start[b], k1 = L->band_start[b + 1], k;
    double cmax = 0.0, e = 0.0, v;
    int sf, i;

    for (k = k0; k < k1; k++)
    {
        double a = x[k] < 0.0f ? -(double)x[k] : (double)x[k];
        e += (double)x[k] * x[k];
        if (a > cmax) cmax = a;
    }
    if (cmax <= 0.0)
    {
        for (k = k0; k < k1; k++) q[k] = 0;
        return 0;
    }
    v = floor(4.0 * log(cmax) / log(2.0) + 0.5) + SF_BIAS;   /* the step closest to the largest coefficient */
    if (v < 0.0) v = 0.0;
    if (v > MMX_SF_COUNT - 1) v = MMX_SF_COUNT - 1;
    sf = (int)v;
    for (i = 0;; i++)
    {
        double step = mmx_sf_step((unsigned int)sf), rec = 0.0;
        int any = 0;
        for (k = k0; k < k1; k++)
        {
            double t = (double)x[k] / step, at = t < 0.0 ? -t : t;
            int qi = at < DEADZONE ? 0 : (int)floor(at + 0.5);
            if (t < 0.0) qi = -qi;
            q[k] = qi;
            if (qi) { any = 1; rec += (double)qi * qi * step * step; }
        }
        if ((any && rec >= keep_rel * e) || i >= steps || sf == 0)
            break;
        sf--;
    }
    return (unsigned int)sf;
}

static void zero_band(MMXFrameSyntax *s, unsigned int c, unsigned int b, unsigned long k0, unsigned long k1)
{
    unsigned long k;
    s->sf[c][b] = 0;
    s->band_zero[c][b] = 1;
    s->band_noise[c][b] = 0;
    for (k = k0; k < k1; k++) s->q[c][k] = 0;
}

/* One band of all channels with the plain quantizer: DEADZONE rounding at the
   threshold-derived step, tightened while the measured noise exceeds the
   allowance; an M/S pair (allowed_lr != NULL) also against the L/R
   allowances, the L/R errors being e_M +- e_S and fully correlated when both
   round to zero. zero[c] says the channel may be dropped (in/out, a dropped
   channel of an M/S pair is un-dropped when the L/R check needs it), sf[c]
   holds the threshold-derived scalefactor (in) and the chosen one (out),
   any[c] whether anything was coded and noise[c] the noise of the result;
   s->q is written in place. */
static void plain_quantize_band(MMXFrameSyntax *s, float *const *x, unsigned long k0, unsigned long k1,
                                const double *energy, const double *allowed, const double *allowed_lr, const double *zw,
                                int *zero, unsigned int *sf, int *any, double *noise)
{
    unsigned int c;
    unsigned long k, n = k1 - k0;

    if (!allowed_lr)
    {
        for (c = 0; c < s->channels; c++)
        {
            any[c] = 0;
            noise[c] = energy[c] * zw[c];
            if (zero[c])
                continue;
            /* start from the step whose uniform noise matches the threshold, then
               verify the actual noise and tighten while it exceeds the threshold
               (bands just above the threshold round mostly to zero, where the error
               is the signal itself, not step^2/12) */
            for (;;)
            {
                noise[c] = quantize_band(x[c], k0, k1, mmx_sf_step(sf[c]), zw[c], s->q[c], &any[c], NULL);
                if (noise[c] <= allowed[c] || sf[c] == 0)
                    break;
                sf[c]--;
            }
        }
        return;
    }
    {
        double e[2][MMX_HOP];
        unsigned int iter;
        any[0] = any[1] = 0;
        for (iter = 0; iter < 200; iter++)
        {
            double el = 0.0, er = 0.0, worst = 0.0;
            int worst_c = 0;
            for (c = 0; c < 2; c++)
            {
                if (zero[c])
                {
                    double sz = zw[c] < 1.0 ? sqrt(zw[c]) : 1.0;
                    noise[c] = energy[c] * zw[c];
                    any[c] = 0;
                    for (k = k0; k < k1; k++) { s->q[c][k] = 0; e[c][k - k0] = (float)(x[c][k] * sz); }
                }
                else
                    noise[c] = quantize_band(x[c], k0, k1, mmx_sf_step(sf[c]), zw[c], s->q[c], &any[c], e[c]);
            }
            for (k = 0; k < n; k++)
            {
                double l = e[0][k] + e[1][k], r = e[0][k] - e[1][k];
                el += l * l;
                er += r * r;
            }
            if (noise[0] <= allowed[0] && noise[1] <= allowed[1] && el <= allowed_lr[0] && er <= allowed_lr[1])
                break;
            /* tighten the channel that contributes most, relative to its allowance */
            for (c = 0; c < 2; c++)
            {
                double rel = noise[c] / (allowed[c] + 1e-30);
                if (rel > worst) { worst = rel; worst_c = (int)c; }
            }
            if (el > allowed_lr[0] || er > allowed_lr[1])
            {
                /* both channels' errors add up in L/R: tighten the louder error */
                worst_c = noise[0] >= noise[1] ? 0 : 1;
            }
            if (zero[worst_c])
                zero[worst_c] = 0;
            else if (sf[worst_c] > 0)
                sf[worst_c]--;
            else
                break;
        }
    }
}

static unsigned int fclass_of(unsigned int band);
static unsigned int nbclass_of(int a, int b);

/* ------------------------------------------------------------------ RDOQ */

/* Rate-distortion optimized quantization (encoder side only). The plain
   quantizer above spends a band's noise budget with one rounding rule
   (DEADZONE) at the step whose uniform noise matches the threshold, tightened
   until the measured noise fits. Here its result is the first candidate and
   the band is tried at the steps around it - two finer, up to eight coarser
   while nearest rounding still fits the budget - and at every step the
   rounding of each coefficient (the two levels around its value) is chosen
   by a Viterbi trellis over the coder's neighbour-class chain that minimizes
   distortion + lambda * bits, with the exact bit costs of the adaptive
   probabilities the frame will be coded with (previous-frame class
   included); lambda is bisected so that the real noise of the band just
   stays inside the budget. The cheapest feasible result wins, scalefactor
   delta included. The budget (see rdo_quantize_band) never exceeds the
   threshold, the M/S pair keeps its L/R check and the decoder is not
   involved. MMX_RDOQ=0 restores the plain quantizer bit for bit. */

#define RDO_CANDS 2                 /* rdo_prepare never makes more than two: the two levels around the value */
#define RDO_UEG_TAB 32              /* exp-Golomb tails cached per band (magnitudes 3 .. 34) */
#define RDO_SF_UP 8                 /* coarser steps tried above the given one while nearest rounding fits (worth 0.1 %) */
#define RDO_SF_DOWN 2               /* finer steps tried below it: more zeros at a finer step can be cheaper */
#define RDO_MAX_ERR 1.0             /* largest error of a candidate in steps: the two levels around the value */
#define RDO_LAMBDA_MIN 0.015625     /* 2^-6: rounds to nearest in all but name */
#define RDO_LAMBDA_MAX 8.0          /* rounds a value down (at most RDO_MAX_ERR steps, one step of distortion) for 1/8 bit saved */
#define RDO_LAMBDA_STEPS 7          /* bisection steps between the two (log scale) */
#define RDO_SLACK_DB 1.5            /* a band may be filled up to this far below its threshold (see rdo_quantize_band) */
#define RDO_JOINT_TIGHTEN 0.7       /* budget cut when the L/R check of an M/S band fails: -1.5 dB, one scalefactor step */
#define RDO_JOINT_ITER 60

/* The rate loop's cheap probes switch the trellis off for the duration of a
   probe encode (mmx_frame_rdoq_set_enabled); a probe only has to predict the
   SIZE, and the trellis is six sevenths of the encode time. The final encode
   always runs with it on again. */
static int rdoq_forced_off = 0;

void mmx_frame_rdoq_set_enabled(int on)
{
    rdoq_forced_off = !on;
}

static int rdoq_enabled(void)
{
    static int enabled = -1;
    if (rdoq_forced_off)
        return 0;
    if (enabled < 0)
    {
        const char *e = mmx_lab_getenv("MMX_RDOQ");
        enabled = e == NULL || atoi(e) != 0;
    }
    return enabled;
}

/* Budget ceiling relative to the threshold: RDO_SLACK_DB, or MMX_RDOQ_SLACK=<dB>
   for listening experiments (0 = the whole allowance, large = the plain
   quantizer's noise only). */
static double rdoq_slack(void)
{
    static double slack = -1.0;
    if (slack < 0.0)
    {
        const char *e = mmx_lab_getenv("MMX_RDOQ_SLACK");
        slack = pow(10.0, -(e ? atof(e) : RDO_SLACK_DB) / 10.0);
    }
    return slack;
}

static float cost_tab[2][1 << MMX_PROB_BITS];   /* bits of a decision: [bit][probability of a 0] */

static void cost_tab_init(void)
{
    unsigned int p;
    if (cost_tab[0][1] != 0.0f)
        return;
    for (p = 1; p < (1U << MMX_PROB_BITS); p++)
    {
        double p0 = (double)p / (double)(1U << MMX_PROB_BITS);
        cost_tab[0][p] = (float)(-log(p0) / log(2.0));
        cost_tab[1][p] = (float)(-log(1.0 - p0) / log(2.0));
    }
    cost_tab[0][0] = cost_tab[0][1];
}
#define BITCOST(p, bit) ((double)cost_tab[(bit) ? 1 : 0][p])

static double ueg_cost(const MMXProb *ctx, unsigned long value)
{
    unsigned long v = value + 1;
    unsigned int len = 0, i;
    double bits;
    while ((v >> len) > 1)
        len++;
    bits = (double)len;    /* suffix, bypass coded */
    for (i = 0; i < len; i++)
        bits += BITCOST(ctx[i < 19 ? i : 19], 1);
    return bits + BITCOST(ctx[len < 19 ? len : 19], 0);
}

static double seg_cost(const MMXProb *ctx, long long value)
{
    if (value == 0)
        return BITCOST(ctx[0], 0);
    return BITCOST(ctx[0], 1) + BITCOST(ctx[1], value < 0) + ueg_cost(ctx + 2, (unsigned long)(value < 0 ? -value : value) - 1);
}

/* The bit costs of one frequency class, with the per-magnitude cost tabulated
   over the 18 (previous class, neighbour class) pairs. The cost of magnitude a
   in context (nb, pc), sign included, used to be computed from the raw
   probabilities for every coefficient, candidate and neighbour class - six
   times per candidate, exp-Golomb tail included. It only ever takes four
   shapes, and only the tail depends on the magnitude:
     a == 0                    zero[nb][pc] as a 0                       -> m0
     a == 1        zero as a 1, sign, gt1[nb][pc] as a 0                 -> m1
     a == 2        zero, sign, gt1 as a 1, gt2[nb][pc] as a 0            -> m2
     a >= 3        zero, sign, gt1 as a 1, gt2 as a 1                    -> m3
                   + ueg_cost(eg, a - 3)                                 -> ueg
   so they are tabulated once per band here (and the tails of the common
   magnitudes with them). The arithmetic is the same and in the same order,
   so the tabulated values are bit for bit the ones the old coef_cost
   returned. */
typedef struct
{
    const MMXProb (*zero)[3], (*gt1)[3], (*gt2)[3];   /* [neighbour class][previous class] of one frequency class */
    const MMXProb *eg;
    float m0[3][6], m1[3][6], m2[3][6];   /* [previous class][neighbour class], as stored in the trellis */
    double m3[3][6];                      /* magnitude >= 3 without the tail: the tail is added before rounding */
    double ueg[RDO_UEG_TAB];
} RdoCosts;

static void rdo_costs_init(RdoCosts *r, const MMXCodecContexts *ctx, unsigned int mode, unsigned int fc)
{
    unsigned int nb, pc, i;
    r->zero = ctx->coef_zero[mode][fc];
    r->gt1 = ctx->coef_gt1[mode][fc];
    r->gt2 = ctx->coef_gt2[mode][fc];
    r->eg = ctx->coef_eg[mode][fc];
    for (pc = 0; pc < 3; pc++)
        for (nb = 0; nb < 6; nb++)
        {
            double one = BITCOST(r->zero[nb][pc], 1) + 1.0;
            r->m0[pc][nb] = (float)BITCOST(r->zero[nb][pc], 0);
            r->m1[pc][nb] = (float)(one + BITCOST(r->gt1[nb][pc], 0));
            r->m2[pc][nb] = (float)(one + BITCOST(r->gt1[nb][pc], 1) + BITCOST(r->gt2[nb][pc], 0));
            r->m3[pc][nb] = one + BITCOST(r->gt1[nb][pc], 1) + BITCOST(r->gt2[nb][pc], 1);
        }
    for (i = 0; i < RDO_UEG_TAB; i++)
        r->ueg[i] = ueg_cost(r->eg, i);
}

/* Trellis work area of the band being quantized (one at a time). */
static struct
{
    int mag[MMX_HOP][RDO_CANDS];                      /* candidate magnitudes, nearest first */
    unsigned char nc[MMX_HOP];                        /* number of candidates */
    float dist[MMX_HOP][RDO_CANDS];                   /* squared error in units of step^2 */
    float bits[MMX_HOP][RDO_CANDS][6];                /* bit cost per candidate and neighbour class */
    unsigned char bp[MMX_HOP][RDO_CANDS][RDO_CANDS];  /* [k][candidate at k][candidate at k-1] -> candidate at k-2 */
    unsigned char sel[MMX_HOP];                       /* chosen candidate per coefficient */
    unsigned long n;
    int single;                                       /* every coefficient has one candidate: one path, lambda cannot move it */
    int cached;                                       /* that one path has been walked since the last rdo_prepare */
    double cd, cb;                                    /* and these are its distortion and bits */
    int ca;
} tr;

static const unsigned char nb_tab[9] = { 0, 1, 2, 3, 3, 4, 4, 4, 4 };   /* nbclass_of by |q1| + |q2|, 5 from 9 on */

/* Candidates, distortions and bit costs of the band at one step. pc is indexed
   like x (previous-frame or previous-group magnitude class per coefficient). */
static void rdo_prepare(const float *x, unsigned long k0, unsigned long k1, double step, double zw, const RdoCosts *r, const unsigned char *pc)
{
    double inv = 1.0 / step;
    unsigned long k;
    int single = 1;
    tr.n = k1 - k0;
    for (k = 0; k < tr.n; k++)
    {
        double v = fabs((double)x[k0 + k]) * inv;
        int a = (int)floor(v + 0.5);
        unsigned int nc = 1, c, p = pc[k0 + k];
        tr.mag[k][0] = a;
        tr.mag[k][1] = 0;   /* the second slot is never used when nc == 1, but rdo_run copies it blindly */
        if (a >= 1 && v - (double)(a - 1) <= RDO_MAX_ERR) { tr.mag[k][1] = a - 1; nc = 2; single = 0; }
        tr.nc[k] = (unsigned char)nc;
        for (c = 0; c < nc; c++)
        {
            int m = tr.mag[k][c];
            double d = v - (double)m;
            tr.dist[k][c] = (float)(d * d * (m ? 1.0 : zw));
            if (m < 3)
                memcpy(tr.bits[k][c], m == 0 ? r->m0[p] : m == 1 ? r->m1[p] : r->m2[p], 6 * sizeof(float));
            else
            {
                unsigned long t = (unsigned long)(m - 3);
                double e = t < RDO_UEG_TAB ? r->ueg[t] : ueg_cost(r->eg, t);
                unsigned int nb;
                for (nb = 0; nb < 6; nb++)
                    tr.bits[k][c][nb] = (float)(r->m3[p][nb] + e);
            }
        }
    }
    tr.single = single;
    tr.cached = 0;
}

/* Viterbi over the candidates with the cost dist + lambda * bits; the context
   chain starts after the magnitudes q1 (previous coefficient) and q2. Fills
   tr.sel and returns the path's distortion (in step^2); *bits receives its
   bit cost, *any whether any coefficient is nonzero. */
static double rdo_run(double lambda, int q1, int q2, double *bits, int *any)
{
    double J[RDO_CANDS][RDO_CANDS], Jn[RDO_CANDS][RDO_CANDS];   /* [candidate at k][candidate at k-1] */
    int pm1[RDO_CANDS], pm2[RDO_CANDS], m1 = q1 < 0 ? -q1 : q1, m2 = q2 < 0 ? -q2 : q2;
    unsigned int n1 = 1, n2 = 1, c, i, j, bc = 0, bi = 0;
    unsigned long k;
    double best = 1e300, d = 0.0, b = 0.0;

    /* A band whose coefficients all have a single candidate has a single path,
       whatever lambda is: the search over the lambda ladder at one step then
       asks the same question up to nine times. Walk it once and keep the
       answer until the next rdo_prepare. */
    if (tr.single && tr.cached)
    {
        *bits = tr.cb;
        *any = tr.ca;
        return tr.cd;
    }
    *any = 0;
    if (tr.single)
    {
        for (k = 0; k < tr.n; k++)
        {
            int sum = m1 + m2, m = tr.mag[k][0];
            unsigned int nb = sum > 8 ? 5 : nb_tab[sum];
            tr.sel[k] = 0;
            d += tr.dist[k][0];
            b += tr.bits[k][0][nb];
            if (m) *any = 1;
            m2 = m1;
            m1 = m;
        }
        tr.cd = d;
        tr.cb = b;
        tr.ca = *any;
        tr.cached = 1;
        *bits = b;
        return d;
    }

    pm1[0] = m1; pm1[1] = 0;
    pm2[0] = m2; pm2[1] = 0;
    J[0][0] = 0.0;
    for (k = 0; k < tr.n; k++)
    {
        unsigned int nc = tr.nc[k];
        if (n1 == 1 && n2 == 1)
        {
            /* the common run: one predecessor pair, so no minimum to take and
               nothing to shuffle - the two state copies below are what the
               compiler turned into memcpy/memset calls */
            int sum = pm1[0] + pm2[0];
            unsigned int nb = sum > 8 ? 5 : nb_tab[sum];
            double base = J[0][0];
            double c0 = base + tr.dist[k][0] + lambda * tr.bits[k][0][nb];
            tr.bp[k][0][0] = 0;
            if (nc == 2)
            {
                J[1][0] = base + tr.dist[k][1] + lambda * tr.bits[k][1][nb];
                tr.bp[k][1][0] = 0;
            }
            J[0][0] = c0;
            pm2[0] = pm1[0];
            n2 = 1;
            pm1[0] = tr.mag[k][0];
            pm1[1] = tr.mag[k][1];
            n1 = nc;
            continue;
        }
        for (i = 0; i < n1; i++)
        {
            double jc0 = 1e300, jc1 = 1e300;
            unsigned char jb0 = 0, jb1 = 0;
            for (j = 0; j < n2; j++)
            {
                int sum = pm1[i] + pm2[j];
                unsigned int nb = sum > 8 ? 5 : nb_tab[sum];
                double base = J[i][j];
                double cost = base + tr.dist[k][0] + lambda * tr.bits[k][0][nb];
                if (cost < jc0) { jc0 = cost; jb0 = (unsigned char)j; }
                if (nc == 2)
                {
                    cost = base + tr.dist[k][1] + lambda * tr.bits[k][1][nb];
                    if (cost < jc1) { jc1 = cost; jb1 = (unsigned char)j; }
                }
            }
            Jn[0][i] = jc0; tr.bp[k][0][i] = jb0;
            if (nc == 2) { Jn[1][i] = jc1; tr.bp[k][1][i] = jb1; }
        }
        J[0][0] = Jn[0][0];                             /* only the states the next step can read */
        if (n1 == 2) J[0][1] = Jn[0][1];
        if (nc == 2) { J[1][0] = Jn[1][0]; if (n1 == 2) J[1][1] = Jn[1][1]; }
        pm2[0] = pm1[0]; pm2[1] = pm1[1];
        n2 = n1;
        pm1[0] = tr.mag[k][0];
        pm1[1] = tr.mag[k][1];
        n1 = nc;
    }
    for (c = 0; c < n1; c++)
        for (i = 0; i < n2; i++)
            if (J[c][i] < best) { best = J[c][i]; bc = c; bi = i; }
    for (k = tr.n; k-- > 0;)
    {
        unsigned int bj = tr.bp[k][bc][bi];
        tr.sel[k] = (unsigned char)bc;
        bc = bi;
        bi = bj;
    }
    for (k = 0; k < tr.n; k++)
    {
        int sum = m1 + m2, m = tr.mag[k][tr.sel[k]];
        unsigned int nb = sum > 8 ? 5 : nb_tab[sum];
        d += tr.dist[k][tr.sel[k]];
        b += tr.bits[k][tr.sel[k]][nb];
        if (m) *any = 1;
        m2 = m1;
        m1 = m;
    }
    *bits = b;
    return d;
}

/* Bits of a given quantization (band-relative q) with the prepared tables;
   1e300 when a value is not among the candidates. */
static double rdo_bits_of(const int *q, int q1, int q2, int *any)
{
    int m1 = q1 < 0 ? -q1 : q1, m2 = q2 < 0 ? -q2 : q2;
    unsigned long k;
    double b = 0.0;
    *any = 0;
    for (k = 0; k < tr.n; k++)
    {
        int m = q[k] < 0 ? -q[k] : q[k], sum = m1 + m2;
        unsigned int c, nb = sum > 8 ? 5 : nb_tab[sum];
        for (c = 0; c < tr.nc[k] && tr.mag[k][c] != m; c++)
            ;
        if (c == tr.nc[k])
            return 1e300;
        b += tr.bits[k][c][nb];
        if (m) *any = 1;
        m2 = m1;
        m1 = m;
    }
    return b;
}

/* Materializes the trellis selection as signed coefficients. */
static void rdo_take(const float *x, unsigned long k0, int *q)
{
    unsigned long k;
    for (k = 0; k < tr.n; k++)
    {
        int m = tr.mag[k][tr.sel[k]];
        q[k] = x[k0 + k] < 0.0f ? -m : m;
    }
}

/* MMX_DEBUG_RDO=1: trial-codes every chosen band with a copy of the contexts
   and reports the estimated against the real bits (checks the cost model). */
static double rdo_actual_bits(const RdoCosts *r, const unsigned char *pc, unsigned long k0, unsigned long k1, const int *q, int q1, int q2)
{
    MMXProb zero[6][3], gt1[6][3], gt2[6][3], eg[20];
    MMXRangeEncoder rc;
    unsigned long k;
    double p0, bits;
    memcpy(zero, r->zero, sizeof(zero)); memcpy(gt1, r->gt1, sizeof(gt1)); memcpy(gt2, r->gt2, sizeof(gt2)); memcpy(eg, r->eg, sizeof(eg));
    mmx_rc_enc_init(&rc);
    p0 = mmx_rc_enc_bits(&rc);
    for (k = k0; k < k1; k++)
    {
        int v = q[k - k0], a = v < 0 ? -v : v;
        unsigned int nb = nbclass_of(q1, q2), c = pc[k];
        mmx_rc_enc_bit(&rc, &zero[nb][c], a != 0);
        if (a)
        {
            mmx_rc_enc_bypass(&rc, v < 0, 1);
            mmx_rc_enc_bit(&rc, &gt1[nb][c], a > 1);
            if (a > 1)
            {
                mmx_rc_enc_bit(&rc, &gt2[nb][c], a > 2);
                if (a > 2) mmx_rc_enc_ueg(&rc, eg, (unsigned long)(a - 3));
            }
        }
        q2 = q1; q1 = v;
    }
    bits = mmx_rc_enc_bits(&rc) - p0;
    mmx_rc_enc_free(&rc);
    return bits;
}

static void rdo_debug_check(const RdoCosts *r, const unsigned char *pc, unsigned long k0, unsigned long k1, const int *q, int q1, int q2, double est)
{
    static double sum_est = 0.0, sum_act = 0.0;
    static unsigned long calls = 0;
    static int on = -1;
    if (on < 0) on = mmx_lab_getenv("MMX_DEBUG_RDO") != NULL;
    if (!on) return;
    sum_est += est;
    sum_act += rdo_actual_bits(r, pc, k0, k1, q, q1, q2);
    if (++calls % 20000 == 0)
        fprintf(stderr, "rdo: %lu bands, estimated %.0f bits, real %.0f bits (%.2f %%)\n", calls, sum_est, sum_act, 100.0 * (sum_act / sum_est - 1.0));
}

/* Quantizes one band of one channel against a noise budget (energy units).
   On entry *sf, q (band-relative) and *noise hold a quantization to start
   from (the plain quantizer's, or the previous round of the M/S loop); when
   its noise fits the budget it is the first candidate and the search can
   only improve on its bits. On exit they hold the choice, *bits its bit
   cost and *any whether anything is nonzero (an all-zero result is
   acceptable when its noise, the band energy, fits the budget, as with the
   plain quantizer). Returns 1 when the budget is met, 0 when even the
   finest step misses it (then the result of that step is delivered, as with
   the plain quantizer). */
static int rdo_band(const float *x, unsigned long k0, unsigned long k1, double zw, const RdoCosts *r, const unsigned char *pc,
                    int q1, int q2, double budget, const MMXProb *sfctx, int prev_sf,
                    unsigned int *sf, int *q, double *noise, double *bits, int *any)
{
    unsigned int s = *sf, top, prepared;
    double best_total = 1e300, d, b, step;
    int ok, a, seed = *noise <= budget;

    if (seed)
    {
        /* the given quantization is the first candidate when it is acceptable */
        rdo_prepare(x, k0, k1, mmx_sf_step(s), zw, r, pc);
        b = rdo_bits_of(q, q1, q2, &a);
        seed = b < 1e300;
    }
    if (seed)
    {
        best_total = b + seg_cost(sfctx, (long long)s - prev_sf);
        *bits = b; *any = a;
        prepared = s;
    }
    else
    {
        /* the finest step to try: down from the given one until nearest rounding fits */
        for (;;)
        {
            step = mmx_sf_step(s);
            rdo_prepare(x, k0, k1, step, zw, r, pc);
            d = rdo_run(0.0, q1, q2, &b, &a) * step * step;
            ok = d <= budget;
            if (ok || s == 0)
                break;
            s--;
        }
        best_total = b + seg_cost(sfctx, (long long)s - prev_sf);
        *sf = s; *noise = d; *bits = b; *any = a;
        rdo_take(x, k0, q);
        if (!ok)
            return 0;
        prepared = s;
    }

    for (top = s > RDO_SF_DOWN ? s - RDO_SF_DOWN : 0; top <= s + RDO_SF_UP && top < MMX_SF_COUNT; top++)
    {
        double lo = RDO_LAMBDA_MIN, hi = RDO_LAMBDA_MAX, side = seg_cost(sfctx, (long long)top - prev_sf), total;
        unsigned int it;
        step = mmx_sf_step(top);
        if (top != prepared)
        {
            rdo_prepare(x, k0, k1, step, zw, r, pc);
            prepared = top;
        }
        /* nearest rounding is the least noise this step can make */
        d = rdo_run(0.0, q1, q2, &b, &a) * step * step;
        if (d > budget)
        {
            if (top > s) break;
            continue;
        }
        if ((total = b + side) < best_total) { best_total = total; *sf = top; *noise = d; *bits = b; *any = a; rdo_take(x, k0, q); }
        /* nothing to trade when the budget is spent or everything rounds to zero already */
        if (d >= budget * 0.999 || !a)
            continue;
        /* coarsest rounding first, then bisect lambda so the noise just fits */
        d = rdo_run(hi, q1, q2, &b, &a) * step * step;
        if (d <= budget)
        {
            if ((total = b + side) < best_total) { best_total = total; *sf = top; *noise = d; *bits = b; *any = a; rdo_take(x, k0, q); }
            continue;
        }
        for (it = 0; it < RDO_LAMBDA_STEPS; it++)
        {
            double lam = sqrt(lo * hi);
            d = rdo_run(lam, q1, q2, &b, &a) * step * step;
            if (d <= budget)
            {
                lo = lam;
                if ((total = b + side) < best_total) { best_total = total; *sf = top; *noise = d; *bits = b; *any = a; rdo_take(x, k0, q); }
            }
            else
                hi = lam;
        }
    }
    rdo_debug_check(r, pc, k0, k1, q, q1, q2, *bits);
    return 1;
}

/* The budget of a band: the plain quantizer's noise, or the allowance less
   RDO_SLACK_DB when the plain quantizer left more slack than that - never
   the whole allowance: a band filled to its threshold moves the model's
   numbers (title A q7: over 3.6 -> 5.3 %) and is a worse source for every
   frame that references it (its residual no longer drops below the zeroing
   rule), and a flat margin wastes bits on bands the plain quantizer had
   already tight. Measured on title A, level 3: plain noise only -0.6 %
   (over 3.2 %), slack 2 dB -0.8 % (3.3 %), 1.5 dB -1.1 % (3.5 %), 1 dB
   -1.6 % (3.8 %); the whole allowance -3.4 % (5.3 %). */
static double rdo_budget(double plain_noise, double allowed)
{
    double slack = allowed * rdoq_slack(), budget = plain_noise < allowed ? plain_noise : allowed;
    return slack > budget ? slack : budget;
}

/* One band of all channels after plain_quantize_band (zero[c], sf[c] and
   plain_noise[c] are its results): single channels against their budget, an
   M/S pair (allowed_lr != NULL) also against the L/R allowances, the L/R
   errors being e_M +- e_S. sf[c] and any[c] receive the choice, s->q is
   written in place, bits_out[c] (optional) the coefficient bits of the
   channel's band. pc[c], q1/q2 and prev_sf are the context chains of the
   channels as the coder will see them. */
static void rdo_quantize_band(MMXFrameSyntax *s, float *const *x, unsigned long k0, unsigned long k1,
                              const double *energy, const double *allowed, const double *plain_noise, const double *allowed_lr,
                              const double *zw, const RdoCosts *costs, const unsigned char *const *pc, const MMXProb *sfctx,
                              const int *prev_sf, const int *q1, const int *q2, int *zero, unsigned int *sf, int *any, double *bits_out)
{
    unsigned int c;
    unsigned long k, n = k1 - k0;
    double noise[MMX_MAX_CH], budget[MMX_MAX_CH], bits[MMX_MAX_CH];

    for (c = 0; c < s->channels; c++)
    {
        budget[c] = rdo_budget(plain_noise[c], allowed[c]);
        noise[c] = plain_noise[c];
        bits[c] = 0.0;
    }
    if (!allowed_lr)
    {
        for (c = 0; c < s->channels; c++)
        {
            any[c] = 0;
            if (zero[c])
                continue;
            rdo_band(x[c], k0, k1, zw[c], costs, pc[c], q1[c], q2[c], budget[c], sfctx, prev_sf[c],
                     &sf[c], s->q[c] + k0, &noise[c], &bits[c], &any[c]);
        }
        if (bits_out)
            for (c = 0; c < s->channels; c++) bits_out[c] = bits[c];
        return;
    }
    {
        double e[2][MMX_HOP];
        int dirty[2] = { 1, 1 }, fits[2] = { 1, 1 };
        unsigned int iter;
        for (iter = 0; iter < RDO_JOINT_ITER; iter++)
        {
            double el = 0.0, er = 0.0, worst = 0.0;
            int worst_c = 0;
            for (c = 0; c < 2; c++)
            {
                double step;
                if (!dirty[c])
                    continue;
                dirty[c] = 0;
                if (zero[c])
                {
                    double sz = zw[c] < 1.0 ? sqrt(zw[c]) : 1.0;
                    noise[c] = energy[c] * zw[c];
                    any[c] = 0;
                    bits[c] = 0.0;
                    for (k = k0; k < k1; k++) { s->q[c][k] = 0; e[c][k - k0] = x[c][k] * sz; }
                    continue;
                }
                fits[c] = rdo_band(x[c], k0, k1, zw[c], costs, pc[c], q1[c], q2[c], budget[c], sfctx, prev_sf[c],
                                   &sf[c], s->q[c] + k0, &noise[c], &bits[c], &any[c]);
                step = mmx_sf_step(sf[c]);
                {
                    double sz = zw[c] < 1.0 ? sqrt(zw[c]) : 1.0;
                    for (k = k0; k < k1; k++) e[c][k - k0] = s->q[c][k] ? x[c][k] - s->q[c][k] * step : x[c][k] * sz;
                }
            }
            for (k = 0; k < n; k++)
            {
                double l = e[0][k] + e[1][k], r = e[0][k] - e[1][k];
                el += l * l;
                er += r * r;
            }
            if (noise[0] <= allowed[0] && noise[1] <= allowed[1] && el <= allowed_lr[0] && er <= allowed_lr[1])
                break;
            /* tighten the channel that contributes most, relative to its allowance */
            for (c = 0; c < 2; c++)
            {
                double rel = noise[c] / (allowed[c] + 1e-30);
                if (rel > worst) { worst = rel; worst_c = (int)c; }
            }
            if (el > allowed_lr[0] || er > allowed_lr[1])
                worst_c = noise[0] >= noise[1] ? 0 : 1;   /* both channels' errors add up in L/R: the louder error */
            if (zero[worst_c])
                zero[worst_c] = 0;
            else if (fits[worst_c] && sf[worst_c] > 0)
                budget[worst_c] *= RDO_JOINT_TIGHTEN;
            else
                break;
            dirty[worst_c] = 1;
        }
        if (bits_out)
            for (c = 0; c < 2; c++) bits_out[c] = bits[c];
    }
}

/* Long frames with the coder contexts: see mmx_frame_quantize. With an
   intensity plan, every candidate band above its first band is quantized
   both ways - the two channels' bands as usual, the mid band alone against
   the plan's threshold, both with the rate-distortion quantizer - and coded
   as an intensity band when flag, position and mid band cost fewer bits
   (times the plan's bias) than the two bands with their flags and
   scalefactors. The context chains (previous band's flag, position
   prediction, scalefactor deltas, neighbour classes) are tracked as the coder
   will see them. */
static void quantize_long_rdo(const MMXBandLayout *L, MMXFrameSyntax *s, float *const *x,
                              const float *const *thr, unsigned int cutoff_band, const float *const *thr_lr,
                              const float *const *crest, const MMXCodecContexts *ctx, unsigned int mode,
                              const MMXIntensityPlan *is, const MMXNfZeroWeight *nfz)
{
    unsigned int c, b, is0 = (is && is->first_band && s->channels == 2) ? is->first_band : 0;
    int joint = (s->stereo_ms && s->channels == 2 && thr_lr != NULL);
    int prev_sf[MMX_MAX_CH], q1[MMX_MAX_CH], q2[MMX_MAX_CH], prev_is = 0;
    signed char last_pos = 0;
    const unsigned char *pc[MMX_MAX_CH];
    static int qis[MMX_HOP];

    cost_tab_init();
    for (c = 0; c < s->channels; c++)
    {
        prev_sf[c] = ctx->has_prev ? ctx->prev_sf[c] : SF_BIAS;
        q1[c] = q2[c] = 0;
        pc[c] = ctx->prev_q[c];
    }
    memset(s->band_is, 0, sizeof(s->band_is));
    alloc_knobs();
    for (b = 0; b < L->band_count; b++)
    {
        unsigned long k0 = L->band_start[b], k1 = L->band_start[b + 1], k, n = k1 - k0;
        double energy[MMX_MAX_CH], allowed[MMX_MAX_CH], allowed_lr[2], noise[MMX_MAX_CH], cbits[MMX_MAX_CH], zw[MMX_MAX_CH];
        int zero[MMX_MAX_CH], any[MMX_MAX_CH], ctx_is = prev_is;
        unsigned int sf[MMX_MAX_CH];
        RdoCosts costs;

        prev_is = 0;
        if (b >= cutoff_band || thr[0][b] >= 1e29)
        {
            for (c = 0; c < s->channels; c++) { zero_band(s, c, b, k0, k1); q1[c] = q2[c] = 0; }
            continue;
        }
        rdo_costs_init(&costs, ctx, mode, fclass_of(b));
        for (c = 0; c < s->channels; c++)
        {
            energy[c] = 0.0;
            for (k = k0; k < k1; k++)
                energy[c] += (double)x[c][k] * x[c][k];
            allowed[c] = thr[c][b] * (double)n;
            zero[c] = alloc_zero ? energy[c] * (crest ? (double)crest[c][b] : 1.0) <= allowed[c] : energy[c] <= 0.0;
            sf[c] = mmx_sf_for_threshold(thr[c][b]);
            zw[c] = zero_weight_of(nfz, s, L, c, b);
        }
        if (joint)
        {
            allowed_lr[0] = thr_lr[0][b] * (double)n;
            allowed_lr[1] = thr_lr[1][b] * (double)n;
        }
        plain_quantize_band(s, x, k0, k1, energy, allowed, joint ? allowed_lr : NULL, zw, zero, sf, any, noise);
        rdo_quantize_band(s, x, k0, k1, energy, allowed, noise, joint ? allowed_lr : NULL, zw, &costs, pc, ctx->sf[mode],
                          prev_sf, q1, q2, zero, sf, any, cbits);
        if (alloc_k > 0)
            for (c = 0; c < s->channels; c++)
                if (!zero[c] && any[c]) any[c] = cap_band(s->q[c], k0, k1, alloc_k);
        if (is0 && b >= is0 && is->cand[b])
        {
            /* the mid band alone against the two channels' bands, exact context costs */
            const MMXProb *fctx = &ctx->is_flag[mode][ctx_is][ctx->prev_is[b]];
            double e_is = 0.0, allowed_is = (double)is->thr[b] * (double)n, noise_is, bits_is = 0.0, cost_n, cost_is;
            unsigned int sf_is = mmx_sf_for_threshold(is->thr[b]);
            int zero_is, any_is = 0;
            signed char pred_pos = ctx->prev_is[b] ? ctx->prev_is_pos[b] : last_pos;
            for (k = k0; k < k1; k++) e_is += (double)is->x[k] * is->x[k];
            zero_is = e_is * (crest ? (double)crest[0][b] : 1.0) <= allowed_is;
            cost_n = BITCOST(*fctx, 0);
            cost_is = BITCOST(*fctx, 1) + seg_cost(ctx->is_pos, (long long)is->pos[b] - pred_pos);
            for (c = 0; c < 2; c++)
            {
                int coded = !(zero[c] || !any[c]);
                cost_n += BITCOST(ctx->zero_band[mode][b], !coded);
                if (coded) cost_n += seg_cost(ctx->sf[mode], (long long)sf[c] - prev_sf[c]) + cbits[c];
            }
            noise_is = e_is;
            if (!zero_is)
            {
                for (;;)
                {
                    noise_is = quantize_band(is->x, k0, k1, mmx_sf_step(sf_is), 1.0, qis, &any_is, NULL);
                    if (noise_is <= allowed_is || sf_is == 0)
                        break;
                    sf_is--;
                }
                rdo_band(is->x, k0, k1, 1.0, &costs, pc[0], q1[0], q2[0], rdo_budget(noise_is, allowed_is), ctx->sf[mode], prev_sf[0],
                         &sf_is, qis + k0, &noise_is, &bits_is, &any_is);
            }
            if (zero_is || !any_is)
                cost_is += BITCOST(ctx->zero_band[mode][b], 1);
            else
                cost_is += BITCOST(ctx->zero_band[mode][b], 0) + seg_cost(ctx->sf[mode], (long long)sf_is - prev_sf[0]) + bits_is;
            if (cost_is * is->bias < cost_n)
            {
                s->band_is[b] = 1;
                s->is_pos[b] = is->pos[b];
                last_pos = is->pos[b];
                prev_is = 1;
                if (zero_is || !any_is)
                {
                    zero_band(s, 0, b, k0, k1);
                    q1[0] = q2[0] = 0;
                }
                else
                {
                    memcpy(s->q[0] + k0, qis + k0, sizeof(int) * n);
                    s->sf[0][b] = (unsigned char)sf_is;
                    s->band_zero[0][b] = 0;
                    s->band_noise[0][b] = 0;
                    prev_sf[0] = (int)sf_is;
                    q2[0] = n >= 2 ? s->q[0][k1 - 2] : q1[0];
                    q1[0] = s->q[0][k1 - 1];
                }
                zero_band(s, 1, b, k0, k1);
                q1[1] = q2[1] = 0;
                continue;
            }
        }
        for (c = 0; c < s->channels; c++)
        {
            if (zero[c] || !any[c])
            {
                zero_band(s, c, b, k0, k1);
                q1[c] = q2[c] = 0;
                continue;
            }
            s->sf[c][b] = (unsigned char)sf[c];
            s->band_zero[c][b] = 0;
            s->band_noise[c][b] = 0;
            prev_sf[c] = (int)sf[c];
            q2[c] = n >= 2 ? s->q[c][k1 - 2] : q1[c];
            q1[c] = s->q[c][k1 - 1];
        }
    }
}

/* Short frames with the coder contexts: see mmx_frame_quantize_short. The
   context chains are those of encode_short: the previous group's coefficient
   is the "previous frame" class, neighbours restart at every group. */
static void quantize_short_rdo(const MMXBandLayout *Ls, MMXFrameSyntax *s, float *const *x,
                               const float (*const *thr)[MMX_SHORT_BANDS], unsigned int cutoff_band,
                               const float (*const *thr_lr)[MMX_SHORT_BANDS], const MMXCodecContexts *ctx, unsigned int mode)
{
    static unsigned char pcs[MMX_MAX_CH][MMX_HOP];
    const unsigned char *pc[MMX_MAX_CH];
    unsigned int c, g, b;
    int joint = (s->stereo_ms && s->channels == 2 && thr_lr != NULL);
    int prev_sf[MMX_MAX_CH], q1[MMX_MAX_CH], q2[MMX_MAX_CH];

    cost_tab_init();
    memset(s->band_noise, 0, sizeof(s->band_noise)); /* short frames have no noise bands */
    memset(s->band_is, 0, sizeof(s->band_is));       /* ... and no intensity bands */
    for (c = 0; c < s->channels; c++)
    {
        prev_sf[c] = SF_BIAS;
        pc[c] = pcs[c];
    }
    for (g = 0; g < MMX_SHORT_GROUPS; g++)
    {
        for (c = 0; c < s->channels; c++) q1[c] = q2[c] = 0;
        for (b = 0; b < Ls->band_count && b < MMX_SHORT_BANDS; b++)
        {
            unsigned long k0 = g * MMX_SHORT_M + Ls->band_start[b], k1 = g * MMX_SHORT_M + Ls->band_start[b + 1], k, n = k1 - k0;
            double energy[MMX_MAX_CH], allowed[MMX_MAX_CH], allowed_lr[2], noise[MMX_MAX_CH];
            int zero[MMX_MAX_CH], any[MMX_MAX_CH];
            unsigned int sf[MMX_MAX_CH];
            RdoCosts costs;
            if (b >= cutoff_band || thr[0][g][b] >= 1e29f)
            {
                for (c = 0; c < s->channels; c++)
                {
                    s->sf_s[c][g][b] = 0;
                    s->band_zero_s[c][g][b] = 1;
                    for (k = k0; k < k1; k++) s->q[c][k] = 0;
                    q1[c] = q2[c] = 0;
                }
                continue;
            }
            rdo_costs_init(&costs, ctx, mode + 2, b < 3 ? 0 : b < 10 ? 1 : 2);
            for (c = 0; c < s->channels; c++)
            {
                energy[c] = 0.0;
                for (k = k0; k < k1; k++)
                {
                    int pq = g ? s->q[c][k - MMX_SHORT_M] : 0;
                    energy[c] += (double)x[c][k] * x[c][k];
                    pcs[c][k] = (unsigned char)(pq == 0 ? 0 : (pq == 1 || pq == -1) ? 1 : 2);
                }
                allowed[c] = thr[c][g][b] * (double)n;
                zero[c] = energy[c] <= allowed[c];
                sf[c] = mmx_sf_for_threshold(thr[c][g][b]);
            }
            if (joint)
            {
                allowed_lr[0] = thr_lr[0][g][b] * (double)n;
                allowed_lr[1] = thr_lr[1][g][b] * (double)n;
            }
            plain_quantize_band(s, x, k0, k1, energy, allowed, joint ? allowed_lr : NULL, zw_ones, zero, sf, any, noise);
            rdo_quantize_band(s, x, k0, k1, energy, allowed, noise, joint ? allowed_lr : NULL, zw_ones, &costs, pc, ctx->sf_s[mode],
                              prev_sf, q1, q2, zero, sf, any, NULL);
            for (c = 0; c < s->channels; c++)
            {
                if (zero[c] || !any[c])
                {
                    s->sf_s[c][g][b] = 0;
                    s->band_zero_s[c][g][b] = 1;
                    for (k = k0; k < k1; k++) s->q[c][k] = 0;
                    q1[c] = q2[c] = 0;
                    continue;
                }
                s->sf_s[c][g][b] = (unsigned char)sf[c];
                s->band_zero_s[c][g][b] = 0;
                prev_sf[c] = (int)sf[c];
                q2[c] = n >= 2 ? s->q[c][k1 - 2] : q1[c];
                q1[c] = s->q[c][k1 - 1];
            }
        }
    }
}

void mmx_frame_quantize(const MMXBandLayout *L, MMXFrameSyntax *s, float *const *x,
                        const float *const *thr, unsigned int cutoff_band, const float *const *thr_lr,
                        const float *const *crest, const MMXCodecContexts *ctx, unsigned int mode)
{
    mmx_frame_quantize_ext(L, s, x, thr, cutoff_band, thr_lr, crest, ctx, mode, NULL, NULL);
}

void mmx_frame_quantize_ext(const MMXBandLayout *L, MMXFrameSyntax *s, float *const *x,
                            const float *const *thr, unsigned int cutoff_band, const float *const *thr_lr,
                            const float *const *crest, const MMXCodecContexts *ctx, unsigned int mode,
                            const MMXIntensityPlan *is, const MMXNfZeroWeight *nfz)
{
    unsigned int c, b;
    int joint = (s->stereo_ms && s->channels == 2 && thr_lr != NULL);

    memset(s->nf_level, 0, sizeof(s->nf_level));   /* noise filling levels are set by the encoder after quantization */
    memset(s->band_bwe, 0, sizeof(s->band_bwe));   /* the replicated bands are set by the encoder after quantization */
    memset(s->bwe_mix, 0, sizeof(s->bwe_mix));
    if (ctx && rdoq_enabled())
    {
        quantize_long_rdo(L, s, x, thr, cutoff_band, thr_lr, crest, ctx, mode, is, nfz);
        return;
    }
    memset(s->band_is, 0, sizeof(s->band_is));   /* the plain quantizer codes no intensity bands */
    for (b = 0; b < L->band_count; b++)
    {
        unsigned long k0 = L->band_start[b], k1 = L->band_start[b + 1], k, n = k1 - k0;
        double energy[MMX_MAX_CH], allowed[MMX_MAX_CH], allowed_lr[2], noise[MMX_MAX_CH], zw[MMX_MAX_CH];
        int zero[MMX_MAX_CH], any[MMX_MAX_CH];
        unsigned int sf[MMX_MAX_CH];

        if (b >= cutoff_band || thr[0][b] >= 1e29)
        {
            for (c = 0; c < s->channels; c++)
                zero_band(s, c, b, k0, k1);
            continue;
        }
        for (c = 0; c < s->channels; c++)
        {
            energy[c] = 0.0;
            for (k = k0; k < k1; k++)
                energy[c] += (double)x[c][k] * x[c][k];
            allowed[c] = thr[c][b] * (double)n;
            /* whole band below the masking threshold, also at its temporal peak */
            zero[c] = alloc_zero ? energy[c] * (crest ? (double)crest[c][b] : 1.0) <= allowed[c] : energy[c] <= 0.0;
            sf[c] = mmx_sf_for_threshold(thr[c][b]);
            zw[c] = zero_weight_of(nfz, s, L, c, b);
        }
        if (joint)
        {
            allowed_lr[0] = thr_lr[0][b] * (double)n;
            allowed_lr[1] = thr_lr[1][b] * (double)n;
        }
        plain_quantize_band(s, x, k0, k1, energy, allowed, joint ? allowed_lr : NULL, zw, zero, sf, any, noise);
        if (alloc_k > 0)
            for (c = 0; c < s->channels; c++)
                if (!zero[c] && any[c]) any[c] = cap_band(s->q[c], k0, k1, alloc_k);
        for (c = 0; c < s->channels; c++)
        {
            if (zero[c] || !any[c])
            {
                zero_band(s, c, b, k0, k1);
                continue;
            }
            s->sf[c][b] = (unsigned char)sf[c];
            s->band_zero[c][b] = 0;
            s->band_noise[c][b] = 0;
        }
    }
}

void mmx_frame_quantize_short(const MMXBandLayout *Ls, MMXFrameSyntax *s, float *const *x,
                              const float (*const *thr)[MMX_SHORT_BANDS], unsigned int cutoff_band,
                              const float (*const *thr_lr)[MMX_SHORT_BANDS], const MMXCodecContexts *ctx, unsigned int mode)
{
    unsigned int c, g, b;
    int joint = (s->stereo_ms && s->channels == 2 && thr_lr != NULL);

    memset(s->nf_level, 0, sizeof(s->nf_level));    /* short frames have no noise filling */
    memset(s->band_bwe, 0, sizeof(s->band_bwe));    /* ... and no band replication */
    memset(s->bwe_mix, 0, sizeof(s->bwe_mix));
    if (ctx && rdoq_enabled())
    {
        quantize_short_rdo(Ls, s, x, thr, cutoff_band, thr_lr, ctx, mode);
        return;
    }
    memset(s->band_noise, 0, sizeof(s->band_noise)); /* short frames have no noise bands */
    memset(s->band_is, 0, sizeof(s->band_is));       /* ... and no intensity bands */
    for (g = 0; g < MMX_SHORT_GROUPS; g++)
        for (b = 0; b < Ls->band_count && b < MMX_SHORT_BANDS; b++)
        {
            unsigned long k0 = g * MMX_SHORT_M + Ls->band_start[b], k1 = g * MMX_SHORT_M + Ls->band_start[b + 1], k, n = k1 - k0;
            double energy[MMX_MAX_CH], allowed[MMX_MAX_CH], allowed_lr[2], noise[MMX_MAX_CH];
            int zero[MMX_MAX_CH], any[MMX_MAX_CH];
            unsigned int sf[MMX_MAX_CH];
            if (b >= cutoff_band || thr[0][g][b] >= 1e29f)
            {
                for (c = 0; c < s->channels; c++)
                {
                    s->sf_s[c][g][b] = 0;
                    s->band_zero_s[c][g][b] = 1;
                    for (k = k0; k < k1; k++) s->q[c][k] = 0;
                }
                continue;
            }
            for (c = 0; c < s->channels; c++)
            {
                energy[c] = 0.0;
                for (k = k0; k < k1; k++) energy[c] += (double)x[c][k] * x[c][k];
                allowed[c] = thr[c][g][b] * (double)n;
                zero[c] = energy[c] <= allowed[c];
                sf[c] = mmx_sf_for_threshold(thr[c][g][b]);
            }
            if (joint)
            {
                allowed_lr[0] = thr_lr[0][g][b] * (double)n;
                allowed_lr[1] = thr_lr[1][g][b] * (double)n;
            }
            plain_quantize_band(s, x, k0, k1, energy, allowed, joint ? allowed_lr : NULL, zw_ones, zero, sf, any, noise);
            for (c = 0; c < s->channels; c++)
            {
                if (zero[c] || !any[c]) { s->sf_s[c][g][b] = 0; s->band_zero_s[c][g][b] = 1; for (k = k0; k < k1; k++) s->q[c][k] = 0; }
                else { s->sf_s[c][g][b] = (unsigned char)sf[c]; s->band_zero_s[c][g][b] = 0; }
            }
        }
}

void mmx_frame_dequantize_short(const MMXBandLayout *Ls, const MMXFrameSyntax *s, float *const *out)
{
    unsigned int c, g, b;
    for (c = 0; c < s->channels; c++)
        for (g = 0; g < MMX_SHORT_GROUPS; g++)
            for (b = 0; b < Ls->band_count && b < MMX_SHORT_BANDS; b++)
            {
                unsigned long k0 = g * MMX_SHORT_M + Ls->band_start[b], k1 = g * MMX_SHORT_M + Ls->band_start[b + 1], k;
                double step = mmx_sf_step(s->sf_s[c][g][b]);
                for (k = k0; k < k1; k++)
                    out[c][k] = s->band_zero_s[c][g][b] ? 0.0f : (float)(s->q[c][k] * step);
            }
}

double mmx_frame_estimate_bits_short(const MMXBandLayout *Ls, const float *x, const float (*thr)[MMX_SHORT_BANDS], unsigned int cutoff_band)
{
    unsigned int g, b;
    double bits = 0.0;
    for (g = 0; g < MMX_SHORT_GROUPS; g++)
        for (b = 0; b < Ls->band_count && b < cutoff_band && b < MMX_SHORT_BANDS; b++)
        {
            unsigned long k0 = g * MMX_SHORT_M + Ls->band_start[b], k1 = g * MMX_SHORT_M + Ls->band_start[b + 1], k;
            double t = thr[g][b], energy = 0.0, inv;
            if (t >= 1e29) break;
            for (k = k0; k < k1; k++) energy += (double)x[k] * x[k];
            bits += 0.3;
            if (energy <= t * (double)(k1 - k0)) continue;
            bits += 3.0;
            inv = 1.0 / mmx_sf_step(mmx_sf_for_threshold(t));
            for (k = k0; k < k1; k++)
            {
                double a = fabs((double)x[k]) * inv;
                bits += a < 0.5 ? 0.25 : 2.2 + 1.35 * log(a + 0.5) / log(2.0);
            }
        }
    return bits;
}

/* ------------------------------------------------- noise substitution */

unsigned int mmx_pns_first_band(const MMXBandLayout *L)
{
    unsigned int b;
    for (b = 0; b < L->band_count; b++)
        if (L->band_hz[b] >= (float)MMX_PNS_MIN_HZ)
            return b;
    return L->band_count;
}

unsigned int mmx_pns_level_index(double energy, unsigned long n)
{
    double v;
    if (energy <= 0.0 || n == 0)
        return 0;
    v = floor(2.0 * log(energy / (double)n) / log(2.0) + 0.5) + SF_BIAS;   /* 4 * log2(rms) */
    if (v < 0.0) v = 0.0;
    if (v > MMX_SF_COUNT - 1) v = MMX_SF_COUNT - 1;
    return (unsigned int)v;
}

/* Fills out[k0..k1) with pseudo-random noise of exactly n * rms^2 energy. The
   generator (xorshift32, seeded from frame, channel and band) and the scaling
   are part of the format: the encoder reconstructs the same values. Uniform
   values suffice, the synthesis window sums many of them. */
static void fill_noise(float *out, unsigned long k0, unsigned long k1, double rms, unsigned long frame, unsigned int c, unsigned int b)
{
    unsigned long x = ((frame + 1UL) * 0x9E3779B1UL ^ (unsigned long)(c + 1) * 0x85EBCA77UL ^ (unsigned long)(b + 1) * 0xC2B2AE3DUL) & 0xFFFFFFFFUL;
    double sum = 0.0, g;
    unsigned long k;
    if (x == 0) x = 0x1234567UL;
    for (k = k0; k < k1; k++)
    {
        x ^= (x << 13) & 0xFFFFFFFFUL;
        x ^= x >> 17;
        x ^= (x << 5) & 0xFFFFFFFFUL;
        out[k] = (float)((double)((x >> 8) & 0xFFFFFFUL) / 8388608.0 - 1.0);
        sum += (double)out[k] * out[k];
    }
    g = sum > 0.0 ? rms * sqrt((double)(k1 - k0) / sum) : 0.0;
    for (k = k0; k < k1; k++)
        out[k] = (float)(out[k] * g);
}

/* ------------------------------------------------------ band replication */

/* Source coefficient of a replicated coefficient (part of the format).
   Mode 0 (default): octave transposition, k >> j with the smallest j that
   lands below the crossover, so the source always sits in the octave
   [X/2, X) and a harmonic series maps onto a subset of itself - no beating
   against the coded part. Mode 1: linear shift by the width of that octave
   (the fine structure is copied unchanged, the harmonic spacing is not). */
unsigned long mmx_bwe_source_bin(const MMXBandLayout *L, unsigned long k)
{
    unsigned long x = L->band_start[L->bwe_band], s0 = L->bwe_src0, src;
    if (L->bwe_mode == 1)
    {
        unsigned long span = x - s0;
        src = k;
        while (src >= x && span)
            src -= span;
        if (src < s0) src = s0 + (src % (span ? span : 1));
        return src;
    }
    src = k;
    while (src >= x)
        src >>= 1;
    if (src < s0) src = s0;
    return src;
}

/* Spectral flatness (geometric / arithmetic mean of the bin powers) of
   x[k0..k1), 0 = a single tone, ~1 = flat. Encoder side only. */
static double band_flatness(const float *x, unsigned long k0, unsigned long k1)
{
    double sum = 0.0, logsum = 0.0;
    unsigned long k, n = k1 - k0;
    if (n == 0)
        return 1.0;
    for (k = k0; k < k1; k++)
    {
        double v = (double)x[k] * x[k] + 1e-30;
        sum += v;
        logsum += log(v);
    }
    if (sum <= 0.0)
        return 1.0;
    return exp(logsum / (double)n) / (sum / (double)n);
}

/* Encoder: how much noise a replicated band needs. The patch carries the fine
   structure of the source band; when the target is noisier than the source
   (a cymbal wash regenerated from a tonal guitar octave) a copy alone sounds
   metallic, when the target is more tonal than the source, noise would smear
   it. The index is the share of the band energy that comes from noise,
   interpolating the two flatness values (white noise through this window
   measures about 0.56). */
unsigned int mmx_bwe_mix_index(const MMXBandLayout *L, const float *x, unsigned int b)
{
    unsigned long k0 = L->band_start[b], k1 = L->band_start[b + 1], k;
    double ft, fp, p;
    static float patch[MMX_HOP];
    if (k1 <= k0)
        return 0;
    for (k = k0; k < k1; k++)
        patch[k] = x[mmx_bwe_source_bin(L, k)];
    ft = band_flatness(x, k0, k1);
    fp = band_flatness(patch, k0, k1);
    if (ft <= fp)
        return 0;
    p = (ft - fp) / (1.0 - fp + 1e-9);
    if (p < 0.0) p = 0.0;
    if (p > 1.0) p = 1.0;
    return (unsigned int)floor(p * MMX_BWE_MIX_MAX + 0.5);
}

/* Decoder and encoder closed loop: regenerates every replicated band of a long
   frame. `rec` is the finished reconstruction in the coded channel domain; the
   source coefficients all sit below the crossover and are therefore final.
   Patch and noise are each normalized to unit rms, mixed by the band's index
   and the result scaled so that the band carries exactly n * step(level)^2. */
void mmx_bwe_regenerate(const MMXBandLayout *L, const MMXFrameSyntax *s, float *const *rec, unsigned long frame)
{
    unsigned int c, b;
    static float noise[MMX_HOP];
    if (L->bwe_band >= L->band_count || s->block_type == MMX_BT_SHORT)
        return;
    for (c = 0; c < s->channels; c++)
        for (b = L->bwe_band; b < L->band_count; b++)
        {
            unsigned long k0 = L->band_start[b], k1 = L->band_start[b + 1], k, n = k1 - k0;
            double rms, mix, ep = 0.0, en = 0.0, gp, gn, e = 0.0, g;
            if (!s->band_bwe[c][b])
                continue;
            if (s->band_zero[c][b])
            {
                for (k = k0; k < k1; k++) rec[c][k] = 0.0f;
                continue;
            }
            rms = mmx_sf_step(s->sf[c][b]);
            mix = (double)s->bwe_mix[c][b] / (double)MMX_BWE_MIX_MAX;
            fill_noise(noise, k0, k1, 1.0, frame, c, b);
            for (k = k0; k < k1; k++)
            {
                double v = rec[c][mmx_bwe_source_bin(L, k)];
                ep += v * v;
                en += (double)noise[k] * noise[k];
            }
            gp = ep > 0.0 ? sqrt((1.0 - mix) * (double)n / ep) : 0.0;
            gn = en > 0.0 ? sqrt(mix * (double)n / en) : 0.0;
            if (gp == 0.0 && gn == 0.0)
            {
                gn = en > 0.0 ? sqrt((double)n / en) : 0.0;   /* silent source: pure noise */
                if (gn == 0.0) { for (k = k0; k < k1; k++) rec[c][k] = 0.0f; continue; }
            }
            for (k = k0; k < k1; k++)
            {
                double v = gp * (double)rec[c][mmx_bwe_source_bin(L, k)] + gn * (double)noise[k];
                noise[k] = (float)v;      /* the patch source may lie inside this band's own range? no: it is below the crossover */
                e += v * v;
            }
            g = e > 0.0 ? rms * sqrt((double)n / e) : 0.0;
            for (k = k0; k < k1; k++)
                rec[c][k] = (float)(noise[k] * g);
        }
}

/* ------------------------------------------------------- energy-preserving bands (EPB, revision 9) */

unsigned int mmx_epb_level(double energy, unsigned long n)
{
    return mmx_pns_level_index(energy, n);            /* the same 1.5 dB grid as every other level: 4 * log2(rms) + 48 */
}

/* Every band of a long frame ends up carrying exactly the energy its index says. A coded band (with or
   without a prediction under it) is scaled to it, within +-12 dB so a broken index cannot blow up a band;
   an uncoded band is filled from the reconstruction an octave below (the octave mapping of the band
   replication, source bin k >> 1) - the content the ear expects up there when the coder could not afford
   the band - or with noise where there is no octave below (the lowest bins) or the source is silent. */
/* Every band of a long frame ends up carrying the energy its index says. What the band already has
   (prediction plus coded residual) stays as it is: the coded coefficients keep their amplitudes and a
   prediction is never overwritten. The missing remainder is added at the positions the coder left
   empty (q == 0): from the finished reconstruction an octave below (source bin k >> 1, the mapping of
   the band replication) above the fold floor, as noise under it. A band with more energy than its
   index says is scaled down, at most 12 dB. Encoder closed loop and decoder both run this, so the
   prediction of later frames sees the finished bands. */
void mmx_epb_finish(const MMXBandLayout *L, const MMXFrameSyntax *s, float *const *rec, unsigned long frame)
{
    unsigned int c, b;
    static float noise[MMX_HOP];
    if (!L->lowrate || s->block_type == MMX_BT_SHORT)
        return;
    for (c = 0; c < s->channels; c++)
        for (b = 0; b < L->band_count; b++)
        {
            unsigned long k0 = L->band_start[b], k1 = L->band_start[b + 1], k, n = k1 - k0;
            double target, have = 0.0, rem, es = 0.0, g;
            int fold, coded = !s->band_zero[c][b] && !s->band_noise[c][b];
            if (s->channels == 2 && s->band_is[b] && c == 1)
                continue;                                  /* intensity: channel 1 is derived from channel 0 later */
            if (s->band_bwe[c][b])
                continue;                                  /* replicated bands keep their own level machinery */
            if (coded && !L->epb_coded)
                continue;
            for (k = k0; k < k1; k++) have += (double)rec[c][k] * rec[c][k];
            if (s->nrg[c][b] == 0)
            {
                for (k = k0; k < k1; k++) rec[c][k] = 0.0f;   /* the source band is empty */
                continue;
            }
            target = mmx_sf_step(s->nrg[c][b]);
            target = target * target * (double)n;
            if (have > target)
            {
                g = sqrt(target / have);
                if (g < 0.251) g = 0.251;                  /* -12 dB at most */
                for (k = k0; k < k1; k++) rec[c][k] = (float)(rec[c][k] * g);
                continue;
            }
            rem = target - have;
            if (rem <= 1e-3 * target)
                continue;                                  /* within 0.004 dB: nothing to add */
            fold = (double)k0 * L->sample_rate / (2.0 * MMX_HOP) >= L->epb_fold_hz && k0 >= 16;
            if (!fold && !L->epb_noise)
                continue;
            /* the fill at the empty positions only */
            if (fold)
                for (k = k0; k < k1; k++) noise[k] = (coded && s->q[c][k]) ? 0.0f : rec[c][k >> 1];
            else
            {
                fill_noise(noise, k0, k1, 1.0, frame, c, b);
                if (coded) for (k = k0; k < k1; k++) if (s->q[c][k]) noise[k] = 0.0f;
            }
            for (k = k0; k < k1; k++) es += (double)noise[k] * noise[k];
            if (es <= 1e-12 * target)
            {
                if (fold)                                  /* the octave below is empty too: noise then */
                {
                    fill_noise(noise, k0, k1, 1.0, frame, c, b);
                    if (coded) for (k = k0; k < k1; k++) if (s->q[c][k]) noise[k] = 0.0f;
                    es = 0.0;
                    for (k = k0; k < k1; k++) es += (double)noise[k] * noise[k];
                }
                if (es <= 0.0) continue;
            }
            g = L->epb_fill_gain * sqrt(rem / es);
            for (k = k0; k < k1; k++) rec[c][k] = (float)(rec[c][k] + noise[k] * g);
        }
}

/* ------------------------------------------------------- noise filling */

double mmx_nf_ratio(unsigned int level)
{
    return level ? pow(2.0, ((double)level - MMX_NF_LEVELS) / 4.0) : 0.0;
}

unsigned int mmx_nf_level_index(double ratio)
{
    double v;
    if (ratio <= 0.0)
        return 0;
    v = floor(4.0 * log(ratio) / log(2.0) + 0.5) + MMX_NF_LEVELS;
    if (v < 1.0) return 0;
    if (v > MMX_NF_LEVELS - 1) v = MMX_NF_LEVELS - 1;
    return (unsigned int)v;
}

unsigned int mmx_nf_first_region(const MMXBandLayout *L)
{
    unsigned int b = mmx_pns_first_band(L);
    return b < L->band_count ? L->eq_band[b] : MMX_EQ_BANDS;
}

/* Fills the zero-quantized coefficients of out[k0..k1) with noise of exactly
   rms^2 energy per filled coefficient, same generator and seed as fill_noise
   (a band is either a noise band or a coded band). */
static void fill_zeros(float *out, const int *q, unsigned long k0, unsigned long k1, double rms, unsigned long frame, unsigned int c, unsigned int b)
{
    unsigned long x = ((frame + 1UL) * 0x9E3779B1UL ^ (unsigned long)(c + 1) * 0x85EBCA77UL ^ (unsigned long)(b + 1) * 0xC2B2AE3DUL) & 0xFFFFFFFFUL;
    double sum = 0.0, g;
    unsigned long k, n = 0;
    if (x == 0) x = 0x1234567UL;
    for (k = k0; k < k1; k++)
    {
        if (q[k])
            continue;
        x ^= (x << 13) & 0xFFFFFFFFUL;
        x ^= x >> 17;
        x ^= (x << 5) & 0xFFFFFFFFUL;
        out[k] = (float)((double)((x >> 8) & 0xFFFFFFUL) / 8388608.0 - 1.0);
        sum += (double)out[k] * out[k];
        n++;
    }
    g = sum > 0.0 ? rms * sqrt((double)n / sum) : 0.0;
    for (k = k0; k < k1; k++)
        if (!q[k])
            out[k] = (float)(out[k] * g);
}

void mmx_frame_noise_fill_levels(const MMXBandLayout *L, MMXFrameSyntax *s, float *const *x, unsigned int first_region,
                                 double gain, const unsigned char (*skip)[MMX_EQ_BANDS])
{
    unsigned int c, b, e, pns0 = mmx_pns_first_band(L);
    for (c = 0; c < s->channels; c++)
    {
        double sum[MMX_EQ_BANDS];
        unsigned long cnt[MMX_EQ_BANDS];
        memset(s->nf_level[c], 0, MMX_EQ_BANDS);
        if (s->block_type == MMX_BT_SHORT)
            continue;
        for (e = 0; e < MMX_EQ_BANDS; e++) { sum[e] = 0.0; cnt[e] = 0; }
        for (b = pns0; b < L->band_count; b++)
        {
            unsigned long k0 = L->band_start[b], k1 = L->band_start[b + 1], k;
            double inv;
            e = L->eq_band[b];
            if (s->band_zero[c][b] || s->band_noise[c][b] || s->band_bwe[c][b] || e < first_region || (skip && skip[c][e]))
                continue;
            if (s->channels == 2 && s->band_is[b])
                continue;   /* intensity band: channel 0 holds the mid, not this channel's signal */
            inv = 1.0 / mmx_sf_step(s->sf[c][b]);
            for (k = k0; k < k1; k++)
                if (s->q[c][k] == 0)
                {
                    double v = x[c][k] * inv;
                    sum[e] += v * v;
                    cnt[e]++;
                }
        }
        for (e = first_region; e < MMX_EQ_BANDS; e++)
            if (cnt[e])
                s->nf_level[c][e] = (unsigned char)mmx_nf_level_index(gain * sqrt(sum[e] / (double)cnt[e]));
    }
}

void mmx_frame_dequantize(const MMXBandLayout *L, const MMXFrameSyntax *s, float *const *out, unsigned long frame)
{
    unsigned int c, b, pns0 = mmx_pns_first_band(L);
    for (c = 0; c < s->channels; c++)
        for (b = 0; b < L->band_count; b++)
        {
            unsigned long k0 = L->band_start[b], k1 = L->band_start[b + 1], k;
            double step = mmx_sf_step(s->sf[c][b]);
            unsigned int nf;
            if (s->band_zero[c][b] || s->band_bwe[c][b])
            {
                for (k = k0; k < k1; k++) out[c][k] = 0.0f;   /* replicated bands are regenerated after the prediction */
                continue;
            }
            if (s->band_noise[c][b])
            {
                fill_noise(out[c], k0, k1, step, frame, c, b);
                continue;
            }
            for (k = k0; k < k1; k++)
                out[c][k] = (float)(s->q[c][k] * step);
            /* noise filling: the zeros of the band carry the region's fill level */
            nf = b >= pns0 ? s->nf_level[c][L->eq_band[b]] : 0;
            if (nf)
                fill_zeros(out[c], s->q[c], k0, k1, step * mmx_nf_ratio(nf), frame, c, b);
        }
}


static unsigned int fclass_of(unsigned int band)
{
    return band < 6 ? 0 : band < 20 ? 1 : 2;
}

static unsigned int nbclass_of(int a, int b)
{
    int s = (a < 0 ? -a : a) + (b < 0 ? -b : b);
    return s == 0 ? 0 : s == 1 ? 1 : s == 2 ? 2 : s <= 4 ? 3 : s <= 8 ? 4 : 5;
}

static void encode_gains(MMXRangeEncoder *rc, MMXCodecContexts *ctx, const MMXFrameSyntax *s, unsigned int n_sources)
{
    unsigned int src, c, e;
    for (src = 0; src < n_sources; src++)
        for (c = 0; c < s->channels; c++)
        {
            mmx_rc_enc_bit(rc, &ctx->polarity, s->polarity[src][c] != ctx->prev_polarity[src][c]);
            for (e = 0; e < MMX_EQ_BANDS; e++)
            {
                signed char g = s->gain[src][c][e], prev = ctx->prev_gain[src][c][e];
                int prev_off = (prev == MMX_GAIN_OFF);
                mmx_rc_enc_bit(rc, &ctx->gain_off[prev_off], g == MMX_GAIN_OFF);
                if (g != MMX_GAIN_OFF)
                    mmx_rc_enc_seg(rc, ctx->gain, (long long)g - (prev_off ? 0 : prev));
            }
        }
}

static void decode_gains(MMXRangeDecoder *rc, MMXCodecContexts *ctx, MMXFrameSyntax *s, unsigned int n_sources)
{
    unsigned int src, c, e;
    for (src = 0; src < n_sources; src++)
        for (c = 0; c < s->channels; c++)
        {
            unsigned int flip = mmx_rc_dec_bit(rc, &ctx->polarity);
            s->polarity[src][c] = (unsigned char)(ctx->prev_polarity[src][c] ^ flip);
            for (e = 0; e < MMX_EQ_BANDS; e++)
            {
                signed char prev = ctx->prev_gain[src][c][e];
                int prev_off = (prev == MMX_GAIN_OFF);
                if (mmx_rc_dec_bit(rc, &ctx->gain_off[prev_off]))
                    s->gain[src][c][e] = MMX_GAIN_OFF;
                else
                {
                    long long v = mmx_rc_dec_seg(rc, ctx->gain) + (prev_off ? 0 : prev);
                    if (v < MMX_GAIN_MIN) v = MMX_GAIN_MIN;
                    if (v > MMX_GAIN_MAX) v = MMX_GAIN_MAX;
                    s->gain[src][c][e] = (signed char)v;
                }
            }
        }
}

/* EPB: the energy index a band contributes to prediction. Channel 1 of an intensity band carries no
   energy symbol (it is derived from channel 0), so both sides use channel 0's index for it. */
static int nrg_ref(const MMXFrameSyntax *s, unsigned int c, unsigned int b)
{
    if (c == 1 && s->channels == 2 && s->band_is[b])
        return s->nrg[0][b];
    return s->nrg[c][b];
}

static void remember(MMXCodecContexts *ctx, const MMXFrameSyntax *s, unsigned int n_sources)
{
    unsigned int src, c, b;
    unsigned long k;
    for (c = 0; c < s->channels; c++)
    {
        unsigned char last = ctx->has_prev ? ctx->prev_sf[c] : SF_BIAS;
        for (b = 0; b < MMX_MAX_BANDS; b++)
        {
            if (!s->band_zero[c][b] && !s->band_noise[c][b] && s->sf[c][b])
                last = s->sf[c][b];
            ctx->prev_pns[c][b] = s->block_type == MMX_BT_SHORT ? 0 : s->band_noise[c][b];
            if (s->block_type != MMX_BT_SHORT) ctx->prev_nrg[c][b] = (unsigned char)nrg_ref(s, c, b);
            if (s->block_type == MMX_BT_SHORT)
                ctx->prev_bwe[c][b] = 0;
            else
            {
                ctx->prev_bwe[c][b] = s->band_bwe[c][b];
                if (s->band_bwe[c][b] && !s->band_zero[c][b])
                {
                    ctx->prev_bwe_level[c][b] = s->sf[c][b];
                    ctx->prev_bwe_mix[c][b] = s->bwe_mix[c][b];
                }
            }
            if (c == 0)
            {
                ctx->prev_is[b] = s->block_type == MMX_BT_SHORT || s->channels != 2 ? 0 : s->band_is[b];
                ctx->prev_is_pos[b] = ctx->prev_is[b] ? s->is_pos[b] : 0;
            }
        }
        if (s->block_type == MMX_BT_SHORT)
            memset(ctx->prev_nf[c], 0, MMX_EQ_BANDS);
        else
            memcpy(ctx->prev_nf[c], s->nf_level[c], MMX_EQ_BANDS);
        ctx->prev_sf[c] = last;
        for (k = 0; k < s->m; k++)
        {
            int a = s->q[c][k] < 0 ? -s->q[c][k] : s->q[c][k];
            ctx->prev_q[c][k] = (unsigned char)(s->block_type == MMX_BT_SHORT ? 0 : (a == 0 ? 0 : a == 1 ? 1 : 2));
        }
    }
    for (src = 0; src < n_sources; src++)
        for (c = 0; c < s->channels; c++)
        {
            memcpy(ctx->prev_gain[src][c], s->gain[src][c], MMX_EQ_BANDS);
            ctx->prev_polarity[src][c] = s->polarity[src][c];
        }
    ctx->has_prev = 1;
}

static void enc_block_type(MMXRangeEncoder *rc, MMXCodecContexts *ctx, unsigned int bt)
{
    MMXProb *p = ctx->block_type[ctx->prev_block_type & 3];
    mmx_rc_enc_bit(rc, &p[0], bt >> 1);
    mmx_rc_enc_bit(rc, &p[1 + (bt >> 1)], bt & 1);
    ctx->prev_block_type = (unsigned char)bt;
}

static unsigned int dec_block_type(MMXRangeDecoder *rc, MMXCodecContexts *ctx)
{
    MMXProb *p = ctx->block_type[ctx->prev_block_type & 3];
    unsigned int hi = mmx_rc_dec_bit(rc, &p[0]), bt;
    bt = (hi << 1) | mmx_rc_dec_bit(rc, &p[1 + hi]);
    ctx->prev_block_type = (unsigned char)bt;
    return bt;
}

/* short frames: coefficients of the 8 groups, contexts from the previous group */
static void encode_short(MMXRangeEncoder *rc, MMXCodecContexts *ctx, const MMXBandLayout *Ls,
                         const MMXFrameSyntax *s, unsigned int mode)
{
    unsigned int c, g, b;
    for (c = 0; c < s->channels; c++)
    {
        int prev_sf = SF_BIAS;
        for (g = 0; g < MMX_SHORT_GROUPS; g++)
        {
            int q1 = 0, q2 = 0;
            for (b = 0; b < Ls->band_count && b < MMX_SHORT_BANDS; b++)
            {
                unsigned long k0 = g * MMX_SHORT_M + Ls->band_start[b], k1 = g * MMX_SHORT_M + Ls->band_start[b + 1], k;
                unsigned int fc = b < 3 ? 0 : b < 10 ? 1 : 2;
                mmx_rc_enc_bit(rc, &ctx->zero_band_s[mode - 2][b], s->band_zero_s[c][g][b] != 0);
                if (s->band_zero_s[c][g][b]) { q1 = q2 = 0; continue; }
                mmx_rc_enc_seg(rc, ctx->sf_s[mode - 2], (long long)s->sf_s[c][g][b] - prev_sf);
                prev_sf = s->sf_s[c][g][b];
                for (k = k0; k < k1; k++)
                {
                    int q = s->q[c][k], a = q < 0 ? -q : q, pq = g ? s->q[c][k - MMX_SHORT_M] : 0;
                    unsigned int nb = nbclass_of(q1, q2), pc = pq == 0 ? 0 : (pq == 1 || pq == -1) ? 1 : 2;
                    mmx_rc_enc_bit(rc, &ctx->coef_zero[mode][fc][nb][pc], a != 0);
                    if (a)
                    {
                        mmx_rc_enc_bypass(rc, q < 0, 1);
                        mmx_rc_enc_bit(rc, &ctx->coef_gt1[mode][fc][nb][pc], a > 1);
                        if (a > 1)
                        {
                            mmx_rc_enc_bit(rc, &ctx->coef_gt2[mode][fc][nb][pc], a > 2);
                            if (a > 2) mmx_rc_enc_ueg(rc, ctx->coef_eg[mode][fc], (unsigned long)(a - 3));
                        }
                    }
                    q2 = q1; q1 = q;
                }
            }
        }
    }
}

static int decode_short(MMXRangeDecoder *rc, MMXCodecContexts *ctx, const MMXBandLayout *Ls,
                        MMXFrameSyntax *s, unsigned int mode)
{
    unsigned int c, g, b;
    for (c = 0; c < s->channels; c++)
    {
        int prev_sf = SF_BIAS;
        for (g = 0; g < MMX_SHORT_GROUPS; g++)
        {
            int q1 = 0, q2 = 0;
            for (b = 0; b < Ls->band_count && b < MMX_SHORT_BANDS; b++)
            {
                unsigned long k0 = g * MMX_SHORT_M + Ls->band_start[b], k1 = g * MMX_SHORT_M + Ls->band_start[b + 1], k;
                unsigned int fc = b < 3 ? 0 : b < 10 ? 1 : 2;
                long long sf;
                s->band_zero_s[c][g][b] = (unsigned char)mmx_rc_dec_bit(rc, &ctx->zero_band_s[mode - 2][b]);
                if (s->band_zero_s[c][g][b]) { s->sf_s[c][g][b] = 0; for (k = k0; k < k1; k++) s->q[c][k] = 0; q1 = q2 = 0; continue; }
                sf = mmx_rc_dec_seg(rc, ctx->sf_s[mode - 2]) + prev_sf;
                if (sf < 0 || sf >= MMX_SF_COUNT || rc->failed) return -1;
                s->sf_s[c][g][b] = (unsigned char)sf;
                prev_sf = (int)sf;
                for (k = k0; k < k1; k++)
                {
                    int pq = g ? s->q[c][k - MMX_SHORT_M] : 0, a = 0;
                    unsigned int nb = nbclass_of(q1, q2), pc = pq == 0 ? 0 : (pq == 1 || pq == -1) ? 1 : 2;
                    if (mmx_rc_dec_bit(rc, &ctx->coef_zero[mode][fc][nb][pc]))
                    {
                        unsigned int neg = (unsigned int)mmx_rc_dec_bypass(rc, 1);
                        a = 1;
                        if (mmx_rc_dec_bit(rc, &ctx->coef_gt1[mode][fc][nb][pc]))
                        {
                            a = 2;
                            if (mmx_rc_dec_bit(rc, &ctx->coef_gt2[mode][fc][nb][pc]))
                                a = 3 + (int)mmx_rc_dec_ueg(rc, ctx->coef_eg[mode][fc]);
                        }
                        if (neg) a = -a;
                    }
                    s->q[c][k] = a;
                    q2 = q1; q1 = a;
                }
                if (rc->failed) return -1;
            }
        }
    }
    return 0;
}

void mmx_frame_encode(MMXRangeEncoder *rc, MMXCodecContexts *ctx, const MMXBandLayout *L, const MMXBandLayout *Ls,
                      const MMXFrameSyntax *s, unsigned int n_sources)
{
    unsigned int mode = n_sources ? 1 : 0, c, b, pns0 = mmx_pns_first_band(L), is0 = mmx_is_first_band(L);
    double *book = NULL, pos = 0.0;          /* MMX_DEBUG_BITS: bits per EQ region of this frame */
    double (*bands)[MMX_MAX_BANDS] = band_book, bpos = 0.0;   /* closed loop: bits per band of this frame */
    double *eb = elem_book, ep = 0.0;        /* MMX_DEBUG_EST: bits per syntax element of this frame */

    band_book = NULL;
    elem_book = NULL;
    if (eb) ep = mmx_rc_enc_bits(rc);
    if (bitdump.armed)
    {
        bitdump.armed = 0;
        book = bitdump.bits[bitdump.cls];
        bitdump.frames[bitdump.cls]++;
        pos = mmx_rc_enc_bits(rc);
    }

    enc_block_type(rc, ctx, s->block_type);
    if (s->channels == 2)
        mmx_rc_enc_bit(rc, &ctx->stereo, s->stereo_ms != 0);
    encode_gains(rc, ctx, s, n_sources);
    if (s->block_type == MMX_BT_SHORT)
    {
        ELEM(MMX_ELEM_HDR);
        encode_short(rc, ctx, Ls, s, mode + 2);
        ELEM(MMX_ELEM_SHORT);
        remember(ctx, s, n_sources);
        if (book) book[MMX_EQ_BANDS + 1] += mmx_rc_enc_bits(rc) - pos;
        return;
    }
    encode_tns(rc, ctx, s);
    encode_nf(rc, ctx, L, s, mode);
    ELEM(MMX_ELEM_HDR);
    if (book) { book[MMX_EQ_BANDS] += mmx_rc_enc_bits(rc) - pos; pos = mmx_rc_enc_bits(rc); }

    for (c = 0; c < s->channels; c++)
    {
        int prev_sf = ctx->has_prev ? ctx->prev_sf[c] : SF_BIAS;
        int q1 = 0, q2 = 0, prev_noise = 0, prev_is = 0;
        signed char last_pos = 0;
        if (bands) bpos = mmx_rc_enc_bits(rc);
        for (b = 0; b < L->band_count; b++)
        {
            unsigned long k0 = L->band_start[b], k1 = L->band_start[b + 1], k;
            unsigned int fc = fclass_of(b);

            /* the previous band ends here (its zero and noise flags included) */
            if (bands && b) { double now = mmx_rc_enc_bits(rc); bands[c][b - 1] = now - bpos; bpos = now; }
            /* band replication: a zero flag and, when the band carries energy, its
               level and noise-mix index. No intensity flag, no noise flag, no
               coefficients - the decoder regenerates the band from the octave
               below the crossover. */
            if (b >= L->bwe_band)
            {
                mmx_rc_enc_bit(rc, &ctx->zero_band[mode][b], s->band_zero[c][b] != 0);
                if (!s->band_zero[c][b])
                {
                    unsigned char pl = ctx->prev_bwe[c][b] ? ctx->prev_bwe_level[c][b] : ctx->prev_bwe_last[c];
                    unsigned char pm = ctx->prev_bwe[c][b] ? ctx->prev_bwe_mix[c][b] : 0;
                    mmx_rc_enc_seg(rc, ctx->bwe_level[mode], (long long)s->sf[c][b] - (long long)pl);
                    mmx_rc_enc_seg(rc, ctx->bwe_mix_p[mode], (long long)s->bwe_mix[c][b] - (long long)pm);
                    ctx->prev_bwe_last[c] = s->sf[c][b];
                }
                ELEM(MMX_ELEM_BWE);
                q1 = q2 = 0;
                prev_noise = 0;
                prev_is = 0;
                if (book) { book[L->eq_band[b]] += mmx_rc_enc_bits(rc) - pos; pos = mmx_rc_enc_bits(rc); }
                continue;
            }
            /* intensity band: flag and position with channel 0 (before its band), nothing in channel 1 */
            if (s->channels == 2 && b >= is0)
            {
                if (c == 1)
                {
                    if (s->band_is[b]) { q1 = q2 = 0; prev_noise = 0; continue; }
                }
                else
                {
                    int flag = s->band_is[b] != 0;
                    mmx_rc_enc_bit(rc, &ctx->is_flag[mode][prev_is][ctx->prev_is[b]], flag);
                    if (flag)
                    {
                        signed char pred = ctx->prev_is[b] ? ctx->prev_is_pos[b] : last_pos;
                        mmx_rc_enc_seg(rc, ctx->is_pos, (long long)s->is_pos[b] - pred);
                        last_pos = s->is_pos[b];
                    }
                    prev_is = flag;
                    ELEM(MMX_ELEM_IS);
                }
            }
            mmx_rc_enc_bit(rc, &ctx->zero_band[mode][b], s->band_zero[c][b] != 0);
            ELEM(MMX_ELEM_ZERO);
            if (L->lowrate)
            {
                /* EPB: the band's energy index, predicted from the same band of the previous long frame,
                   else from the band below in this frame */
                int pred = ctx->has_prev ? ctx->prev_nrg[c][b] : (b ? nrg_ref(s, c, b - 1) : 0);
                mmx_rc_enc_seg(rc, ctx->nrg_ctx[mode], (long long)s->nrg[c][b] - pred);
                ELEM(MMX_ELEM_SF);
            }
            if (s->band_zero[c][b])
            {
                q1 = q2 = 0;
                prev_noise = 0;
                continue;
            }
            /* noise band: flag (from the first substitutable band on) and level */
            if (b >= pns0)
            {
                int noise = s->band_noise[c][b] != 0;
                mmx_rc_enc_bit(rc, &ctx->pns_flag[mode][prev_noise][ctx->prev_pns[c][b]], noise);
                prev_noise = noise;
                if (noise)
                {
                    mmx_rc_enc_seg(rc, ctx->pns_level[mode], (long long)s->sf[c][b] - ctx->prev_pns_level[c]);
                    ctx->prev_pns_level[c] = s->sf[c][b];
                    ELEM(MMX_ELEM_PNS);
                    q1 = q2 = 0;
                    continue;
                }
                ELEM(MMX_ELEM_PNS);
            }
            mmx_rc_enc_seg(rc, ctx->sf[mode], (long long)s->sf[c][b] - prev_sf);
            prev_sf = s->sf[c][b];
            ELEM(MMX_ELEM_SF);

            for (k = k0; k < k1; k++)
            {
                int q = s->q[c][k], a = q < 0 ? -q : q;
                unsigned int nb = nbclass_of(q1, q2), pc = ctx->prev_q[c][k];
                mmx_rc_enc_bit(rc, &ctx->coef_zero[mode][fc][nb][pc], a != 0);
                ELEM(MMX_ELEM_CZERO);
                if (a)
                {
                    mmx_rc_enc_bypass(rc, q < 0, 1);
                    ELEM(MMX_ELEM_SIGN);
                    mmx_rc_enc_bit(rc, &ctx->coef_gt1[mode][fc][nb][pc], a > 1);
                    if (a > 1)
                    {
                        mmx_rc_enc_bit(rc, &ctx->coef_gt2[mode][fc][nb][pc], a > 2);
                        ELEM(MMX_ELEM_GT);
                        if (a > 2)
                        {
                            mmx_rc_enc_ueg(rc, ctx->coef_eg[mode][fc], (unsigned long)(a - 3));
                            ELEM(MMX_ELEM_EG);
                        }
                    }
                    else
                        ELEM(MMX_ELEM_GT);
                }
                q2 = q1;
                q1 = q;
            }
            if (book) { book[L->eq_band[b]] += mmx_rc_enc_bits(rc) - pos; pos = mmx_rc_enc_bits(rc); }
        }
        if (bands && L->band_count) bands[c][L->band_count - 1] = mmx_rc_enc_bits(rc) - bpos;
    }
    remember(ctx, s, n_sources);
}

int mmx_frame_decode(MMXRangeDecoder *rc, MMXCodecContexts *ctx, const MMXBandLayout *L, const MMXBandLayout *Ls,
                     MMXFrameSyntax *s, unsigned int n_sources)
{
    unsigned int mode = n_sources ? 1 : 0, c, b, pns0 = mmx_pns_first_band(L), is0 = mmx_is_first_band(L);

    s->block_type = (unsigned char)dec_block_type(rc, ctx);
    s->stereo_ms = (s->channels == 2) ? (int)mmx_rc_dec_bit(rc, &ctx->stereo) : 0;
    decode_gains(rc, ctx, s, n_sources);
    memset(s->tns, 0, sizeof(s->tns));
    memset(s->band_noise, 0, sizeof(s->band_noise));
    memset(s->band_is, 0, sizeof(s->band_is));
    memset(s->band_bwe, 0, sizeof(s->band_bwe));
    memset(s->bwe_mix, 0, sizeof(s->bwe_mix));
    memset(s->nf_level, 0, sizeof(s->nf_level));
    if (s->block_type == MMX_BT_SHORT)
    {
        if (decode_short(rc, ctx, Ls, s, mode + 2) != 0)
            return -1;
        remember(ctx, s, n_sources);
        return 0;
    }
    if (decode_tns(rc, ctx, s) != 0 || decode_nf(rc, ctx, L, s, mode) != 0)
        return -1;

    for (c = 0; c < s->channels; c++)
    {
        int prev_sf = ctx->has_prev ? ctx->prev_sf[c] : SF_BIAS;
        int q1 = 0, q2 = 0, prev_noise = 0, prev_is = 0;
        signed char last_pos = 0;
        for (b = 0; b < L->band_count; b++)
        {
            unsigned long k0 = L->band_start[b], k1 = L->band_start[b + 1], k;
            unsigned int fc = fclass_of(b);
            long long sf;

            if (b >= L->bwe_band)
            {
                s->band_zero[c][b] = (unsigned char)mmx_rc_dec_bit(rc, &ctx->zero_band[mode][b]);
                s->band_bwe[c][b] = 1;
                for (k = k0; k < k1; k++) s->q[c][k] = 0;
                if (s->band_zero[c][b])
                    s->sf[c][b] = 0;
                else
                {
                    unsigned char pl = ctx->prev_bwe[c][b] ? ctx->prev_bwe_level[c][b] : ctx->prev_bwe_last[c];
                    unsigned char pm = ctx->prev_bwe[c][b] ? ctx->prev_bwe_mix[c][b] : 0;
                    long long lvl = mmx_rc_dec_seg(rc, ctx->bwe_level[mode]) + (long long)pl;
                    long long mixv = mmx_rc_dec_seg(rc, ctx->bwe_mix_p[mode]) + (long long)pm;
                    if (lvl < 0 || lvl >= MMX_SF_COUNT || mixv < 0 || mixv > MMX_BWE_MIX_MAX || rc->failed)
                        return -1;
                    s->sf[c][b] = (unsigned char)lvl;
                    s->bwe_mix[c][b] = (unsigned char)mixv;
                    ctx->prev_bwe_last[c] = (unsigned char)lvl;
                }
                q1 = q2 = 0;
                prev_noise = 0;
                prev_is = 0;
                continue;
            }
            if (s->channels == 2 && b >= is0)
            {
                if (c == 1)
                {
                    if (s->band_is[b])
                    {
                        s->band_zero[c][b] = 1;
                        s->sf[c][b] = 0;
                        for (k = k0; k < k1; k++) s->q[c][k] = 0;
                        q1 = q2 = 0;
                        prev_noise = 0;
                        continue;
                    }
                }
                else
                {
                    int flag = (int)mmx_rc_dec_bit(rc, &ctx->is_flag[mode][prev_is][ctx->prev_is[b]]);
                    if (flag)
                    {
                        long long pos = mmx_rc_dec_seg(rc, ctx->is_pos) + (ctx->prev_is[b] ? ctx->prev_is_pos[b] : last_pos);
                        if (pos < -MMX_IS_POS_MAX || pos > MMX_IS_POS_MAX || rc->failed)
                            return -1;
                        s->band_is[b] = 1;
                        s->is_pos[b] = (signed char)pos;
                        last_pos = (signed char)pos;
                    }
                    prev_is = flag;
                }
            }
            s->band_zero[c][b] = (unsigned char)mmx_rc_dec_bit(rc, &ctx->zero_band[mode][b]);
            if (L->lowrate)
            {
                int pred = ctx->has_prev ? ctx->prev_nrg[c][b] : (b ? nrg_ref(s, c, b - 1) : 0);
                long long v = pred + mmx_rc_dec_seg(rc, ctx->nrg_ctx[mode]);
                if (v < 0) v = 0;
                if (v > MMX_SF_COUNT - 1) v = MMX_SF_COUNT - 1;
                s->nrg[c][b] = (unsigned char)v;
            }
            if (s->band_zero[c][b])
            {
                s->sf[c][b] = 0;
                for (k = k0; k < k1; k++) s->q[c][k] = 0;
                q1 = q2 = 0;
                prev_noise = 0;
                continue;
            }
            if (b >= pns0)
            {
                int noise = (int)mmx_rc_dec_bit(rc, &ctx->pns_flag[mode][prev_noise][ctx->prev_pns[c][b]]);
                prev_noise = noise;
                if (noise)
                {
                    long long lvl = mmx_rc_dec_seg(rc, ctx->pns_level[mode]) + ctx->prev_pns_level[c];
                    if (lvl < 0 || lvl >= MMX_SF_COUNT || rc->failed)
                        return -1;
                    s->band_noise[c][b] = 1;
                    s->sf[c][b] = (unsigned char)lvl;
                    ctx->prev_pns_level[c] = (unsigned char)lvl;
                    for (k = k0; k < k1; k++) s->q[c][k] = 0;
                    q1 = q2 = 0;
                    continue;
                }
            }
            sf = mmx_rc_dec_seg(rc, ctx->sf[mode]) + prev_sf;
            if (sf < 0 || sf >= MMX_SF_COUNT || rc->failed)
                return -1;
            s->sf[c][b] = (unsigned char)sf;
            prev_sf = (int)sf;

            for (k = k0; k < k1; k++)
            {
                unsigned int nb = nbclass_of(q1, q2), pc = ctx->prev_q[c][k];
                int a = 0, q;
                if (mmx_rc_dec_bit(rc, &ctx->coef_zero[mode][fc][nb][pc]))
                {
                    unsigned int neg = (unsigned int)mmx_rc_dec_bypass(rc, 1);
                    a = 1;
                    if (mmx_rc_dec_bit(rc, &ctx->coef_gt1[mode][fc][nb][pc]))
                    {
                        a = 2;
                        if (mmx_rc_dec_bit(rc, &ctx->coef_gt2[mode][fc][nb][pc]))
                            a = 3 + (int)mmx_rc_dec_ueg(rc, ctx->coef_eg[mode][fc]);
                    }
                    if (neg) a = -a;
                }
                q = a;
                s->q[c][k] = q;
                q2 = q1;
                q1 = q;
            }
            if (rc->failed)
                return -1;
        }
    }
    remember(ctx, s, n_sources);
    return 0;
}

double mmx_frame_estimate_bits(const MMXBandLayout *L, const float *x, const float *thr, unsigned int cutoff_band)
{
    unsigned int b;
    double bits = 0.0;
    for (b = 0; b < L->band_count && b < cutoff_band; b++)
    {
        unsigned long k0 = L->band_start[b], k1 = L->band_start[b + 1], k;
        double t = thr[b], energy = 0.0, inv;
        if (t >= 1e29)
            break;
        for (k = k0; k < k1; k++)
            energy += (double)x[k] * x[k];
        bits += 0.3; /* band flag */
        if (energy <= t * (double)(k1 - k0))
            continue;
        bits += 3.0; /* scalefactor */
        inv = 1.0 / mmx_sf_step(mmx_sf_for_threshold(t));
        for (k = k0; k < k1; k++)
        {
            double a = fabs((double)x[k]) * inv;
            if (a < 0.5)
                bits += 0.25;
            else
                bits += 2.2 + 1.35 * log(a + 0.5) / log(2.0);
        }
    }
    return bits;
}

/* Binary entropy of n1 ones in n0 + n1 symbols, in bits. */
static double h2_bits(unsigned long n0, unsigned long n1)
{
    double n = (double)(n0 + n1), p;
    if (n0 == 0 || n1 == 0)
        return 0.0;
    p = (double)n1 / n;
    return -n * (p * log(p) + (1.0 - p) * log(1.0 - p)) / log(2.0);
}

double mmx_frame_estimate_bits_coded(const MMXBandLayout *L, const float *x, const float *thr, unsigned int cutoff_band,
                                     const float *crest)
{
    unsigned int b, fc, nb;
    int q[MMX_HOP], q1 = 0, q2 = 0;
    unsigned long zero[3][6][2], gt1[3][6][2], gt2[3][6][2], signs = 0;
    double bits = 0.0, eg = 0.0, e_zero = 0.0, e_sf = 0.0, e_cz = 0.0, e_gt = 0.0;
    double *eb = est_elem_book;

    est_elem_book = NULL;
    alloc_knobs();
    memset(zero, 0, sizeof(zero));
    memset(gt1, 0, sizeof(gt1));
    memset(gt2, 0, sizeof(gt2));
    if (L->lowrate) { bits += MMX_EPB_EST_BITS * L->band_count; e_sf += MMX_EPB_EST_BITS * L->band_count; }   /* EPB: one energy symbol per band */
    for (b = 0; b < L->band_count && b < cutoff_band; b++)
    {
        unsigned long k0 = L->band_start[b], k1 = L->band_start[b + 1], k, n = k1 - k0;
        double t = thr[b], energy = 0.0, allowed;
        unsigned int sf;
        int any;
        if (t >= 1e29)
            break;
        for (k = k0; k < k1; k++)
            energy += (double)x[k] * x[k];
        allowed = t * (double)n;
        bits += 0.3; /* band flag */
        e_zero += 0.3;
        if (alloc_zero ? energy * (crest ? (double)crest[b] : 1.0) <= allowed : energy <= 0.0)
        {
            q1 = q2 = 0;
            continue;
        }
        /* the quantizer of mmx_frame_quantize: dead zone, step tightened until the noise fits */
        sf = mmx_sf_for_threshold(t);
        for (;;)
        {
            double noise = quantize_band(x, k0, k1, mmx_sf_step(sf), 1.0, q, &any, NULL);
            if (noise <= allowed || sf == 0)
                break;
            sf--;
        }
        if (any && alloc_k > 0) any = cap_band(q, k0, k1, alloc_k);
        if (!any)
        {
            q1 = q2 = 0;
            continue;
        }
        bits += 3.0; /* scalefactor */
        e_sf += 3.0;
        fc = fclass_of(b);
        for (k = k0; k < k1; k++)
        {
            int a = q[k] < 0 ? -q[k] : q[k];
            nb = nbclass_of(q1, q2);
            zero[fc][nb][a != 0]++;
            if (a)
            {
                signs++;
                gt1[fc][nb][a > 1]++;
                if (a > 1)
                {
                    gt2[fc][nb][a > 2]++;
                    if (a > 2)
                    {
                        unsigned long v = (unsigned long)(a - 3) + 1;
                        unsigned int len = 0;
                        while ((v >> len) > 1) len++;
                        eg += 2.0 * len + 1.0;
                    }
                }
            }
            q2 = q1;
            q1 = q[k];
        }
    }
    for (fc = 0; fc < 3; fc++)
        for (nb = 0; nb < 6; nb++)
        {
            e_cz += h2_bits(zero[fc][nb][0], zero[fc][nb][1]);
            e_gt += h2_bits(gt1[fc][nb][0], gt1[fc][nb][1]) + h2_bits(gt2[fc][nb][0], gt2[fc][nb][1]);
        }
    bits += e_cz + e_gt;
    if (eb)
    {
        eb[MMX_ELEM_ZERO] += e_zero; eb[MMX_ELEM_SF] += e_sf; eb[MMX_ELEM_CZERO] += e_cz;
        eb[MMX_ELEM_GT] += e_gt; eb[MMX_ELEM_SIGN] += (double)signs; eb[MMX_ELEM_EG] += eg;
    }
    return bits + (double)signs + eg;
}
