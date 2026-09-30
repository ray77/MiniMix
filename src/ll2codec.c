#include <stdlib.h>
#include <string.h>
#include "lab.h"
#include "ll2codec.h"

int mmx_ll2s_init(MMXLl2Stream *s, unsigned int channels, unsigned int decay, unsigned int bank, const unsigned char *xols,
                  unsigned int profile, unsigned int own_taps)
{
    static const unsigned int orders0[4] = { 1024, 256, 32, 16 };
    static const unsigned int orders1[6] = { 1024, 1280, 256, 256, 32, 16 };
    static const int norm1[6] = { 0, 1, 0, 1, 1, 0 };
    unsigned int c, k, own, dmax;
    int cd;
    memset(s, 0, sizeof(*s));
    s->prof = mmx_ll2_profile(profile);
    if (!s->prof) return -1;
    /* the CD profile reads its header exactly as the frozen format did: no own-tap byte (94), memories 6..14 */
    cd = s->prof->id == MMX_LL2_PROFILE_CD;
    dmax = cd ? 14 : 16;
    own = own_taps && !cd ? own_taps : s->prof->own_taps;
    if (own < 4 || own > MMX_LL2_OWN_MAX || s->prof->fut > MMX_LL2_FUT_MAX) return -1;
    s->nfut = s->prof->fut;
    s->channels = channels > 2 ? 2 : channels;
    for (c = 0; c < s->channels; c++)
    {
        mmx_ll2_init(&s->ch[c], own, s->channels == 2 && c == 1 ? MMX_LL2_CROSS + s->nfut : 0, 0,
                     bank ? orders1 : orders0, bank ? 6 : 4);
        if (bank)
            for (k = 0; k < 6; k++)
                if (norm1[k] && mmx_ll2_set_norm_stage(&s->ch[c], k, (int)orders1[k], 3) != 0) return -1;
    }
    for (c = 0; c < s->channels; c++) if (decay >= 6 && decay <= dmax) s->ch[c].ols.decay = (int)decay;
    for (c = 0; c < s->channels; c++)
        for (k = 0; xols && k < 2; k++)
            if (xols[k] && (xols[k] > dmax || mmx_ll2_add_ols(&s->ch[c], xols[k]) != 0)) return -1;
    s->coder = mmx_ll2c_new();
    s->last_src[0] = s->last_src[1] = -1;
    if (s->coder) mmx_ll2c_set_raw_depth(s->coder, s->prof->raw_depth);
    return s->coder ? 0 : -1;
}

void mmx_ll2s_free(MMXLl2Stream *s)
{
    unsigned int c;
    for (c = 0; c < 2; c++) mmx_ll2_free(&s->ch[c]);
    mmx_ll2c_free(s->coder);
    for (c = 0; c < 2; c++) { free(s->res[c]); free(s->praw[c]); }
    memset(s, 0, sizeof(*s));
}

static MMXLl2Profile ll2_profiles[MMX_LL2_PROFILES] = {
    { MMX_LL2_PROFILE_CD, "CD", 20, 8, 16, 0, 0, 0, 0, 0, 0, 0, MMX_LL2_FUT },         /* frozen: 16-bit files stay byte-identical */
    { MMX_LL2_PROFILE_HIRES, "hi-res", 24, 16, 16, 1, 1, 9, 6, 1, 1, 0, 8 }, /* full 24-bit prediction with double OLS weights,
                                                                LMS stages scaled to ~2^9, noise bits raw */
};
static int profiles_env = 0;
static void profiles_lab(void)                      /* lab: MMX_LL2_HR_OWN=n tries other hi-res OLS lengths */
{
    const char *e = mmx_lab_getenv("MMX_LL2_HR_OWN");
    if (profiles_env) return;
    profiles_env = 1;
    if (e && atoi(e) >= 4 && atoi(e) <= 32) ll2_profiles[MMX_LL2_PROFILE_HIRES].own_taps = (unsigned int)atoi(e);
    e = mmx_lab_getenv("MMX_LL2_HR_CSHIFT");                /* lab: the hi-res covariance scale */
    if (e && atoi(e) >= 4 && atoi(e) <= 24) ll2_profiles[MMX_LL2_PROFILE_HIRES].cov_shift = atoi(e);
    e = mmx_lab_getenv("MMX_LL2_HR_KEEP");                  /* lab: 0 = clamp pivots like the CD profile */
    if (e) ll2_profiles[MMX_LL2_PROFILE_HIRES].keep_on_fail = atoi(e) != 0;
    e = mmx_lab_getenv("MMX_LL2_HR_DPRED");                 /* lab: 0 = Q16 weights like the CD profile */
    if (e) ll2_profiles[MMX_LL2_PROFILE_HIRES].dpred = atoi(e) != 0;
    e = mmx_lab_getenv("MMX_LL2_HR_LTARGET");               /* lab: the hi-res stage scale target (0 = fixed) */
    if (e && atoi(e) >= 0 && atoi(e) <= 14) ll2_profiles[MMX_LL2_PROFILE_HIRES].stage_target = atoi(e);
    e = mmx_lab_getenv("MMX_LL2_HR_RAW");                   /* lab: raw refinement bits from this depth (0 = off) */
    if (e && atoi(e) >= 0 && atoi(e) <= 31) ll2_profiles[MMX_LL2_PROFILE_HIRES].raw_depth = atoi(e);
    e = mmx_lab_getenv("MMX_LL2_HR_FNORM");                 /* lab: 0 = integer normalised LMS */
    if (e) ll2_profiles[MMX_LL2_PROFILE_HIRES].fnorm = atoi(e) != 0;
    e = mmx_lab_getenv("MMX_LL2_HR_DCOV");                  /* lab: 0 = integer covariance */
    if (e) ll2_profiles[MMX_LL2_PROFILE_HIRES].dcov = atoi(e) != 0;
    e = mmx_lab_getenv("MMX_LL2_HR_SOLVE");                 /* lab: samples between two OLS solves */
    if (e && atoi(e) >= 1 && atoi(e) <= 1024) ll2_profiles[MMX_LL2_PROFILE_HIRES].solve_every = atoi(e);
    e = mmx_lab_getenv("MMX_LL2_HR_FUT");                   /* lab: channel 1's future taps */
    if (e && atoi(e) >= 0 && atoi(e) <= MMX_LL2_FUT_MAX) ll2_profiles[MMX_LL2_PROFILE_HIRES].fut = (unsigned int)atoi(e);
    e = mmx_lab_getenv("MMX_LL2_HR_PBITS");                 /* lab: the hi-res prediction bits */
    if (e && atoi(e) >= 16 && atoi(e) <= 24) ll2_profiles[MMX_LL2_PROFILE_HIRES].pred_bits = (unsigned int)atoi(e);
}

unsigned int mmx_ll2_profile_for(unsigned int bits, unsigned long sample_rate, unsigned int channels)
{
    const char *e = mmx_lab_getenv("MMX_LL2_PROFILE");     /* lab: force a profile */
    (void)channels;
    if (e && atoi(e) >= 0 && atoi(e) < MMX_LL2_PROFILES) return (unsigned int)atoi(e);
    return (bits && bits > 16) || sample_rate > 48000 ? MMX_LL2_PROFILE_HIRES : MMX_LL2_PROFILE_CD;
}

const MMXLl2Profile *mmx_ll2_profile(unsigned int id)
{
    profiles_lab();
    return id < MMX_LL2_PROFILES ? &ll2_profiles[id] : NULL;
}

static const MMXLl2Profile *stream_profile(const MMXLl2Stream *s) { return s->prof ? s->prof : &ll2_profiles[0]; }

void mmx_ll2s_set_mixer(MMXLl2Stream *s, unsigned int mix, unsigned int bits, unsigned int shift)
{
    unsigned int c, b = (bits ? bits : 16) > shift ? (bits ? bits : 16) - shift : 16, up, pb = stream_profile(s)->pred_bits;
    if (b > pb) b = pb;                                    /* the predictor runs at most at the profile's bits (set_rails) */
    up = b > 16 ? b - 16 : 0;
    if (mix != 1) return;
    for (c = 0; c < s->channels; c++) mmx_ll2_set_experts4(&s->ch[c], 32LL << up, 512LL << up);
}

void mmx_ll2s_set_rails(MMXLl2Stream *s, unsigned int bits, unsigned int shift)
{
    long long lim = ((long long)1 << (bits ? bits - 1 : 15)) - 1;
    unsigned int c, b = (bits ? bits : 16) > shift ? (bits ? bits : 16) - shift : 16, pb = stream_profile(s)->pred_bits;
    s->pshift = b > pb ? (int)(b - pb) : 0;
    s->wrap32 = b >= 32;
    for (c = 0; c < s->channels; c++)
    {
        mmx_ll2_set_cov_shift(&s->ch[c], stream_profile(s)->cov_shift);
        mmx_ll2_set_keep_on_fail(&s->ch[c], stream_profile(s)->keep_on_fail);
        mmx_ll2_set_dpred(&s->ch[c], stream_profile(s)->dpred);
        mmx_ll2_set_stage_target(&s->ch[c], stream_profile(s)->stage_target);
        mmx_ll2_set_float_norm(&s->ch[c], stream_profile(s)->fnorm);
        mmx_ll2_set_dcov(&s->ch[c], stream_profile(s)->dcov);
        mmx_ll2_set_solve_every(&s->ch[c], stream_profile(s)->solve_every);
    }
    s->rail_lo = (-lim - 1) >> shift;
    s->rail_hi = lim >> shift;
    for (c = 0; c < 2; c++) { s->lo[c] = s->rail_lo; s->hi[c] = s->rail_hi; }
}

unsigned long mmx_ll2_pld_frames(unsigned int seconds, unsigned long sample_rate)
{
    unsigned long long f = (unsigned long long)seconds * sample_rate / 1024u;
    return seconds ? (unsigned long)(f ? f : 1) : 0;
}

void mmx_ll2s_set_rate(MMXLl2Stream *s, unsigned long sample_rate)
{
    unsigned int c, n = 16;
    unsigned long r = sample_rate;
    if (!stream_profile(s)->dcov) return;
    if (stream_profile(s)->solve_every) n = (unsigned int)stream_profile(s)->solve_every;   /* lab override */
    else
        while (r >= 176400 && n < 128) { n *= 2; r /= 2; }
    for (c = 0; c < s->channels; c++) mmx_ll2_set_solve_every(&s->ch[c], (int)n);
}

int mmx_ll2s_block_begin(MMXLl2Stream *s, unsigned long n)
{
    unsigned int c;
    if (n > s->cap)
    {
        for (c = 0; c < s->channels; c++)
        {
            long long *r = (long long *)realloc(s->res[c], sizeof(long long) * (n ? n : 1));
            long long *p;
            if (!r) return -1;
            s->res[c] = r;
            p = (long long *)realloc(s->praw[c], sizeof(long long) * (n ? n : 1));
            if (!p) return -1;
            s->praw[c] = p;
        }
        s->cap = n;
    }
    s->used = 0;
    return 0;
}

/* the decoded signal as the decoder has it: positions at or past the frontier read 0, so do negative ones */
/* 32 coded bits: a residual modulo 2^32 in (-2^31, 2^31] (the coder maps that range into 32 bits), and a sample
   back into [-2^31, 2^31) - exact, because every sample is a 32-bit integer */
static long long wrap32(long long d)
{
    d &= 0xFFFFFFFFLL;
    return d > 0x80000000LL ? d - 0x100000000LL : d;
}
static long long to_s32(long long x)
{
    x &= 0xFFFFFFFFLL;
    return x >= 0x80000000LL ? x - 0x100000000LL : x;
}

static long long get(const long long *d, long long pos, long long frontier)
{
    return pos >= 0 && pos < frontier ? d[pos] : 0;
}

/* the regressor taps: like get, but nothing before the segment start (0 = the whole file; lab restart points) */
static long long get_seg(const long long *d, long long pos, long long seg, long long frontier)
{
    return pos >= seg ? get(d, pos, frontier) : 0;
}

void mmx_ll2s_frame_ch(MMXLl2Stream *s, unsigned int c, long long *const *dec, long long start, unsigned long count,
                       long long src, long long front, int decode, long long *res, long long lo, long long hi)
{
    const int ps = s->pshift;
    unsigned long n;
    int new_repeat = src >= 0 && src != (s->last_src[c] >= 0 ? s->last_src[c] + 1024 : -2);
    MMXLl2Chan *ch = &s->ch[c];
    long long *own = dec[c];
    const long long *oth = s->channels == 2 ? dec[1 - c] : NULL;
    long long oth_front = c == 0 ? start : front;     /* channel 1 of a block comes after channel 0 of the whole block */
    for (n = 0; n < count; n++)
    {
        long long t = start + (long long)n, cross[MMX_LL2_CROSS + MMX_LL2_FUT_MAX], pr, x;
        int i;
        if (s->channels == 2 && c == 1)
        {
            for (i = 0; i < MMX_LL2_CROSS; i++) cross[i] = get_seg(oth, t - i, s->seg_start, oth_front) >> ps;
            for (i = 1; i <= (int)s->nfut; i++) cross[MMX_LL2_CROSS + i - 1] = get_seg(oth, t + i, s->seg_start, oth_front) >> ps;
        }
        pr = mmx_ll2_predict(ch, cross, NULL);
        if (src >= 0)
        {
            /* the source windows through the same getter: own channel before t, other channel before its frontier */
            long long sw[MMX_LL2_SRC_N + MMX_LL2_OWN_MAX], so[MMX_LL2_SRC_N + MMX_LL2_CROSS + 2 * MMX_LL2_FUT_MAX + 2];
            const int no = (int)ch->n_own;
            long long base = src + (long long)n;
            int k;
            for (k = 0; k < MMX_LL2_SRC_N + no; k++)
                sw[k] = get(own, base - MMX_LL2_SRC_K - no + k, t) >> ps;
            if (oth && c == 1)                            /* channel 0 has no cross taps: it never reads channel 1 */
                for (k = 0; k < MMX_LL2_SRC_N + MMX_LL2_CROSS + 2 * (int)s->nfut + 2; k++)
                    so[k] = get(oth, base - MMX_LL2_SRC_K - MMX_LL2_CROSS - 1 + k, oth_front) >> ps;
            pr = oth && c == 1 ? mmx_ll2_predict_src2(ch, sw + no + MMX_LL2_SRC_K, so + MMX_LL2_CROSS + 1 + MMX_LL2_SRC_K,
                                            c == 1, c == 1 ? s->nfut : 0, new_repeat && n == 0)
                     : mmx_ll2_predict_src(ch, sw + no + MMX_LL2_SRC_K, new_repeat && n == 0);
        }
        else
            pr = mmx_ll2_predict_src(ch, NULL, 0);
        if (ps) pr = pr * (1LL << ps) + (1LL << (ps - 1));   /* back to full scale, the centre of the cell */
        if (!decode && s->praw[c]) s->praw[c][(res - s->res[c]) + (long)n] = pr;
        if (pr < lo) pr = lo;
        else if (pr > hi) pr = hi;
        if (decode)
        {
            x = res[n] + pr;
            if (s->wrap32) x = to_s32(x);
            if (t >= s->store_from) own[t] = x;
        }
        else
        {
            x = own[t];
            res[n] = s->wrap32 ? wrap32(x - pr) : x - pr;
        }
        mmx_ll2_update(ch, x >> ps);
    }
    s->last_src[c] = src;
}

void mmx_ll2s_skip_frame(MMXLl2Stream *s, long long src)
{
    s->last_src[0] = s->last_src[1] = src;
}

void mmx_ll2s_frame(MMXLl2Stream *s, long long *const *dec, long long start, unsigned long count, long long src,
                    long long front, int decode)
{
    unsigned int c;
    for (c = 0; c < s->channels; c++)
        mmx_ll2s_frame_ch(s, c, dec, start, count, src, front, decode, s->res[c] + s->used, s->rail_lo, s->rail_hi);
    s->used += count;
}

/* a bound: zigzag, its bit length in 6 bits, then the bits (two bypass calls, so 32-bit unsigned long suffices) */
static void put_bound(MMXRangeEncoder *rc, long long v)
{
    unsigned long long u = v < 0 ? 2ull * (unsigned long long)(-(v + 1)) + 1ull : 2ull * (unsigned long long)v;
    unsigned int nb = 0;
    while (nb < 63 && (u >> nb)) nb++;
    mmx_rc_enc_bypass(rc, nb, 6);
    if (nb > 32) mmx_rc_enc_bypass(rc, (unsigned long)(u >> 32), nb - 32);
    mmx_rc_enc_bypass(rc, (unsigned long)(u & 0xFFFFFFFFull), nb > 32 ? 32 : nb);
}

static long long get_bound(MMXRangeDecoder *rc)
{
    unsigned int nb = (unsigned int)mmx_rc_dec_bypass(rc, 6);
    unsigned long long u = 0;
    if (nb > 63) nb = 63;
    if (nb > 32) u = (unsigned long long)mmx_rc_dec_bypass(rc, nb - 32) << 32;
    u |= (unsigned long long)mmx_rc_dec_bypass(rc, nb > 32 ? 32 : nb);
    return (u & 1ull) ? -(long long)(u >> 1) - 1 : (long long)(u >> 1);
}

static int bitlen(unsigned long long u) { int b = 0; while (u) { b++; u >>= 1; } return b; }
static long long clampll(long long v, long long lo, long long hi) { return v < lo ? lo : v > hi ? hi : v; }

void mmx_ll2s_finish_block(MMXLl2Stream *s)
{
    unsigned int c;
    unsigned long n;
    for (c = 0; c < s->channels; c++)
    {
        long long lo = 0, hi = 0, save = 0, cost;
        s->expl[c] = 0;
        s->lo[c] = s->rail_lo; s->hi[c] = s->rail_hi;
        if (!s->used) continue;
        for (n = 0; n < s->used; n++)           /* the samples: residual + prediction clamped to the rails */
        {
            long long x = s->res[c][n] + clampll(s->praw[c][n], s->rail_lo, s->rail_hi);
            if (s->wrap32) x = to_s32(x);
            if (n == 0 || x < lo) lo = x;
            if (n == 0 || x > hi) hi = x;
        }
        for (n = 0; n < s->used; n++)
        {
            long long p = s->praw[c][n];
            if (p < lo || p > hi)
            {
                long long x = s->res[c][n] + clampll(p, s->rail_lo, s->rail_hi), a, b = s->res[c][n];
                if (s->wrap32) x = to_s32(x);
                a = s->wrap32 ? wrap32(x - clampll(p, lo, hi)) : x - clampll(p, lo, hi);
                save += bitlen((unsigned long long)(b < 0 ? -b : b)) - bitlen((unsigned long long)(a < 0 ? -a : a));
            }
        }
        cost = 12 + 2 * bitlen((unsigned long long)(hi < 0 ? -hi : hi)) + 2 * bitlen((unsigned long long)(lo < 0 ? -lo : lo));
        if (save > cost)
        {
            s->expl[c] = 1;
            for (n = 0; n < s->used; n++)
            {
                long long x = s->res[c][n] + clampll(s->praw[c][n], s->rail_lo, s->rail_hi);
                if (s->wrap32) x = to_s32(x);
                s->res[c][n] = s->wrap32 ? wrap32(x - clampll(s->praw[c][n], lo, hi)) : x - clampll(s->praw[c][n], lo, hi);
            }
            s->lo[c] = lo; s->hi[c] = hi;
        }
    }
}

void mmx_ll2s_encode_block(MMXLl2Stream *s, MMXRangeEncoder *rc)
{
    unsigned int c;
    for (c = 0; c < s->channels; c++)
    {
        mmx_rc_enc_bypass(rc, (unsigned long)s->expl[c], 1);
        if (s->expl[c]) { put_bound(rc, s->lo[c]); put_bound(rc, s->hi[c]); }
    }
    for (c = 0; c < s->channels; c++)
        mmx_ll2c_encode_cont64(s->coder, rc, s->res[c], (int)s->used);
}

void mmx_ll2s_decode_bounds(const MMXLl2Stream *s, MMXRangeDecoder *rc, long long *lo, long long *hi)
{
    unsigned int c;
    for (c = 0; c < s->channels; c++)
    {
        if (mmx_rc_dec_bypass(rc, 1)) { lo[c] = get_bound(rc); hi[c] = get_bound(rc); }
        else { lo[c] = s->rail_lo; hi[c] = s->rail_hi; }
    }
}

int mmx_ll2s_decode_block(MMXLl2Stream *s, MMXRangeDecoder *rc, unsigned long n)
{
    unsigned int c;
    if (mmx_ll2s_block_begin(s, n) != 0) return -1;
    mmx_ll2s_decode_bounds(s, rc, s->lo, s->hi);
    for (c = 0; c < s->channels; c++)
        mmx_ll2c_decode_cont64(s->coder, rc, s->res[c], (int)n);
    s->used = 0;                       /* the frames consume the buffers from the start */
    return 0;
}
