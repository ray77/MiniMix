/* See ll2coder.h. Port of the design of sac's BitplaneCoder (MIT licence, (c) 2024 Sebastian Lehmann). */
#include <stdlib.h>
#include <string.h>
#include "ll2coder.h"

#define PBITS 15
#define PSCALE (1 << PBITS)
#define PSCALEm (PSCALE - 1)
#define WBITS 16
#define WSCALE (1 << WBITS)
#define WRANGE (1 << (WBITS + 3))
#define RBITS 16
#define LBITS 8
#define DMAX (8 * (1 << LBITS) - 1)
#define DMIN (-DMAX)
#define USHIFT (LBITS + PBITS - WBITS + RBITS)

#define RATE_P 150
#define RATE_SIG 300
#define RATE_REF 150
#define RATE_SSE 150
#define MIXRATE_REF 328                 /* 0.005 * 2^16 */
#define MIXRATE_SIG 328
#define MIXRATE_SSE 98                  /* 0.0015 * 2^16 */
#define SSE_N 16                        /* bins of 256 logit units over [-2048, 2048): shifts instead of divisions */
#define SSE_SHIFT 8
#define MAXCNT 301

/* ------------------------------------------------------------------ logit domain (integer only) */
/* squash anchors p = 32768 / (1 + e^(-x/256)) at x = -2048, -1920, ..., 2048, rounded and clamped once */
static const int squash_anchor[33] = {
    11, 18, 30, 49, 81, 133, 219, 360, 589, 961, 1554, 2486, 3906, 5978, 8813, 12371, 16384,
    20397, 23955, 26790, 28862, 30282, 31214, 31807, 32179, 32408, 32549, 32635, 32687, 32719, 32738, 32750, 32757
};
static int inv_tab[DMAX - DMIN + 1];    /* squash: logit -> probability */
static int fwd_tab[PSCALE];             /* stretch: probability -> logit */
static int div_tab[MAXCNT + 1];
static int tables_ready = 0;

static void tables_init(void)
{
    int x, p, i;
    if (tables_ready) return;
    for (x = DMIN; x <= DMAX; x++)
    {
        int k = (x + 2048) >> 7, f = (x + 2048) & 127, v;
        v = squash_anchor[k] + (((squash_anchor[k + 1] - squash_anchor[k]) * f + 64) >> 7);
        inv_tab[x - DMIN] = v < 1 ? 1 : v > PSCALEm ? PSCALEm : v;
    }
    /* stretch(p) = the smallest x with squash(x) >= p (monotone inverse); below/above the range: the ends */
    x = DMIN;
    for (p = 0; p < PSCALE; p++)
    {
        while (x < DMAX && inv_tab[x - DMIN] < p) x++;
        fwd_tab[p] = x;
    }
    for (i = 0; i <= MAXCNT; i++) div_tab[i] = PSCALE / (i + 3);
    tables_ready = 1;
}

static int squash(int x)
{
    if (x < DMIN) return 1;
    if (x > DMAX) return PSCALEm;
    return inv_tab[x - DMIN];
}
static int stretch(int p) { return fwd_tab[p < 0 ? 0 : p > PSCALEm ? PSCALEm : p]; }

static int clampi(int v, int lo, int hi) { return v < lo ? lo : v > hi ? hi : v; }
static long long idiv_signed(long long v, int s)
{
    long long h = 1LL << (s - 1);
    return v < 0 ? -(((-v) + h) >> s) : ((v + h) >> s);
}

/* ------------------------------------------------------------------ models */
typedef struct { unsigned short p1, cnt; } Ctr;               /* sac's LinearCounterLimit */

static void ctr_init(Ctr *c) { c->p1 = PSCALE >> 1; c->cnt = 0; }
static void ctr_update(Ctr *c, int bit, int limit)
{
    int dp, p;
    if (c->cnt < limit) c->cnt++;
    dp = bit ? ((PSCALE - c->p1) * div_tab[c->cnt]) >> PBITS : -((c->p1 * div_tab[c->cnt]) >> PBITS);
    p = c->p1 + dp;
    c->p1 = (unsigned short)clampi(p, 1, PSCALEm);
}

typedef struct { int n, pd; int w[5]; int x[5]; } Mix;        /* sac's LogMixer */

static void mix_init(Mix *m, int n, int uniform)
{
    int i;
    m->n = n; m->pd = 0;
    for (i = 0; i < n; i++) { m->w[i] = uniform ? WSCALE / n : 0; m->x[i] = 0; }
}
static int mix_predict(Mix *m, const int *p)
{
    long long sum = 0;
    int i;
    for (i = 0; i < m->n; i++)
    {
        m->x[i] = stretch(p[i]);
        sum += (long long)m->w[i] * m->x[i];
    }
    m->pd = clampi(squash((int)idiv_signed(sum, WBITS)), 1, PSCALEm);
    return m->pd;
}
static void mix_update(Mix *m, int bit, int rate)
{
    int err = (bit << PBITS) - m->pd, i;
    for (i = 0; i < m->n; i++)
    {
        int dw = (int)idiv_signed((long long)m->x[i] * err * rate, USHIFT);
        m->w[i] = clampi(m->w[i] + dw, -WRANGE, WRANGE - 1);
    }
}

typedef struct { Ctr map[2][SSE_N + 1]; int lb, pq, tscale, xscale; } Sse;   /* sac's SSENL<15> */

static void sse_init(Sse *s)
{
    int i;
    s->tscale = SSE_N << (SSE_SHIFT - 1);           /* 2048 */
    s->xscale = 1 << SSE_SHIFT;
    for (i = 0; i <= SSE_N; i++)
    {
        int x = squash(i * s->xscale - s->tscale);
        s->map[0][i].p1 = s->map[1][i].p1 = (unsigned short)x;
        s->map[0][i].cnt = s->map[1][i].cnt = 0;
    }
    s->lb = 0; s->pq = 0;
}
static int sse_predict(Sse *s, int p1)
{
    int pq = clampi(stretch(p1) + s->tscale, 0, 2 * s->tscale - 1), mod, pl, ph;
    s->pq = pq >> SSE_SHIFT;
    mod = pq & (s->xscale - 1);
    pl = s->map[s->lb][s->pq].p1;
    ph = s->map[s->lb][s->pq + 1].p1;
    return clampi((pl * (s->xscale - mod) + ph * mod) >> SSE_SHIFT, 1, PSCALEm);
}
static void sse_update(Sse *s, int bit, int rate)
{
    ctr_update(&s->map[s->lb][s->pq], bit, rate);
    ctr_update(&s->map[s->lb][s->pq + 1], bit, rate);
    s->lb = bit;
}

/* ------------------------------------------------------------------ the coder */
#define N_CSIG0 (1 << 16)
#define N_CSIG1 128
#define N_CREF0 64
#define N_CREF1 256
#define N_CREF2 64
#define N_CREF3 16
#define N_SSE 256

struct MMXLl2Coder
{
    Ctr csig0[N_CSIG0], csig1[N_CSIG1], cref0[N_CREF0], cref1[N_CREF1], cref2[N_CREF2], cref3[N_CREF3];
    Ctr p_laplace[32];
    Mix lmixref[256], lmixsig[256], ssemix;
    Sse sse[N_SSE];
    Sse *psse1, *psse2;
    Ctr *pc1, *pc2, *pc3, *pc4, *pl;
    Mix *plmix;
    unsigned int *buf;                  /* the unsigned residuals, known planes only while decoding */
    int *msb;
    int nalloc, n, sample, bpn, pestimate;
    unsigned int state;
    int sigst[17];
    /* the +-32 running window of known magnitudes */
    int scount;
    unsigned long long sum;
    int n1, n2;                         /* significant neighbours within +-32 (any plane / above the current one) */
    int started;                        /* the model has been initialised once */
    int raw_depth;                      /* 0: every bit through the model; else refinement bits raw_depth or more planes
                                           below the sample's top bit go raw (measured 1.0000 bit/bit from depth 6) */
};

void mmx_ll2c_set_raw_depth(MMXLl2Coder *c, int depth) { c->raw_depth = depth; }

MMXLl2Coder *mmx_ll2c_new(void)
{
    MMXLl2Coder *c = (MMXLl2Coder *)calloc(1, sizeof(MMXLl2Coder));
    tables_init();
    return c;
}

void mmx_ll2c_free(MMXLl2Coder *c)
{
    if (!c) return;
    free(c->buf);
    free(c->msb);
    free(c);
}

/* p_laplace initial values: round((1 - 1 / (1 + 0.99^(2^i))) * 32768), clamped */
static const unsigned short laplace_init[32] = {
    16302, 16219, 16055, 15726, 15070, 13772, 11289, 7093, 2323, 190, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1
};

static int reset_model(MMXLl2Coder *c, int n, int model)
{
    int i;
    if (n > c->nalloc)
    {
        free(c->buf); free(c->msb);
        c->buf = (unsigned int *)malloc(sizeof(unsigned int) * (size_t)n);
        c->msb = (int *)malloc(sizeof(int) * (size_t)n);
        if (!c->buf || !c->msb) { c->nalloc = 0; return -1; }
        c->nalloc = n;
    }
    c->n = n;
    if (!model && c->started)
    {
        memset(c->msb, 0, sizeof(int) * (size_t)n);
        return 0;
    }
    c->started = 1;
    for (i = 0; i < N_CSIG0; i++) ctr_init(&c->csig0[i]);
    for (i = 0; i < N_CSIG1; i++) ctr_init(&c->csig1[i]);
    for (i = 0; i < N_CREF0; i++) ctr_init(&c->cref0[i]);
    for (i = 0; i < N_CREF1; i++) ctr_init(&c->cref1[i]);
    for (i = 0; i < N_CREF2; i++) ctr_init(&c->cref2[i]);
    for (i = 0; i < N_CREF3; i++) ctr_init(&c->cref3[i]);
    for (i = 0; i < 32; i++) { c->p_laplace[i].p1 = laplace_init[i]; c->p_laplace[i].cnt = 0; }
    for (i = 0; i < 256; i++) { mix_init(&c->lmixref[i], 5, 0); mix_init(&c->lmixsig[i], 3, 0); }
    mix_init(&c->ssemix, 2, 1);
    for (i = 0; i < N_SSE; i++) sse_init(&c->sse[i]);
    memset(c->msb, 0, sizeof(int) * (size_t)n);
    return 0;
}

static void get_sig_state(MMXLl2Coder *c, int i)
{
    int k, n = c->n;
    c->sigst[0] = c->msb[i];
    for (k = 1; k <= 8; k++)
    {
        c->sigst[2 * k - 1] = i - k >= 0 ? c->msb[i - k] : 0;
        c->sigst[2 * k] = i + k < n ? c->msb[i + k] : 0;
    }
}

/* the +-32 window sum of the known magnitudes (higher planes everywhere, the current plane left of the sample) */
static void update_avg(MMXLl2Coder *c)
{
    const int r = 32, s = c->sample, n = c->n;
    const unsigned int plane = 1u << c->bpn, cmask = ~(plane - 1u), hmask = cmask ^ plane;
    if (s == 0)
    {
        int k;
        c->sum = 0;
        c->scount = (r < n - 1 ? r : n - 1) + 1;
        for (k = 0; k < c->scount; k++) c->sum += c->buf[k] & hmask;
    }
    else
    {
        int out = s - r - 1, in = s + r;
        c->sum += c->buf[s - 1] & plane;
        if (out >= 0) { c->sum -= c->buf[out] & cmask; c->scount--; }
        if (in < n) { c->sum += c->buf[in] & hmask; c->scount++; }
    }
}

static int predict_laplace(const MMXLl2Coder *c)
{
    long long x;
    if (c->scount <= 0 || c->sum == 0) return 1;
    /* p = e^(-plane/mean) / (1 + e^(-plane/mean)) = squash(-plane/mean) in logit units of 1/256 */
    x = -(((long long)1 << c->bpn) * 256 * c->scount) / (long long)c->sum;
    if (x < DMIN) return 1;
    return clampi(squash((int)x), 1, PSCALEm);
}

static int predict_ref(MMXLl2Coder *c)
{
    const unsigned int *pa = c->buf;
    int s = c->sample, n = c->n, b = c->bpn;
    /* 64-bit: a mapped residual can use all 32 bits (32-bit sources); below 2^31 the values are the 32-bit ones */
    long long val = pa[s], lval = s > 0 ? pa[s - 1] : 0, lval2 = s > 1 ? pa[s - 2] : 0;
    long long nval = s < n - 1 ? pa[s + 1] : 0, nval2 = s < n - 2 ? pa[s + 2] : 0;
    long long b0 = val >> (b + 1), b1 = lval >> b, b2 = nval >> (b + 1), b3 = lval2 >> b, b4 = nval2 >> (b + 1);
    int c0 = (b0 << 1) < b1, c1 = b0 < b2, c2 = (b0 << 1) < b3, c3 = b0 < b4;
    long long x0 = (val >> (b + 1)) << 1, x1 = lval >> b, x2 = (nval >> (b + 1)) << 1, x3 = lval2 >> b, x4 = (nval2 >> (b + 1)) << 1;
    long long xm = (x0 + x1 + x2 + x3 + x4) / 5;
    int d0 = x0 > xm, d1 = x1 > xm;
    int ctx1 = (int)((b0 & 15) + ((b1 & 15) << 4) + ((b2 & 15) << 8));
    int ctx2 = (c0 + (c1 << 1) + (c2 << 2) + (c3 << 3)) + (d0 << 4) + (d1 << 5);
    int ctx3 = c->sigst[1] + c->sigst[2] + c->sigst[3] + c->sigst[4] + c->sigst[5] + c->sigst[6] + c->sigst[7] + c->sigst[8];
    int p[5], pctx;
    c->pl = &c->p_laplace[b];
    c->pc1 = &c->cref0[c->msb[s] & (N_CREF0 - 1)];
    c->pc2 = &c->cref1[ctx1 & 255];
    c->pc3 = &c->cref2[ctx2 & (N_CREF2 - 1)];
    c->pc4 = &c->cref3[ctx3 & (N_CREF3 - 1)];
    pctx = ((((c->pestimate >> 12) << 1) + d0) << 1) + (b0 & 1);
    c->plmix = &c->lmixref[pctx & 255];
    p[0] = c->pestimate; p[1] = c->pl->p1; p[2] = c->pc1->p1; p[3] = c->pc2->p1; p[4] = c->pc3->p1;
    return mix_predict(c->plmix, p);
}

static void update_ref(MMXLl2Coder *c, int bit)
{
    ctr_update(c->pl, bit, RATE_P);
    ctr_update(c->pc1, bit, RATE_REF);
    ctr_update(c->pc2, bit, RATE_REF);
    ctr_update(c->pc3, bit, RATE_REF);
    ctr_update(c->pc4, bit, RATE_REF);
    mix_update(c->plmix, bit, MIXRATE_REF);
    c->state = c->state << 1;
}

/* The significance counts of the +-32 window around the sample, kept incrementally: identical to recounting the
   window at every sample (sac's CountSig), because within one plane only the sample just coded can change. */
static void sig_window(MMXLl2Coder *c)
{
    const int r = 32, s = c->sample, n = c->n, b = c->bpn;
    if (s == 0)
    {
        int i;
        c->n1 = c->n2 = 0;
        for (i = 1; i <= r && i < n; i++) { if (c->msb[i]) c->n1++; if (c->msb[i] > b) c->n2++; }
    }
    else
    {
        int out = s - r - 1, in = s + r, prev = s - 1;
        if (out >= 0) { if (c->msb[out]) c->n1--; if (c->msb[out] > b) c->n2--; }
        if (in < n) { if (c->msb[in]) c->n1++; if (c->msb[in] > b) c->n2++; }
        if (c->msb[prev]) c->n1++;                                  /* the previous sample joins the window ... */
        if (c->msb[prev] > b) c->n2++;
        if (c->msb[s]) c->n1--;                                     /* ... and the current one leaves it */
        if (c->msb[s] > b) c->n2--;
    }
}

static void count_sig(const MMXLl2Coder *c, int r, int *n1, int *n2)
{
    (void)r;
    *n1 = c->n1;
    *n2 = c->n2;
}

static int predict_sig(MMXLl2Coder *c)
{
    int ctx1 = 0, i, n1, n2, p[3], mixctx;
    for (i = 0; i < 16; i++) if (c->sigst[i + 1]) ctx1 += 1 << i;
    count_sig(c, 32, &n1, &n2);
    c->pl = &c->p_laplace[c->bpn];
    c->pc1 = &c->csig0[ctx1];
    c->pc2 = &c->csig1[n2 & (N_CSIG1 - 1)];
    mixctx = (int)((c->state & 15) << 3) + ((n1 >= 3 ? 3 : n1) << 1) + (n2 > 0 ? 1 : 0);
    c->plmix = &c->lmixsig[mixctx & 255];
    p[0] = c->pl->p1; p[1] = c->pc1->p1; p[2] = c->pc2->p1;
    return mix_predict(c->plmix, p);
}

static void update_sig(MMXLl2Coder *c, int bit)
{
    ctr_update(c->pl, bit, RATE_P);
    ctr_update(c->pc1, bit, RATE_SIG);
    ctr_update(c->pc2, bit, RATE_SIG);
    mix_update(c->plmix, bit, MIXRATE_SIG);
    c->state = (c->state << 1) + 1;
}

static int predict_sse(MMXLl2Coder *c, int p1)
{
    int ctx1 = ((c->pestimate >> 11) << 1) + (c->sigst[0] ? 1 : 0);
    int ctx2 = 32 + (c->sigst[0] ? 1 : 0) + ((c->sigst[1] ? 1 : 0) << 1) + ((c->sigst[2] ? 1 : 0) << 2) + ((c->sigst[3] ? 1 : 0) << 3)
             + ((c->sigst[4] ? 1 : 0) << 4) + ((c->sigst[5] ? 1 : 0) << 5) + ((c->sigst[6] ? 1 : 0) << 6);
    int pr1, pr2, p[2];
    c->psse1 = &c->sse[ctx1 & (N_SSE - 1)];
    c->psse2 = &c->sse[ctx2 & (N_SSE - 1)];
    pr1 = sse_predict(c->psse1, p1);
    pr2 = sse_predict(c->psse2, pr1);
    p[0] = (pr1 + pr2 + 1) >> 1; p[1] = p1;
    return mix_predict(&c->ssemix, p);
}

static void update_sse(MMXLl2Coder *c, int bit)
{
    sse_update(c->psse1, bit, RATE_SSE);
    sse_update(c->psse2, bit, RATE_SSE);
    mix_update(&c->ssemix, bit, MIXRATE_SSE);
}

static int max_plane(const unsigned int *u, int n)
{
    unsigned int m = 0;
    int i, b = 0;
    for (i = 0; i < n; i++) if (u[i] > m) m = u[i];
    while (m >>= 1) b++;
    return b;
}

/* residual <-> unsigned: 0, -1, +1, -2, +2 ... -> 0, 2, 1, 4, 3 ...; any value in [-(2^31 - 1), 2^31] fits 32 bits
   (a 32-bit source's residual is reduced to that range modulo 2^32 by the codec) */
static unsigned int zz(long long r)
{
    return r < 0 ? (unsigned int)(2ull * (unsigned long long)(-r)) : r > 0 ? (unsigned int)(2ull * (unsigned long long)r - 1ull) : 0u;
}
static long long unzz(unsigned int u)
{
    return (u & 1u) ? (long long)(((unsigned long long)u + 1ull) >> 1) : -(long long)(u >> 1);
}

#ifdef LL2C_STATS
#include <stdio.h>
#include <math.h>
static double st_cost[34], st_n[34];
static void st_dump(void)
{
    int d;
    for (d = 0; d < 34; d++)
        if (st_n[d] > 0) fprintf(stderr, "depth %2d: %12.0f bits, %.4f bit/bit\n", d - 1, st_n[d], st_cost[d] / st_n[d]);
}
static void st_add(MMXLl2Coder *c, int p, int bit)
{
    static int reg = 0;
    int d = c->sigst[0] ? c->msb[c->sample] - c->bpn + 1 : 0;   /* 0 = significance, else depth below the msb */
    double q = (double)p / 32768.0;
    if (!reg) { reg = 1; atexit(st_dump); }
    if (d > 33) d = 33;
    st_cost[d] += -log2(bit ? q : 1.0 - q);
    st_n[d] += 1;
}
#endif

/* codes c->buf[0..n) (the mapped residuals) plane by plane */
static void encode_planes(MMXLl2Coder *c, MMXRangeEncoder *rc, int n)
{
    int maxbpn, k;
    maxbpn = max_plane(c->buf, n);
    for (k = 4; k >= 0; k--) mmx_rc_enc_bypass(rc, (unsigned long)((maxbpn >> k) & 1), 1);   /* 5 bits: the top plane */
    for (c->bpn = maxbpn; c->bpn >= 0; c->bpn--)
    {
        c->state = 0;
        for (c->sample = 0; c->sample < n; c->sample++)
        {
            int bit, p;
            update_avg(c);
            sig_window(c);
            bit = (int)((c->buf[c->sample] >> c->bpn) & 1u);
            if (c->raw_depth && c->msb[c->sample] - c->bpn >= c->raw_depth)
            {   /* deep refinement bit: noise, coded raw; the model does not see it */
                mmx_rc_enc_bypass(rc, (unsigned long)bit, 1);
                c->state <<= 1;
                continue;
            }
            c->pestimate = predict_laplace(c);
            get_sig_state(c, c->sample);
            if (c->sigst[0])
            {
                p = predict_sse(c, predict_ref(c));
#ifdef LL2C_STATS
                st_add(c, p, bit);
#endif
                mmx_rc_enc_p15(rc, (unsigned int)p, (unsigned int)bit);
                update_ref(c, bit);
                update_sse(c, bit);
            }
            else
            {
                p = predict_sse(c, predict_sig(c));
#ifdef LL2C_STATS
                st_add(c, p, bit);
#endif
                mmx_rc_enc_p15(rc, (unsigned int)p, (unsigned int)bit);
                update_sig(c, bit);
                update_sse(c, bit);
                if (bit) c->msb[c->sample] = c->bpn;
            }
        }
    }
}

/* decodes n mapped residuals into c->buf */
static void decode_planes(MMXLl2Coder *c, MMXRangeDecoder *rc, int n)
{
    int maxbpn = 0, k;
    memset(c->buf, 0, sizeof(unsigned int) * (size_t)n);
    for (k = 0; k < 5; k++) maxbpn = (maxbpn << 1) | (int)mmx_rc_dec_bypass(rc, 1);
    for (c->bpn = maxbpn; c->bpn >= 0; c->bpn--)
    {
        c->state = 0;
        for (c->sample = 0; c->sample < n; c->sample++)
        {
            int bit;
            update_avg(c);
            sig_window(c);
            if (c->raw_depth && c->msb[c->sample] - c->bpn >= c->raw_depth)
            {
                if (mmx_rc_dec_bypass(rc, 1)) c->buf[c->sample] |= 1u << c->bpn;
                c->state <<= 1;
                continue;
            }
            c->pestimate = predict_laplace(c);
            get_sig_state(c, c->sample);
            if (c->sigst[0])
            {
                bit = (int)mmx_rc_dec_p15(rc, (unsigned int)predict_sse(c, predict_ref(c)));
                update_ref(c, bit);
                update_sse(c, bit);
            }
            else
            {
                bit = (int)mmx_rc_dec_p15(rc, (unsigned int)predict_sse(c, predict_sig(c)));
                update_sig(c, bit);
                update_sse(c, bit);
                if (bit) c->msb[c->sample] = c->bpn;
            }
            if (bit) c->buf[c->sample] |= 1u << c->bpn;
        }
    }
}

void mmx_ll2c_encode(MMXLl2Coder *c, MMXRangeEncoder *rc, const int *res, int n)
{
    int i;
    if (n <= 0 || reset_model(c, n, 1) != 0) return;
    for (i = 0; i < n; i++) c->buf[i] = zz(res[i]);
    encode_planes(c, rc, n);
}
void mmx_ll2c_decode(MMXLl2Coder *c, MMXRangeDecoder *rc, int *res, int n)
{
    int i;
    if (n <= 0 || reset_model(c, n, 1) != 0) return;
    decode_planes(c, rc, n);
    for (i = 0; i < n; i++) res[i] = (int)unzz(c->buf[i]);
}
void mmx_ll2c_encode_cont(MMXLl2Coder *c, MMXRangeEncoder *rc, const int *res, int n)
{
    int i;
    if (n <= 0 || reset_model(c, n, 0) != 0) return;
    for (i = 0; i < n; i++) c->buf[i] = zz(res[i]);
    encode_planes(c, rc, n);
}
void mmx_ll2c_decode_cont(MMXLl2Coder *c, MMXRangeDecoder *rc, int *res, int n)
{
    int i;
    if (n <= 0 || reset_model(c, n, 0) != 0) return;
    decode_planes(c, rc, n);
    for (i = 0; i < n; i++) res[i] = (int)unzz(c->buf[i]);
}
void mmx_ll2c_encode_cont64(MMXLl2Coder *c, MMXRangeEncoder *rc, const long long *res, int n)
{
    int i;
    if (n <= 0 || reset_model(c, n, 0) != 0) return;
    for (i = 0; i < n; i++) c->buf[i] = zz(res[i]);
    encode_planes(c, rc, n);
}
void mmx_ll2c_decode_cont64(MMXLl2Coder *c, MMXRangeDecoder *rc, long long *res, int n)
{
    int i;
    if (n <= 0 || reset_model(c, n, 0) != 0) return;
    decode_planes(c, rc, n);
    for (i = 0; i < n; i++) res[i] = unzz(c->buf[i]);
}
