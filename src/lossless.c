#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "lab.h"
#include "lossless.h"

#define HIST MMX_LL_HIST
#define SIDE_K MMX_LL_SIDE_K
#define LPC_MAX_SHIFT 15
#define LPC_COEF_LIMIT (1LL << 20)       /* decoder sanity bound on |coefficient| */
#define LOG2_LUT_SIZE 4096
#define PI 3.14159265358979323846

/* Encoder search: the Levinson-Durbin error energies rank the LPC orders,
   the best LPC_SEARCH_ORDERS of them are evaluated exactly (integer residual
   after coefficient quantization) at LPC_BASE_PRECISION bits, the winner is
   then re-quantized at the other precisions. Precisions are bits including
   sign; 6..8 win most often at 1024 samples per frame, the side information
   of 12 coefficients at 12 bits rarely pays off. */
#ifndef LPC_SEARCH_ORDERS
#define LPC_SEARCH_ORDERS 3
#endif
#ifndef LPC_BASE_PRECISION
#define LPC_BASE_PRECISION 7
#endif
#ifndef LPC_PRECISIONS
#define LPC_PRECISIONS 6, 8, 10
#endif
static const unsigned int lpc_precisions[] = { LPC_PRECISIONS };
#define N_PRECISIONS (sizeof(lpc_precisions) / sizeof(lpc_precisions[0]))

/* Filter cascade, long to short (Monkey's Audio "insane": 1024/15, 256/13,
   16/11). Measured on 60 s of title A against the LPC alone: 256/13+32/10
   -1.00 %, 512/14+32/10 -1.21 %, 1024/15+32/10 -1.28 %, this cascade
   -1.36 %; 1024 taps at shift 14 are worse than at 15, and the short stage
   in front of the long one is worse. */
static const unsigned int nlms_order_def[MMX_LL_NLMS_STAGES] = { 1024, 256, 16 };
static const unsigned int nlms_shift_def[MMX_LL_NLMS_STAGES] = { 15, 13, 11 };
/* Error-normalised step (revision 8, see nlms.h): the step at the mean error is half the sign-sign
   step on the 1024-tap stage and the full step on the short ones. Prototype on twelve 30 s excerpts
   (Q6 multiplier capped at 2, mean |err| at rate 1/32): -0.35 % against sign-sign; this build,
   lossless at analysis 4 on 30 s excerpts: title E -0.54 %, title G -0.25 %, title H -0.33 %,
   decoder CPU time +10 % (title E). */
static const unsigned int nlms_step_def[MMX_LL_NLMS_STAGES] = { 8, 7, 7 };

/* Experiment (MMX_NLMS_ORDER / MMX_NLMS_SHIFT / MMX_NLMS_STEP, comma separated, one value per
   stage): the cascade is part of the bitstream - encoder and decoder must build the same filters -
   so a changed value only decodes in the same environment. Orders are clamped to
   MMX_NLMS_MIN_ORDER..MAX_ORDER; MMX_NLMS_STEP is the step shift of the error-normalised update
   (0 = plain sign-sign, clamped to MMX_NLMS_MAX_STEP_SHIFT).

   Motive and RESULT, so nobody repeats it: a further order-32 LPC still pulls 0.22-0.99 %
   (mean 0.61 %) out of the finished residual of four tracks, and most of it sits at low order,
   i.e. in the range the last stage is supposed to cover. Swept on title F, title A,
   title E and title G, lossless, analysis 3, total bytes against the default 1024,256,16 at
   15,13,11: last stage shift 10 +0.004 %, 32 taps +0.023 %, shift 12 +0.041 %, 32 taps at
   shift 12 +0.049 %, shift 9 +0.135 %. Every variation is neutral or worse - the cascade is at
   its optimum.

   And the "residual structure" itself was a measurement artefact: the 0.61 % was a
   VARIANCE reduction, 0.5 * log2(var_in / var_out), which is only an entropy on a stationary signal.
   A residual is heteroscedastic - a few loud passages own the variance while the coder pays per
   sample - and a quadratic feature r * |r| games exactly that proxy. Re-measured with the empirical
   entropy of the same residual: a global 12-parameter least-squares fit (8 linear + 4 quadratic taps)
   gains +0.004 bit/sample, an ideal per-frame fit +0.015, and a backward-adaptive integer stage
   built behind the LPC (correct, bit-exact, then removed) -0.005. There is nothing there. */
static void nlms_cfg(const char *env, const unsigned int *def, unsigned int *out, long long min)
{
    const char *e = mmx_lab_getenv(env);
    unsigned int i = 0;
    for (i = 0; i < MMX_LL_NLMS_STAGES; i++)
        out[i] = def[i];
    if (!e)
        return;
    for (i = 0; i < MMX_LL_NLMS_STAGES && *e; i++)
    {
        long long v = strtoll(e, NULL, 10);
        if (v >= min && v <= 0x7fffffffLL && (v > 0 || *e == '0'))
            out[i] = (unsigned int)v;
        while (*e && *e != ',') e++;
        if (*e == ',') e++;
    }
}

MMXLosslessStats mmx_ll_stats;

void mmx_ll_state_init(MMXLosslessContexts *c)
{
    unsigned int ch, st;
    memset(c, 0, sizeof(*c));
    {
        unsigned int ord[MMX_LL_NLMS_STAGES], sh[MMX_LL_NLMS_STAGES], mu[MMX_LL_NLMS_STAGES];
        nlms_cfg("MMX_NLMS_ORDER", nlms_order_def, ord, 1);
        nlms_cfg("MMX_NLMS_SHIFT", nlms_shift_def, sh, 1);
        nlms_cfg("MMX_NLMS_STEP", nlms_step_def, mu, 0);
        for (ch = 0; ch < MMX_MAX_CH; ch++)
            for (st = 0; st < MMX_LL_NLMS_STAGES; st++)
                mmx_nlms_init(&c->nlms[ch][st], ord[st], sh[st], mu[st]);
    }
    /* The residual model is stream state like the filter cascade: a block
       averages four frames, far too few for 40 x 69 counters to converge,
       and the decoder walks the blocks in file order anyway. */
    mmx_ctr_init_array(&c->res[0][0][0], 2UL * MMX_LL_RES_CLASSES * MMX_LL_RES_CTX);
    memset(c->res_state, 0, sizeof(c->res_state));
    /* the joint-stereo LS stage starts from zero weights and covariance (memset above);
       a decoder of a revision 6-7 file switches it off again */
    c->xls_on = 1;
    {
        MMXProb *p = (MMXProb *)c;
        size_t n = offsetof(MMXLosslessContexts, prev_gain) / sizeof(MMXProb), i;
        for (i = 0; i < n; i++)
            p[i] = MMX_PROB_INIT;
    }
    mmx_ll_contexts_init(c);
}

void mmx_ll_state_set_revision(MMXLosslessContexts *c, unsigned int bitstream_rev)
{
    unsigned int ch, st;
    if (bitstream_rev >= 8)
        return;
    for (ch = 0; ch < MMX_MAX_CH; ch++)
        for (st = 0; st < MMX_LL_NLMS_STAGES; st++)
            c->nlms[ch][st].step_shift = 0;   /* revisions 6-7: plain sign-sign cascade */
    c->xls_on = bitstream_rev >= 8;   /* the joint-stereo LS stage of channel 1 exists since revision 8 */
}

/* Per block: resets the block state. The probabilities of the side
   information are stream state since revision 6 - a block averages four
   frames on real material, far too few for the 12 x 22 LPC coefficient
   contexts to converge (title A -0.02 %, title B -0.01 %,
   title I +-0). */
void mmx_ll_contexts_init(MMXLosslessContexts *c)
{
    size_t off = offsetof(MMXLosslessContexts, prev_gain);
    memset((char *)c + off, 0, offsetof(MMXLosslessContexts, nlms) - off);
}

/* ------------------------------------------------------------- signals */

/* A signal buffer has HIST valid samples before index 0 (history of the
   previous frame in the same coded domain). */
static long long *sig_alloc(unsigned long count)
{
    long long *b = (long long *)calloc(count + HIST, sizeof(long long));
    return b ? b + HIST : NULL;
}

static void sig_free(long long *s)
{
    if (s)
        free(s - HIST);
}

/* shifts the last samples of s into the history h (HIST entries, oldest first) */
static void hist_push(long long *h, const long long *s, unsigned long n)
{
    if (n >= HIST)
        memcpy(h, s + n - HIST, sizeof(long long) * HIST);
    else if (n > 0)
    {
        memmove(h, h + n, sizeof(long long) * (HIST - n));
        memcpy(h + HIST - n, s, sizeof(long long) * n);
    }
}

static long long apply_gain(long long v, long long gain_q12)
{
    long long p = (long long)v * gain_q12;
    long long half = 1LL << (MMX_LL_GAIN_Q - 1);
    return (long long)((p >= 0 ? (p + half) : (p - half + 1)) >> MMX_LL_GAIN_Q);
}

static long long fit_gain(const long long *x, const long long *src, unsigned long n)
{
    double xy = 0.0, yy = 0.0, g;
    unsigned long i;
    for (i = 0; i < n; i++)
    {
        xy += (double)x[i] * src[i];
        yy += (double)src[i] * src[i];
    }
    if (yy <= 0.0)
        return 0;
    g = xy / yy * (double)(1 << MMX_LL_GAIN_Q);
    if (g > MMX_LL_GAIN_MAX) g = MMX_LL_GAIN_MAX;
    if (g < -MMX_LL_GAIN_MAX) g = -MMX_LL_GAIN_MAX;
    return (long long)floor(g + 0.5);
}

/* history of channel c in the reference-residual domain */
static void load_history(const MMXLosslessContexts *ctx, unsigned int c, int use_pred, long long gain,
                         int use2, long long gain2, long long *s)
{
    unsigned int k;
    for (k = 0; k < HIST; k++)
        s[(int)k - HIST] = ctx->hist_x[c][k] - (use_pred ? apply_gain(ctx->hist_src[c][k], gain) : 0)
                                             - (use2 ? apply_gain(ctx->hist_src2[c][k], gain2) : 0);
}

static void store_history(MMXLosslessContexts *ctx, unsigned int channels, long long *const *x, long long *const *src,
                          long long *const *src2, unsigned long count)
{
    unsigned int c;
    for (c = 0; c < channels; c++)
    {
        hist_push(ctx->hist_x[c], x[c], count);
        if (src && src[c])
            hist_push(ctx->hist_src[c], src[c], count);
        else
            memset(ctx->hist_src[c], 0, sizeof(ctx->hist_src[c]));
        if (src2 && src2[c])
            hist_push(ctx->hist_src2[c], src2[c], count);
        else
            memset(ctx->hist_src2[c], 0, sizeof(ctx->hist_src2[c]));
    }
}

/* joint stereo: coded pair (a, b) from (l, r); works in place */
static void joint_forward(int mode, long long *a, long long *b, const long long *l, const long long *r, long long from, long long to)
{
    long long i;
    for (i = from; i < to; i++)
    {
        long long lv = l[i], rv = r[i];
        switch (mode)
        {
        case 1: a[i] = (lv + rv) >> 1; b[i] = lv - rv; break;
        case 2: a[i] = lv; b[i] = lv - rv; break;
        case 3: a[i] = rv; b[i] = lv - rv; break;
        default: a[i] = lv; b[i] = rv; break;
        }
    }
}

static void joint_inverse(int mode, long long *a, long long *b, unsigned long count)
{
    unsigned long i;
    for (i = 0; i < count; i++)
    {
        long long av = a[i], bv = b[i];
        switch (mode)
        {
        case 1: { long long sum = av * 2 + (bv & 1LL); a[i] = (sum + bv) >> 1; b[i] = (sum - bv) >> 1; } break;
        case 2: b[i] = av - bv; break;
        case 3: a[i] = av + bv; b[i] = av; break;
        default: break;
        }
    }
}

/* ---------------------------------------------------------- predictors */

/* s points at the sample to predict; s[-1..-HIST] are valid */
static long long predict_fixed(const long long *s, unsigned int order)
{
    switch (order)
    {
    case 0: return 0;
    case 1: return s[-1];
    case 2: return 2 * s[-1] - s[-2];
    default: return 3 * s[-1] - 3 * s[-2] + s[-3];
    }
}

static long long predict_lpc(const long long *s, const long long *coef, unsigned int order, unsigned int shift)
{
    long long sum = 0;
    unsigned int j;
    for (j = 0; j < order; j++)
        sum += (long long)coef[j] * (long long)s[-1 - (int)j];
    return (long long)(sum >> shift);
}

static long long predict(const long long *s, const MMXLosslessPred *p)
{
    return p->lpc ? predict_lpc(s, p->coef, p->order, p->shift) : predict_fixed(s, p->order);
}

/* ------------------------------------------------------- cost proxies */

static double log2_lut[LOG2_LUT_SIZE];
static double tukey_win[MMX_HOP];
static unsigned long tukey_n = 0;
static int tables_ready = 0;

static void tables_init(void)
{
    unsigned long i;
    if (tables_ready)
        return;
    for (i = 0; i < LOG2_LUT_SIZE; i++)
        log2_lut[i] = log((double)i + 1.0) / log(2.0);
    tables_ready = 1;
}

/* log2(1 + |r|): the bypass part of the residual code grows like this, the
   adaptive part adds a roughly constant amount per sample */
static double bits_of(long long r)
{
    unsigned long v = (unsigned long)(r < 0 ? -r : r);
    if (v < LOG2_LUT_SIZE)
        return log2_lut[v];
    {
        unsigned int len = 0;
        while ((v >> len) > 1)
            len++;
        return (double)len + (double)(v - (1UL << len)) / (double)(1UL << len);
    }
}

static double cost_pred(const long long *s, unsigned long n, const MMXLosslessPred *p)
{
    double c = 0.0;
    unsigned long i;
    if (p->lpc)
        for (i = 0; i < n; i++)
            c += bits_of(s[i] - predict_lpc(s + i, p->coef, p->order, p->shift));
    else
        for (i = 0; i < n; i++)
            c += bits_of(s[i] - predict_fixed(s + i, p->order));
    return c;
}

/* ------------------------------------------------------- LPC analysis */

typedef struct
{
    unsigned int fixed_order;           /* best fixed predictor and its proxy cost */
    double fixed_cost;
    unsigned int maxo;                  /* highest usable LPC order (0: none) */
    double lpc[MMX_LL_LPC_MAX_ORDER][MMX_LL_LPC_MAX_ORDER]; /* lpc[o-1][0..o-1]: coefficients of order o */
    double err[MMX_LL_LPC_MAX_ORDER];   /* prediction error energies of the windowed signal */
} SigAnalysis;

/* Tukey(0.5) window (as FLAC's default apodization), cached per length */
static double tukey(unsigned long i, unsigned long n)
{
    unsigned long taper = n / 4;
    if (i < taper)
        return 0.5 * (1.0 - cos(PI * (double)i / (double)taper));
    if (i >= n - taper)
        return 0.5 * (1.0 - cos(PI * (double)(n - 1 - i) / (double)taper));
    return 1.0;
}

/* Best fixed predictor, then windowed autocorrelation and Levinson-Durbin
   recursion in double precision for the LPC orders 1..12. */
static void analyze(const long long *s, unsigned long n, double *w, SigAnalysis *a)
{
    double R[MMX_LL_LPC_MAX_ORDER + 1], k[MMX_LL_LPC_MAX_ORDER], e;
    unsigned long i, lag;
    unsigned int order, j;
    MMXLosslessPred p;

    memset(&p, 0, sizeof(p));
    a->fixed_cost = 1e300;
    a->fixed_order = 0;
    for (p.order = 0; p.order <= MMX_LL_MAX_ORDER; p.order++)
    {
        double c = cost_pred(s, n, &p);
        if (c < a->fixed_cost) { a->fixed_cost = c; a->fixed_order = p.order; }
    }
    a->maxo = 0;
    if (n <= MMX_LL_LPC_MAX_ORDER + 1)
        return;
    if (n <= MMX_HOP)
    {
        if (tukey_n != n)
        {
            for (i = 0; i < n; i++)
                tukey_win[i] = tukey(i, n);
            tukey_n = n;
        }
        for (i = 0; i < n; i++)
            w[i] = (double)s[i] * tukey_win[i];
    }
    else
        for (i = 0; i < n; i++)
            w[i] = (double)s[i] * tukey(i, n);
    for (lag = 0; lag <= MMX_LL_LPC_MAX_ORDER; lag++)
    {
        double acc = 0.0;
        for (i = lag; i < n; i++)
            acc += w[i] * w[i - lag];
        R[lag] = acc;
    }
    if (R[0] <= 0.0)
        return;
    e = R[0];
    for (order = 0; order < MMX_LL_LPC_MAX_ORDER; order++)
    {
        double r = -R[order + 1];
        for (j = 0; j < order; j++)
            r -= k[j] * R[order - j];
        r /= e;
        k[order] = r;
        for (j = 0; j < (order >> 1); j++)
        {
            double t = k[j];
            k[j] += r * k[order - 1 - j];
            k[order - 1 - j] += r * t;
        }
        if (order & 1)
            k[j] += k[j] * r;
        e *= (1.0 - r * r);
        for (j = 0; j <= order; j++)
            a->lpc[order][j] = -k[j];
        a->err[order] = e;
        a->maxo = order + 1;
        if (e <= 0.0)
            return;
    }
}

/* Estimated bits of order o from its error energy, on the scale of the
   log2(1+|r|) proxy: a Gaussian residual of deviation s costs about
   log2(s) - 0.9 proxy bits per sample and the Tukey window takes about 0.2
   off the energy estimate, hence EST_OFFSET. */
#ifndef EST_OFFSET
#define EST_OFFSET 0.7
#endif
static double lpc_est_bits(const SigAnalysis *a, unsigned long n, unsigned int o)
{
    return (0.5 * log(a->err[o - 1] / (double)n + 0.1) / log(2.0) - EST_OFFSET) * (double)n + (double)o * (LPC_BASE_PRECISION + 2) + 5.0;
}

/* Estimated bits of a channel signal for the stereo and reference decisions
   (deciding by exact predictor searches instead measured no gain). */
static double est_bits(const SigAnalysis *a, unsigned long n)
{
    double c = a->fixed_cost + 1.0;
    unsigned int o;
    for (o = 1; o <= a->maxo; o++)
    {
        double e = lpc_est_bits(a, n, o);
        if (e < c) c = e;
    }
    return c;
}

/* FLAC-style quantization: `precision` bits including sign, the shift is
   chosen so the largest coefficient just fits, rounding with error feedback. */
static int lpc_quantize(const double *lpc, unsigned int order, unsigned int precision, long long *q, unsigned int *shift)
{
    double cmax = 0.0, e = 0.0;
    long long qmax = (1LL << (precision - 1)) - 1, qmin = -qmax - 1;
    int log2cmax, sh;
    unsigned int j;
    for (j = 0; j < order; j++)
        if (fabs(lpc[j]) > cmax)
            cmax = fabs(lpc[j]);
    if (cmax <= 0.0)
        return -1;
    (void)frexp(cmax, &log2cmax);
    log2cmax--;
    sh = (int)precision - 2 - log2cmax;
    if (sh > LPC_MAX_SHIFT)
        sh = LPC_MAX_SHIFT;
    if (sh < 0)
        return -1;
    for (j = 0; j < order; j++)
    {
        long long v;
        e += lpc[j] * (double)(1LL << sh);
        v = (long long)floor(e + 0.5);
        if (v > qmax) v = qmax;
        else if (v < qmin) v = qmin;
        e -= (double)v;
        q[j] = v;
    }
    *shift = (unsigned int)sh;
    return 0;
}

/* estimated side information of an LPC predictor in bits */
static double lpc_side_bits(const MMXLosslessPred *p)
{
    double b = 4.0 + 1.0;
    unsigned int j;
    for (j = 0; j < p->order; j++)
        b += bits_of(p->coef[j]) + 3.0;
    return b;
}

static void try_lpc(const long long *s, unsigned long n, const SigAnalysis *a, unsigned int o, unsigned int precision,
                    MMXLosslessPred *best, double *bc)
{
    MMXLosslessPred p;
    double c;
    memset(&p, 0, sizeof(p));
    p.lpc = 1;
    p.order = o;
    if (lpc_quantize(a->lpc[o - 1], o, precision, p.coef, &p.shift) != 0)
        return;
    c = cost_pred(s, n, &p) + lpc_side_bits(&p);
    if (c < *bc)
    {
        *bc = c;
        *best = p;
    }
}

/* Picks the predictor for one channel by estimated bits = sum log2(1+|residual|)
   + side information: the best fixed order against the exactly evaluated
   LPC candidates (see the search notes at the top). Returns that cost. */
static double choose_predictor(const long long *s, unsigned long n, const SigAnalysis *a, MMXLosslessPred *best)
{
    double est[MMX_LL_LPC_MAX_ORDER + 1], bc;
    unsigned int o, k, tried = 0, used[MMX_LL_LPC_MAX_ORDER + 1];

    memset(best, 0, sizeof(*best));
    best->order = a->fixed_order;
    bc = a->fixed_cost + 1.0;
    if (a->maxo == 0)
        return bc;
    memset(used, 0, sizeof(used));
    for (o = 1; o <= a->maxo; o++)
        est[o] = lpc_est_bits(a, n, o);
    while (tried < LPC_SEARCH_ORDERS && tried < a->maxo)
    {
        unsigned int bo = 0;
        for (o = 1; o <= a->maxo; o++)
            if (!used[o] && (bo == 0 || est[o] < est[bo]))
                bo = o;
        used[bo] = 1;
        tried++;
        try_lpc(s, n, a, bo, LPC_BASE_PRECISION, best, &bc);
    }
    if (best->lpc)
        for (k = 0; k < N_PRECISIONS; k++)
            try_lpc(s, n, a, best->order, lpc_precisions[k], best, &bc);
    return bc;
}

/* ------------------------------------------------------ filter cascade */

/* One sample through the cascade: every stage predicts its input from its own
   history and adapts to it, so the state after a frame does not depend on
   whether the output is the coded signal. Returns the last stage's residual. */
static long long cascade_step(MMXNlms *f, long long v)
{
    unsigned int s;
    for (s = 0; s < MMX_LL_NLMS_STAGES; s++)
    {
        long long e = v - mmx_nlms_predict(&f[s]);
        mmx_nlms_update(&f[s], v, e);
        v = e;
    }
    return v;
}

/* Inverse of cascade_step for the residual e of the last stage. */
static long long cascade_inverse(MMXNlms *f, long long e)
{
    unsigned int s;
    for (s = MMX_LL_NLMS_STAGES; s-- > 0;)
    {
        long long v = e + mmx_nlms_predict(&f[s]);
        mmx_nlms_update(&f[s], v, e);
        e = v;
    }
    return e;
}

/* -------------------------------------------- joint-stereo LS stage */

/* Coded channel 1 predicted from its own past and channel 0's first difference
   around t, weights by exponentially weighted integer least squares (see
   lossless.h). The arithmetic is the measured prototype's (px_ldl_step /
   px_ldl_solve): only the regressor and the target are clamped to 2^24, which
   no 16-bit signal reaches (|first difference of L-R| < 2^18) and which keeps
   every product of a 24- or 32-bit source inside int64. */
#define XLS_N MMX_LL_XLS_N
#define XLS_DECAY 9                     /* R -= R >> 9: a memory of about 512 samples */
#define XLS_SOLVE 16                    /* channel-1 samples between two solves */
#define XLS_REG 14                      /* ridge: trace >> 14 on the diagonal */
#define XLS_LQ 20                       /* L in Q20 */
#define XLS_WQ 16                       /* weights in Q16 */
#define XLS_W_MAX (1LL << 22)
#define XLS_IN_MAX (1LL << 24)

static long long xls_clamp(long long v, long long m)
{
    return v > m ? m : v < -m ? -m : v;
}

/* Fixed-point LDL^T solve of (R + reg I) w = r: R normalised so its largest
   diagonal entry fits 2^28, regularised by trace >> XLS_REG, every
   intermediate clamped, so the solve is deterministic for any input. */
static void xls_solve(MMXLosslessXls *o)
{
    long long A[XLS_N][XLS_N], L[XLS_N][XLS_N], D[XLS_N], V[XLS_N], z[XLS_N], b[XLS_N];
    long long mx = 1, tr = 0, reg;
    int i, j, k, sh = 0;
    for (i = 0; i < XLS_N; i++)
    {
        if (o->R[i][i] > mx) mx = o->R[i][i];
        tr += o->R[i][i];
    }
    while ((mx >> sh) > (1LL << 28))
        sh++;
    reg = ((tr >> sh) >> XLS_REG) + 1;
    for (i = 0; i < XLS_N; i++)
    {
        for (j = 0; j <= i; j++)
            A[i][j] = o->R[i][j] >> sh;
        A[i][i] += reg;
        b[i] = xls_clamp(o->r[i] >> sh, 1LL << 31);
    }
    for (j = 0; j < XLS_N; j++)
    {
        long long d = A[j][j];
        for (k = 0; k < j; k++)
        {
            V[k] = xls_clamp((L[j][k] * D[k]) >> XLS_LQ, 1LL << 31);   /* L_jk D_k, on the scale of A */
            d -= (L[j][k] * V[k]) >> XLS_LQ;
        }
        if (d < 1)
            d = 1;
        D[j] = d;
        L[j][j] = 1LL << XLS_LQ;
        for (i = j + 1; i < XLS_N; i++)
        {
            long long t = A[i][j];
            for (k = 0; k < j; k++)
                t -= (L[i][k] * V[k]) >> XLS_LQ;
            t = xls_clamp(t, 1LL << 38);
            L[i][j] = xls_clamp((t * (1LL << XLS_LQ)) / d, 1LL << 29);
        }
    }
    for (i = 0; i < XLS_N; i++)
    {
        long long t = b[i];
        for (k = 0; k < i; k++)
            t -= (L[i][k] * z[k]) >> XLS_LQ;
        z[i] = xls_clamp(t, 1LL << 32);
    }
    for (i = XLS_N - 1; i >= 0; i--)
    {
        long long t = (z[i] * (1LL << XLS_WQ)) / D[i];
        for (k = i + 1; k < XLS_N; k++)
            t -= (L[k][i] * o->w[k]) >> XLS_LQ;
        o->w[i] = xls_clamp(t, XLS_W_MAX);
    }
}

/* Channel 0's first difference in the coded domain for the stage of channel 1:
   xu[i] = v0[i] - v0[i-1] for i in [1 - MMX_LL_XLS_CROSS, n) (v0 carries the
   history of the previous frame in the current domain on both sides), zero
   for the taps past the frame end - channel 0's whole frame is known before
   channel 1 on both sides. xu needs n + MMX_LL_XLS_FUT entries after 0. */
static void xls_cross(const long long *v0, unsigned long n, long long *xu)
{
    long long i;
    for (i = 1 - MMX_LL_XLS_CROSS; i < (long long)n; i++)
        xu[i] = xls_clamp(v0[i] - v0[i - 1], XLS_IN_MAX);
    for (i = 0; i < MMX_LL_XLS_FUT; i++)
        xu[(long long)n + i] = 0;
}

/* regressor at frame index i: y1[t-1..t-8], u0[t..t-7], u0[t+1..t+4]. Every entry is clamped
   to 2^24, so it is held as int: the 32 x 32 -> 64 bit products below vectorize (the decoder
   runs this for every sample of channel 1). */
static void xls_phi(const MMXLosslessXls *o, const long long *xu, long long i, int *phi)
{
    int k, m = 0;
    for (k = 0; k < MMX_LL_XLS_OWN; k++)
        phi[m++] = (int)o->own[k];
    for (k = 0; k < MMX_LL_XLS_CROSS; k++)
        phi[m++] = (int)xu[i - k];
    for (k = 1; k <= MMX_LL_XLS_FUT; k++)
        phi[m++] = (int)xu[i + k];
}

static long long xls_predict(const MMXLosslessXls *o, const int *phi)
{
    long long p = 0;
    int i;
    for (i = 0; i < XLS_N; i++)
        p += o->w[i] * phi[i];
    return (p + (1LL << (XLS_WQ - 1))) >> XLS_WQ;
}

/* adapts to the stage input y (the target the prediction was made for) */
static void xls_update(MMXLosslessXls *o, const int *phi, long long y)
{
    int i, j;
    y = xls_clamp(y, XLS_IN_MAX);
    for (i = 0; i < XLS_N; i++)
    {
        long long a = phi[i], *R = o->R[i];
        for (j = 0; j <= i; j++)
            R[j] += a * phi[j] - (R[j] >> XLS_DECAY);
        o->r[i] += a * y - (o->r[i] >> XLS_DECAY);
    }
    memmove(o->own + 1, o->own, sizeof(long long) * (MMX_LL_XLS_OWN - 1));
    o->own[0] = y;
    if (++o->phase == XLS_SOLVE)
    {
        o->phase = 0;
        xls_solve(o);
    }
}

static long long sat16l(long long v)
{
    return v > 32767 ? 32767 : v < -32768 ? -32768 : v;
}

/* Source stage: u points at the pre-whitened source at the current sample,
   taps past the frame end (k >= avail) read zeros on both sides. */
static long long side_predict(const MMXLosslessSide *f, const long long *u, long long avail)
{
    long long sum = 0;
    int k;
    for (k = -SIDE_K; k <= SIDE_K; k++)
        if (k < avail)
            sum += (long long)f->w[k + SIDE_K] * sat16l(u[k]);
    return (long long)((sum + (1LL << 12)) >> 13);
}

/* sign-sign adaptation with the step classes of the main filters */
static void side_update(MMXLosslessSide *f, const long long *u, long long avail, long long err)
{
    int k;
    long long a = u[0] < 0 ? -u[0] : u[0];
    if (err != 0)
        for (k = -SIDE_K; k <= SIDE_K; k++)
        {
            long long x = k < avail ? u[k] : 0, step;
            long long ax = x < 0 ? -x : x;
            if (ax == 0)
                continue;
            step = ax > 3 * f->avg ? 32 : ax > (4 * f->avg) / 3 ? 16 : 8;
            if ((err > 0) != (x > 0))
                step = -step;
            f->w[k + SIDE_K] = (short)sat16l(f->w[k + SIDE_K] + step);
        }
    f->avg += (a - f->avg) / 16;
}

/* Source stage availability of coded channel c: every physical channel that
   contributes to it must use the reference gain. */
static int src_stage_ok(const MMXLosslessFrame *fr, unsigned int channels, unsigned int c, long long *const *src, const int *use)
{
    if (!src || !src[c])
        return 0;
    if (channels == 2 && c < 2)
        switch (fr->stereo_ms)
        {
        case 1: return use[0] && use[1];
        case 2: return c == 0 ? use[0] : use[0] && use[1];
        case 3: return c == 0 ? use[1] : use[0] && use[1];
        default: break;
        }
    return use[c];
}

/* Pre-whitened source in the coded domain: js[c] gets the joint source with
   history, us[c] its first differences (us[-SIDE_K..count) valid). */
static void source_signals(const MMXLosslessFrame *fr, unsigned int channels, long long *const *src,
                           const long long (*hist)[MMX_LL_HIST], unsigned long count, long long **js, long long **us)
{
    unsigned int c, k;
    long long i;
    for (c = 0; c < channels; c++)
    {
        for (k = 0; k < HIST; k++)
            js[c][(int)k - HIST] = hist[c][k];
        memcpy(js[c], src[c], sizeof(long long) * count);
    }
    if (channels == 2 && fr->stereo_ms)
        joint_forward(fr->stereo_ms, js[0], js[1], js[0], js[1], -HIST, (long long)count);
    for (c = 0; c < channels; c++)
        for (i = -SIDE_K; i < (long long)count; i++)
            us[c][i] = js[c][i] - js[c][i - 1];
}

/* The chain of one coded channel in front of the predictor: first
   difference, source stage (when active), filter cascade. */
typedef struct
{
    MMXNlms *cascade;
    MMXLosslessSide *src;               /* NULL: no source stage */
    const long long *us;
    MMXLosslessSide *src2;              /* NULL: no second source stage */
    const long long *us2;
    MMXLosslessXls *xls;                /* NULL: no joint-stereo LS stage (channel 0, not stereo, revision < 8) */
    const long long *xu;                /* channel 0's first difference for the stage (see xls_cross) */
} Chain;

static void chain_setup(Chain *ch, MMXLosslessContexts *ctx, const MMXLosslessFrame *fr, unsigned int channels,
                        unsigned int c, long long *const *src, long long *const *us, long long *const *src2, long long *const *us2,
                        const long long *xu)
{
    ch->xls = xu && c == 1 ? &ctx->xls : NULL;
    ch->xu = xu;
    ch->cascade = ctx->nlms[c];
    ch->src = src_stage_ok(fr, channels, c, src, fr->use_pred) ? &ctx->sside[c] : NULL;
    ch->us = ch->src ? us[c] : NULL;
    ch->src2 = src_stage_ok(fr, channels, c, src2, fr->use_src2) ? &ctx->sside2[c] : NULL;
    ch->us2 = ch->src2 ? us2[c] : NULL;
}

/* Runs the chain over v[0..n) (v[-1] valid) into e. */
static void chain_forward(const Chain *ch, const long long *v, long long *e, unsigned long n)
{
    unsigned long i;
    for (i = 0; i < n; i++)
    {
        long long u = v[i] - v[(long long)i - 1], sp = ch->src ? side_predict(ch->src, ch->us + i, (long long)(n - i)) : 0;
        long long q = 0;
        int phi[XLS_N];
        if (ch->src2)
            sp += side_predict(ch->src2, ch->us2 + i, (long long)(n - i));
        if (ch->xls)
        {
            xls_phi(ch->xls, ch->xu, (long long)i, phi);
            q = xls_predict(ch->xls, phi);
        }
        e[i] = cascade_step(ch->cascade, u - sp - q);
        if (ch->xls)
            xls_update(ch->xls, phi, u - sp);
        if (ch->src)
            side_update(ch->src, ch->us + i, (long long)(n - i), u - sp);
        if (ch->src2)
            side_update(ch->src2, ch->us2 + i, (long long)(n - i), u - sp);
    }
}

/* ------------------------------------------------------ entropy coding */

static void enc_tree(MMXRangeEncoder *rc, MMXProb *p, unsigned int value, unsigned int bits)
{
    unsigned int node = 1, i;
    for (i = bits; i-- > 0;)
    {
        unsigned int b = (value >> i) & 1;
        mmx_rc_enc_bit(rc, &p[node], b);
        node = (node << 1) | b;
    }
}

static unsigned int dec_tree(MMXRangeDecoder *rc, MMXProb *p, unsigned int bits)
{
    unsigned int node = 1, i;
    for (i = 0; i < bits; i++)
        node = (node << 1) | mmx_rc_dec_bit(rc, &p[node]);
    return node - (1U << bits);
}

/* mode symbol: 0..3 fixed order, 4..15 LPC order 1..12; then shift and coefficients */
static void enc_pred(MMXRangeEncoder *rc, MMXLosslessContexts *ctx, unsigned int c, const MMXLosslessPred *p)
{
    unsigned int j;
    enc_tree(rc, ctx->mode[ctx->prev_lpc[c] ? 1 : 0], p->lpc ? 3 + p->order : p->order, MMX_LL_MODE_BITS);
    if (p->lpc)
    {
        enc_tree(rc, ctx->lpc_shift, p->shift, 4);
        for (j = 0; j < p->order; j++)
            mmx_rc_enc_seg(rc, ctx->lpc_coef[j], p->coef[j]);
    }
    ctx->prev_lpc[c] = p->lpc;
}

static int dec_pred(MMXRangeDecoder *rc, MMXLosslessContexts *ctx, unsigned int c, MMXLosslessPred *p)
{
    unsigned int m = dec_tree(rc, ctx->mode[ctx->prev_lpc[c] ? 1 : 0], MMX_LL_MODE_BITS), j;
    memset(p, 0, sizeof(*p));
    if (m <= MMX_LL_MAX_ORDER)
        p->order = m;
    else
    {
        p->lpc = 1;
        p->order = m - 3;
        if (p->order > MMX_LL_LPC_MAX_ORDER)
            return -1;
        p->shift = dec_tree(rc, ctx->lpc_shift, 4);
        for (j = 0; j < p->order; j++)
        {
            p->coef[j] = mmx_rc_dec_seg(rc, ctx->lpc_coef[j]);
            if (p->coef[j] > LPC_COEF_LIMIT || p->coef[j] < -LPC_COEF_LIMIT)
                return -1;
        }
    }
    ctx->prev_lpc[c] = p->lpc;
    return rc->failed ? -1 : 0;
}

/* Residual context class: half an octave of the running mean of |residual|,
   over its whole range. The previous ladder (one class per octave, 12 of
   them, on the Q5 mean) saturated at a mean |residual| of 128 while the
   tracks measured run at 500-1500, so every sample landed in the last class
   and the context carried nothing; only the adaptation of the exponent
   probabilities tracked the level. */
static unsigned int res_class(const MMXLosslessResState *st)
{
    unsigned long m = (unsigned long)st->mean;
    unsigned int l = 0, cl;
    if (m < 2)
        return 0;
    while ((m >> l) > 1)
        l++;
    cl = l * 2 + (unsigned int)((m >> (l - 1)) & 1);
    if (cl >= MMX_LL_RES_CLASSES)
        cl = MMX_LL_RES_CLASSES - 1;
    return cl;
}

static void res_update(MMXLosslessResState *st, long long r)
{
    long long a = r < 0 ? -r : r;
    st->mean += ((a << MMX_LL_RES_RATE) - st->mean) >> MMX_LL_RES_RATE;
}

/* Signed residual. Layout of the MMX_LL_RES_CTX counters of a class:
   [0] zero flag, [1..20] exponent unary by position, [21..36] first mantissa
   bit by exponent, [37..68] second mantissa bit by exponent and first bit.
   The sign and the remaining mantissa bits are bypass coded: measured over
   1.76 M residuals of four tracks the sign carries 0.99997 bit and mantissa
   bit three 0.99720 bit, so a model for either can only pay for its own
   adaptation noise. The exponent is capped at RES_MAX_LEN on both sides (no
   terminator there; a longer value fails the encoder), which matters for
   32-bit sources whose residuals exceed 32 bits. */
#define RC_ZERO 0
#define RC_EXP 1
#define RC_BIT1 21
#define RC_BIT2 37
#define RES_MAX_LEN 62

static void enc_res(MMXRangeEncoder *rc, MMXCtr *ctx, long long r)
{
    unsigned long v;
    unsigned int len = 0, i, le, b1;
    if (r == 0)
    {
        mmx_rc_enc_ctr(rc, &ctx[RC_ZERO], 0);
        return;
    }
    mmx_rc_enc_ctr(rc, &ctx[RC_ZERO], 1);
    mmx_rc_enc_bypass(rc, r < 0 ? 1UL : 0UL, 1);
    v = (unsigned long)(r < 0 ? -r : r);
    while ((v >> len) > 1)
        len++;
    if (len > RES_MAX_LEN)
    {
        rc->failed = 1;
        return;
    }
    for (i = 0; i < len; i++)
        mmx_rc_enc_ctr(rc, &ctx[RC_EXP + (i < 19 ? i : 19)], 1);
    if (len < RES_MAX_LEN)
        mmx_rc_enc_ctr(rc, &ctx[RC_EXP + (len < 19 ? len : 19)], 0);
    if (len == 0)
        return;
    le = len < 16 ? len : 15;
    b1 = (v >> (len - 1)) & 1;
    mmx_rc_enc_ctr(rc, &ctx[RC_BIT1 + le], b1);
    if (len >= 2)
    {
        mmx_rc_enc_ctr(rc, &ctx[RC_BIT2 + 2 * le + b1], (unsigned int)((v >> (len - 2)) & 1));
        if (len >= 3)
            mmx_rc_enc_bypass(rc, v & ((1UL << (len - 2)) - 1), len - 2);
    }
}

static long long dec_res(MMXRangeDecoder *rc, MMXCtr *ctx)
{
    unsigned long v;
    unsigned int len = 0, neg, le, b1;
    if (!mmx_rc_dec_ctr(rc, &ctx[RC_ZERO]))
        return 0;
    neg = (unsigned int)mmx_rc_dec_bypass(rc, 1);
    while (len < RES_MAX_LEN && mmx_rc_dec_ctr(rc, &ctx[RC_EXP + (len < 19 ? len : 19)]))
        len++;
    v = 1UL << len;
    if (len >= 1)
    {
        le = len < 16 ? len : 15;
        b1 = mmx_rc_dec_ctr(rc, &ctx[RC_BIT1 + le]);
        v |= (unsigned long)b1 << (len - 1);
        if (len >= 2)
        {
            v |= (unsigned long)mmx_rc_dec_ctr(rc, &ctx[RC_BIT2 + 2 * le + b1]) << (len - 2);
            if (len >= 3)
                v |= mmx_rc_dec_bypass(rc, len - 2);
        }
    }
    return neg ? -(long long)v : (long long)v;
}

/* predictor residuals of s into r */
static void residuals(const long long *s, unsigned long n, const MMXLosslessPred *p, long long *r)
{
    unsigned long i;
    for (i = 0; i < n; i++)
        r[i] = s[i] - predict(s + i, p);
}

static void code_residuals(MMXRangeEncoder *rc, MMXCtr (*ctx)[MMX_LL_RES_CTX], MMXLosslessResState *st,
                           const long long *r, unsigned long n)
{
    unsigned long i;
    for (i = 0; i < n; i++)
    {
        enc_res(rc, ctx[res_class(st)], r[i]);
        res_update(st, r[i]);
    }
}

/* Decodes the residuals of one channel frame and rebuilds the coded signal s
   (s[-1..-HIST] valid) and the chain output e (e[-1..-HIST] valid): with
   `use` the predictor worked on e and s comes out of the inverse chain,
   otherwise on s and the chain follows s. */
static int decode_residuals(MMXRangeDecoder *rc, const Chain *ch, MMXCtr (*ctx)[MMX_LL_RES_CTX], MMXLosslessResState *st,
                            long long *s, long long *e, unsigned long n, const MMXLosslessPred *p, int use)
{
    unsigned long i;
    for (i = 0; i < n; i++)
    {
        long long r = dec_res(rc, ctx[res_class(st)]), sp, v, q = 0;
        int phi[XLS_N];
        res_update(st, r);
        sp = ch->src ? side_predict(ch->src, ch->us + i, (long long)(n - i)) : 0;
        if (ch->src2)
            sp += side_predict(ch->src2, ch->us2 + i, (long long)(n - i));
        if (ch->xls)
        {
            xls_phi(ch->xls, ch->xu, (long long)i, phi);
            q = xls_predict(ch->xls, phi);
        }
        if (use)
        {
            e[i] = r + predict(e + i, p);
            v = cascade_inverse(ch->cascade, e[i]) + q;
            s[i] = v + sp + s[(long long)i - 1];
        }
        else
        {
            s[i] = r + predict(s + i, p);
            v = s[i] - s[(long long)i - 1] - sp;
            e[i] = cascade_step(ch->cascade, v - q);
        }
        if (ch->xls)
            xls_update(ch->xls, phi, v);
        if (ch->src)
            side_update(ch->src, ch->us + i, (long long)(n - i), v);
        if (ch->src2)
            side_update(ch->src2, ch->us2 + i, (long long)(n - i), v);
        if (rc->failed)
            return -1;
    }
    return 0;
}

/* --------------------------------------------------------------- frames */

int mmx_ll_encode_frame(MMXRangeEncoder *rc, MMXLosslessContexts *ctx, unsigned int channels,
                        long long *const *x, long long *const *src, long long *const *src2, unsigned long count, MMXLosslessFrame *fr)
{
    long long *d[MMX_MAX_CH], *ms[2], *tmp, *r, *e, *js[MMX_MAX_CH], *us[MMX_MAX_CH], *js2[MMX_MAX_CH], *us2[MMX_MAX_CH], *xu = NULL;
    SigAnalysis *an, *coded[MMX_MAX_CH];    /* an[c]: d[c]; an[MMX_MAX_CH], an[MMX_MAX_CH + 1]: joint pair */
    double *w;
    unsigned int c;
    unsigned long i;
    int rcode = -1;
    static int no_nlms = -1;                /* MMX_NO_NLMS: never code the cascade output (measurements) */
    /* Stereo mode decision with the joint-stereo LS stage (revision 8). The stage and the cascade of
       channel 1 learn one mode's signals, and every switch throws that away: on 30 s excerpts the
       per-frame decision (title E 66 switches, title H 244) leaves most of the stage's gain on the
       table - stage with per-frame decision / fixed M/S / fixed L/R against revision 7: title E
       -0.97 / -1.14 / -0.93 %, title G -0.22 / -0.23 / -0.11 %, title H -0.03 / -0.06 / -0.26 %,
       title A 60-90 s -0.29 / -0.34 / -0.16 %. The encoder therefore decides on the estimated bits
       of each mode summed over the past with a decay of 2^-7 (about three seconds): 1.14 / 0.23 /
       0.26 / 0.34 %, the best fixed mode of every excerpt. Encoder only, the mode is still written
       per frame and the decoder needs no setting.
       MMX_LL_STEREO_SMOOTH=k sets the decay 2^-k (0: the plain per-frame decision of revision 7);
       MMX_LL_STEREO=lr|ms|ls|rs fixes the mode of every frame (trials). */
    static int force_stereo = -2;
    static int smooth_stereo = 7;

    tables_init();
    if (no_nlms < 0)
        no_nlms = mmx_lab_getenv("MMX_NO_NLMS") != NULL;
    if (force_stereo == -2)
    {
        const char *fs = mmx_lab_getenv("MMX_LL_STEREO");
        force_stereo = !fs ? -1 : !strcmp(fs, "lr") ? 0 : !strcmp(fs, "ms") ? 1 : !strcmp(fs, "ls") ? 2 : !strcmp(fs, "rs") ? 3 : -1;
        fs = mmx_lab_getenv("MMX_LL_STEREO_SMOOTH");
        if (fs)
            smooth_stereo = atoi(fs) < 0 ? 0 : atoi(fs) > 20 ? 20 : atoi(fs);
    }
    memset(fr, 0, sizeof(*fr));
    memset(d, 0, sizeof(d));
    memset(js, 0, sizeof(js));
    memset(us, 0, sizeof(us));
    memset(js2, 0, sizeof(js2));
    memset(us2, 0, sizeof(us2));
    ms[0] = ms[1] = NULL;
    tmp = sig_alloc(count);
    e = sig_alloc(count);
    r = (long long *)malloc(sizeof(long long) * (count ? count : 1));
    w = (double *)malloc(sizeof(double) * (count ? count : 1));
    an = (SigAnalysis *)malloc(sizeof(SigAnalysis) * (MMX_MAX_CH + 3));
    if (!tmp || !e || !r || !w || !an)
        goto done;
    for (c = 0; c < channels; c++)
    {
        d[c] = sig_alloc(count);
        js[c] = sig_alloc(count);
        us[c] = sig_alloc(count);
        js2[c] = sig_alloc(count);
        us2[c] = sig_alloc(count);
        if (!d[c] || !js[c] || !us[c] || !js2[c] || !us2[c])
            goto done;
        load_history(ctx, c, 0, 0, 0, 0, d[c]);
        memcpy(d[c], x[c], sizeof(long long) * count);
        analyze(d[c], count, w, &an[c]);
        coded[c] = &an[c];
        /* reference prediction with a broadband gain, kept only when it lowers the cost */
        if (src && src[c])
        {
            long long g = fit_gain(x[c], src[c], count);
            if (g != 0)
            {
                SigAnalysis *at = &an[MMX_MAX_CH + 2];
                load_history(ctx, c, 1, g, 0, 0, tmp);
                for (i = 0; i < count; i++)
                    tmp[i] = x[c][i] - apply_gain(src[c][i], g);
                analyze(tmp, count, w, at);
                if (est_bits(at, count) + 17.0 < est_bits(&an[c], count))
                {
                    memcpy(d[c] - HIST, tmp - HIST, sizeof(long long) * (count + HIST));
                    an[c] = *at;
                    fr->use_pred[c] = 1;
                    fr->gain[c] = g;
                }
            }
            /* the second source of a two-source block: its gain is fitted on what the first
               source leaves, and it stays only when it pays for its own side information */
            if (fr->use_pred[c] && src2 && src2[c])
            {
                long long g2 = fit_gain(d[c], src2[c], count);
                if (g2 != 0)
                {
                    SigAnalysis *at = &an[MMX_MAX_CH + 2];
                    load_history(ctx, c, 1, fr->gain[c], 1, g2, tmp);
                    for (i = 0; i < count; i++)
                        tmp[i] = d[c][i] - apply_gain(src2[c][i], g2);
                    analyze(tmp, count, w, at);
                    if (est_bits(at, count) + 17.0 < est_bits(&an[c], count))
                    {
                        memcpy(d[c] - HIST, tmp - HIST, sizeof(long long) * (count + HIST));
                        an[c] = *at;
                        fr->use_src2[c] = 1;
                        fr->gain2[c] = g2;
                    }
                }
            }
        }
    }

    /* stereo decision on the reference residuals: L/R, M/S, L/S or R/S */
    if (channels == 2)
    {
        double cl, cr, cm, cs, best;
        int mode;
        ms[0] = sig_alloc(count);
        ms[1] = sig_alloc(count);
        if (!ms[0] || !ms[1])
            goto done;
        joint_forward(1, ms[0], ms[1], d[0], d[1], -HIST, (long long)count);
        analyze(ms[0], count, w, &an[MMX_MAX_CH]);
        analyze(ms[1], count, w, &an[MMX_MAX_CH + 1]);
        cl = est_bits(&an[0], count);
        cr = est_bits(&an[1], count);
        cm = est_bits(&an[MMX_MAX_CH], count);
        cs = est_bits(&an[MMX_MAX_CH + 1], count);
        best = cl + cr; mode = 0;
        if (cm + cs < best) { best = cm + cs; mode = 1; }
        if (cl + cs < best) { best = cl + cs; mode = 2; }
        if (cr + cs < best) { best = cr + cs; mode = 3; }
        if (smooth_stereo > 0)   /* decayed sums per mode, stream state like the stage */
        {
            double c4[4], *acc = ctx->stereo_acc;
            int m;
            c4[0] = cl + cr; c4[1] = cm + cs; c4[2] = cl + cs; c4[3] = cr + cs;
            for (m = 0; m < 4; m++)
                acc[m] += c4[m] - acc[m] / (double)(1 << smooth_stereo);
            for (m = 1, mode = 0; m < 4; m++)
                if (acc[m] < acc[mode])
                    mode = m;
        }
        if (force_stereo >= 0)
            mode = force_stereo;
        fr->stereo_ms = mode;
        if (mode)
        {
            joint_forward(mode, ms[0], ms[1], d[0], d[1], -HIST, (long long)count);
            coded[0] = mode == 1 ? &an[MMX_MAX_CH] : &an[mode == 2 ? 0 : 1];
            coded[1] = &an[MMX_MAX_CH + 1];
        }
    }

    for (c = 0; c < channels; c++)
    {
        mmx_rc_enc_bit(rc, &ctx->use_pred[ctx->prev_used[c]], fr->use_pred[c]);
        if (fr->use_pred[c])
        {
            mmx_rc_enc_seg(rc, ctx->gain, fr->gain[c] - ctx->prev_gain[c]);
            ctx->prev_gain[c] = fr->gain[c];
            if (src2 && src2[c])            /* both sides know the block carries a second source */
            {
                mmx_rc_enc_bit(rc, &ctx->use_src2[ctx->prev_used2[c]], fr->use_src2[c]);
                if (fr->use_src2[c])
                {
                    mmx_rc_enc_seg(rc, ctx->gain2, fr->gain2[c] - ctx->prev_gain2[c]);
                    ctx->prev_gain2[c] = fr->gain2[c];
                }
                ctx->prev_used2[c] = fr->use_src2[c];
            }
        }
        ctx->prev_used[c] = fr->use_pred[c];
    }
    if (channels == 2)
        enc_tree(rc, ctx->stereo, (unsigned int)fr->stereo_ms, 2);
    if (src)
        source_signals(fr, channels, src, ctx->hist_src, count, js, us);
    if (src2)
        source_signals(fr, channels, src2, ctx->hist_src2, count, js2, us2);
    /* the LS stage of channel 1 reads channel 0's coded signal, all of the frame */
    if (channels == 2 && ctx->xls_on)
    {
        xu = sig_alloc(count + MMX_LL_XLS_FUT);
        if (!xu)
            goto done;
        xls_cross(fr->stereo_ms ? ms[0] : d[0], count, xu);
    }

    for (c = 0; c < channels; c++)
    {
        const long long *v = (fr->stereo_ms && c < 2) ? ms[c] : d[c];
        MMXLosslessPred *p = &fr->pred[c], pe;
        SigAnalysis *ae = &an[MMX_MAX_CH + 2];
        Chain ch;
        double cv, ce;
        unsigned int k;

        /* the chain output against the coded signal itself, each with its best predictor */
        for (k = 0; k < HIST; k++)
            e[(int)k - HIST] = ctx->hist_e[c][k];
        chain_setup(&ch, ctx, fr, channels, c, src, us, src2, us2, xu);
        chain_forward(&ch, v, e, count);
        analyze(e, count, w, ae);
        ce = choose_predictor(e, count, ae, &pe);
        cv = choose_predictor(v, count, coded[c], p);
        fr->nlms[c] = !no_nlms && ce < cv;
        if (fr->nlms[c])
            *p = pe;
        hist_push(ctx->hist_e[c], e, count);

        mmx_rc_enc_bit(rc, &ctx->nlms_flag[ctx->prev_nlms[c]], (unsigned int)fr->nlms[c]);
        ctx->prev_nlms[c] = fr->nlms[c];
        enc_pred(rc, ctx, c, p);
        residuals(fr->nlms[c] ? e : v, count, p, r);
        code_residuals(rc, ctx->res[c < 2 ? c : 1], &ctx->res_state[c], r, count);
        mmx_ll_stats.channel_frames++;
        if (p->lpc) { mmx_ll_stats.lpc_frames++; mmx_ll_stats.lpc_order_hist[p->order]++; }
        else mmx_ll_stats.fixed_frames++;
        if (fr->nlms[c]) mmx_ll_stats.nlms_frames++;
    }
    store_history(ctx, channels, x, src, src2, count);
    rcode = rc->failed ? -1 : 0;

done:
    for (c = 0; c < channels; c++) { sig_free(d[c]); sig_free(js[c]); sig_free(us[c]); sig_free(js2[c]); sig_free(us2[c]); }
    sig_free(ms[0]); sig_free(ms[1]); sig_free(tmp); sig_free(e); sig_free(xu);
    free(w); free(an); free(r);
    return rcode;
}

int mmx_ll_decode_frame(MMXRangeDecoder *rc, MMXLosslessContexts *ctx, unsigned int channels,
                        long long *const *x, long long *const *src, long long *const *src2, unsigned long count, MMXLosslessFrame *fr)
{
    long long *d[MMX_MAX_CH], *e, *js[MMX_MAX_CH], *us[MMX_MAX_CH], *js2[MMX_MAX_CH], *us2[MMX_MAX_CH], *xu = NULL;
    unsigned int c;
    unsigned long i;
    int rcode = -1;

    memset(fr, 0, sizeof(*fr));
    memset(d, 0, sizeof(d));
    memset(js, 0, sizeof(js));
    memset(us, 0, sizeof(us));
    memset(js2, 0, sizeof(js2));
    memset(us2, 0, sizeof(us2));
    e = sig_alloc(count);
    if (!e)
        goto done;
    if (channels == 2 && ctx->xls_on)
    {
        xu = sig_alloc(count + MMX_LL_XLS_FUT);
        if (!xu)
            goto done;
    }
    for (c = 0; c < channels; c++)
    {
        fr->use_pred[c] = (int)mmx_rc_dec_bit(rc, &ctx->use_pred[ctx->prev_used[c]]);
        if (fr->use_pred[c])
        {
            fr->gain[c] = ctx->prev_gain[c] + mmx_rc_dec_seg(rc, ctx->gain);
            ctx->prev_gain[c] = fr->gain[c];
            if (src2 && src2[c])
            {
                fr->use_src2[c] = (int)mmx_rc_dec_bit(rc, &ctx->use_src2[ctx->prev_used2[c]]);
                if (fr->use_src2[c])
                {
                    fr->gain2[c] = ctx->prev_gain2[c] + mmx_rc_dec_seg(rc, ctx->gain2);
                    ctx->prev_gain2[c] = fr->gain2[c];
                }
                ctx->prev_used2[c] = fr->use_src2[c];
            }
        }
        ctx->prev_used[c] = fr->use_pred[c];
        if (fr->use_pred[c] && (!src || !src[c]))
            goto done;
        if (fr->use_src2[c] && (!src2 || !src2[c]))
            goto done;
    }
    fr->stereo_ms = channels == 2 ? (int)dec_tree(rc, ctx->stereo, 2) : 0;

    for (c = 0; c < channels; c++)
    {
        d[c] = sig_alloc(count);
        js[c] = sig_alloc(count);
        us[c] = sig_alloc(count);
        js2[c] = sig_alloc(count);
        us2[c] = sig_alloc(count);
        if (!d[c] || !js[c] || !us[c] || !js2[c] || !us2[c])
            goto done;
        load_history(ctx, c, fr->use_pred[c], fr->gain[c], fr->use_src2[c], fr->gain2[c], d[c]);
    }
    if (fr->stereo_ms)
        joint_forward(fr->stereo_ms, d[0], d[1], d[0], d[1], -HIST, 0);
    if (src)
        source_signals(fr, channels, src, ctx->hist_src, count, js, us);
    if (src2)
        source_signals(fr, channels, src2, ctx->hist_src2, count, js2, us2);
    for (c = 0; c < channels; c++)
    {
        Chain ch;
        unsigned int k;
        fr->nlms[c] = (int)mmx_rc_dec_bit(rc, &ctx->nlms_flag[ctx->prev_nlms[c]]);
        ctx->prev_nlms[c] = fr->nlms[c];
        if (dec_pred(rc, ctx, c, &fr->pred[c]) != 0)
            goto done;
        for (k = 0; k < HIST; k++)
            e[(int)k - HIST] = ctx->hist_e[c][k];
        if (xu && c == 1)
            xls_cross(d[0], count, xu);   /* channel 0 is complete, still in the coded domain */
        chain_setup(&ch, ctx, fr, channels, c, src, us, src2, us2, xu);
        if (decode_residuals(rc, &ch, ctx->res[c < 2 ? c : 1], &ctx->res_state[c], d[c], e, count, &fr->pred[c], fr->nlms[c]) != 0)
            goto done;
        hist_push(ctx->hist_e[c], e, count);
    }
    if (fr->stereo_ms)
        joint_inverse(fr->stereo_ms, d[0], d[1], count);
    for (c = 0; c < channels; c++)
        for (i = 0; i < count; i++)
            x[c][i] = d[c][i] + (fr->use_pred[c] ? apply_gain(src[c][i], fr->gain[c]) : 0)
                              + (fr->use_src2[c] ? apply_gain(src2[c][i], fr->gain2[c]) : 0);
    store_history(ctx, channels, x, src, src2, count);
    rcode = rc->failed ? -1 : 0;

done:
    for (c = 0; c < channels; c++) { sig_free(d[c]); sig_free(js[c]); sig_free(us[c]); sig_free(js2[c]); sig_free(us2[c]); }
    sig_free(e); sig_free(xu);
    return rcode;
}

double mmx_ll_estimate_bits(const float *coefs, unsigned long m, unsigned int source_bits)
{
    /* Rate of an ideal predictive coder: mean over bins of 0.5*log2(12 * S_k / 512),
       S_k = coefficient power in integer units (white noise of variance v gives
       coefficient power 512 v). One coefficient per sample. */
    double scale = pow(2.0, 2.0 * (double)(source_bits ? source_bits - 1 : 15)) * 12.0 / 512.0;
    double bits = 0.0;
    unsigned long k;
    for (k = 0; k < m; k++)
    {
        double p = (double)coefs[k] * coefs[k] * scale;
        if (p > 1.0)
            bits += 0.5 * log(p) / log(2.0);
    }
    return bits + 0.3 * (double)m;
}
