#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "lab.h"
#include "ll2.h"

#ifdef _MSC_VER
#pragma fp_contract(off)
#else
#pragma STDC FP_CONTRACT OFF
#endif

static int ll2_dsolve(void)   /* experiment switch MMX_LL2_ISOLVE=1: the fixed-point solve */
{
    static int v = -1;
    if (v < 0) { const char *e = mmx_lab_getenv("MMX_LL2_ISOLVE"); v = e ? !atoi(e) : 1; }
    return v;
}

#define DECAY 9                         /* R -= R >> 9: lambda = 1 - 2^-9 = 0.998 */
#define LQ 20                           /* L of the LDL^T in Q20 */
#define WQ 16                           /* OLS weights in Q16 */
#define W_MAX (1LL << 22)
#define IN_MAX (1LL << 24)              /* regressor clamp: 24 bits (the hi-res profile); 16-bit signals never reach 2^20 */
#define REG 14                          /* ridge: trace >> 14 */
#define PROJ 3                          /* projection towards the blended prediction: 1/8 */
#define MIX_MU 328                      /* 0.005 in Q16 */
#define MIX_RATE 4                      /* running means adapt by 1/16 */

/* Tunable parameters (step 4, encoder search). Defaults are the measured design; MMX_LL2_P="decay,proj,mixmu,mixrate,
   srcmu_shift,erate" overrides them (experiments and the encoder search only, until the header carries them). */
typedef struct { int decay, proj, mixmu, mixrate, srcmu, erate; } Ll2Knobs;
static Ll2Knobs K = { 9, 3, 328, 4, 12, 1 };
static int knobs_read = 0;
static void knobs_init(void)
{
    const char *e;
    if (knobs_read) return;
    if ((e = mmx_lab_getenv("MMX_LL2_P")) != NULL)
        sscanf(e, "%d,%d,%d,%d,%d,%d", &K.decay, &K.proj, &K.mixmu, &K.mixrate, &K.srcmu, &K.erate);
    knobs_read = 1;
}

/* lab switches read from the environment once (ridge, side-tap loading, IRLS off) */
static double g_reg = -1.0, g_regd = -1.0;
static int g_noirls = -1;
static void env_statics(void)
{
    if (g_reg < 0.0) { const char *e = mmx_lab_getenv("MMX_LL2_REG"); g_reg = e ? atof(e) : 1e-10; }
    if (g_regd < 0.0) { const char *e = mmx_lab_getenv("MMX_LL2_REGD"); g_regd = e ? atof(e) : 0.0; }
    if (g_noirls < 0) { const char *e = mmx_lab_getenv("MMX_LL2_NOIRLS"); g_noirls = e ? atoi(e) : 0; }
}

void mmx_ll2_statics(void)
{
    knobs_init();
    (void)ll2_dsolve();
    env_statics();
}

static long long clampl(long long v, long long m)
{
    return v > m ? m : v < -m ? -m : v;
}

/* a * b modulo 2^64 (two's complement): the CD profile's frozen mixer arithmetic. Its unbounded normalised-LMS weights
   can overflow these products on pathological input (near silence into a full-scale onset - UBSan on the synthetic
   loop vector); every machine we build for wrapped them, so the explicit wrap keeps the bits and defines them. */
static long long wrapmul(long long a, long long b)
{
    return (long long)((unsigned long long)a * (unsigned long long)b);
}

/* a * b, exact whenever |a * b| < 4e18, else saturated at +-2^62 (hi-res profile only, see mmx_ll2_predict): the
   mixer's products stay far below that at 24 bits (the fast path), only 32-bit sources with scaled stages reach it */
static long long mulsat(long long a, long long b)
{
    const long long lim = 1LL << 31;
    double p;
    if (a > -lim && a < lim && b > -lim && b < lim) return a * b;
    p = (double)a * (double)b;
    if (p > 4.0e18) return 1LL << 62;
    if (p < -4.0e18) return -(1LL << 62);
    return a * b;
}

/* IRLS weight (esum + 2)^-0.75 in Q12 from the Q4 running mean |error|: indexed by 8 log2 of (esum/16 + 2),
   table entry max(1, round(4096 * 2^(-0.75 * k / 8))), k = 8 log2((esum + 2) / 2). */
static const unsigned short irls_tab[] = {
    4096, 3838, 3597, 3371, 3158, 2960, 2774, 2599, 2435, 2282, 2139, 2004, 1878, 1760, 1649, 1545,
    1448, 1357, 1272, 1192, 1117, 1046, 981, 919, 861, 807, 756, 709, 664, 622, 583, 546,
    512, 480, 450, 421, 395, 370, 347, 325, 304, 285, 267, 251, 235, 220, 206, 193,
    181, 170, 159, 149, 140, 131, 123, 115, 108, 101, 95, 89, 83, 78, 73, 68,
    64, 60, 56, 53, 49, 46, 43, 41, 38, 36, 33, 31, 29, 27, 26, 24,
    23, 21, 20, 19, 17, 16, 15, 14, 13, 13, 12, 11, 10, 10, 9, 9,
    8, 7, 7, 7, 6, 6, 5, 5, 5, 4, 4, 4, 4, 3, 3, 3,
    3, 3, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1
};

static long long irls_weight(long long esum_q4)
{
    unsigned long long v = (unsigned long long)(esum_q4 + 32);   /* (esum/16 + 2) in Q4 */
    unsigned int l = 0, k;
    while ((v >> l) > 1) l++;                                     /* floor(log2 v) */
    /* 8 * log2(v / 16): integer part (l - 4) * 8 plus three fraction bits of the mantissa */
    k = (l >= 4 ? (l - 4) * 8 : 0) + (unsigned int)(l >= 3 ? ((v >> (l - 3)) & 7) : 0);
    if (k >= sizeof(irls_tab) / sizeof(irls_tab[0])) k = sizeof(irls_tab) / sizeof(irls_tab[0]) - 1;
    k = k >= 8 ? k - 8 : 0;                                       /* (esum + 2) with esum = 0 is 2^1: start the table there */
    return irls_tab[k];
}

/* Fixed-point LDL^T solve of (R + reg I) w = r, as in the joint-stereo stage (src/lossless.c xls_solve). */
static void ols_solve(MMXLl2Ols *o)
{
    long long A[MMX_LL2_MAX_N][MMX_LL2_MAX_N], L[MMX_LL2_MAX_N][MMX_LL2_MAX_N], D[MMX_LL2_MAX_N], V[MMX_LL2_MAX_N],
        z[MMX_LL2_MAX_N], b[MMX_LL2_MAX_N];
    long long mx = 1, tr = 0, reg;
    int n = o->n, i, j, k, sh = 0;
    for (i = 0; i < n; i++)
    {
        if (o->R[i][i] > mx) mx = o->R[i][i];
        tr += o->R[i][i];
    }
    while ((mx >> sh) > (1LL << 28))
        sh++;
    reg = ((tr >> sh) >> REG) + 1;
    for (i = 0; i < n; i++)
    {
        for (j = 0; j <= i; j++)
            A[i][j] = o->R[i][j] >> sh;
        A[i][i] += reg;
        b[i] = clampl(o->r[i] >> sh, 1LL << 31);
    }
    for (j = 0; j < n; j++)
    {
        long long d = A[j][j];
        for (k = 0; k < j; k++)
        {
            V[k] = clampl((L[j][k] * D[k]) >> LQ, 1LL << 31);
            d -= (L[j][k] * V[k]) >> LQ;
        }
        if (d < 1)
            d = 1;
        D[j] = d;
        L[j][j] = 1LL << LQ;
        for (i = j + 1; i < n; i++)
        {
            long long t = A[i][j];
            for (k = 0; k < j; k++)
                t -= (L[i][k] * V[k]) >> LQ;
            t = clampl(t, 1LL << 38);
            L[i][j] = clampl((t * (1LL << LQ)) / d, 1LL << 29);
        }
    }
    for (i = 0; i < n; i++)
    {
        long long t = b[i];
        for (k = 0; k < i; k++)
            t -= (L[i][k] * z[k]) >> LQ;
        z[i] = clampl(t, 1LL << 32);
    }
    for (i = n - 1; i >= 0; i--)
    {
        long long t = (z[i] * (1LL << WQ)) / D[i];
        for (k = i + 1; k < n; k++)
            t -= (L[k][i] * o->w[k]) >> LQ;
        o->w[i] = clampl(t, W_MAX);
    }
}

/* Double-precision LDL^T solve of (R + reg I) w = r. Only +, -, * and / in a fixed order, no library calls and no
   contraction (see the pragma at the top of the file): IEEE-754 double gives the same bits on every machine. The
   integer R and r are exact; the weights leave as Q16 integers, so the prediction itself stays integer. */
static double rget(const MMXLl2Ols *o, int i, int j) { return o->dcov ? o->Rd[i][j] : (double)o->R[i][j]; }

static void ols_solve_d(MMXLl2Ols *o)
{
    double A[MMX_LL2_MAX_N][MMX_LL2_MAX_N], D[MMX_LL2_MAX_N], z[MMX_LL2_MAX_N], sc = 1.0, tr = 0.0, reg;
    int n = o->n, i, j, k, live[MMX_LL2_MAX_N];
    for (i = 0; i < n; i++) tr += rget(o, i, i);
    if (tr <= 0.0) return;
    sc = (double)n / tr;                                  /* normalise: mean diagonal 1 */
    env_statics();
    reg = g_reg;                                          /* ridge relative to the mean diagonal */
    {
        const double rd = g_regd;
        for (i = 0; i < n; i++)
        {
            for (j = 0; j <= i; j++) A[i][j] = rget(o, i, j) * sc;
            A[i][i] += reg + (i >= o->n_plain ? rd * A[i][i] : 0.0);   /* side taps: loading relative to their own energy */
        }
    }
    for (j = 0; j < n; j++)
    {
        double d = A[j][j];
        int dead = rget(o, j, j) * sc < 1e-9;       /* an input that has been (almost) zero for the whole memory */
        for (k = 0; k < j; k++) d -= A[j][k] * A[j][k] * D[k];
        if (dead)
        {   /* take it out of the system: unit pivot, no coupling, weight 0 below */
            D[j] = 1.0;
            for (i = j + 1; i < n; i++) A[i][j] = 0.0;
            for (k = 0; k < j; k++) A[j][k] = 0.0;
            live[j] = 0;
            continue;
        }
        live[j] = 1;
        if (d < 1e-12)
        {
            if (o->keep_on_fail) return;              /* untrustworthy: keep the previous weights */
            d = 1e-12;
        }
        D[j] = d;
        for (i = j + 1; i < n; i++)
        {
            double t = A[i][j];
            for (k = 0; k < j; k++) t -= A[i][k] * A[j][k] * D[k];
            A[i][j] = t / d;
        }
    }
    for (i = 0; i < n; i++)
    {
        double t = live[i] ? (o->dcov ? o->rd[i] : (double)o->r[i]) * sc : 0.0;
        for (k = 0; k < i; k++) t -= A[i][k] * z[k];
        z[i] = t;
    }
    for (i = 0; i < n; i++) z[i] /= D[i];
    for (i = n - 1; i >= 0; i--)
    {
        double t = z[i];
        for (k = i + 1; k < n; k++) t -= A[k][i] * z[k];
        z[i] = t;
    }
    if (o->keep_on_fail)
        for (i = 0; i < n; i++)
            if (!(z[i] * 65536.0 <= (double)W_MAX && z[i] * 65536.0 >= -(double)W_MAX)) return;   /* NaN or beyond the clamp */
    for (i = 0; i < n; i++)
    {
        double wq = z[i] * 65536.0;
        long long q;
        if (wq > (double)W_MAX) wq = (double)W_MAX;
        if (wq < -(double)W_MAX) wq = -(double)W_MAX;
        /* A NaN (an ill-conditioned moment: a clamped pivot overflows to inf - inf) must become 0 explicitly: converting
           it is undefined in C and differs by machine (ARM gives 0, x86 INT64_MIN). 0 is what the ARM encoders wrote, so
           every existing file keeps its bits. Found by the MSVC decoder check in CI. */
        if (wq != wq) q = 0;
        else q = (long long)(wq >= 0.0 ? wq + 0.5 : wq - 0.5);  /* round half away from zero, no libm */
        o->w[i] = q;
        if (o->dpred) o->wd[i] = z[i];
    }
}

/* The prediction sum(w x) >> 20 over the window was computed by the previous update (0 before the first). */
static long long norm_predict(MMXLl2Norm *f)
{
    f->pred = f->pnext;
    return f->pred;
}

/* adapts towards the target v (the stage's input stream), pushes v into the history and computes the next prediction
   in the same pass over the weights: the next window is v followed by the current window without its oldest sample,
   so tap i of the next prediction is the updated w[i] times x[i-1] (tap 0: v). Integer sums, so the result is the
   same as a separate pass. */
/* sum of w[i] x[i] over n (a multiple of 8) taps in 8 lanes, combined in a fixed tree: vectorizes, and IEEE
   single precision without contraction (pragma at the top) gives the same bits on every machine */
static double lanes8_dot(const float *w, const float *x, int n)
{
    float a[8] = { 0, 0, 0, 0, 0, 0, 0, 0 };
    int i, k;
    for (i = 0; i < n; i += 8)
        for (k = 0; k < 8; k++) a[k] += w[i + k] * x[i + k];
    return (double)(((a[0] + a[1]) + (a[2] + a[3])) + ((a[4] + a[5]) + (a[6] + a[7])));
}

static long long round_pred(double s)
{
    if (!(s < 1099511627776.0 && s > -1099511627776.0)) return 0;   /* beyond 2^40 (never for real signals) or NaN */
    return (long long)(s >= 0.0 ? s + 0.5 : s - 0.5);
}

static void norm_adapt_f(float *restrict w, const float *restrict x, int n, float g)
{
    int i;
    for (i = 0; i < n; i++) w[i] += g * x[i];
}

/* float mode: push v, adapt w towards it on the old window (x_old[i] = xn[i + 1]), then sum the next prediction on
   the new window xn (two vectorized passes; w stays in the first-level cache) */
static void norm_update_f(MMXLl2Norm *f, long long v)
{
    const int n = f->n;
    long long e = v - f->pred;
    int xin = (int)clampl(v, 32767), xout;
    float *w = f->wf, *xn;
    if (f->fpos == 0)
    {
        memmove(f->xf + n + 1, f->xf, sizeof(float) * (size_t)n);
        f->fpos = n + 1;
    }
    f->fpos--;
    xn = f->xf + f->fpos;
    xn[0] = (float)xin;
    xout = (int)xn[n];
    if (f->pow > 0 && e != 0)
        norm_adapt_f(w, xn + 1, n, (float)((double)clampl(e, 1LL << 20) / (double)(f->pow + 1) / (double)(1LL << f->mu_shift)));
    f->pnext = round_pred(lanes8_dot(w, xn, n));
    f->pow += (long long)xin * xin - (long long)xout * xout;
}

static void norm_update(MMXLl2Norm *f, long long v)
{
    int *x = f->x + f->pos, i, xin, xout, n = f->n;
    long long e = v - f->pred, s;
    long long *w = f->w;
    if (f->fmode) { norm_update_f(f, v); return; }
    xin = (int)clampl(v, 32767);
    if (f->pow > 0 && e != 0)
    {
        /* g = mu * e / pow in Q40 (mu = 2^-mu_shift); w (Q20) += (g * x) >> 20 */
        long long g = (clampl(e, 1LL << 20) * (1LL << (40 - f->mu_shift))) / (f->pow + 1);
        w[0] += (g * x[0]) >> 20;
        s = w[0] * xin;
        for (i = 1; i < n; i++)
        {
            w[i] += (g * x[i]) >> 20;
            s += w[i] * x[i - 1];
        }
    }
    else
    {
        s = w[0] * xin;
        for (i = 1; i < n; i++) s += w[i] * x[i - 1];
    }
    f->pnext = (s + (1LL << 19)) >> 20;
    xout = x[f->n - 1];
    f->pow += (long long)xin * xin - (long long)xout * xout;
    if (f->pos == 0)
    {
        memmove(f->x + f->n + 1, f->x, sizeof(int) * (size_t)(f->n - 1));
        f->pos = f->n + 1;
    }
    f->pos--;
    f->x[f->pos] = xin;
}

int mmx_ll2_set_norm_stage(MMXLl2Chan *c, unsigned int s, int n, int mu_shift)
{
    MMXLl2Norm *f;
    if (s >= c->nst || n < 1 || n > MMX_LL2_NLMS_MAX) return -1;
    f = (MMXLl2Norm *)calloc(1, sizeof(MMXLl2Norm));
    if (!f) return -1;
    f->n = n;
    f->mu_shift = mu_shift;
    c->norm[s] = f;
    return 0;
}

void mmx_ll2_set_cov_shift(MMXLl2Chan *c, int cshift)
{
    unsigned int k;
    c->ols.cshift = cshift;
    for (k = 0; k < c->nxols; k++) c->olsx[k]->cshift = cshift;
}

void mmx_ll2_set_stage_target(MMXLl2Chan *c, int ltarget) { c->ltarget = ltarget; }

void mmx_ll2_set_dcov(MMXLl2Chan *c, int on)
{
    unsigned int k;
    c->ols.dcov = on != 0;
    for (k = 0; k < c->nxols; k++) c->olsx[k]->dcov = on != 0;
}

void mmx_ll2_set_solve_every(MMXLl2Chan *c, int n)
{
    unsigned int k;
    c->ols.solve_every = n;
    for (k = 0; k < c->nxols; k++) c->olsx[k]->solve_every = n;
}

void mmx_ll2_set_float_norm(MMXLl2Chan *c, int on)
{
    unsigned int s;
    for (s = 0; s < c->nst; s++)
        if (c->norm[s] && c->norm[s]->n % 8 == 0)
        {
            c->norm[s]->fmode = on != 0;
            c->norm[s]->fpos = c->norm[s]->n + 1;
        }
}

/* the stage inputs move to 2^-d times their scale: histories, running sums and the pending prediction follow,
   the weights stay (integer operations, the same on both sides) */
static void norm_rescale(MMXLl2Norm *f, int d)
{
    int i;
    long long s = 0;
    if (f->fmode)
    {
        for (i = 0; i < 2 * MMX_LL2_NLMS_MAX + 1; i++)
        {
            int v = (int)f->xf[i];
            f->xf[i] = (float)(d > 0 ? (v + (1 << (d - 1))) >> d : (int)clampl((long long)v * (1LL << -d), 32767));
        }
        f->pow = 0;
        for (i = 0; i < f->n; i++) f->pow += (long long)f->xf[f->fpos + i] * (long long)f->xf[f->fpos + i];
        f->pnext = round_pred(lanes8_dot(f->wf, f->xf + f->fpos, f->n));
        return;
    }
    for (i = 0; i < 2 * MMX_LL2_NLMS_MAX; i++)
        f->x[i] = d > 0 ? (f->x[i] + (1 << (d - 1))) >> d : (int)clampl((long long)f->x[i] * (1LL << -d), 32767);
    f->pow = 0;
    for (i = 0; i < f->n; i++) f->pow += (long long)f->x[f->pos + i] * f->x[f->pos + i];
    for (i = 0; i < f->n; i++) s += f->w[i] * f->x[f->pos + i];
    f->pnext = (s + (1LL << 19)) >> 20;
}

#define LSTAGE_EVERY 4096               /* samples between two decisions on the stage scale */
static void stage_scale(MMXLl2Chan *c, long long r)
{
    long long a = r < 0 ? -r : r, m;
    unsigned int s;
    int l;
    if (a > (1LL << 40)) a = 1LL << 40;
    c->lmean += ((a << 4) - c->lmean) >> 10;
    if (++c->lcount < LSTAGE_EVERY) return;
    c->lcount = 0;
    m = c->lmean >> 4;
    l = c->lshift;
    while (l < 16 && (m >> l) >= (1LL << (c->ltarget + 1))) l++;   /* hysteresis: a factor 4 band around 2^ltarget */
    while (l > 0 && (m >> l) < (1LL << (c->ltarget - 1))) l--;
    if (l == c->lshift) return;
    for (s = 0; s < c->nst; s++)
        if (c->norm[s]) norm_rescale(c->norm[s], l - c->lshift); else mmx_nlms_rescale(&c->st[s], l - c->lshift);
    c->lshift = l;
}

void mmx_ll2_set_dpred(MMXLl2Chan *c, int on)
{
    unsigned int k;
    c->ols.dpred = on;
    for (k = 0; k < c->nxols; k++) c->olsx[k]->dpred = on;
}

void mmx_ll2_set_keep_on_fail(MMXLl2Chan *c, int on)
{
    unsigned int k;
    c->ols.keep_on_fail = on;
    for (k = 0; k < c->nxols; k++) c->olsx[k]->keep_on_fail = on;
}

void mmx_ll2_set_experts4(MMXLl2Chan *c, long long d1, long long d2)
{
    MMXLl2Mix *m = &c->mix;
    int e, k;
    m->ne = 4;
    for (e = 2; e < 4; e++)
    {
        for (k = 0; k < m->n; k++) m->w[e][k] = m->w[0][k];
        m->score[e] = m->score[0];
    }
    for (e = 0; e < 4; e++) m->bw[e] = (1LL << 16) / 4;
    m->delta[0] = 0; m->delta[1] = d1; m->delta[2] = d2; m->delta[3] = 1LL << 40;
}

int mmx_ll2_add_ols(MMXLl2Chan *c, unsigned int decay)
{
    MMXLl2Ols *o;
    if (c->nxols >= MMX_LL2_XOLS || decay < 6 || decay > 16) return -1;
    o = (MMXLl2Ols *)calloc(1, sizeof(MMXLl2Ols));
    if (!o) return -1;
    o->n = c->ols.n;
    o->n_plain = c->ols.n_plain;
    o->decay = (int)decay;
    o->cshift = c->ols.cshift;
    o->keep_on_fail = c->ols.keep_on_fail;
    o->dpred = c->ols.dpred;
    c->olsx[c->nxols++] = o;
    c->mix.n = (int)(c->nst + c->nxols);
    return 0;
}

void mmx_ll2_free(MMXLl2Chan *c)
{
    unsigned int s;
    for (s = 0; s < MMX_LL2_MAX_STAGES; s++) { free(c->norm[s]); c->norm[s] = NULL; }
    for (s = 0; s < MMX_LL2_XOLS; s++) { free(c->olsx[s]); c->olsx[s] = NULL; }
    c->nxols = 0;
}

void mmx_ll2_init(MMXLl2Chan *c, unsigned int n_own, unsigned int n_cross, unsigned int n_side,
                  const unsigned int *orders, unsigned int nst)
{
    static const unsigned int sh[MMX_LL2_MAX_STAGES] = { 15, 13, 11, 11, 11, 11 };
    static const unsigned int st[MMX_LL2_MAX_STAGES] = { 8, 7, 7, 7, 7, 7 };
    unsigned int s;
    knobs_init();
    memset(c, 0, sizeof(*c));
    if (n_own > MMX_LL2_OWN_MAX) n_own = MMX_LL2_OWN_MAX;
    if (n_own + n_cross + n_side > MMX_LL2_MAX_N) n_side = MMX_LL2_MAX_N - n_own - n_cross;
    if (nst > MMX_LL2_MAX_STAGES) nst = MMX_LL2_MAX_STAGES;
    c->n_own = n_own; c->n_cross = n_cross; c->n_side = n_side; c->nst = nst;
    c->ols.n = (int)(n_own + n_cross + n_side);
    c->ols.n_plain = (int)(n_own + n_cross);
    c->ols.decay = K.decay;
    c->ols.cshift = 8;
    for (s = 0; s < nst; s++)
        mmx_nlms_init(&c->st[s], orders[s], orders[s] >= 1024 ? sh[0] : orders[s] >= 256 ? sh[1] : orders[s] >= 32 ? sh[2] : sh[3],
                      orders[s] >= 1024 ? st[0] : st[1]);
    c->mix.n = (int)nst;
    for (s = 0; s < nst; s++) c->mix.w[0][s] = c->mix.w[1][s] = (1LL << 16) / (long long)(nst ? nst : 1);
    c->mix.ne = 2;
    c->mix.bw0 = 1LL << 15;
    c->mix.score[0] = c->mix.score[1] = 1LL << 16;
}

/* the OLS prediction with double weights: fixed order, no contraction (pragma at the top), rounded half away from
   zero; out of range (never for real signals) predicts 0 */
static long long dpred_dot(const double *w, const int *phi, unsigned int n)
{
    double s = 0.0;
    unsigned int k;
    for (k = 0; k < n; k++) s += w[k] * (double)phi[k];
    if (!(s < 4.0e18 && s > -4.0e18)) return 0;
    return (long long)(s >= 0.0 ? s + 0.5 : s - 0.5);
}

long long mmx_ll2_predict(MMXLl2Chan *c, const long long *cross, const long long *side)
{
    MMXLl2Ols *o = &c->ols;
    MMXLl2Mix *m = &c->mix;
    long long p = 0;
    unsigned int k, n = 0;
    int e;
    for (k = 0; k < c->n_own; k++) c->phi[n++] = (int)clampl(c->own[k], IN_MAX);
    for (k = 0; k < c->n_cross; k++) c->phi[n++] = (int)clampl(cross[k], IN_MAX);
    for (k = 0; k < c->n_side; k++) c->phi[n++] = (int)clampl(side[k], IN_MAX);
    for (k = 0; k < n; k++) p += o->w[k] * c->phi[k];
    c->p_ols = o->dpred ? dpred_dot(o->wd, c->phi, n) : (p + (1LL << (WQ - 1))) >> WQ;
    for (k = 0; k < c->nst; k++)
    {
        c->p_raw[k] = c->norm[k] ? norm_predict(c->norm[k]) : mmx_nlms_predict(&c->st[k]);
        /* scaled stages (hi-res profile): bounded so the mixer's sums stay in 64 bits even for 32-bit sources; the CD
           profile keeps the unbounded arithmetic of its frozen format */
        c->p_st[k] = c->ltarget ? clampl(c->p_raw[k] * (1LL << c->lshift), 1LL << 31) : c->p_raw[k];
    }
    for (k = 0; k < c->nst; k++) m->x[k] = c->p_st[k];
    for (k = 0; k < c->nxols; k++)
    {
        const MMXLl2Ols *ox = c->olsx[k];
        long long px = 0;
        unsigned int i;
        for (i = 0; i < n; i++) px += ox->w[i] * c->phi[i];
        c->p_olsx[k] = ox->dpred ? dpred_dot(ox->wd, c->phi, n) : (px + (1LL << (WQ - 1))) >> WQ;
        m->x[c->nst + k] = c->p_olsx[k] - c->p_ols;
    }
    for (e = 0; e < m->ne; e++)
    {
        long long s = 0;
        for (k = 0; k < (unsigned int)m->n; k++) s += m->w[e][k] * m->x[k];
        m->ep[e] = s;
    }
    if (m->ne == 2) m->pred = (m->bw0 * (m->ep[0] >> 8) + ((1LL << 16) - m->bw0) * (m->ep[1] >> 8)) >> 8;
    else
    {
        long long pp = 0;
        for (e = 0; e < m->ne; e++) pp += m->bw[e] * (m->ep[e] >> 8);
        m->pred = pp >> 8;
    }
    c->pred = c->p_ols + ((m->pred + (1LL << 15)) >> 16);
    return c->pred;
}

static long long predict_src_core(MMXLl2Chan *c, const long long *src, const long long *so, int is_ch1, unsigned int nfut,
                                  int new_block);

long long mmx_ll2_predict_src(MMXLl2Chan *c, const long long *src, int new_block)
{
    return predict_src_core(c, src, NULL, 0, 0, new_block);
}

long long mmx_ll2_predict_src2(MMXLl2Chan *c, const long long *src, const long long *src_other, int is_ch1,
                               unsigned int n_future, int new_block)
{
    return predict_src_core(c, src, src_other, is_ch1, n_future, new_block);
}

static long long predict_src_core(MMXLl2Chan *c, const long long *src, const long long *so, int is_ch1, unsigned int nfut,
                                  int new_block)
{
    MMXLl2Src *q = &c->src;
    long long p = 0;
    int k, i;
    if (!src)
    {
        q->active = 0;
        q->pred = 0;
        return c->pred;
    }
    if (new_block || !q->active)
    {
        memset(q->w, 0, sizeof(q->w));
        q->w[MMX_LL2_SRC_K] = 1LL << 16;                 /* prior: the residual repeats */
        q->pw = 0;
        q->e_on = q->e_off = 0;                          /* gate open until the stage proves worse */
    }
    q->active = 1;
    /* whiten the source with the current own-tap weights of the OLS (the part of the source the OLS would have
       predicted is removed, what stays corresponds to the target's OLS residual) */
    for (k = -MMX_LL2_SRC_K; k <= MMX_LL2_SRC_K; k++)
    {
        long long acc = 0, v;
        for (i = 1; i <= (int)c->n_own; i++) acc += c->ols.w[i - 1] * src[k - i];
        if (so)
        {   /* the cross taps as mmx_ll2_predict saw them: the other channel at t-1-i (channel 0) or t-i (channel 1),
               then channel 1's future taps t+1..t+nfut */
            unsigned int nx = c->n_cross - (is_ch1 ? nfut : 0), j;
            for (j = 0; j < nx; j++) acc += c->ols.w[c->n_own + j] * so[k - (is_ch1 ? 0 : 1) - (int)j];
            if (is_ch1)
                for (j = 1; j <= nfut; j++) acc += c->ols.w[c->n_own + nx + j - 1] * so[k + (int)j];
        }
        v = src[k] - ((acc + (1LL << 15)) >> 16);
        q->x[k + MMX_LL2_SRC_K] = clampl(v, 1LL << 20);
    }
    for (k = 0; k < MMX_LL2_SRC_N; k++) p += q->w[k] * q->x[k];
    q->pred = (p + (1LL << 15)) >> 16;
    q->use = q->e_on <= q->e_off;
    if (q->use) c->pred += q->pred;
    return c->pred;
}

/* The OLS (main and extras, same regressor): each one's running mean |err| sets the IRLS weight cw of this sample in
   its exponentially forgetting covariance; the product phi_i phi_j is formed once for all of them (cw * (phi_i phi_j)
   is the same integer as (phi_i cw) phi_j). The weights are solved every MMX_LL2_SOLVE samples. */
/* double covariance (hi-res profile): R = (R - R 2^-decay) + c (phi_i phi_j), c = IRLS weight 2^-cshift; the row
   loops vectorize, IEEE double without contraction gives the same bits on every machine */
static void cov_row_d(double *restrict R, const double *restrict ph, int len, double pi, double c, double dec)
{
    int j;
    for (j = 0; j < len; j++) R[j] = (R[j] - R[j] * dec) + c * (pi * ph[j]);
}

static void ols_learn_d(MMXLl2Ols *const *ols, const long long *err, unsigned int m, const int *phi, long long y)
{
    double ph[MMX_LL2_MAX_N], c[1 + MMX_LL2_XOLS], dec[1 + MMX_LL2_XOLS], yd = (double)y;
    unsigned int k;
    int i, n = ols[0]->n;
    for (k = 0; k < m; k++)
    {
        MMXLl2Ols *o = ols[k];
        long long ae = err[k] < 0 ? -err[k] : err[k];
        o->esum += ((ae << 4) - o->esum) >> K.erate;
        c[k] = (double)(g_noirls ? 4096 : irls_weight(o->esum)) / (double)(1LL << o->cshift);
        dec[k] = 1.0 / (double)(1LL << o->decay);
    }
    for (i = 0; i < n; i++) ph[i] = (double)phi[i];
    for (k = 0; k < m; k++)
    {
        MMXLl2Ols *o = ols[k];
        for (i = 0; i < n; i++)
        {
            cov_row_d(o->Rd[i], ph, i + 1, ph[i], c[k], dec[k]);
            o->rd[i] = (o->rd[i] - o->rd[i] * dec[k]) + c[k] * (ph[i] * yd);
        }
        if (++o->phase >= (o->solve_every ? o->solve_every : MMX_LL2_SOLVE))
        {
            o->phase = 0;
            ols_solve_d(o);
        }
    }
}

static void ols_learn(MMXLl2Ols *const *ols, const long long *err, unsigned int m, const int *phi, long long y)
{
    long long cw[1 + MMX_LL2_XOLS];
    unsigned int k;
    int i, j, n = ols[0]->n;
    int nw;
    env_statics();
    if (ols[0]->dcov) { ols_learn_d(ols, err, m, phi, y); return; }
    nw = g_noirls;
    for (k = 0; k < m; k++)
    {
        MMXLl2Ols *o = ols[k];
        long long ae = err[k] < 0 ? -err[k] : err[k];
        o->esum += ((ae << 4) - o->esum) >> K.erate;                               /* fast mean |error|, Q4 (sac: 0.6/0.4) */
        cw[k] = nw ? 4096 : irls_weight(o->esum);
    }
    if (m == 1)
        for (i = 0; i < n; i++)
        {
            MMXLl2Ols *o = ols[0];
            long long *R = o->R[i], a = (long long)phi[i] * cw[0];
            const int d = o->decay, sh = o->cshift;
            for (j = 0; j <= i; j++)
                R[j] += ((a * phi[j]) >> sh) - (R[j] >> d);
            o->r[i] += ((a * y) >> sh) - (o->r[i] >> d);
        }
    else
        for (i = 0; i < n; i++)
        {
            const long long pi = phi[i];
            long long *R0 = ols[0]->R[i], *R1 = ols[1]->R[i], *R2 = m > 2 ? ols[2]->R[i] : NULL;
            const long long c0 = cw[0], c1 = cw[1], c2 = m > 2 ? cw[2] : 0;
            const int d0 = ols[0]->decay, d1 = ols[1]->decay, d2 = m > 2 ? ols[2]->decay : 0;
            const int h0 = ols[0]->cshift, h1 = ols[1]->cshift, h2 = m > 2 ? ols[2]->cshift : 8;
            if (R2)
                for (j = 0; j <= i; j++)
                {
                    const long long p = pi * phi[j];
                    R0[j] += ((c0 * p) >> h0) - (R0[j] >> d0);
                    R1[j] += ((c1 * p) >> h1) - (R1[j] >> d1);
                    R2[j] += ((c2 * p) >> h2) - (R2[j] >> d2);
                }
            else
                for (j = 0; j <= i; j++)
                {
                    const long long p = pi * phi[j];
                    R0[j] += ((c0 * p) >> h0) - (R0[j] >> d0);
                    R1[j] += ((c1 * p) >> h1) - (R1[j] >> d1);
                }
            for (k = 0; k < m; k++)
            {
                MMXLl2Ols *o = ols[k];
                o->r[i] += ((pi * cw[k] * y) >> o->cshift) - (o->r[i] >> o->decay);
            }
        }
    for (k = 0; k < m; k++)
        if (++ols[k]->phase >= MMX_LL2_SOLVE)
        {
            ols[k]->phase = 0;
            if (ll2_dsolve()) ols_solve_d(ols[k]); else ols_solve(ols[k]);
        }
}

void mmx_ll2_update(MMXLl2Chan *c, long long x)
{
    MMXLl2Ols *o = &c->ols;
    MMXLl2Mix *m = &c->mix;
    long long r = x - c->p_ols - (c->src.active && c->src.use ? c->src.pred : 0), prefix = 0, y;
    unsigned int s;
    int i, e;

    /* bank: stage s learns the projected target */
    for (s = 0; s < c->nst; s++)
    {
        long long px = prefix - (prefix >> K.proj) + (m->pred >> K.proj);         /* Q16 */
        long long bt = r - ((px + (1LL << 15)) >> 16);
        long long w = m->ne == 2 ? (m->bw0 * m->w[0][s] + ((1LL << 16) - m->bw0) * m->w[1][s]) >> 16
                    : (m->bw[0] * m->w[0][s] + m->bw[1] * m->w[1][s] + m->bw[2] * m->w[2][s] + m->bw[3] * m->w[3][s]) >> 16;
        long long bs = c->lshift ? (bt + (1LL << (c->lshift - 1))) >> c->lshift : bt;
        if (c->norm[s]) norm_update(c->norm[s], bs); else mmx_nlms_update(&c->st[s], bs, bs - c->p_raw[s]);
        if (w > 0) prefix += w * c->p_st[s];
    }
    /* mixer experts: 0 = sign error, the last = error, between them Huber losses (ne = 4); gradient normalised by its
       running mean magnitude */
    for (e = 0; e < m->ne; e++)
    {
        long long err_q = r * 65536 - m->ep[e], err = err_q >> 16, aerr = err_q < 0 ? -err_q : err_q;
        int sign = e == 0;
        long long dl = m->ne == 2 ? 1LL << 40 : m->delta[e], ec = err > dl ? dl : err < -dl ? -dl : err;
        for (s = 0; s < (unsigned int)m->n; s++)
        {
            long long g = sign ? (err_q > 0 ? m->x[s] : err_q < 0 ? -m->x[s] : 0)
                               : c->ltarget ? clampl(mulsat(ec, m->x[s]), 1LL << 54) : wrapmul(ec, m->x[s]);   /* as p_st above */
            long long ag = g < 0 ? wrapmul(g, -1) : g;
            m->m[e][s] += (ag - m->m[e][s]) >> K.mixrate;
            m->w[e][s] += wrapmul(K.mixmu, g) / (m->m[e][s] + 1);
            m->w[e][s] = clampl(m->w[e][s], 1LL << 20);
        }
        m->score[e] += (aerr - m->score[e]) >> K.mixrate;
    }
    if (m->ne == 4)
    {   /* inverse-square blend relative to the best expert, so any signal scale works (24-bit residuals too) */
        long long sc[4], smin, inv[4], tot = 0;
        for (e = 0; e < 4; e++) sc[e] = (m->score[e] >> 8) + 1;
        smin = sc[0];
        for (e = 1; e < 4; e++) if (sc[e] < smin) smin = sc[e];
        for (e = 0; e < 4; e++)
        {
            long long q = ((sc[e] << 16) / smin) >> 8;        /* ratio to the best, Q8, >= 256 */
            inv[e] = q > (1LL << 23) ? 0 : (1LL << 46) / (q * q);
            tot += inv[e];
        }
        for (e = 0; e < 4; e++) m->bw[e] = (inv[e] << 16) / tot;
    }
    else
    {
        long long s0 = (m->score[0] >> 8) + 1, s1 = (m->score[1] >> 8) + 1;   /* Q8 */
        long long a = s1 * s1, b = s0 * s0;                                   /* inverse-square weights, cross-multiplied */
        while (a > (1LL << 40) || b > (1LL << 40)) { a >>= 1; b >>= 1; }
        m->bw0 = (a << 16) / (a + b + 1);
    }
    /* source stage: normalised LMS on the whitened source towards the OLS residual */
    if (c->src.active)
    {
        MMXLl2Src *q = &c->src;
        long long e = (x - c->p_ols) - q->pred, e2 = 0, r0 = x - c->p_ols;
        q->e_on += ((e < 0 ? -e : e) * 16 - q->e_on) >> 5;          /* Q4 running means over ~32 samples */
        q->e_off += ((r0 < 0 ? -r0 : r0) * 16 - q->e_off) >> 5;
        for (i = 0; i < MMX_LL2_SRC_N; i++) e2 += q->x[i] * q->x[i];
        q->pw += (e2 - q->pw) >> 4;
        if (q->pw > 0)
            for (i = 0; i < MMX_LL2_SRC_N; i++)
                q->w[i] = clampl(q->w[i] + (e * q->x[i] * (1LL << K.srcmu)) / (q->pw + 1), 1LL << 20);   /* mu = 2^(srcmu-16) */
    }
    /* OLS: IRLS-weighted covariance update (main OLS on the bank's target, the extra ones on their own error),
       then the own history */
    y = clampl(x, IN_MAX);
    {
        MMXLl2Ols *list[1 + MMX_LL2_XOLS];
        long long errs[1 + MMX_LL2_XOLS];
        list[0] = o; errs[0] = r;
        for (s = 0; s < c->nxols; s++) { list[1 + s] = c->olsx[s]; errs[1 + s] = x - c->p_olsx[s]; }
        ols_learn(list, errs, 1 + c->nxols, c->phi, y);
    }
    if (c->ltarget) stage_scale(c, r);
    if (c->n_own)
    {
        memmove(c->own + 1, c->own, sizeof(long long) * (MMX_LL2_OWN_MAX - 1));
        c->own[0] = x;
    }
}
