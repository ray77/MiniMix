#ifndef MMX_LL2_H
#define MMX_LL2_H
/* LL2: the integer predictor chain of the second lossless core (bitstream revision 10).
 *
 * Per coded channel, in the signal domain (not the first difference - measured 0.6 % better):
 *
 *   1. OLS stage: exponentially weighted least squares (decay 1 - 2^-9) on a regressor of the channel's own past,
 *      the other channel (channel 1 sees channel 0 at t) and optional side taps (reference sources), with an IRLS
 *      weight (esum + 2)^-0.75 from a table (sac's L1-leaning weighting), solved every MMX_LL2_SOLVE samples by
 *      the fixed-point LDL^T of the joint-stereo stage.
 *   2. Bank: integer sign-LMS stages (src/nlms.c) on the OLS residual. Stage s learns the projected target
 *      r - round((1 - 1/8) prefix_s + 1/8 mix), prefix_s = the mixer-weighted sum of the stages before it (sac).
 *   3. Mixer: two experts (sign error and error, gradient normalised by its running mean |gradient|) over the stage
 *      outputs, blended by inverse-square weighting of their running mean |error|. All Q16, integer only.
 *
 * Prediction = OLS + mixer. Encoder and decoder run exactly the same integer operations. Every design choice was
 * measured first on a fixed set of 30-s excerpts.
 */
#include "nlms.h"

#define MMX_LL2_MAX_N 48                /* OLS regressor length */
#define MMX_LL2_MAX_STAGES 6
#define MMX_LL2_XOLS 2                  /* extra OLS with other memories (header bytes 90, 91) */
#define MMX_LL2_MIX_MAX (MMX_LL2_MAX_STAGES + MMX_LL2_XOLS)   /* mixer inputs: the stages, then the extra OLS */
#define MMX_LL2_SOLVE 16                /* samples between two OLS solves */
#define MMX_LL2_OWN_MAX 32

/* integer normalised LMS stage (sac's NLMS_Stream in fixed point): prediction sum(w x) >> 20, update
   w += mu * e * x / (sum of x^2 over the taps), one division per sample. History newest first. */
#define MMX_LL2_NLMS_MAX 1280
typedef struct
{
    int n;
    int mu_shift;                       /* mu = 2^-mu_shift, relative to the tap count */
    long long w[MMX_LL2_NLMS_MAX];      /* Q20 */
    int x[2 * MMX_LL2_NLMS_MAX];        /* ring of 2n, the window x[pos .. pos+n) is contiguous, newest first */
    int pos;
    long long pow;                      /* sum of x^2 over the window */
    long long pred;
    long long pnext;                    /* the next prediction, computed in the same pass as the update */
    /* float mode (hi-res profile): the same filter with float weights and history, summed in 8 fixed lanes */
    int fmode;
    int fpos;                           /* window xf[fpos .. fpos+n) newest first; xf[fpos+n] the sample that left */
    float wf[MMX_LL2_NLMS_MAX];
    float xf[2 * MMX_LL2_NLMS_MAX + 1];
} MMXLl2Norm;

typedef struct
{
    int n, n_plain, phase;              /* n_plain: own and cross taps; the side taps after them get a diagonal loading */
    int decay;                          /* R -= R >> decay (lambda = 1 - 2^-decay); per file, header byte 88 */
    int cshift;                         /* covariance scale: R += (w x x') >> cshift (profile: 8 for up to 20 bits, 16 at 24) */
    int keep_on_fail;                   /* 1: an ill-conditioned solve keeps the previous weights (hi-res profile, as sac) */
    int dpred;                          /* 1: predict with the solver's double weights (hi-res profile) instead of Q16 */
    double wd[MMX_LL2_MAX_N];           /* the solver's weights (dpred) */
    long long esum;                     /* running mean |error| in Q4 */
    long long R[MMX_LL2_MAX_N][MMX_LL2_MAX_N];
    long long r[MMX_LL2_MAX_N];
    long long w[MMX_LL2_MAX_N];         /* Q16 */
    int dcov;                           /* 1: the covariance in double, Rd and rd instead of R and r (hi-res profile) */
    int solve_every;                    /* samples between two solves in the double path (0 = MMX_LL2_SOLVE) */
    double Rd[MMX_LL2_MAX_N][MMX_LL2_MAX_N];
    double rd[MMX_LL2_MAX_N];
} MMXLl2Ols;

/* The mixer: ne experts, each a weight set over the same inputs learnt with its own loss, blended by the inverse
   square of their running mean |error|. ne = 2 (header byte 92 = 0): sign error and error. ne = 4 (byte 92 = 1): Huber
   losses with thresholds 0 (sign), delta[1], delta[2] and none (error) - sac's blend of Huber experts (measured:
   classical -0.04..-0.055 %, rock neutral). */
#define MMX_LL2_EXPERTS 4
typedef struct
{
    int n, ne;
    long long w[MMX_LL2_EXPERTS][MMX_LL2_MIX_MAX];   /* Q16 */
    long long m[MMX_LL2_EXPERTS][MMX_LL2_MIX_MAX];   /* running mean |gradient| */
    long long ep[MMX_LL2_EXPERTS];      /* expert predictions, Q16 */
    long long score[MMX_LL2_EXPERTS];   /* running mean |error|, Q16 */
    long long bw0;                      /* ne = 2: weight of expert 0, Q16 */
    long long bw[MMX_LL2_EXPERTS];      /* ne = 4: blend weights, Q16 */
    long long delta[MMX_LL2_EXPERTS];   /* ne = 4: Huber thresholds */
    long long pred;                     /* blended prediction, Q16 */
    long long x[MMX_LL2_MIX_MAX];
} MMXLl2Mix;

#define MMX_LL2_SRC_K 2                 /* source stage: whitened source at t-K..t+K */
#define MMX_LL2_SRC_N (2 * MMX_LL2_SRC_K + 1)

typedef struct
{
    long long w[MMX_LL2_SRC_N];         /* Q16, reset to identity on the centre tap when the source changes */
    long long x[MMX_LL2_SRC_N];         /* the whitened source values of the current sample */
    long long pw;                       /* running mean of x^2 over the taps (normalisation), integer */
    long long pred;                     /* the stage's own prediction of the OLS residual */
    long long e_on, e_off;              /* running mean |error| with and without the stage (gating, no side info) */
    int active, use;
} MMXLl2Src;

typedef struct
{
    unsigned int n_own, n_cross, n_side, nst;
    MMXLl2Src src;
    MMXLl2Ols ols;
    MMXLl2Ols *olsx[MMX_LL2_XOLS];      /* extra OLS on the same regressor with other memories; each correction
                                           p_olsx - p_ols is one more mixer input (the bank still learns on the
                                           main OLS residual) */
    long long p_olsx[MMX_LL2_XOLS];
    unsigned int nxols;
    MMXNlms st[MMX_LL2_MAX_STAGES];
    MMXLl2Norm *norm[MMX_LL2_MAX_STAGES];    /* non-NULL: stage s is a normalised LMS stage instead of sign-sign */
    MMXLl2Mix mix;
    long long own[MMX_LL2_OWN_MAX];     /* own past samples, newest first */
    int phi[MMX_LL2_MAX_N];
    long long p_ols;                    /* cached for the update */
    long long p_st[MMX_LL2_MAX_STAGES];      /* stage predictions at the signal scale */
    long long p_raw[MMX_LL2_MAX_STAGES];     /* the same at the stages' scale (p_st >> lshift) */
    int lshift;                         /* the stages see the target >> lshift: their inputs saturate at 16 bits */
    int ltarget;                        /* 0: lshift stays 0 (CD profile); else lshift follows mean |r| to about 2^ltarget */
    int lcount;
    long long lmean;                    /* running mean |r| (OLS residual), Q4 */
    long long pred;
} MMXLl2Chan;

/* n_own + n_cross + n_side <= MMX_LL2_MAX_N, nst <= MMX_LL2_MAX_STAGES, orders as in mmx_nlms_init. */
void mmx_ll2_init(MMXLl2Chan *c, unsigned int n_own, unsigned int n_cross, unsigned int n_side,
                  const unsigned int *orders, unsigned int nst);
/* Prediction of the next sample. cross: n_cross samples of the other channel (newest first), side: n_side values. */
long long mmx_ll2_predict(MMXLl2Chan *c, const long long *cross, const long long *side);
/* Source stage (references): src points at the source sample aligned with t; src[-K-n_own .. K] must be readable
   (decoded audio). NULL = no reference for this sample. new_block = 1 restarts the stage (a new repeat). Call it
   right after mmx_ll2_predict; it returns the prediction including the source stage. */
long long mmx_ll2_predict_src(MMXLl2Chan *c, const long long *src, int new_block);
/* The same with the other channel's source (src_other aligned like src, readable where the cross and future taps of
   this channel reach): the source is whitened with the full OLS (own, cross and future taps), so an exact repeat of
   both channels leaves exactly the target's OLS residual. is_ch1: this channel sees the other one at t (and t+1..). */
long long mmx_ll2_predict_src2(MMXLl2Chan *c, const long long *src, const long long *src_other, int is_ch1,
                               unsigned int n_future, int new_block);
/* Adapts to the actual sample x (after mmx_ll2_predict). */
void mmx_ll2_update(MMXLl2Chan *c, long long x);
/* Adds an extra OLS with memory decay (6..14) on the same regressor as the main OLS; its correction p - p_ols joins
   the mixer with weight 0. Call after mmx_ll2_init, before the first prediction; at most MMX_LL2_XOLS. Measured:
   memories 2^7 and 2^12 beside 2^9, twelve 30-s excerpts -0.45 % against OptimFROG (-0.05 % against the best
   single memory per file). */
int mmx_ll2_add_ols(MMXLl2Chan *c, unsigned int decay);
/* The covariance scale of all OLS of the channel (main and extras): 8 keeps the revision-10 CD numbers; a 24-bit
   regressor needs 16 so that the exponentially forgetting sums stay below 2^63 (x^2 w reaches 2^58). */
void mmx_ll2_set_cov_shift(MMXLl2Chan *c, int cshift);
void mmx_ll2_set_stage_target(MMXLl2Chan *c, int ltarget);
void mmx_ll2_set_float_norm(MMXLl2Chan *c, int on);    /* before the first sample */
void mmx_ll2_set_dcov(MMXLl2Chan *c, int on);          /* before the first sample */
void mmx_ll2_set_solve_every(MMXLl2Chan *c, int n);    /* the double path's solve interval (0 = MMX_LL2_SOLVE) */
/* 1: when a solve meets a pivot it cannot trust (or a non-finite weight), the OLS keeps its previous weights instead
   of clamping the pivot (0, the CD behaviour, frozen). Oversampled hi-res material makes the covariance nearly
   singular; clamped pivots then give wild weights. All OLS of the channel. */
void mmx_ll2_set_keep_on_fail(MMXLl2Chan *c, int on);
/* 1: the OLS predicts with the solver's double weights, summed in a fixed order without contraction (the same bits on
   every IEEE-754 machine), instead of Q16 weights clamped to +-64. A 24-bit sample times a Q16 rounding error is
   already 64 LSB per tap, and oversampled material wants weights far beyond 64. Hi-res profile; all OLS. */
void mmx_ll2_set_dpred(MMXLl2Chan *c, int on);
/* Switches the mixer to four Huber experts with thresholds d1, d2 (in units of the coded integers). Call after
   mmx_ll2_init and mmx_ll2_add_ols, before the first prediction. */
void mmx_ll2_set_experts4(MMXLl2Chan *c, long long d1, long long d2);
/* Turns stage s into an integer normalised LMS stage of n taps (heap; mmx_ll2_free releases it). */
int mmx_ll2_set_norm_stage(MMXLl2Chan *c, unsigned int s, int n, int mu_shift);
void mmx_ll2_free(MMXLl2Chan *c);
/* Reads the module's environment switches once; call before predictors run on several threads (the encoder's
   parallel candidates, the decoder's channel threads), so no thread initialises a shared static. */
void mmx_ll2_statics(void);

#endif
