#ifndef MMX_LOSSLESS_H
#define MMX_LOSSLESS_H

/* MiniMix lossless frame codec (codec id 3): integer time-domain coding on
   the same 1024-sample frame grid and the same global reference plan as the
   MDCT codec. Frame f owns samples [(f-1)*1024, f*1024); a referenced frame
   predicts them as round(gain * source[s + n]) from already decoded samples,
   the integer residual (optionally joint stereo: M/S, L/S or R/S) goes
   through a per-channel predictor - a fixed polynomial of order 0..3 or an
   adaptive LPC of order 1..12 with quantized coefficients evaluated in
   64-bit integer arithmetic (FLAC style) - and the adaptive range coder.
   Bit-exact, so "MMX lossless" can be compared with FLAC to show what the
   global analysis alone saves.

   Filter cascade (Monkey's Audio principle, no side information): per coded
   channel the first difference of the coded signal, minus an adaptive
   two-sided 9-tap window on the reference source for referenced frames,
   runs through a cascade of backward-adaptive sign-sign LMS filters (1024,
   256 and 16 taps, see nlms.h; since revision 8 with the error-normalised
   step, 8/7/7); a flag per channel frame says whether the
   cascade output (then usually with a fixed order 0 or 1 predictor) or the
   coded signal itself goes to the LPC/fixed predictor. The filters adapt on
   the actual signal in every frame, so their state does not depend on the
   flag, and the cascade is carried across blocks (blocks average 4 frames
   on real material, a filter needs thousands of samples to converge); the
   decoder walks the blocks in file order anyway. The source stage restarts
   with every block: a block is one repeat with its own alignment, and
   weights carried over from the previous repeat cost 0.23 %. Measured on
   title A: the cascade after the forward LPC-12 saves 0.6 %, in front of
   it 1.4 %, the source stage another 1.9 % (the single broadband gain leaves
   fractional-delay and EQ structure that the window catches).

   Predictor history: frames coded with one context set are contiguous in
   time (the block is the unit whose contexts are reset), so the coder keeps
   the last MMX_LL_HIST raw target and source samples of the previous frame
   in the contexts and derives the history in the current coded domain from
   them (reference gain and joint stereo applied). The first samples of a
   frame are therefore predicted from real signal instead of zero history or
   verbatim FLAC-style warm-up samples; only the first frame of a block
   starts from zeros. Measured on real material this saves 0.3 % against
   zero history (about `order` badly predicted samples per channel frame).

   Residuals: signed values coded as zero flag, sign, exponent in unary with
   adaptive contexts, the two mantissa bits below the leading one adaptive
   (per exponent), the rest bypass. Contexts are selected by a running mean of
   |residual|, which beats the previous "last two residuals" classes and the
   plain Exp-Golomb mantissa by 0.5 % together.

   The residual model uses MMXCtr counters (15-bit probability, rate 1/(n+2)
   up to 1/226) instead of the 11-bit MMXProb of the rest of the bitstream,
   the class ladder is half-octave and covers the whole dynamic range of the
   running mean (the old 12 octave classes saturated at a mean |residual| of
   128, while dense rock runs at 500-1500, so the class carried no
   information), and both the counters and the running mean are carried
   across blocks like the filter cascade. The sign is a bypass bit: measured
   over 1.76 M residuals its conditional entropy is 0.99997 bit, so an
   adaptive model can only lose (it cost 1.010 bit). Together -1.05 % on the
   four measured tracks at unchanged decoder speed. Measured and rejected on
   top of this: logistic context mixing of two or three models (coarse class,
   fine class, fast/slow energy ratio) and an SSE/APM stage, both +-0.02 %,
   because every model is a function of the same local scale.

   Joint-stereo least-squares stage (bitstream revision 8, stereo only): coded
   channel 1 is predicted from a regressor of 20 integers - its own stage
   input y1[t-1..t-8], channel 0's first difference u0[t..t-7] and, because
   channel 0's whole frame is reconstructed before channel 1 on both sides,
   u0[t+1..t+4] (taps past the frame end read 0, like the source stage).
   Prediction round(w . phi / 2^16) with Q16 weights; the covariance R and
   the cross vector r are exponentially weighted int64 sums (R -= R >> 9,
   R += phi phi^T on the lower triangle), and every 16 channel-1 samples a
   fixed-point LDL^T solve (normalised to 2^28, regularised by trace >> 14,
   clamped) refreshes w. The filter cascade then gets y1 - p. Integer only,
   no side information; the state is stream state like the cascade (carried
   across blocks, in file order). Channel 0 has no stage. Revision 6-7 files
   decode with the stage off (xls_on = 0). */

#include "rangecoder.h"
#include "framecodec.h"
#include "nlms.h"

#define MMX_LL_MAX_ORDER 3              /* fixed polynomial predictors 0..3 */
#define MMX_LL_LPC_MAX_ORDER 12         /* adaptive LPC orders 1..12 */
#define MMX_LL_HIST MMX_LL_LPC_MAX_ORDER
#define MMX_LL_MODE_BITS 4              /* mode symbol: 0..3 fixed order, 4..15 LPC order 1..12 */
#define MMX_LL_RES_CLASSES 40           /* magnitude classes of the residual contexts, half an octave each */
#define MMX_LL_RES_CTX 69               /* counters per residual class */
#define MMX_LL_RES_RATE 5               /* running mean of |residual|: Q5, adaptation 1/32 */
#define MMX_LL_GAIN_Q 12
#define MMX_LL_GAIN_MAX (8 << MMX_LL_GAIN_Q)
#define MMX_LL_NLMS_STAGES 3            /* filter cascade per coded channel, long to short */
#define MMX_LL_SIDE_K 4                 /* source stage: taps on the source at t-K..t+K (K < MMX_LL_HIST) */
#define MMX_LL_SIDE_TAPS (2 * MMX_LL_SIDE_K + 1)
#define MMX_LL_XLS_OWN 8                /* joint-stereo LS stage: own taps y1[t-1..t-8] */
#define MMX_LL_XLS_CROSS 8              /* channel 0 taps u0[t..t-7] (< MMX_LL_HIST) */
#define MMX_LL_XLS_FUT 4                /* channel 0 taps u0[t+1..t+4] */
#define MMX_LL_XLS_N (MMX_LL_XLS_OWN + MMX_LL_XLS_CROSS + MMX_LL_XLS_FUT)

typedef struct
{
    int lpc;                            /* 0: fixed polynomial of `order` (0..3); 1: LPC of `order` (1..12) */
    unsigned int order;
    unsigned int shift;                 /* LPC: arithmetic right shift of the 64-bit coefficient sum, 0..15 */
    long long coef[MMX_LL_LPC_MAX_ORDER];    /* LPC: quantized coefficients, coef[0] weights the previous sample */
} MMXLosslessPred;

typedef struct
{
    long long mean;                          /* running mean of |residual|, Q MMX_LL_RES_RATE */
} MMXLosslessResState;

/* Source stage: predicts a channel's pre-whitened input from the window
   u[t-K..t+K] of the pre-whitened reference source, sign-sign adapted like
   the main filters (Q13 weights). */
typedef struct
{
    long long avg;                           /* running average of |u[t]| */
    short w[MMX_LL_SIDE_TAPS];
} MMXLosslessSide;

/* Joint-stereo LS stage of coded channel 1 (stream state, see above). */
typedef struct
{
    long long R[MMX_LL_XLS_N][MMX_LL_XLS_N]; /* weighted covariance of the regressor, lower triangle */
    long long r[MMX_LL_XLS_N];               /* weighted cross vector regressor x target */
    long long w[MMX_LL_XLS_N];               /* weights, Q16 */
    long long own[MMX_LL_XLS_OWN];           /* last stage inputs, own[0] = y1[t-1] */
    unsigned int phase;                      /* channel-1 samples since the last solve */
} MMXLosslessXls;

typedef struct
{
    /* probabilities: everything before prev_gain is initialized to MMX_PROB_INIT */
    MMXProb use_pred[2];                /* [previous frame used prediction] */
    MMXProb gain[22];                   /* signed delta of the Q12 gain vs previous frame */
    MMXProb stereo[4];                  /* 2-bit tree: 0 L/R, 1 M/S, 2 L/S, 3 R/S */
    MMXProb mode[2][1 << MMX_LL_MODE_BITS]; /* [previous channel frame was LPC] 4-bit tree: 0..3 fixed order, 4..15 LPC order 1..12 */
    MMXProb lpc_shift[16];              /* 4-bit tree for the coefficient shift */
    MMXProb lpc_coef[MMX_LL_LPC_MAX_ORDER][22]; /* [coefficient index] signed EG contexts */
    MMXProb nlms_flag[2];               /* [previous channel frame coded the cascade output] */
    MMXProb use_src2[2];                /* [previous channel frame used the second source] (two-source blocks only) */
    MMXProb gain2[22];                  /* signed delta of the Q12 gain of the second source vs previous frame */
    /* block state: zeroed by mmx_ll_contexts_init, carried across the frames of a block */
    long long prev_gain[MMX_MAX_CH];
    int prev_used[MMX_MAX_CH];
    int prev_lpc[MMX_MAX_CH];
    int prev_nlms[MMX_MAX_CH];
    long long hist_x[MMX_MAX_CH][MMX_LL_HIST];   /* last raw target samples of the previous frame, oldest first */
    long long hist_src[MMX_MAX_CH][MMX_LL_HIST]; /* last source samples of the previous frame (zero without source) */
    long long hist_e[MMX_MAX_CH][MMX_LL_HIST];   /* last cascade outputs of the previous frame */
    MMXLosslessSide sside[MMX_MAX_CH];      /* source stage per coded channel (a block is one repeat) */
    /* second source of a two-source block: its own gain chain, history and side stage (block state) */
    long long prev_gain2[MMX_MAX_CH];
    int prev_used2[MMX_MAX_CH];
    long long hist_src2[MMX_MAX_CH][MMX_LL_HIST];
    MMXLosslessSide sside2[MMX_MAX_CH];
    /* stream state: set up by mmx_ll_state_init, carried across blocks */
    MMXNlms nlms[MMX_MAX_CH][MMX_LL_NLMS_STAGES]; /* filter cascade per coded channel */
    MMXCtr res[2][MMX_LL_RES_CLASSES][MMX_LL_RES_CTX]; /* [channel][magnitude class] residual contexts */
    MMXLosslessResState res_state[MMX_MAX_CH];
    MMXLosslessXls xls;                 /* joint-stereo LS stage of coded channel 1 */
    int xls_on;                         /* 1 (revision >= 8, set by mmx_ll_state_init); the decoder clears it for older files */
    double stereo_acc[4];               /* encoder only: decayed sums of the estimated bits per stereo mode */
} MMXLosslessContexts;

typedef struct
{
    int use_pred[MMX_MAX_CH];
    long long gain[MMX_MAX_CH];          /* Q12 */
    int use_src2[MMX_MAX_CH];       /* the second source of a two-source block is subtracted too */
    long long gain2[MMX_MAX_CH];         /* Q12, fitted on what the first source leaves */
    MMXLosslessPred pred[MMX_MAX_CH];
    int stereo_ms;                  /* joint stereo mode: 0 L/R, 1 M/S, 2 L/S, 3 R/S (nonzero = joint) */
    int nlms[MMX_MAX_CH];           /* coded channel codes the filter cascade output */
} MMXLosslessFrame;

/* Encoder statistics (process wide, for tests and tuning): channel frames
   coded by each predictor family. Reset them yourself before measuring. */
typedef struct
{
    unsigned long channel_frames;
    unsigned long lpc_frames;
    unsigned long fixed_frames;
    unsigned long lpc_order_hist[MMX_LL_LPC_MAX_ORDER + 1];
    unsigned long nlms_frames;      /* channel frames that coded the cascade output */
} MMXLosslessStats;

extern MMXLosslessStats mmx_ll_stats;

/* Once per stream (before the first block): clears everything and sets up the
   filter cascade. The contexts are about 200 KB, keep them off small stacks. */
void mmx_ll_state_init(MMXLosslessContexts *c);

/* Decoder of an older file: right after mmx_ll_state_init, switches the
   cascade to the update rule of that bitstream revision (6-7: sign-sign;
   8 and later: the error-normalised step, which the encoder always writes). */
void mmx_ll_state_set_revision(MMXLosslessContexts *c, unsigned int bitstream_rev);

/* Per block: resets the block state; the probabilities, the residual model
   and the filters are stream state and survive. */
void mmx_ll_contexts_init(MMXLosslessContexts *c);

/* Encodes one frame. x[ch][n]: target integers (n < count). src[ch][n]: source
   integers (NULL when not referenced). Chooses gain, predictor and stereo
   mode itself, writes the decision into `fr`. Returns 0 on success. Frames
   coded with the same contexts must be consecutive in time (see above). */
int mmx_ll_encode_frame(MMXRangeEncoder *rc, MMXLosslessContexts *ctx, unsigned int channels,
                        long long *const *x, long long *const *src, long long *const *src2, unsigned long count, MMXLosslessFrame *fr);

/* Decodes one frame into x[ch][n]. Returns 0 on success. */
int mmx_ll_decode_frame(MMXRangeDecoder *rc, MMXLosslessContexts *ctx, unsigned int channels,
                        long long *const *x, long long *const *src, long long *const *src2, unsigned long count, MMXLosslessFrame *fr);

/* Analysis-side bit estimate for the lossless coder from MDCT coefficients of
   a 2048 window (target or residual): rate of an ideal predictive coder at the
   given integer resolution, for one frame of one channel. */
double mmx_ll_estimate_bits(const float *coefs, unsigned long m, unsigned int source_bits);

#endif
