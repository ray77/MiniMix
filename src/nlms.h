#ifndef MMX_NLMS_H
#define MMX_NLMS_H

/* Backward-adaptive integer sign-sign LMS predictor, the "NN filter" of
   Monkey's Audio: the next input is predicted from the last `order` inputs
   (saturated to 16 bit) with 16-bit weights, and every weight moves by a
   small step towards sign(error) * sign(input) after each sample. No side
   information, integer arithmetic only (the dot product is summed modulo
   2^32 like Monkey's, the weights saturate), so encoder and decoder run the
   same state bit-exactly. The step of a tap is 8, 16 or 32 by the magnitude
   of its input against a running average and is halved at lags 1, 2 and 8,
   so recent inputs adapt faster than old ones. A weight of 1 << shift is
   unity gain. Measured: the steps 8/16/32 beat 4/8/16 and 16/32/64, and a
   step proportional to the input (sign-error LMS) gains nothing over the
   sign.

   Error-normalised step (bitstream revision 8): the tap step d is scaled by
   m = |err| / mean|err|, capped at 2 and quantized in Q6 (mq = 0..128), so a
   sample the filter already predicts well barely moves the weights and a
   badly predicted one moves them up to twice as far. mean|err| is a running
   average (Q4, rate 1/32) taken before the current sample. Every weight
   moves by round(d * (2 mq + 1) / 2^step_shift) towards sign(err): the
   multiplier is odd and |d| <= 32, so for step_shift >= 7 the product never
   lies exactly halfway between two integers and the rounding is symmetric
   in the sign of the error (half-up rounding of an even multiplier made
   the weights drift and cost 2 %). step_shift 7 gives today's step d at the
   mean error, 8 half of it. |d * (2 mq + 1)| <= 32 * 257 plus the rounding
   term fits 16-bit lanes, so the loop stays as wide as the sign-sign one;
   the cost is one division per sample and stage, skipped at the cap.
   step_shift 0 is the plain sign-sign rule of revisions 6-7. Measured on
   twelve 30 s excerpts with 8/7/7 on the 1024/256/16 cascade: -0.35 %. */

#define MMX_NLMS_MAX_ORDER 1024
#define MMX_NLMS_MIN_ORDER 16
#define MMX_NLMS_WINDOW 512             /* new samples between buffer rolls */
#define MMX_NLMS_MAX_STEP_SHIFT 12      /* error-normalised step: 1..12; 0 = sign-sign */

typedef struct
{
    unsigned int order;                 /* taps, MMX_NLMS_MIN_ORDER..MMX_NLMS_MAX_ORDER */
    unsigned int shift;                 /* weight scale: prediction = round(sum >> shift) */
    unsigned int pos;                   /* next input index in x and d: order..order+WINDOW */
    unsigned int step_shift;            /* error-normalised step (see above); 0 = sign-sign */
    long long avg;                           /* running average of |input| */
    long long eavg;                     /* running average of |err|, Q4 (error-normalised step) */
    short w[MMX_NLMS_MAX_ORDER];        /* weights, oldest tap first */
    short x[MMX_NLMS_MAX_ORDER + MMX_NLMS_WINDOW]; /* saturated inputs */
    short d[MMX_NLMS_MAX_ORDER + MMX_NLMS_WINDOW]; /* adaptation steps, signed like the input */
} MMXNlms;

/* step_shift: 0 = sign-sign (revisions 6-7), otherwise the error-normalised
   step, clamped to MMX_NLMS_MAX_STEP_SHIFT. */
void mmx_nlms_init(MMXNlms *f, unsigned int order, unsigned int shift, unsigned int step_shift);

/* Prediction of the next input from the current state. */
long long mmx_nlms_predict(const MMXNlms *f);

/* Adapts to the input `in` that was predicted with error `err` (in - prediction)
   and appends it to the history. */
void mmx_nlms_update(MMXNlms *f, long long in, long long err);

/* Rescales the state to inputs 2^-d as large (d < 0: 2^-d as large, saturating): the history and the running
   averages move, the weights stay, so the prediction scales with the input. */
void mmx_nlms_rescale(MMXNlms *f, int d);

#endif
