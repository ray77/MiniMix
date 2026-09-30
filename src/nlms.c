#include <string.h>
#include "nlms.h"

#define WINDOW MMX_NLMS_WINDOW
#define STEP 8                          /* smallest adaptation step; 16 and 32 for inputs above the running average */

/* error-normalised step (revision 8): |err| / mean|err| in Q6, capped at 2.0 */
#define MQ_BITS 6
#define MQ_CAP (2 << MQ_BITS)
#define EAVG_Q 4                        /* mean |err| in Q4 ... */
#define EAVG_RATE 5                     /* ... adapting by 1/32 */
#define ERR_CLAMP (1LL << 40)           /* far above any real error (32-bit sources stay below 2^34); keeps a
                                           corrupt stream from overflowing the Q4 mean or the Q6 quotient */

/* The weight update runs in 16-bit lanes: the largest tap step (4 * STEP) times the largest odd
   multiplier (2 * MQ_CAP + 1) plus the rounding term of the largest shift must fit a short. */
typedef char mmx_nlms_lanes_fit_int16[(4 * STEP) * (2 * MQ_CAP + 1) + (1 << (MMX_NLMS_MAX_STEP_SHIFT - 1)) <= 32767 ? 1 : -1];

static short sat16(long long v)
{
    return (short)(v > 32767 ? 32767 : v < -32768 ? -32768 : v);
}

void mmx_nlms_init(MMXNlms *f, unsigned int order, unsigned int shift, unsigned int step_shift)
{
    memset(f, 0, sizeof(*f));
    if (order < MMX_NLMS_MIN_ORDER) order = MMX_NLMS_MIN_ORDER;
    if (order > MMX_NLMS_MAX_ORDER) order = MMX_NLMS_MAX_ORDER;
    if (step_shift > MMX_NLMS_MAX_STEP_SHIFT) step_shift = MMX_NLMS_MAX_STEP_SHIFT;
    f->order = order;
    f->shift = shift;
    f->step_shift = step_shift;
    f->pos = order;
}

long long mmx_nlms_predict(const MMXNlms *f)
{
    const short *x = f->x + f->pos - f->order, *w = f->w;
    unsigned int sum = 0, i;
    long long v;
    /* 16 x 16 bit products summed modulo 2^32 (defined for unsigned, and the
       same on both sides): the weights stay far from the overflow in practice.
       The loop vectorizes; a 64-bit accumulator did not (2.3x slower). */
    for (i = 0; i < f->order; i++)
        sum += (unsigned int)((int)w[i] * (int)x[i]);
    v = sum >= 0x80000000u ? -(long long)(~sum) - 1 : (long long)sum;
    return (v + (1LL << (f->shift - 1))) >> f->shift;
}

/* w += sign * d with 16-bit saturation (restrict: the loop vectorizes) */
static void adapt(short *restrict w, const short *restrict d, unsigned int n, int sign)
{
    unsigned int i;
    for (i = 0; i < n; i++)
    {
        int t = w[i] + sign * d[i];
        w[i] = (short)(t > 32767 ? 32767 : t < -32768 ? -32768 : t);
    }
}

/* w += round(d * m / 2^sh) with 16-bit saturation. m is odd and carries the sign of the error;
   d * m + 2^(sh-1) fits a short (see the lane check above), the shift is arithmetic like every
   other right shift of a signed value in the core. The shift count is one of two constants in
   practice; the specialised copies let the compiler use an immediate vector shift. */
#define ADAPT_SCALED(name, SH)                                                              \
    static void name(short *restrict w, const short *restrict d, unsigned int n, short m)   \
    {                                                                                       \
        const short r = (short)(1 << ((SH) - 1));                                           \
        unsigned int i;                                                                     \
        for (i = 0; i < n; i++)                                                             \
        {                                                                                   \
            short x = (short)((short)(d[i] * m + r) >> (SH));                               \
            int t = w[i] + x;                                                               \
            w[i] = (short)(t > 32767 ? 32767 : t < -32768 ? -32768 : t);                    \
        }                                                                                   \
    }
ADAPT_SCALED(adapt_scaled7, 7)
ADAPT_SCALED(adapt_scaled8, 8)

static void adapt_scaled(short *restrict w, const short *restrict d, unsigned int n, short m, unsigned int sh)
{
    const short r = (short)(1 << (sh - 1));
    unsigned int i;
    for (i = 0; i < n; i++)
    {
        short x = (short)((short)(d[i] * m + r) >> sh);
        int t = w[i] + x;
        w[i] = (short)(t > 32767 ? 32767 : t < -32768 ? -32768 : t);
    }
}

void mmx_nlms_update(MMXNlms *f, long long in, long long err)
{
    unsigned int n = f->order, p = f->pos;
    long long a = in < 0 ? -in : in;
    int step;

    if (f->step_shift == 0)
    {
        /* sign-sign adaptation (revisions 6-7): every weight moves by its tap's step towards sign(err) * sign(input) */
        if (err > 0)
            adapt(f->w, f->d + p - n, n, 1);
        else if (err < 0)
            adapt(f->w, f->d + p - n, n, -1);
    }
    else
    {
        /* error-normalised step: the tap step scaled by m = |err| / mean|err| (Q6, capped at 2),
           the mean taken before this sample; the division is skipped at the cap and for err == 0 */
        long long ae = err < 0 ? -err : err, ea = f->eavg >> EAVG_Q;
        int mq = 0;
        if (ae > ERR_CLAMP)
            ae = ERR_CLAMP;
        if (ae != 0)
            mq = ea <= 0 || ae >= 2 * ea ? MQ_CAP : (int)(((unsigned long long)ae << MQ_BITS) / (unsigned long long)ea);
        f->eavg += ((ae << EAVG_Q) - f->eavg) >> EAVG_RATE;
        if (mq > 0)
        {
            short m = (short)(err > 0 ? 2 * mq + 1 : -(2 * mq + 1));
            if (f->step_shift == 8)
                adapt_scaled8(f->w, f->d + p - n, n, m);
            else if (f->step_shift == 7)
                adapt_scaled7(f->w, f->d + p - n, n, m);
            else
                adapt_scaled(f->w, f->d + p - n, n, m, f->step_shift);
        }
    }

    /* step of the new input by its magnitude against the running average */
    if (a > 3 * f->avg)
        step = 4 * STEP;
    else if (a > (4 * f->avg) / 3)
        step = 2 * STEP;
    else if (a > 0)
        step = STEP;
    else
        step = 0;
    f->avg += (a - f->avg) / 16;

    f->x[p] = sat16(in);
    f->d[p] = (short)(in < 0 ? -step : step);
    f->d[p - 1] >>= 1;
    f->d[p - 2] >>= 1;
    f->d[p - 8] >>= 1;
    if (++p == n + WINDOW)
    {
        memmove(f->x, f->x + WINDOW, sizeof(short) * n);
        memmove(f->d, f->d + WINDOW, sizeof(short) * n);
        p = n;
    }
    f->pos = p;
}

void mmx_nlms_rescale(MMXNlms *f, int d)
{
    unsigned int i;
    if (d == 0) return;
    for (i = 0; i < MMX_NLMS_MAX_ORDER + WINDOW; i++)
        f->x[i] = d > 0 ? (short)((f->x[i] + (1 << (d - 1))) >> d) : sat16((long long)f->x[i] * (1LL << -d));
    if (d > 0) { f->avg >>= d; f->eavg >>= d; }
    else { f->avg *= 1LL << -d; f->eavg *= 1LL << -d; }
}
