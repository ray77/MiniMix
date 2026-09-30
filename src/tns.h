#ifndef MMX_TNS_H
#define MMX_TNS_H

/* Temporal noise shaping: linear prediction along the frequency axis of an
   MDCT frame. The encoder filters the coefficients with the whitening FIR
   A(z), the decoder runs the inverse IIR 1/A(z); quantization noise then
   follows the temporal envelope of the signal inside the window instead of
   being spread uniformly over 46 ms (pre-echo control without block
   switching). Reflection coefficients are quantized to 4 bits, which keeps
   the decoder filter stable and lets both sides derive identical filters. */

#define MMX_TNS_MAX_ORDER 8
#define MMX_TNS_COEF_BITS 4

typedef struct
{
    int active;
    unsigned int order;
    unsigned char coef[MMX_TNS_MAX_ORDER];   /* quantized reflection coefficient indices */
} MMXTns;

/* Analyzes coefficients x[k0..k1) and fills `t`; returns the prediction gain
   (linear power ratio, >= 1). `t->active` is set when the gain exceeds
   `min_gain`. */
double mmx_tns_analyze(const float *x, unsigned long k0, unsigned long k1, unsigned int max_order,
                       double min_gain, MMXTns *t);

/* Applies the analysis (whitening) filter in place over x[k0..k1). */
void mmx_tns_filter(const MMXTns *t, float *x, unsigned long k0, unsigned long k1);

/* Applies the synthesis (inverse) filter in place over x[k0..k1). */
void mmx_tns_inverse(const MMXTns *t, float *x, unsigned long k0, unsigned long k1);

/* Temporal power envelope of the synthesis filter over an MDCT window of n
   samples (m = n/2 coefficients): a filter along the frequency index acts as
   a multiplication in time, env[t] = |1/A(e^{j pi (t + 0.5 + m/2) / m})|^2. */
void mmx_tns_time_envelope(const MMXTns *t, unsigned long n, double *env);

/* Mean power gain of the synthesis filter (energy of its impulse response):
   noise coded in the filtered domain grows by this factor after the inverse. */
double mmx_tns_noise_gain(const MMXTns *t);

#endif
