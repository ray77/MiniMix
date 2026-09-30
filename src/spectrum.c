#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "spectrum.h"

#define MMX_PI 3.14159265358979323846

int mmx_spectrum_init(MMXSpectrum *s, unsigned long n)
{
    unsigned long i;
    double sum_w2 = 0.0;

    memset(s, 0, sizeof(*s));
    if (mmx_fft_init(&s->fft, n) != 0)
        return -1;
    s->n = n;
    s->window = (double *)malloc(sizeof(double) * n);
    s->re = (double *)malloc(sizeof(double) * n);
    s->im = (double *)malloc(sizeof(double) * n);
    if (!s->window || !s->re || !s->im)
    {
        mmx_spectrum_free(s);
        return -1;
    }
    for (i = 0; i < n; i++)
    {
        double t = 2.0 * MMX_PI * (double)i / (double)(n - 1);
        s->window[i] = 0.35875 - 0.48829 * cos(t) + 0.14128 * cos(2.0 * t) - 0.01168 * cos(3.0 * t);
        sum_w2 += s->window[i] * s->window[i];
    }
    /* stationary noise of variance v: FFT bin power = v * sum(w^2); MDCT
       coefficient power (sine window, length n) = v * n / 4 */
    s->scale = ((double)n / 4.0) / sum_w2;
    return 0;
}

void mmx_spectrum_free(MMXSpectrum *s)
{
    if (!s)
        return;
    mmx_fft_free(&s->fft);
    free(s->window);
    free(s->re);
    free(s->im);
    memset(s, 0, sizeof(*s));
}

void mmx_spectrum_analyze(MMXSpectrum *s, const float *x, float *amp)
{
    unsigned long i, half = s->n / 2;
    for (i = 0; i < s->n; i++)
    {
        s->re[i] = (double)x[i] * s->window[i];
        s->im[i] = 0.0;
    }
    mmx_fft_forward(&s->fft, s->re, s->im);
    for (i = 0; i < half; i++)
        amp[i] = (float)sqrt((s->re[i] * s->re[i] + s->im[i] * s->im[i]) * s->scale);
}
