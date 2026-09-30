#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "mdct.h"

#define MMX_PI 3.14159265358979323846

/* Kaiser-Bessel-derived window (alpha = 4): same perfect reconstruction as the
   sine window, far better stopband rejection, so quantization noise of loud
   bands does not spread into distant quiet bands (the AAC long-block choice). */
#define KBD_ALPHA 4.0

static double bessel_i0(double x)
{
    double sum = 1.0, term = 1.0, k = 1.0;
    while (term > 1e-14 * sum)
    {
        term *= (x / (2.0 * k)) * (x / (2.0 * k));
        sum += term;
        k += 1.0;
    }
    return sum;
}

static int kbd_window(double *w, unsigned long n)
{
    unsigned long m = n / 2, i;
    double *kaiser = (double *)malloc(sizeof(double) * (m + 1)), cum = 0.0, total = 0.0;
    if (!kaiser)
        return -1;
    for (i = 0; i <= m; i++)
    {
        double r = 2.0 * (double)i / (double)m - 1.0;
        kaiser[i] = bessel_i0(MMX_PI * KBD_ALPHA * sqrt(1.0 - r * r)) / bessel_i0(MMX_PI * KBD_ALPHA);
        total += kaiser[i];
    }
    for (i = 0; i < m; i++)
    {
        cum += kaiser[i];
        w[i] = sqrt(cum / total);
        w[n - 1 - i] = w[i];
    }
    free(kaiser);
    return 0;
}

int mmx_mdct_init(MMXMdct *t, unsigned long n)
{
    unsigned long m = n / 2, h = m / 2, i;

    memset(t, 0, sizeof(*t));
    if (n < 16 || (n & (n - 1)))
        return -1;
    t->n = n;
    t->m = m;
    t->window = (double *)malloc(sizeof(double) * n);
    t->pre_re = (double *)malloc(sizeof(double) * h);
    t->pre_im = (double *)malloc(sizeof(double) * h);
    t->post_re = (double *)malloc(sizeof(double) * h);
    t->post_im = (double *)malloc(sizeof(double) * h);
    t->work_re = (double *)malloc(sizeof(double) * h);
    t->work_im = (double *)malloc(sizeof(double) * h);
    t->fold = (double *)malloc(sizeof(double) * m);
    if (!t->window || !t->pre_re || !t->pre_im || !t->post_re || !t->post_im ||
        !t->work_re || !t->work_im || !t->fold || mmx_fft_init(&t->fft, h) != 0)
    {
        mmx_mdct_free(t);
        return -1;
    }
    if (kbd_window(t->window, n) != 0)
    {
        mmx_mdct_free(t);
        return -1;
    }
    for (i = 0; i < h; i++)
    {
        double a = -MMX_PI * ((double)i + 0.25) / (double)m;
        double b = -MMX_PI * (double)i / (double)m;
        t->pre_re[i] = cos(a); t->pre_im[i] = sin(a);
        t->post_re[i] = cos(b); t->post_im[i] = sin(b);
    }
    return 0;
}

void mmx_mdct_free(MMXMdct *t)
{
    if (!t)
        return;
    free(t->window);
    free(t->pre_re); free(t->pre_im);
    free(t->post_re); free(t->post_im);
    free(t->work_re); free(t->work_im);
    free(t->fold);
    mmx_fft_free(&t->fft);
    memset(t, 0, sizeof(*t));
}

/* DCT-IV of t->fold (m points) -> out (m points), via m/2 complex FFT. */
static void dct4(MMXMdct *t, double *out)
{
    unsigned long m = t->m, h = m / 2, i;
    const double *v = t->fold;

    for (i = 0; i < h; i++)
    {
        double re = v[2 * i], im = v[m - 1 - 2 * i];
        t->work_re[i] = re * t->pre_re[i] - im * t->pre_im[i];
        t->work_im[i] = re * t->pre_im[i] + im * t->pre_re[i];
    }
    mmx_fft_forward(&t->fft, t->work_re, t->work_im);
    for (i = 0; i < h; i++)
    {
        double re = t->work_re[i] * t->post_re[i] - t->work_im[i] * t->post_im[i];
        double im = t->work_re[i] * t->post_im[i] + t->work_im[i] * t->post_re[i];
        out[2 * i] = re;
        out[m - 1 - 2 * i] = -im;
    }
}

void mmx_mdct_forward_w(MMXMdct *t, const float *x, const double *window, float *out)
{
    unsigned long m = t->m, q = m / 2, i;
    double *v = t->fold;
    double tmp[8192];         /* m <= 8192 */

    for (i = 0; i < q; i++)
        v[i] = -x[3 * q - 1 - i] * window[3 * q - 1 - i] - x[3 * q + i] * window[3 * q + i];
    for (i = q; i < m; i++)
        v[i] = x[i - q] * window[i - q] - x[3 * q - 1 - i] * window[3 * q - 1 - i];

    dct4(t, tmp);
    for (i = 0; i < m; i++)
        out[i] = (float)tmp[i];
}

void mmx_mdct_inverse_w(MMXMdct *t, const float *in, const double *window, float *y)
{
    unsigned long m = t->m, q = m / 2, i;
    double scale = 2.0 / (double)m;
    double tmp[8192];

    for (i = 0; i < m; i++)
        t->fold[i] = in[i];
    dct4(t, tmp);
    for (i = 0; i < m; i++)
        tmp[i] *= scale;

    for (i = 0; i < q; i++)
    {
        y[3 * q - 1 - i] = (float)(-tmp[i] * window[3 * q - 1 - i]);
        y[3 * q + i] = (float)(-tmp[i] * window[3 * q + i]);
    }
    for (i = q; i < m; i++)
    {
        y[i - q] = (float)(tmp[i] * window[i - q]);
        y[3 * q - 1 - i] = (float)(-tmp[i] * window[3 * q - 1 - i]);
    }
}

void mmx_mdct_forward(MMXMdct *t, const float *x, float *out)
{
    mmx_mdct_forward_w(t, x, t->window, out);
}

void mmx_mdct_inverse(MMXMdct *t, const float *in, float *y)
{
    mmx_mdct_inverse_w(t, in, t->window, y);
}

unsigned long mmx_mdct_short_offset(const MMXMdct *long_t, const MMXMdct *short_t)
{
    /* the eight short windows cover the centre of the long window: they start
       at n/4 - ns/4 and end at 3n/4 + ns/4 */
    return long_t->n / 4 - short_t->n / 4;
}

int mmx_mdct_window_start(const MMXMdct *long_t, const MMXMdct *short_t, double *window)
{
    unsigned long n = long_t->n, ns = short_t->n, half = n / 2, i;
    unsigned long flat_end = 3 * n / 4 - ns / 4;      /* where the short slope begins */
    if (ns >= n || ns < 2)
        return -1;
    for (i = 0; i < half; i++)
        window[i] = long_t->window[i];
    for (i = half; i < flat_end; i++)
        window[i] = 1.0;
    for (i = 0; i < ns / 2; i++)
        window[flat_end + i] = short_t->window[ns / 2 + i];
    for (i = flat_end + ns / 2; i < n; i++)
        window[i] = 0.0;
    return 0;
}

int mmx_mdct_window_stop(const MMXMdct *long_t, const MMXMdct *short_t, double *window)
{
    unsigned long n = long_t->n, i;
    double *tmp = (double *)malloc(sizeof(double) * n);
    if (!tmp || mmx_mdct_window_start(long_t, short_t, tmp) != 0)
    {
        free(tmp);
        return -1;
    }
    for (i = 0; i < n; i++)
        window[i] = tmp[n - 1 - i];
    free(tmp);
    return 0;
}
