#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "fft.h"

#define MMX_PI 3.14159265358979323846

int mmx_fft_init(MMXFft *f, unsigned long n)
{
    unsigned long i, bits = 0, t;

    memset(f, 0, sizeof(*f));
    if (n < 2 || (n & (n - 1)))
        return -1;
    f->n = n;
    f->cos_table = (double *)malloc(sizeof(double) * n / 2);
    f->sin_table = (double *)malloc(sizeof(double) * n / 2);
    f->bitrev = (unsigned long *)malloc(sizeof(unsigned long) * n);
    if (!f->cos_table || !f->sin_table || !f->bitrev)
    {
        mmx_fft_free(f);
        return -1;
    }
    for (i = 0; i < n / 2; i++)
    {
        f->cos_table[i] = cos(2.0 * MMX_PI * (double)i / (double)n);
        f->sin_table[i] = sin(2.0 * MMX_PI * (double)i / (double)n);
    }
    for (t = n; t > 1; t >>= 1)
        bits++;
    for (i = 0; i < n; i++)
    {
        unsigned long r = 0, b, x = i;
        for (b = 0; b < bits; b++)
        {
            r = (r << 1) | (x & 1);
            x >>= 1;
        }
        f->bitrev[i] = r;
    }
    return 0;
}

void mmx_fft_free(MMXFft *f)
{
    if (!f)
        return;
    free(f->cos_table);
    free(f->sin_table);
    free(f->bitrev);
    memset(f, 0, sizeof(*f));
}

void mmx_fft_forward(const MMXFft *f, double *re, double *im)
{
    unsigned long n = f->n, i, len;

    for (i = 0; i < n; i++)
    {
        unsigned long j = f->bitrev[i];
        if (i < j)
        {
            double t = re[i]; re[i] = re[j]; re[j] = t;
            t = im[i]; im[i] = im[j]; im[j] = t;
        }
    }
    for (len = 2; len <= n; len <<= 1)
    {
        unsigned long half = len / 2, step = n / len, k;
        for (i = 0; i < n; i += len)
        {
            for (k = 0; k < half; k++)
            {
                double wr = f->cos_table[k * step], wi = -f->sin_table[k * step];
                double xr = re[i + k + half], xi = im[i + k + half];
                double vr = xr * wr - xi * wi, vi = xr * wi + xi * wr;
                re[i + k + half] = re[i + k] - vr;
                im[i + k + half] = im[i + k] - vi;
                re[i + k] += vr;
                im[i + k] += vi;
            }
        }
    }
}
