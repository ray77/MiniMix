#ifndef MMX_FFT_H
#define MMX_FFT_H

/* Radix-2 complex FFT with cached twiddles. Sizes must be powers of two. */

typedef struct
{
    unsigned long n;
    double *cos_table;   /* n/2 entries */
    double *sin_table;
    unsigned long *bitrev;
} MMXFft;

int mmx_fft_init(MMXFft *f, unsigned long n);
void mmx_fft_free(MMXFft *f);

/* In-place forward transform (e^{-i...}). */
void mmx_fft_forward(const MMXFft *f, double *re, double *im);

#endif
