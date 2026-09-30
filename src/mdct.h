#ifndef MMX_MDCT_H
#define MMX_MDCT_H

#include "fft.h"

/* MDCT of size N (N/2 coefficients) with sine window, TDAC perfect reconstruction.
   forward: N windowed input samples -> N/2 coefficients
   inverse: N/2 coefficients -> N windowed output samples for overlap-add */

typedef struct
{
    unsigned long n;          /* window length */
    unsigned long m;          /* n/2 coefficients */
    double *window;           /* n */
    double *pre_re, *pre_im;  /* m/2 pre-twiddles */
    double *post_re, *post_im;/* m/2 post-twiddles */
    double *work_re, *work_im;/* m/2 */
    double *fold;             /* m */
    MMXFft fft;               /* size m/2 */
} MMXMdct;

int mmx_mdct_init(MMXMdct *t, unsigned long n);
void mmx_mdct_free(MMXMdct *t);

/* x: n samples (unwindowed). out: m coefficients. Uses the transform's own (KBD) window. */
void mmx_mdct_forward(MMXMdct *t, const float *x, float *out);

/* in: m coefficients. y: n samples, already windowed, to be overlap-added. */
void mmx_mdct_inverse(MMXMdct *t, const float *in, float *y);

/* Same with an explicit window of n samples (block switching: start/stop shapes). */
void mmx_mdct_forward_w(MMXMdct *t, const float *x, const double *window, float *out);
void mmx_mdct_inverse_w(MMXMdct *t, const float *in, const double *window, float *y);

/* Block switching window shapes for a long transform of n samples with short
   transforms of ns samples (AAC style): the START window is the long rising
   half, a flat part, the short falling slope and zeros; STOP is its mirror.
   Both satisfy the TDAC condition against a LONG window on one side and the
   eight short windows on the other. */
int mmx_mdct_window_start(const MMXMdct *long_t, const MMXMdct *short_t, double *window);
int mmx_mdct_window_stop(const MMXMdct *long_t, const MMXMdct *short_t, double *window);

/* Offset of the first short window inside the long frame: (n - ns) / 2 - ns / 4 ...
   concretely (n/2 - ns/2) / 2 + n/4 - ns/2 = for n = 2048, ns = 256: 448. Short
   window k (0..n/ns/2 - 1 = 7) starts at offset + k * ns / 2. */
unsigned long mmx_mdct_short_offset(const MMXMdct *long_t, const MMXMdct *short_t);

#endif
