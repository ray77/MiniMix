#ifndef MMX_SPECTRUM_H
#define MMX_SPECTRUM_H

/* Leak-free spectral analysis for the psychoacoustic model: Blackman-Harris
   windowed FFT (sidelobes -92 dB). MDCT coefficients themselves are not usable
   for masking analysis, their sine window leaks loud low-frequency content
   into the high bands at -24 dB. Power is scaled to the MDCT calibration of
   the band layout (per-coefficient power of stationary noise), so thresholds
   derived from it apply directly to MDCT coefficients of the same window
   length. */

#include "fft.h"

typedef struct MMXSpectrum
{
    unsigned long n;        /* window length, spectrum has n/2 bins */
    MMXFft fft;
    double *window;
    double *re, *im;
    double scale;
} MMXSpectrum;

int mmx_spectrum_init(MMXSpectrum *s, unsigned long n);
void mmx_spectrum_free(MMXSpectrum *s);

/* x: n samples. amp: n/2 values, sqrt of the scaled power (usable as `coefs`
   for mmx_psy_analyze, which squares them). */
void mmx_spectrum_analyze(MMXSpectrum *s, const float *x, float *amp);

#endif
