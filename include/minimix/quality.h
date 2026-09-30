#ifndef MMX_QUALITY_H
#define MMX_QUALITY_H

/* Perceptual quality evaluation: noise-to-mask ratio per band against the
   psychoacoustic model of the original. The safety barrier of MiniMix:
   anything above 0 dB NMR is audible by definition of
   the model and must not be produced. */

#include "minimix/audio_buffer.h"
#include "minimix/codec.h"

/* A band more than this far below the original counts as a hole (see below). */
#define MMX_WARBLE_HOLE_DB 10.0

typedef struct
{
    double worst_nmr_db;        /* maximum band NMR over the whole signal */
    double mean_nmr_db;         /* mean over coded bands with signal */
    double bands_over_percent;  /* percentage of (frame, band) pairs with NMR > 0 dB */
    double frames_over_percent; /* percentage of frames with any band above 0 dB */
    unsigned long frames;
    double snr_db;              /* plain sample SNR for reference */
    double max_diff_lsb;        /* at the source bit depth */
    int passed;                 /* worst NMR within tolerance */
    unsigned long worst_frame;
    unsigned int worst_channel;
    unsigned int worst_band;
    /* pre-echo metric: time-domain SNR in the 46 ms before every detected attack */
    unsigned long attacks;
    unsigned long attacks_pre_echo;   /* attacks whose pre-attack SNR is below 10 dB */
    double worst_pre_attack_snr_db;
    double worst_pre_attack_seconds;
    /* the same three numbers per EQ band region (<200, 200-500, 500-1k, 1k-2k,
       2k-4k, 4k-8k, 8k-12k, >12k Hz; regions above the evaluated range are empty) */
    double eq_over_percent[MMX_EQ_BANDS];
    double eq_worst_nmr_db[MMX_EQ_BANDS];
    double eq_mean_nmr_db[MMX_EQ_BANDS];
    unsigned long eq_cells[MMX_EQ_BANDS];   /* evaluated (sub-frame, band) cells per region */
    /* energy match per region: the band level of the decoded file against the
       original, per (second, region) cell above 30 dB SPL. A regenerated high
       band is a different waveform of the right energy - the NMR above counts
       it as error, these two numbers say whether the energy is right. */
    double eq_level_abs_db[MMX_EQ_BANDS];   /* mean |decoded - original| level per region, dB */
    double eq_level_bias_db[MMX_EQ_BANDS];  /* mean signed level difference per region, dB */
    unsigned long eq_level_cells[MMX_EQ_BANDS];
    /* what the masking model does not represent (see quality.c, "beyond the model") */
    unsigned long clipped_decoded;    /* decoded samples at full scale of the source bit depth */
    unsigned long clipped_original;   /* ... in the original */
    unsigned long clipped_new;        /* decoded at full scale where the original is not: flat tops the codec made */
    unsigned long attacks_ghost;      /* attacks with a 5.8 ms block in the 46 ms before them where the error
                                         exceeds the signal (above -70 dBFS): a ghost note or pre-echo */
    double worst_ghost_db;            /* error over signal in that block, dB */
    double worst_ghost_seconds;
    double dc_worst_db;               /* DC of the error signal per second, worst, dBFS */
    double border_1024_db;            /* first-difference energy of the error within +-8 samples of the
                                         1024 grid over its mean everywhere, dB (0 = no border effects) */
    double border_128_db;             /* same for the 128 grid (short windows) */
    double level_worst_db;            /* band level per second, decoded against original: worst deviation */
    double level_worst_seconds;
    unsigned int level_worst_band;    /* EQ region of the worst deviation */
    double level_over_1db_percent;    /* (second, region) cells deviating by more than 1 dB */
    double level_over_3db_percent;
    /* gurgling ("Gluckern"): the band levels of the decoded signal against the
       original *inside* one second. Per (23 ms sub-window, band) cell whose
       original is above the absolute threshold of hearing the level difference
       decoded - original is taken (both floored at that threshold, so a band
       that goes silent counts as far down as it was audible). The warble index
       of a second is the rms over the bands of the standard deviation of that
       difference inside the second: a steady level error is not a warble, a
       fluctuating one is. A hole is a cell more than MMX_WARBLE_HOLE_DB too
       quiet, a toggle a band whose hole state changes from one sub-window to
       the next inside the second - the band zeroed in one frame and coded in
       the next, which is what the ear hears as bubbling. Opus never zeroes a
       band (0.0 toggles/s by construction). MMX_DEBUG_WARBLE=1 lists the worst
       seconds. */
    double warble_db;                 /* mean warble index over the seconds */
    double warble_worst_db;
    double warble_worst_seconds;
    double hole_percent;              /* share of the evaluated cells that are holes */
    double toggles_per_second;
    double toggles_worst;             /* toggles in the worst second */
    double toggles_worst_seconds;
    /* stereo image (stereo input only): per (sub-frame, band) cell whose original holds signal in both
       channels, the L/R energy ratio (ILD, dB) and the normalized cross-correlation of the complex
       spectra (coherence, -1..1), decoded against original; per EQ region and over all cells, and over
       the cells from 4 kHz up, where intensity coding works. Intensity stereo keeps the ratio and
       raises the coherence to 1, a widened or collapsed image shows here and not in the NMR. */
    unsigned long stereo_cells[MMX_EQ_BANDS];
    double stereo_ild_err_db[MMX_EQ_BANDS];       /* mean |ILD_dec - ILD_orig| */
    double stereo_ild_over_percent[MMX_EQ_BANDS]; /* cells off by more than 2 dB */
    double stereo_coh_err[MMX_EQ_BANDS];          /* mean |coh_dec - coh_orig| */
    double stereo_coh_over_percent[MMX_EQ_BANDS]; /* cells off by more than 0.3 */
    double stereo_ild_err_db_all, stereo_coh_err_all;
    double stereo_ild_err_db_hi, stereo_coh_err_hi;
    /* ---- perceptually weighted second score (see quality.c, "weighted score";
       never a replacement for bands_over_percent, always reported next to it) ----
       Same cells, same 0 dB criterion, two differences: every cell carries the
       perceptual weight w = (ERB width) x (specific loudness of the original in
       that cell), and above w_crossover_hz the cell's error is the band *energy*
       error, not the waveform error, because the ear has no usable temporal fine
       structure there. */
    unsigned int w_crossover_hz;      /* fine-structure / band-energy crossover used */
    double w_over_percent;            /* share of the perceptual weight whose cell is above the threshold */
    double w_mean_nmr_db;             /* weight-weighted mean NMR (energy rule above the crossover) */
    double w_noise_loudness_percent;  /* loudness of the unmasked noise as a share of the music's loudness */
    double eq_w_over_percent[MMX_EQ_BANDS];  /* the weighted error share per EQ region */
    double eq_weight_percent[MMX_EQ_BANDS];  /* where the perceptual weight sits (sums to 100) */
    /* high band energy error: how far the decoded band energy above the crossover
       deviates from the original, in dB - what matters when a coder substitutes
       noise there (an energy-matched substitution is 0 dB, a hole is -inf). */
    unsigned long hi_energy_cells;
    double hi_energy_err_db;          /* mean |10 log10(E_dec / E_orig)| */
    double hi_energy_bias_db;         /* mean signed value: negative = too dull, positive = too bright */
    double hi_energy_over3_percent;   /* cells off by more than 3 dB */
    double hi_energy_worst_db;        /* worst single cell (an outlier, not a summary) */
} MMXQualityReport;

/* Frame-level NMR (coded in L/R): orig/rec are MDCT coefficients per channel.
   skip[band] (optional) excludes bands, e.g. substituted noise bands. */
double mmx_quality_frame_nmr(const MMXCodec *codec, const MMXFramePsy *psy,
                             const float *orig, const float *rec, const unsigned char *skip, unsigned int *bands_over);

/* Whole-signal comparison of two PCM buffers (same format). Tolerance in dB
   above 0 dB NMR still accepted (rounding of scalefactors). */
int mmx_quality_compare(const MMXAudioBuffer *original, const MMXAudioBuffer *decoded, unsigned int quality,
                        unsigned int max_hz, double tolerance_db, MMXQualityReport *report);
/* max_hz limits the evaluated bands (0 = bandwidth of the quality level), so
   codecs with different lowpass filters can be judged on the same range. */

#endif
