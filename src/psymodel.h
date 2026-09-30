#ifndef MMX_PSYMODEL_H
#define MMX_PSYMODEL_H

/* Psychoacoustic model: bark-scaled band layout over the MDCT bins, masking
   threshold per band (spreading, tonality, absolute threshold, transient
   tightening) and the quality-level mapping. All quality decisions in MiniMix
   derive from these thresholds. */

#define MMX_MAX_BANDS 64
#define MMX_EQ_BANDS 8
#define MMX_QUALITY_MAX 10

/* Sub-window grid of the frame analysis: five 1024-sample windows every 512
   samples, the first starting 512 samples before the frame's 2048 window.
   MMXFramePsy.sub_thr[w] is stored in frame (2048-MDCT) units, weighted by the
   share of the frame's noise that lands in sub-window w: 2 for the three inner
   ones (a 1024-MDCT of the same time-domain noise has half the power per
   coefficient), 8 for the two outer ones, which only see the window tails. */
#define MMX_SUB_WINDOWS 5
#define MMX_SUB_WIN 1024
#define MMX_SUB_STEP 512
#define MMX_SUB_WEIGHT(w) (((w) == 0 || (w) == MMX_SUB_WINDOWS - 1) ? 8.0f : 2.0f)

typedef struct
{
    unsigned long sample_rate;
    unsigned long m;                          /* coefficients per frame */
    unsigned int band_count;
    unsigned int band_start[MMX_MAX_BANDS + 1];
    float band_hz[MMX_MAX_BANDS];             /* center frequency */
    float band_bark[MMX_MAX_BANDS];
    float abs_thr[MMX_MAX_BANDS];             /* absolute hearing threshold, power per coefficient */
    unsigned char eq_band[MMX_MAX_BANDS];     /* which of the 8 EQ/gain bands this band belongs to */
    float spread[MMX_MAX_BANDS][MMX_MAX_BANDS];/* normalized spreading matrix */
    unsigned int cutoff_band[MMX_QUALITY_MAX + 1]; /* first band zeroed at each quality */
    double fs_peak_power;                     /* MDCT peak power of a full-scale sine (calibration) */
    /* band replication (BWE, bitstream revision 6): from bwe_band on the bands carry
       only an energy level and a noise-mix index; the decoder regenerates them from
       the already reconstructed source region [bwe_src0, band_start[bwe_band]). Off
       when bwe_band == band_count. Part of the format: set from the file header. */
    unsigned int bwe_band;
    unsigned int bwe_src0;
    unsigned int bwe_mode;                    /* 0 = octave transposition, 1 = linear shift */
    unsigned int lowrate;                     /* EPB (revision 9): every long-frame band carries its energy, see mmx_epb_finish */
    double epb_fold_hz;                       /* EPB fill policy (file header bytes 83-85): fold floor, ... */
    unsigned int epb_noise, epb_coded, epb_ref;   /* ... noise under the floor, fill inside coded bands, fill kept out of the reference ... */
    double epb_fill_gain;                     /* ... and the fill level as a factor */
} MMXBandLayout;

typedef struct
{
    float thr[MMX_MAX_BANDS];     /* allowed noise power per coefficient */
    float energy[MMX_MAX_BANDS];  /* total band energy (sum of squares) */
    float crest[MMX_MAX_BANDS];   /* temporal concentration of the band's energy inside the
                                     window: peak sub-window energy / mean (>= 1). A band may only
                                     be dropped when energy * crest stays below the threshold. */
    float frame_thr[MMX_MAX_BANDS];   /* masking threshold of the whole window (no sub-window rule) */
    float sub_thr[5][MMX_MAX_BANDS];  /* weighted thresholds of the five overlapping sub-windows, in
                                         frame units; thr = min(frame_thr, sub_thr[w]) unless the
                                         encoder shapes the noise in time (TNS) */
    float flat[MMX_MAX_BANDS];
    float prom[MMX_MAX_BANDS];   /* strongest spectral line of the band over the local median of its neighbours, dB (0 = none) */    /* spectral flatness inside the band (geometric / arithmetic mean of
                                     the bin powers, 0..1): ~0.56 for white noise through the leak-free
                                     window, far lower when the band holds a tone (noise substitution) */
    float tonality;               /* 0 = noise ... 1 = tonal */
    float tonality_eq[MMX_EQ_BANDS]; /* tonality used per EQ region (see psymodel.c, region tonality) */
    int transient;
    float attack_db;              /* strongest attack inside the window: energy of the attack
                                     sub-block over the mean before it (0 when none) */
    unsigned int attack_pos;      /* sample offset of the attack sub-block inside the window */
} MMXFramePsy;

int mmx_bands_init(MMXBandLayout *L, unsigned long sample_rate, unsigned long m);

/* Band replication crossover: the first band at or above `hz` becomes the first
   regenerated band (hz == 0 switches it off). The source region is the octave
   below the crossover. Returns the band index that was set. */
unsigned int mmx_bands_set_bwe(MMXBandLayout *L, double hz, unsigned int mode);
void mmx_bands_set_epb(MMXBandLayout *L, unsigned int lowrate, unsigned int fold_250hz, unsigned int flags, int fill_db);

/* Threshold offset in dB and bandwidth for a quality level 1..10. */
double mmx_quality_offset_db(unsigned int quality);
double mmx_quality_bandwidth_hz(unsigned int quality);

/* coefs: m MDCT coefficients of one channel. window: the n = 2m time samples
   the frame was computed from (for transient detection), may be NULL. */
void mmx_psy_analyze(const MMXBandLayout *L, const float *coefs, const float *window,
                     unsigned int quality, MMXFramePsy *out);

/* Pre-echo rule with the signal before the attack (replaces the transient
   scale of mmx_psy_analyze while it is active, see mmx_psy_pre_attack_db): the
   noise a window with an attack (out->transient) spreads before the attack is
   masked only by the signal there. `pre` holds the MMX_PRE_ATTACK samples right
   before out->attack_pos, `pre_spec` / `pre_bands` analyse them (a spectrum of
   MMX_PRE_ATTACK samples, a layout of MMX_PRE_ATTACK / 2 coefficients). Every
   threshold of `out` is limited to margin_db below that signal's own masking
   threshold. */
#define MMX_PRE_ATTACK 512
struct MMXSpectrum;
void mmx_psy_pre_attack(const MMXBandLayout *L, const MMXBandLayout *pre_bands, struct MMXSpectrum *pre_spec,
                        const float *pre, unsigned int quality, double margin_db, MMXFramePsy *out);
/* Margin of the pre-attack rule in dB, or a negative value when the rule is
   off and mmx_psy_analyze applies the transient scale itself. */
double mmx_psy_pre_attack_db(void);

/* Resolves every environment knob of the model once, so the per-frame analysis
   can be run by several threads without writing to a shared cache. */
void mmx_psy_prewarm(void);

/* Limits threshold cliffs between neighbouring bands (encoder-side constraint,
   see psymodel.c). Apply after all other threshold adjustments. */
void mmx_psy_limit_cliffs(MMXFramePsy *p, unsigned int band_count);

/* Temporal consistency: quantization noise of a frame spreads over its whole
   2048-sample window, i.e. into the neighbouring frames. Noise before an attack
   is not masked (pre-echo), noise after a loud frame is partly post-masked.
   Call after all frames are analyzed, with prev/next of the same channel
   (NULL at the edges). */
void mmx_psy_temporal(const MMXFramePsy *prev, MMXFramePsy *cur, const MMXFramePsy *next);

/* Threshold for a mid/side coded pair: conservative combination of both channels. */
void mmx_psy_stereo_threshold(const MMXBandLayout *L, const MMXFramePsy *left, const MMXFramePsy *right,
                              float *thr_out);

/* Noise-to-mask ratio in dB for a band given the noise energy in that band. */
double mmx_psy_nmr_db(const MMXBandLayout *L, const MMXFramePsy *p, unsigned int band, double noise_energy);

/* Signal level of an MDCT power relative to full scale, in dB SPL (96 dB = 0 dBFS sine). */
double mmx_psy_spl_db(const MMXBandLayout *L, double power);

#endif
