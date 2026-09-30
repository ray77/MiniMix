#ifndef MMX_FRAMECODEC_H
#define MMX_FRAMECODEC_H

/* Frame syntax of the MiniMix native MDCT codec. A frame is 1024 new samples
   (2048-sample window). AUDIO frames and residual frames share the syntax; the
   only difference is whether a prediction is added at the decoder. */

#include "rangecoder.h"
#include "psymodel.h"
#include "tns.h"

#define MMX_HOP 1024
#define MMX_WIN 2048
#define MMX_MAX_CH 8
#define MMX_MAX_SOURCES 2
#define MMX_GAIN_OFF (-128)   /* gain index meaning exactly zero */
#define MMX_GAIN_MIN (-72)    /* -36 dB */
#define MMX_GAIN_MAX 24       /* +12 dB */
#define MMX_SF_COUNT 128

/* Block switching (transients, pre-echo). A frame is coded with one
   2048-sample transform (LONG, or START/STOP with asymmetric windows next to
   short frames) or with eight 256-sample transforms (SHORT). Short frames
   store their 8 x 128 coefficients group-interleaved in q[ch][g*128 + j]. */
#define MMX_BT_LONG 0
#define MMX_BT_START 1
#define MMX_BT_SHORT 2
#define MMX_BT_STOP 3
#define MMX_SHORT_GROUPS 8
#define MMX_SHORT_M 128
#define MMX_SHORT_BANDS 32

/* Perceptual noise substitution (noise-like high bands carry only
   their energy). A long-frame band from the first band at MMX_PNS_MIN_HZ on may
   be flagged as a noise band: it carries a level index instead of coefficients
   and the decoder fills it with deterministic pseudo-random noise of that
   energy (seeded from frame, channel and band, so the encoder's closed loop
   reconstructs the same noise). The prediction is not used in a noise band. */
#define MMX_PNS_MIN_HZ 2000

/* Intensity stereo (at low rates the ear localizes the highs by
   level only). A band of a long stereo frame from the first band at
   MMX_IS_MIN_HZ on may be flagged as an intensity band: channel 0 carries the
   band's mid coefficients once (scaled so that the band keeps the energy of
   both channels), a position index gives the L/R energy ratio in
   MMX_IS_STEP_DB steps and channel 1 carries nothing for the band; the decoder
   reconstructs L = gl * mid and R = gr * mid with gl^2 + gr^2 = 2 (see
   mmx_is_gains). The prediction of the band is used as the mid of the two
   channels' predictions. Not combined with noise bands or TNS. */
#define MMX_IS_MIN_HZ 2000
#define MMX_IS_STEP_DB 1.5
#define MMX_IS_POS_MAX 16       /* +-24 dB */
/* Noise filling (the USAC / Opus principle: a coded band keeps its energy at
   low rates). In a coded long-frame band from the first substitutable band on
   the coefficients that quantized to zero are filled by the decoder with
   deterministic noise whose rms is a transmitted fraction of the band's step:
   one level per channel and EQ region (from the region of MMX_PNS_MIN_HZ on),
   0 = off, else rms = step * 2^((level - MMX_NF_LEVELS) / 4), i.e. 1.5 dB
   steps from 0.07 to 0.84 steps. The noise is seeded like the noise bands,
   so the encoder's closed loop reconstructs it. A band that quantized to zero
   as a whole keeps its energy as a noise band instead (see the encoder). */
/* Band replication (BWE, bitstream revision 6). Above a crossover carried in the
   file header the encoder stops coding coefficients: every band of a long frame
   from MMXBandLayout.bwe_band on carries a zero flag and, when it is not zero,
   an energy level (the same 1.5 dB scalefactor grid as a noise band) and a
   2-bit noise-mix index. The decoder regenerates the band from the already
   reconstructed source region an octave below the crossover, mixes in
   deterministic noise according to the index and scales the result to the
   transmitted energy. The band's prediction is not used (like a noise band). */
#define MMX_BWE_MIX_MAX 3       /* mix index 0..3: noise share 0, 1/3, 2/3, 1 of the band energy */

#define MMX_NF_LEVELS 16
#define MMX_NF_LEVEL_DEFAULT 9      /* 0.30 steps, about the rms of the dead zone's zeros: the first level of a region is coded against it */

typedef struct
{
    int stereo_ms;                                        /* coded channel domain */
    unsigned char block_type;                             /* MMX_BT_* */
    unsigned char sf[MMX_MAX_CH][MMX_MAX_BANDS];          /* quantizer step index per band */
    unsigned char band_zero[MMX_MAX_CH][MMX_MAX_BANDS];   /* band has no coded coefficients */
    unsigned char band_noise[MMX_MAX_CH][MMX_MAX_BANDS];  /* noise band: sf holds the level index (rms = step), q = 0 */
    unsigned char band_is[MMX_MAX_BANDS];                 /* intensity band: channel 0 holds the mid, channel 1 nothing */
    unsigned char band_bwe[MMX_MAX_CH][MMX_MAX_BANDS];    /* replicated band: sf holds the level index, q = 0 */
    unsigned char nrg[MMX_MAX_CH][MMX_MAX_BANDS];         /* EPB: energy index of the band (rms per bin on the sf grid, 0 = silent) */
    unsigned char bwe_mix[MMX_MAX_CH][MMX_MAX_BANDS];     /* noise share of a replicated band, 0..MMX_BWE_MIX_MAX */
    signed char is_pos[MMX_MAX_BANDS];                    /* position index of an intensity band (L/R energy ratio) */
    unsigned char nf_level[MMX_MAX_CH][MMX_EQ_BANDS];     /* noise filling level per EQ region (long frames), 0 = off */
    unsigned char sf_s[MMX_MAX_CH][MMX_SHORT_GROUPS][MMX_SHORT_BANDS];       /* short frames: per group */
    unsigned char band_zero_s[MMX_MAX_CH][MMX_SHORT_GROUPS][MMX_SHORT_BANDS];
    signed char gain[MMX_MAX_SOURCES][MMX_MAX_CH][MMX_EQ_BANDS]; /* gain index or MMX_GAIN_OFF */
    unsigned char polarity[MMX_MAX_SOURCES][MMX_MAX_CH];  /* 1 = inverted source */
    MMXTns tns[MMX_MAX_CH];                               /* temporal noise shaping per coded channel */
    int *q[MMX_MAX_CH];                                   /* m quantized coefficients per channel */
    unsigned int channels;
    unsigned long m;
} MMXFrameSyntax;

typedef struct
{
    MMXProb stereo;
    MMXProb block_type[4][4];        /* [previous type] 2-bit tree */
    MMXProb zero_band[2][MMX_MAX_BANDS];
    MMXProb sf[2][22];
    MMXProb zero_band_s[2][MMX_SHORT_BANDS];
    MMXProb sf_s[2][22];
    MMXProb coef_zero[4][3][6][3];   /* [mode: audio, residual, short audio, short residual][freq class][neighbour class][previous class] */
    MMXProb coef_gt1[4][3][6][3];
    MMXProb coef_gt2[4][3][6][3];
    MMXProb coef_eg[4][3][20];
    MMXProb gain_off[2];
    MMXProb gain[22];
    MMXProb polarity;
    MMXProb tns_flag[2];
    MMXProb tns_order[20];
    MMXProb pns_flag[2][2][2];       /* [mode][previous band noise][same band noise in the previous frame] */
    MMXProb pns_level[2][22];        /* level delta against the channel's previous noise level */
    MMXProb is_flag[2][2][2];        /* [mode][previous band intensity][same band intensity in the previous frame] */
    MMXProb is_pos[22];              /* position delta against the prediction (previous frame's, else the previous band's) */
    MMXProb bwe_level[2][22];        /* level delta against the band's level in the previous frame */
    MMXProb bwe_mix_p[2][22];        /* mix-index delta against the band's mix in the previous frame */
    MMXProb nf_flag[2][2];           /* [mode][the region had a level in the previous frame] */
    MMXProb nf_level[2][22];         /* level delta against the region's previous level (MMX_NF_LEVEL_DEFAULT when it had none) */
    MMXProb nrg_ctx[2][22];          /* EPB: energy index delta against the band's index in the previous frame (else the band below) */
    /* inter-frame state (reset at block start, identical in encoder and decoder) */
    int has_prev;
    unsigned char prev_sf[MMX_MAX_CH];      /* last coded scalefactor of the previous frame */
    unsigned char prev_q[MMX_MAX_CH][MMX_HOP]; /* magnitude class (0,1,2) of the previous frame's coefficients */
    signed char prev_gain[MMX_MAX_SOURCES][MMX_MAX_CH][MMX_EQ_BANDS];
    unsigned char prev_polarity[MMX_MAX_SOURCES][MMX_MAX_CH];
    unsigned char prev_tns[MMX_MAX_CH];
    unsigned char prev_block_type;
    unsigned char prev_pns[MMX_MAX_CH][MMX_MAX_BANDS];  /* noise flags of the previous frame */
    unsigned char prev_pns_level[MMX_MAX_CH];           /* last coded noise level of the channel */
    unsigned char prev_is[MMX_MAX_BANDS];               /* intensity flags of the previous frame */
    signed char prev_is_pos[MMX_MAX_BANDS];             /* ... and their positions */
    unsigned char prev_nf[MMX_MAX_CH][MMX_EQ_BANDS];    /* noise filling levels of the previous frame (0 after a short frame) */
    unsigned char prev_bwe[MMX_MAX_CH][MMX_MAX_BANDS];        /* the band was replicated in the previous long frame */
    unsigned char prev_bwe_level[MMX_MAX_CH][MMX_MAX_BANDS];  /* ... and its level */
    unsigned char prev_bwe_mix[MMX_MAX_CH][MMX_MAX_BANDS];    /* ... and its mix index */
    unsigned char prev_bwe_last[MMX_MAX_CH];                  /* last replicated level of the channel (prediction for a new band) */
    unsigned char prev_nrg[MMX_MAX_CH][MMX_MAX_BANDS];        /* EPB: energy index of the band in the previous long frame */
} MMXCodecContexts;

int mmx_frame_syntax_init(MMXFrameSyntax *s, unsigned int channels, unsigned long m);
void mmx_frame_syntax_free(MMXFrameSyntax *s);
void mmx_frame_syntax_copy(MMXFrameSyntax *dst, const MMXFrameSyntax *src);

void mmx_contexts_init(MMXCodecContexts *c);

/* Quantizer step for a scalefactor index. */
double mmx_sf_step(unsigned int sf);
/* Largest scalefactor whose step keeps uniform quantization noise below thr (power per coefficient). */
unsigned int mmx_sf_for_threshold(double thr);
double mmx_gain_value(signed char index, unsigned char polarity);
signed char mmx_gain_index(double gain, unsigned char *polarity);

/* Rate loop only: switches the trellis quantizer off (0) and on again (1) for
   the duration of a cheap probe encode. MMX_RDOQ=0 still wins. */
void mmx_frame_rdoq_set_enabled(int on);

/* Quantizes coded-domain residual/audio coefficients `x[ch][m]` against the
   thresholds `thr[ch][band]`, filling sf, band_zero and q. Bands from
   `cutoff_band` upwards are zeroed. The actual noise of every band is measured
   and the step tightened until it is below the threshold. For mid/side frames
   `thr_lr[2][band]` (the L/R thresholds) must be given: the L/R noise that
   results from the M/S errors is checked as well. With the coder contexts
   `ctx` (the state the frame will be coded with; `mode` 0 = audio, 1 =
   residual) the quantization is rate-distortion optimized: per band the
   coarsest scalefactor and per coefficient the rounding that cost the fewest
   bits while the measured noise stays below the threshold (MMX_RDOQ=0 turns
   it off). ctx NULL = plain quantizer. */
void mmx_frame_quantize(const MMXBandLayout *L, MMXFrameSyntax *s, float *const *x,
                        const float *const *thr, unsigned int cutoff_band, const float *const *thr_lr,
                        const float *const *crest, const MMXCodecContexts *ctx, unsigned int mode);
/* crest[ch][band] (may be NULL): temporal concentration of the band's energy;
   a band is only dropped when energy * crest stays below the allowed noise. */
/* Noise filling (encoder side, long frames): from `first_band` on, in regions
   no source predicts (every region with `resid`), the distortion of a
   coefficient that quantizes to zero counts alpha[EQ region] times its energy
   in the quantizer, because the decoder fills it with noise of that energy.
   Passed per call to mmx_frame_quantize_ext; NULL (or alpha 1) = plain. */
typedef struct
{
    double alpha[MMX_EQ_BANDS];   /* per EQ region; < 1 discounts the zeros there */
    unsigned int first_band;      /* no discount below this band */
    int resid;                    /* discount the regions a source predicts as well */
} MMXNfZeroWeight;

/* Intensity stereo (encoder side): the candidate bands of a long stereo frame
   for mmx_frame_quantize_ext. A candidate band is coded as an intensity band
   when the mid band plus its flag and position cost fewer bits than the two
   channels' bands (the decision is taken with the rate-distortion quantizer,
   so it needs the coder contexts; the plain quantizer ignores the plan). */
typedef struct
{
    unsigned int first_band;            /* 0 = no intensity in this frame */
    unsigned char cand[MMX_MAX_BANDS];  /* the band may be coded as an intensity band */
    signed char pos[MMX_MAX_BANDS];     /* its position index */
    float thr[MMX_MAX_BANDS];           /* allowed noise per coefficient of its mid band */
    float err[2][MMX_MAX_BANDS];        /* energy of the dropped difference per channel (L - gl mid, R - gr mid) */
    const float *x;                     /* the mid target (residual) coefficients, m values */
    double bias;                        /* intensity wins when its bits times this are below the two bands' bits (1 = by bits alone) */
    unsigned int n_range, n_rho, n_slack; /* debug: bands with signal in the range, rejected by coherence, by the slack guard */
} MMXIntensityPlan;

/* mmx_frame_quantize with the encoder's two extra tools: intensity candidates
   (`is`) and the noise-filling zero weight (`nfz`); either may be NULL. */
void mmx_frame_quantize_ext(const MMXBandLayout *L, MMXFrameSyntax *s, float *const *x,
                            const float *const *thr, unsigned int cutoff_band, const float *const *thr_lr,
                            const float *const *crest, const MMXCodecContexts *ctx, unsigned int mode,
                            const MMXIntensityPlan *is, const MMXNfZeroWeight *nfz);

/* First band that may be an intensity band (band center >= MMX_IS_MIN_HZ). */
unsigned int mmx_is_first_band(const MMXBandLayout *L);
/* Position index of a band with the L/R energies el and er (MMX_IS_STEP_DB steps of their ratio). */
signed char mmx_is_position(double el, double er);
/* Gains of L and R for a position index: gl^2 = 2 r / (1 + r), gr^2 = 2 / (1 + r) with
   r the energy ratio of the index, so that the band keeps the energy of the coded mid twice. */
void mmx_is_gains(signed char pos, float *gl, float *gr);

/* Dequantizes into out[ch][m] (coded domain). `frame` is the frame index on
   the timeline: it seeds the noise of noise bands (see MMX_PNS_MIN_HZ). */
void mmx_frame_dequantize(const MMXBandLayout *L, const MMXFrameSyntax *s, float *const *out, unsigned long frame);
/* EPB (revision 9): after the prediction is added, every long-frame band is normalised to its energy index; an
   uncoded band is folded from the reconstruction an octave below (noise in the lowest bands, noise when the source
   is empty). Encoder closed loop and decoder both run this. */
void mmx_epb_finish(const MMXBandLayout *L, const MMXFrameSyntax *s, float *const *rec, unsigned long frame);
unsigned int mmx_epb_level(double energy, unsigned long n);

/* Encoder, minimum allocation (the energy-preservation rule against holes and
   gurgling, see encoder.c): quantizes band `b` of `x` into `q[k0..k1)` so that
   it carries at least one coded coefficient. The step starts at the one
   closest to the band's largest coefficient (so a single coefficient
   reconstructs to about that value) and is refined by 1.5 dB at most `steps`
   times while the reconstructed band energy stays below `keep_rel` of the
   band's own. Returns the scalefactor; 0 with everything zeroed for an empty
   band. */
unsigned int mmx_frame_keep_band(const MMXBandLayout *L, unsigned int b, const float *x, int *q,
                                 double keep_rel, int steps);

/* First band that may be a noise band (band center >= MMX_PNS_MIN_HZ). */
unsigned int mmx_pns_first_band(const MMXBandLayout *L);
/* Level index whose noise rms is closest to sqrt(energy / n). */
unsigned int mmx_pns_level_index(double energy, unsigned long n);

/* Band replication. mmx_bwe_source_bin maps a coefficient of the replicated
   region onto its source coefficient below the crossover (part of the format).
   mmx_bwe_regenerate overwrites every replicated band of a long frame in `rec`
   (the finished reconstruction of the coded region, coded channel domain) with
   the patched, noise-mixed and energy-matched band; it runs in both the
   encoder's closed loop and the decoder. mmx_bwe_mix_index picks the encoder's
   noise share from the spectral flatness of source and target band. */
unsigned long mmx_bwe_source_bin(const MMXBandLayout *L, unsigned long k);
void mmx_bwe_regenerate(const MMXBandLayout *L, const MMXFrameSyntax *s, float *const *rec, unsigned long frame);
unsigned int mmx_bwe_mix_index(const MMXBandLayout *L, const float *x, unsigned int b);

/* Noise filling: rms of the fill in units of the band's step for a level (0
   for level 0), the level closest to a ratio (0 when the ratio is below the
   lowest level), and the first EQ region whose bands may be filled (the
   region of mmx_pns_first_band). */
double mmx_nf_ratio(unsigned int level);
unsigned int mmx_nf_level_index(double ratio);
unsigned int mmx_nf_first_region(const MMXBandLayout *L);
/* Encoder: sets s->nf_level to the levels that preserve the energy the zeros
   of the coded bands lost, per channel and region from `first_region` on
   (rms of the zeroed coefficients in step units over the region, scaled by
   `gain`); regions below first_region, regions with skip[c][e] set (may be
   NULL) and short frames get 0. Call after quantization (and after the noise
   bands are decided). */
void mmx_frame_noise_fill_levels(const MMXBandLayout *L, MMXFrameSyntax *s, float *const *x, unsigned int first_region,
                                 double gain, const unsigned char (*skip)[MMX_EQ_BANDS]);

/* Short frames: x[ch] holds 8 groups of 128 coefficients, thr[ch][g][b] the
   thresholds per group and short band (Ls = the 128-bin layout). */
void mmx_frame_quantize_short(const MMXBandLayout *Ls, MMXFrameSyntax *s, float *const *x,
                              const float (*const *thr)[MMX_SHORT_BANDS], unsigned int cutoff_band,
                              const float (*const *thr_lr)[MMX_SHORT_BANDS], const MMXCodecContexts *ctx, unsigned int mode);
void mmx_frame_dequantize_short(const MMXBandLayout *Ls, const MMXFrameSyntax *s, float *const *out);
double mmx_frame_estimate_bits_short(const MMXBandLayout *Ls, const float *x, const float (*thr)[MMX_SHORT_BANDS], unsigned int cutoff_band);

/* Entropy coding. n_sources tells how many gain sets are present. Ls is the
   short-block layout (used when s->block_type == MMX_BT_SHORT). */
void mmx_frame_encode(MMXRangeEncoder *rc, MMXCodecContexts *ctx, const MMXBandLayout *L, const MMXBandLayout *Ls,
                      const MMXFrameSyntax *s, unsigned int n_sources);
int mmx_frame_decode(MMXRangeDecoder *rc, MMXCodecContexts *ctx, const MMXBandLayout *L, const MMXBandLayout *Ls,
                     MMXFrameSyntax *s, unsigned int n_sources);

/* Fast bit estimate for coefficients x[m] under thresholds thr[band] (no coding). */
double mmx_frame_estimate_bits(const MMXBandLayout *L, const float *x, const float *thr, unsigned int cutoff_band);
/* Estimate that mirrors the coder: every band is quantized exactly like
   mmx_frame_quantize does (dead zone, step tightened until the measured noise
   is under the threshold, dropped when it lies under the threshold, crest as
   there; NULL = none), and the quantized values are charged at the empirical
   entropy of the coder's own contexts inside the frame (zero / >1 / >2 flags
   per frequency and neighbour class), plus the signs and the Exp-Golomb
   magnitudes. Measured on title A the fast estimate charges residuals 27 %
   and audio 13 % less than the range coder (the flat 0.25 bit per zero misses
   the zeros next to non-zeros); this one is meant for ranking candidates
   in the analyzer, where that difference decides between reference and
   audio. Slower (the quantizer runs). */
double mmx_frame_estimate_bits_coded(const MMXBandLayout *L, const float *x, const float *thr, unsigned int cutoff_band,
                                     const float *crest);

/* Debug (MMX_DEBUG_BITS): books the next mmx_frame_encode call as a frame of
   class `cls` and attributes its bits to the EQ band regions (plus frame side
   info and short frames). Classes: 0 AUDIO stationary, 1 AUDIO transient,
   2 REF stationary, 3 REF transient. The print goes to stderr. */
#define MMX_BITDUMP_CLASSES 4
void mmx_frame_bitdump_next(unsigned int cls);
void mmx_frame_bitdump_print(void);

/* Books the exact bits of every band of the next mmx_frame_encode call (long
   frames: band flag, noise flag and level, scalefactor and coefficients) into
   out[ch][band]; the frame side information is not attributed. For the
   encoder's closed-loop cost decisions (arithmetic code positions, no
   estimate). A short frame books nothing. */
void mmx_frame_encode_book_bands(double (*out)[MMX_MAX_BANDS]);

/* M0: bits per syntax element of the next mmx_frame_encode / mmx_frame_estimate_bits_coded call (MMX_DEBUG_EST) */
enum { MMX_ELEM_HDR, MMX_ELEM_BWE, MMX_ELEM_IS, MMX_ELEM_ZERO, MMX_ELEM_PNS, MMX_ELEM_SF,
       MMX_ELEM_CZERO, MMX_ELEM_SIGN, MMX_ELEM_GT, MMX_ELEM_EG, MMX_ELEM_SHORT, MMX_ELEM_COUNT };
extern const char *const mmx_elem_names[MMX_ELEM_COUNT];
void mmx_frame_encode_book_elems(double *out);
void mmx_frame_estimate_book_elems(double *out);

#endif
