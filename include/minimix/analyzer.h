#ifndef MMX_ANALYZER_H
#define MMX_ANALYZER_H

/* Global analysis of the complete track:
   every frame is compared with every earlier position of the file, candidates
   are aligned sample-exactly, scored in bits under the masking model, and a
   global dynamic program picks the reference plan. Nothing here is streamed. */

#include "minimix/audio_buffer.h"
#include "minimix/codec.h"

#define MMX_ANALYSIS_MAX 9
#define MMX_GAIN_SIDE_BITS 42.0   /* side information of a reference per source and channel (8 band gains with flag and delta, measured) */

typedef struct
{
    unsigned char n_sources;
    long long src_start[MMX_MAX_SOURCES];  /* source window starts for this frame */
    float ref_bits;                        /* estimated bits with this plan */
    float audio_bits;                      /* estimated bits as plain audio */
    float best_bits;                       /* cheapest candidate seen for this frame (achievable) */
    float band_gain_db[MMX_EQ_BANDS];      /* prediction gain of the chosen candidate per EQ band */
    unsigned char depth;
    /* the planner's runner-up at this frame (the other lineage with the cheapest
       path through the frame, 0 sources = none): the encoder can trial-code a
       block with both lineages and keep the cheaper one (an experiment,
       MMX_TRIAL_PLAN=1; measured, it does not pay) */
    unsigned char alt_n_sources;
    long long alt_src_start[MMX_MAX_SOURCES];
    float alt_bits;                        /* its path cost, for the record */
} MMXFramePlan;

typedef struct
{
    unsigned int level;        /* 1..9 */
    unsigned int quality;      /* 1..10 */
    unsigned char max_depth;
    int verbose;
    int lossless;              /* cost model of the lossless coder, single broadband gain per channel */
    unsigned long seg_frames;  /* > 0 (lossless --pld): a source, with the 256 samples around it the LL2 source stage
                                  reads, must lie inside the frame's own segment of seg_frames frames */
    int tns;                   /* relax the sub-window rule where temporal noise shaping will be used */
    unsigned int source_bits;
    const MMXAudioBuffer *sources; /* signal the references are scored against: NULL = the input, or the
                                      decoded output of a first closed-loop pass (what the encoder will
                                      really predict from, quantization noise included) */
    double cal_audio[2], cal_ref[2]; /* calibration of the bit estimates (audio frames / residuals, [stationary,
                                        transient]) against the real coder, measured in a first pass; 0 = 1 */
} MMXAnalysisParams;

/* A repeated section found on the section level (0.75 s fingerprints). */
typedef struct
{
    double start_s, end_s;      /* the later occurrence */
    double source_s;            /* where it repeats from */
    double distance;            /* mean fingerprint distance (0 = identical spectra) */
} MMXRepeat;

typedef struct
{
    unsigned long frame_count;
    unsigned int channels;
    float *coefs;              /* [(f*ch + c)*HOP + k] original MDCT coefficients (L/R) */
    MMXFramePsy *psy;          /* [f*ch + c] */
    float *model_thr;          /* [(f*ch + c)*MMX_MAX_BANDS + b] the masking thresholds as the model
                                  left them: before the rate loop's offset (thr_scale) and before the
                                  clean-samples scaling. What the encoder's structured-error guard
                                  judges a leaking prediction against (see encoder.c). */
    unsigned char *transient;  /* [f] */
    unsigned char *silent;     /* [f] */
    MMXFramePlan *plan;        /* [f] */
    /* statistics for the analyzer report */
    unsigned long candidates_tested;
    unsigned long candidates_aligned;
    unsigned long planned_ref_frames;
    unsigned long planned_ref2_frames;
    unsigned long planned_blocks;
    unsigned long sections;
    unsigned long silent_frames;
    unsigned long transient_frames;
    double est_audio_bits;
    double est_plan_bits;
    double est_best_bits;      /* sum of the cheapest candidate per frame: the achievable bound of this search */
    int coded_estimate;        /* the plan was scored with mmx_frame_estimate_bits_coded (levels 7-9), else the fast estimate */
    double seconds;
    unsigned long cand_per_frame;
    double thr_scale;          /* factor the encoder applied to every thr[] (bitrate target); the
                                  window-shape thresholds of short/start/stop frames are derived from
                                  the unscaled frame_thr/sub_thr and apply it themselves */
    MMXRepeat *repeats;        /* section-level repeat map (song decomposition) */
    unsigned long repeat_count;
} MMXAnalysis;

void mmx_analysis_params_default(MMXAnalysisParams *p);

int mmx_analyzer_run(const MMXAudioBuffer *audio, MMXCodec *codec, const MMXAnalysisParams *params, MMXAnalysis *out);

void mmx_analysis_free(MMXAnalysis *a);

/* Convenience accessors. */
const float *mmx_analysis_coefs(const MMXAnalysis *a, unsigned long f, unsigned int c);
const MMXFramePsy *mmx_analysis_psy(const MMXAnalysis *a, unsigned long f, unsigned int c);
/* The model thresholds of one frame band row (never NULL: the current thr[] when
   no snapshot was taken). */
const float *mmx_analysis_model_thr(const MMXAnalysis *a, unsigned long f, unsigned int c);

#endif
