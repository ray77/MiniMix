#ifndef MMX_PERCEPT_H
#define MMX_PERCEPT_H

/* Perceptual repeats ("what sounds the same is played the same" where the
   waveform search finds nothing). Every frame gets a descriptor of its
   spectral envelope (half-bark band levels with their masking thresholds)
   and its temporal envelope (the three inner 23 ms sub-windows per EQ band).
   Two frames are compared after a level fit per EQ band (the gain the
   syntax can express): the distance is the RMS deviation in dB over all
   (band) and (sub-window, EQ band) cells, every cell floored at the target's
   masking threshold, so detail the ear does not get is free and energy the
   source carries where the target is quiet (a ghost note) counts in full.
   Sections of 32 frames every 8 frames are matched against every earlier
   lag of the file; the map says how much of the file has a match under a
   given distance, the plan turns the matches into runs with one lag each.
   This is an analysis tool (mmx analyze --percept): it reports what repeats
   perceptually, it does not encode anything. */

#include "minimix/audio_buffer.h"
#include "psymodel.h"

#define MMX_PERCEPT_DIM 48          /* half-bark bands in the descriptor (44.1 kHz: 43 bands, 42 below the quality-7 cutoff) */
#define MMX_PERCEPT_SUBW 3          /* inner sub-windows (offsets 0, 512, 1024 of the 2048 window) */
#define MMX_PERCEPT_CHROMA 12       /* pitch classes of the 55 Hz - 4 kHz spectrum (harmony: a riff in another chord is another riff) */
#define MMX_PERCEPT_SECTION 32      /* frames per section (0.74 s) */
#define MMX_PERCEPT_HOP 8           /* section hop (0.19 s) */
#define MMX_PERCEPT_DEFAULT_DB 4.0  /* a section counts as a repeat of its match under this distance */
#define MMX_PERCEPT_MIN_LAG_S 1.0   /* shortest lag the map reports (a 46 ms loop is stationarity, not a repeat) */

typedef struct
{
    float spec[MMX_PERCEPT_DIM];                        /* band level, dB */
    float thr[MMX_PERCEPT_DIM];                         /* masking threshold of the band (power * n), dB */
    float sub[MMX_PERCEPT_SUBW][MMX_EQ_BANDS];          /* sub-window level per EQ band, dB */
    float chroma[MMX_PERCEPT_CHROMA];                   /* pitch class level, dB below the strongest class (0 = strongest) */
    unsigned char audible[MMX_EQ_BANDS];                /* EQ band holds at least one band above its threshold */
} MMXPerceptFrame;

typedef struct
{
    unsigned long lag;         /* frames, 0 = no match */
    float dist;                /* RMS dB */
    float eq_dist[MMX_EQ_BANDS + 1]; /* per EQ band at the best match; slot MMX_EQ_BANDS = the pitch classes */
} MMXPerceptMatch;

typedef struct
{
    unsigned long f0, f1;      /* frames [f0, f1) play */
    unsigned long lag;         /* frames */
    long long src_start;       /* source window start of frame f0 (sample-exact, refined by the onset envelope) */
    long long offset;               /* the refinement in samples against the frame grid */
    double dist;               /* mean section distance of the run */
    float eq_dist[MMX_EQ_BANDS + 1];   /* per EQ band over the run's frames (slot MMX_EQ_BANDS = pitch classes) */
} MMXPerceptRun;

typedef struct
{
    unsigned long frame_count;
    unsigned int channels;
    unsigned long sample_rate;
    unsigned int dim;
    unsigned char eq_of[MMX_PERCEPT_DIM];
    MMXPerceptFrame *frames;   /* [f * channels + c] */
    unsigned char *silent;     /* [f] */
    /* the map */
    unsigned long sections;
    MMXPerceptMatch *best;     /* [section] any lag >= 2 frames */
    MMXPerceptMatch *best_1s;  /* [section] lag >= 1 s */
    MMXPerceptMatch *best_4s;  /* [section] lag >= 4 s */
    double adjacent_db;        /* median distance of a frame to its predecessor (stationarity reference) */
    double random_db;          /* median distance of random pairs with a lag >= 4 s */
    double adjacent_eq[MMX_EQ_BANDS + 1], random_eq[MMX_EQ_BANDS + 1];   /* the same per EQ band (mean of the per-band RMS) */
    double seconds;
    /* the plan */
    MMXPerceptRun *runs;
    unsigned long run_count;
} MMXPercept;

/* Descriptors and the all-lag map. `quality` sets the masking thresholds. */
int mmx_percept_analyze(const MMXAudioBuffer *audio, unsigned int quality, int verbose, MMXPercept *P);

/* Runs of frames that match an earlier section: sections with a match under
   max_db (lag >= min_lag_s), consecutive sections with the same lag merged,
   the run's offset refined to the sample by the energy envelope, sources
   that lie inside an earlier run redirected to that run's source. */
int mmx_percept_plan(const MMXAudioBuffer *audio, MMXPercept *P, double max_db, double min_lag_s);

/* Human-readable map to stdout (or JSON). */
void mmx_percept_print(const MMXPercept *P, double max_db, int json);

void mmx_percept_free(MMXPercept *P);

#endif
