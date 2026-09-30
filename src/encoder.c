#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include "minimix/encoder.h"
#include "fft.h"
#include "minimix/codec.h"
#include "minimix/quality.h"
#include "reference_graph.h"
#include "threads.h"
#include "ll2codec.h"
#include "lossless.h"
#include "spectrum.h"
#include "crc32.h"
#include "mmx_io.h"
#include "log.h"

#define BASELINE_SEGMENT 64
#define TNS_MIN_GAIN 1.4
#define RATE_ITERATIONS 8           /* hard cap on probe encodes of one search */
#define RATE_TOLERANCE 0.015
#define RATE_PROBE_TOLERANCE 0.002  /* how close the cheap search drives its ESTIMATE before it hands
                                       the offset to the full encode; tighter than this buys noise,
                                       because the cheap/full factor itself drifts by 0.05-0.3 % */
#define RATE_PROBE_RATIO 0.983      /* full bytes / cheap bytes, the starting guess. Measured over 25
                                       probe points on three 20-s excerpts at 128 and 320 kbit/s and
                                       offsets -17..+15 dB: 0.9726-0.9953, mean 0.9832. It varies by
                                       track (about 0.003 over a whole search on one track), so the
                                       guess only has to land the first full encode near the target;
                                       that encode then measures the factor exactly. */
#define RATE_VERIFY_TOLERANCE 0.005 /* the verified full encode aims three times tighter than the
                                       loop's tolerance, so the file lands where the old bisection's
                                       exact probes used to land */
#define RATE_SLOPE0 (-0.045)        /* first guess for d ln(size) / d offset_db */
#define RATE_FULL_ENCODES 2         /* at most this many full-cost encodes per attempt: the factor
                                       re-measured at the first one is good to about 0.3 %, which is what
                                       a third encode would have to beat, and it cannot */
#define MMX_SECOND_PASS_LEVEL 7   /* analysis levels that plan twice (second_pass) */

/* ------------------------------------------------- tilted rate offset ----

   The bitrate target bisects ONE dB offset and lifts every masking threshold
   by it (scale_thresholds, analysis->thr_scale). That spreads the damage of a
   low rate evenly over the spectrum, while Opus keeps bass and mids almost
   perfect and pays with the highs: at 96 kbit/s on dense live rock MiniMix had
   4.9-14.7 % of the cells over the threshold below 500 Hz where Opus had
   0.5-1.9 %, and 65 % of all MiniMix bits sat above 4 kHz.

   The tilt keeps the loop one-dimensional - it still searches a single scalar
   and still converges on the total size - and only shapes where that scalar
   lands: the per band factor is the global factor raised to a fixed exponent
   profile, thr_b = thr_b * factor^tilt(f_b), with tilt NEGATIVE below
   TILT_LO_HZ (the loop's lift becomes a cut there: bass, mids and the presence
   region are coded FINER than the model asks), the exponent rising linearly in
   log frequency, crossing zero near 4.9 kHz and reaching TILT_HI above
   TILT_HI_HZ (the highs absorb the whole cut). In dB, with the default
   profile: a +1.9 dB offset (title I at 128 kbit/s) becomes
   -2.8 dB below 3.5 kHz, 0 dB at 4.9 kHz, +4.9 dB at 9 kHz and above.

   The low side is bounded by TILT_MIN_DB: a band is never coded more than
   3 dB finer than the model asks, however large the offset grows. That bound
   is what makes the shape usable on material where the loop has to lift a
   lot (title A and title B need +7 to +10 dB where dense rock
   needs +1.9).

   Only a lifting loop is tilted (factor > 1). At a high target the loop
   tightens the thresholds below the model (320 kbit/s), and there a tilt would
   move bits the other way for no reason; those operating points stay exactly
   as they were - as does every quality-fixed file, which never calls the loop.

   THE FALLBACK. Protecting the low bands costs bits that no longer shrink
   when the loop raises the offset, so on a file whose protected bands alone
   are already over budget the bisection saturates and the tilt has made the
   rate target worse instead of better. The loop detects exactly that (the
   same 5 % the "not reached" warning uses), sets tilt_off_override and
   searches again with the uniform offset. The 16 kbit/s case of the roundtrip
   test goes that way and lands on the untilted coder's own result (+27.8 %),
   byte for byte.

   MMX_TILT=0 switches it off (byte-identical to the untilted coder),
   MMX_TILT_LO_HZ / MMX_TILT_HI_HZ move the corners, MMX_TILT_LO / MMX_TILT_HI
   the exponents, MMX_TILT_MIN_DB the low bound, MMX_TILT_MAX_ADD_DB the high
   one. MMX_TILT_MAX_CUT_DB is the older low rail (the tilt may take at most
   that many dB of the offset away from a band); the fallback replaced it, so
   its default is inert. */

#define TILT_LO_HZ 3500.0   /* below this the offset is scaled by TILT_LO */
#define TILT_HI_HZ 9000.0   /* above this by TILT_HI */
#define TILT_LO (-1.50)     /* negative: the lift becomes a cut in the ear's own region */
#define TILT_HI 2.60
#define TILT_MIN_DB (-3.0)  /* never code a band more than this far below the model */
#define TILT_MAX_CUT_DB 60.0 /* inert (see THE FALLBACK); kept as a knob */
#define TILT_MAX_ADD_DB 12.0 /* never lift a band more than this far above the offset */

typedef struct
{
    int on;
    int mirror;   /* band replication: a fixed dB profile instead of the multiplied one */
    double lo_hz, hi_hz, lo, hi, max_cut_db, max_add_db, min_db, bwe_lo_db, bwe_hi_db;
} TiltProfile;

/* Set by the bitrate loop when the tilted profile cost the bisection its
   authority over this file (see "the fallback" below): every later
   tilt_profile_init of the same encode then reports the uniform offset. One
   encode at a time, like the rest of the encoder's state. */
static int tilt_off_override = 0;

/* Set while the rate loop runs a CHEAP PROBE: the trellis quantizer and the two
   closed-loop trials (block type, M/S) are switched off, because a probe only
   has to predict the SIZE of the file, not produce it. Those three stages are
   roughly six sevenths of the encode time and change the size by about two
   percent - a nearly constant factor the loop corrects for and then verifies
   with a full-cost encode (see the rate loop at the end of this file).
   One encode at a time, like the rest of the encoder's state. */
static int rate_fast_probe = 0;

/* THE PROFILE OF A REPLICATION FILE. The run-6 tilt multiplies the loop's
   offset by an exponent per band, so its protection of the ear's region is
   proportional to how hard the loop has to lift. Band replication removes so
   many bits from the highs that the loop stops lifting: title I,
   full track, 121 kbit/s with the replication lands at an offset of exactly
   0.00 dB - and a multiplied profile times zero is no profile at all (the
   regions below 4 kHz measured 1.5/2.3/3.3/3.8/6.1 %, against 0.1/0.1/0.2/0.1
   for the run-6 coder at 128 kbit/s, which needed +2.34 dB and got -3.5 dB in
   the bass out of it). With replication on, the shaping is therefore a FIXED
   profile added to the offset in dB - TILT_BWE_LO_DB below MMX_TILT_LO_HZ,
   TILT_BWE_HI_DB above MMX_TILT_HI_HZ, linear in log frequency in between -
   and the loop searches a uniform offset on top of it. The shaping no longer
   depends on the offset at all, which is the honest reading: with the highs
   described instead of coded, the offset is no longer a measure of how much
   stress the file is under. Only files that use the replication take this
   path, so every other file stays byte-identical. MMX_TILT_MIRROR=0 switches
   the fixed profile off (uniform offset), MMX_TILT_BWE_LO / MMX_TILT_BWE_HI
   move it. */
static int tilt_mirror_on = 0;

/* The fixed profile of the replication files, in dB: below TILT_LO_HZ the
   thresholds are this much under the loop's offset, above TILT_HI_HZ this much
   over it, linear in log frequency in between. */
#define TILT_BWE_LO_DB (-4.0)
#define TILT_BWE_HI_DB 4.0

static void tilt_profile_init(TiltProfile *t)
{
    const char *e;
    t->on = !tilt_off_override;
    t->mirror = tilt_mirror_on;
    t->lo_hz = TILT_LO_HZ;
    t->hi_hz = TILT_HI_HZ;
    t->lo = TILT_LO;
    t->hi = TILT_HI;
    t->max_cut_db = TILT_MAX_CUT_DB;
    t->max_add_db = TILT_MAX_ADD_DB;
    t->min_db = TILT_MIN_DB;
    t->bwe_lo_db = TILT_BWE_LO_DB;
    t->bwe_hi_db = TILT_BWE_HI_DB;
    if ((e = getenv("MMX_TILT")) != NULL && atoi(e) == 0) t->on = 0;
    if ((e = getenv("MMX_TILT_LO_HZ")) != NULL) t->lo_hz = atof(e);
    if ((e = getenv("MMX_TILT_HI_HZ")) != NULL) t->hi_hz = atof(e);
    if ((e = getenv("MMX_TILT_LO")) != NULL) t->lo = atof(e);
    if ((e = getenv("MMX_TILT_HI")) != NULL) t->hi = atof(e);
    if ((e = getenv("MMX_TILT_MAX_CUT_DB")) != NULL) t->max_cut_db = atof(e);
    if ((e = getenv("MMX_TILT_MAX_ADD_DB")) != NULL) t->max_add_db = atof(e);
    if ((e = getenv("MMX_TILT_MIN_DB")) != NULL) t->min_db = atof(e);
    if ((e = getenv("MMX_TILT_MIRROR")) != NULL) t->mirror = atoi(e) != 0;
    if ((e = getenv("MMX_TILT_BWE_LO")) != NULL) t->bwe_lo_db = atof(e);
    if ((e = getenv("MMX_TILT_BWE_HI")) != NULL) t->bwe_hi_db = atof(e);
    if (t->max_cut_db < 0.0) t->max_cut_db = 0.0;
    if (t->max_add_db < 0.0) t->max_add_db = 0.0;
    if (t->lo_hz < 20.0) t->lo_hz = 20.0;
    if (t->hi_hz <= t->lo_hz) t->hi_hz = t->lo_hz * 1.001;
    if (t->hi < 0.0) t->hi = 0.0;
}

/* The exponent at one frequency: flat below/above the corners, linear in
   log frequency in between (the ear's own scale). */
static double tilt_exponent(const TiltProfile *t, double hz)
{
    double u;
    if (!t->on) return 1.0;
    if (hz <= t->lo_hz) return t->lo;
    if (hz >= t->hi_hz) return t->hi;
    u = log(hz / t->lo_hz) / log(t->hi_hz / t->lo_hz);
    return t->lo + u * (t->hi - t->lo);
}

/* The factor the rate loop applies in a band at `hz`: the offset in dB times
   the band's exponent, inside the guard rails. */
static double tilt_band_factor(const TiltProfile *t, double factor, double hz)
{
    double db, tdb;
    if (!t->on)
        return factor;
    db = 10.0 * log10(factor);
    if (t->mirror)
    {
        /* band replication: the offset plus a fixed profile, no rails needed
           (the profile is bounded by its own two constants) */
        double u, fix;
        if (hz <= t->lo_hz) fix = t->bwe_lo_db;
        else if (hz >= t->hi_hz) fix = t->bwe_hi_db;
        else
        {
            u = log(hz / t->lo_hz) / log(t->hi_hz / t->lo_hz);
            fix = t->bwe_lo_db + u * (t->bwe_hi_db - t->bwe_lo_db);
        }
        return pow(10.0, (db + fix) / 10.0);
    }
    if (factor <= 1.0)
        return factor;   /* no offset, or a loop that tightens: nothing to shape */
    tdb = db * tilt_exponent(t, hz);
    if (tdb < db - t->max_cut_db) tdb = db - t->max_cut_db;
    if (tdb > db + t->max_add_db) tdb = db + t->max_add_db;
    if (tdb < t->min_db) tdb = t->min_db;
    return pow(10.0, tdb / 10.0);
}

/* Per-band factors of one band layout (bands beyond band_count keep the plain
   factor; their thresholds are the cutoff's 1e30 and are never scaled). */
static void tilt_band_scales(const TiltProfile *t, const MMXBandLayout *L, double factor, double *out, unsigned int n)
{
    unsigned int b;
    for (b = 0; b < n; b++)
        out[b] = b < L->band_count ? tilt_band_factor(t, factor, (double)L->band_hz[b]) : factor;
}

/* The crossover of the band replication for a bitrate target. Above
   MMX_BWE_DEFAULT_MAX_KBPS nothing is replicated; below it the crossover
   falls with the budget, because the bits a described high band frees are
   worth more the less there is to spend. Tuned on title I at 96 and
   128 kbit/s. */
unsigned int mmx_bwe_default_hz(unsigned int target_kbps)
{
    const char *e = getenv("MMX_BWE_MAX_KBPS");
    unsigned int max_kbps = e ? (unsigned int)atoi(e) : MMX_BWE_DEFAULT_MAX_KBPS;
    if (!target_kbps || target_kbps > max_kbps)
        return 0;
    if (target_kbps >= 96) return 11000;
    if (target_kbps >= 80) return 10000;
    if (target_kbps >= 64) return 8000;
    return 7000;
}

/* The crossover as the header carries it (byte 36, units of 250 Hz). Encoder
   and decoder must derive the same band from the same number, so the encoder
   works with the quantized value, not with what the user typed. */
static unsigned int bwe_header_hz(unsigned int hz)
{
    unsigned int u = (hz + 125) / 250;
    if (u > 255) u = 255;
    return u * 250;
}

void mmx_encoder_params_default(MMXEncoderParams *p)
{
    memset(p, 0, sizeof(*p));
    p->quality = 7;
    p->analysis = 5;
    p->noise_fill = MMX_NF_DEFAULT;
    p->max_ref_depth = MMX_DEFAULT_MAX_REF_DEPTH;
}

/* ---------------------------------------------------------------- blocks */

typedef struct
{
    MMXRangeEncoder rc;
    MMXCodecContexts ctx;
    MMXLosslessContexts llctx;
    MMXBlockEntry entry;
    int open;
} BlockWriter;

static void block_begin(BlockWriter *w, unsigned long f, const MMXFramePlan *p)
{
    memset(&w->entry, 0, sizeof(w->entry));
    w->entry.start_frame = f;
    w->entry.n_sources = p->n_sources;
    w->entry.src_start[0] = p->src_start[0];
    w->entry.src_start[1] = p->src_start[1];
    w->entry.depth = p->depth;
    mmx_rc_enc_init(&w->rc);
    mmx_contexts_init(&w->ctx);
    mmx_ll_contexts_init(&w->llctx);
    w->open = 1;
}

static int block_end(BlockWriter *w, MMXFile *out, MMXStatistics *stats)
{
    if (!w->open)
        return 0;
    mmx_rc_enc_finish(&w->rc);
    if (w->rc.failed)
        return -1;
    w->entry.payload = w->rc.data;
    w->entry.payload_size = w->rc.size;
    w->entry.crc32 = mmx_crc32(w->rc.data, w->rc.size);
    w->rc.data = NULL;
    mmx_rc_enc_free(&w->rc);
    stats->payload_bytes[w->entry.n_sources] += w->entry.payload_size;
    stats->block_count++;
    w->open = 0;
    return mmx_file_add_block(out, &w->entry) < 0 ? -1 : 0;
}

static void baseline_segment(BlockWriter *base, MMXStatistics *stats, unsigned long f, int flush_only)
{
    if (base->open && (flush_only || (f % BASELINE_SEGMENT) == 0))
    {
        mmx_rc_enc_finish(&base->rc);
        stats->baseline_bytes += base->rc.size;
        mmx_rc_enc_free(&base->rc);
        base->open = 0;
    }
    if (!base->open && !flush_only)
    {
        mmx_rc_enc_init(&base->rc);
        mmx_contexts_init(&base->ctx);
        mmx_ll_contexts_init(&base->llctx);
        base->open = 1;
    }
}

static int same_lineage(const MMXFramePlan *prev, const MMXFramePlan *cur)
{
    unsigned int s;
    if (prev->n_sources != cur->n_sources)
        return 0;
    for (s = 0; s < cur->n_sources; s++)
        if (prev->src_start[s] + MMX_HOP != cur->src_start[s])
            return 0;
    return 1;
}

/* Strict mode at the deep analysis levels plans twice (second_pass): only
   there the real bits of the first pass calibrate the estimates, so only
   there (or for the MMX_DEBUG_EST table) the closed loop books the coded
   estimate of every long frame. MMX_SINGLE_PASS: experiment, one pass. */
static int second_pass_wanted(const MMXEncoderParams *params)
{
    return params->quality != 0 && params->analysis >= MMX_SECOND_PASS_LEVEL && params->mode == MMX_MODE_NORMAL &&
           !params->pns_hz && !params->reuse && !params->target_kbps && !getenv("MMX_SINGLE_PASS");
}

/* ---------- debug: real bits against the estimates (MMX_DEBUG_EST) ----------
   Per frame class (AUDIO block, REF with prediction, REF with the gains
   switched off by the trial), position in the block (first, second, third,
   later) and transient flag: the encoder's estimate for the syntax that was
   written, the analyzer's estimate for the plan (original sources) and the
   real bits of the range coder. The first frames of a block pay the context
   reset; their surplus over the later frames is the real switch cost. */
#define EST_POS 4
static struct
{
    int on;                                  /* print the table (MMX_DEBUG_EST); the sums are always kept */
    unsigned long n[3][EST_POS][2];
    double est[3][EST_POS][2], plan[3][EST_POS][2], real[3][EST_POS][2];
    double est2[3][EST_POS][2];              /* the estimate that mirrors the coder (long frames, MMX_DEBUG_EST only) */
    double sq[3][EST_POS][2], sq2[3][EST_POS][2];   /* sum of squared log ratios real/estimate: per-frame scatter */
    unsigned long n2[3][EST_POS][2];
    double elem_est[3][MMX_ELEM_COUNT], elem_real[3][MMX_ELEM_COUNT];   /* M0: per syntax element (long frames with a coded estimate) */
    unsigned long elem_frames[3];
} est_debug;

/* Calibration of the estimates from the sums of the last closed-loop pass:
   real bits over estimated bits for AUDIO frames and for residuals, per
   [stationary, transient]; frames whose gains the trial switched off are
   left out (their estimate is the residual's, their bits are the audio's).
   The estimate is the one the analyzer scores with (analysis->coded_estimate),
   so the factors land on the same scale. */
static void est_calibration(double *cal_audio, double *cal_ref)
{
    unsigned int cls, t, p;
    for (cls = 0; cls < 2; cls++)
        for (t = 0; t < 2; t++)
        {
            double est = 0.0, real = 0.0;
            for (p = 0; p < EST_POS; p++) { est += est_debug.est[cls][p][t]; real += est_debug.real[cls][p][t]; }
            (cls ? cal_ref : cal_audio)[t] = est > 0.0 && real > 0.0 ? real / est : 1.0;
        }
}

static void est_debug_add(unsigned int cls, unsigned long pos, int transient, double est, double plan, double real, double est2)
{
    unsigned int p = pos < EST_POS - 1 ? (unsigned int)pos : EST_POS - 1, t = transient ? 1 : 0;
    if (est <= 0.0)
        return;   /* no estimate on the analyzer's scale for this frame (short frame with the coded estimate) */
    est_debug.n[cls][p][t]++;
    est_debug.est[cls][p][t] += est;
    est_debug.plan[cls][p][t] += plan;
    est_debug.real[cls][p][t] += real;
    if (est2 > 0.0 && est > 0.0 && real > 0.0)
    {
        double l1 = log(real / est), l2 = log(real / est2);
        est_debug.n2[cls][p][t]++;
        est_debug.est2[cls][p][t] += est2;
        est_debug.sq[cls][p][t] += l1 * l1;
        est_debug.sq2[cls][p][t] += l2 * l2;
    }
}

static void est_debug_print(void)
{
    static const char *cls_name[3] = { "AUDIO", "REF", "REF gains off" };
    static const char *pos_name[EST_POS] = { "1st", "2nd", "3rd", "4th+" };
    unsigned int c, p, t, e;
    for (c = 0; c < 2; c++)
    {
        double te = 0.0, tr = 0.0;
        if (!est_debug.elem_frames[c]) continue;
        fprintf(stderr, "bits per syntax element, %s long frames (%lu frames, per frame; estimate = the coded estimator, 0 = not estimated):\n",
                cls_name[c], est_debug.elem_frames[c]);
        fprintf(stderr, "  %-18s %10s %10s %9s\n", "element", "estimated", "actual", "error");
        for (e = 0; e < MMX_ELEM_COUNT; e++)
        {
            double es = est_debug.elem_est[c][e] / est_debug.elem_frames[c], re = est_debug.elem_real[c][e] / est_debug.elem_frames[c];
            te += es; tr += re;
            if (es <= 0.0 && re <= 0.0) continue;
            fprintf(stderr, "  %-18s %10.1f %10.1f %8.1f%%\n", mmx_elem_names[e], es, re, es > 0.0 ? 100.0 * (re - es) / es : 0.0);
        }
        fprintf(stderr, "  %-18s %10.1f %10.1f %8.1f%%\n", "total", te, tr, te > 0.0 ? 100.0 * (tr - te) / te : 0.0);
    }
    fprintf(stderr, "real bits against the estimates (mean per frame; surplus = real - encoder estimate; coded = the estimate that mirrors the coder,\n"
                    "  long frames only; rms = root mean square of log(real / estimate) per frame, i.e. the scatter of each estimate)\n");
    fprintf(stderr, "  %-14s %-4s %-10s %7s %9s %9s %9s %8s %8s %8s %9s %7s %7s\n", "class", "pos", "kind", "frames", "plan est", "enc est", "real", "real/est", "surplus", "coded", "real/cod", "rms", "rms cod");
    for (c = 0; c < 3; c++)
        for (t = 0; t < 2; t++)
            for (p = 0; p < EST_POS; p++)
            {
                unsigned long n = est_debug.n[c][p][t];
                if (!n)
                    continue;
                fprintf(stderr, "  %-14s %-4s %-10s %7lu %9.0f %9.0f %9.0f %8.3f %8.0f %8.0f %9.3f %6.1f%% %6.1f%%\n", cls_name[c], pos_name[p], t ? "transient" : "stationary", n,
                        est_debug.plan[c][p][t] / n, est_debug.est[c][p][t] / n, est_debug.real[c][p][t] / n,
                        est_debug.est[c][p][t] > 0.0 ? est_debug.real[c][p][t] / est_debug.est[c][p][t] : 0.0,
                        (est_debug.real[c][p][t] - est_debug.est[c][p][t]) / n,
                        est_debug.n2[c][p][t] ? est_debug.est2[c][p][t] / est_debug.n2[c][p][t] : 0.0,
                        est_debug.est2[c][p][t] > 0.0 ? est_debug.real[c][p][t] / est_debug.est2[c][p][t] : 0.0,
                        est_debug.n2[c][p][t] ? 100.0 * sqrt(est_debug.sq[c][p][t] / est_debug.n2[c][p][t]) : 0.0,
                        est_debug.n2[c][p][t] ? 100.0 * sqrt(est_debug.sq2[c][p][t] / est_debug.n2[c][p][t]) : 0.0);
            }
}

static int metadata_append(MMXFile *f, const char *line)
{
    size_t old = f->metadata ? strlen(f->metadata) : 0, add = strlen(line);
    char *m = (char *)realloc(f->metadata, old + add + 1);
    if (!m)
        return -1;
    memcpy(m + old, line, add + 1);
    f->metadata = m;
    f->metadata_len = (unsigned long)(old + add);
    return 0;
}

static void fill_header(MMXFile *out, const MMXAudioBuffer *audio, const MMXEncoderParams *params)
{
    out->sample_rate = audio->sample_rate;
    out->channels = audio->channels;
    out->source_bits = audio->source_bits;
    out->frame_count = audio->frame_count;
    out->hop = MMX_HOP;
    out->quality = (unsigned char)params->quality;
    out->analysis_level = (unsigned char)params->analysis;
    out->max_ref_depth = params->max_ref_depth;
    out->codec_id = params->quality == 0 ? MMX_CODEC_LOSSLESS : MMX_CODEC_MDCT;
    out->bitstream_rev = MMX_BITSTREAM_REVISION;
    out->bwe_hz = params->quality == 0 ? 0 : bwe_header_hz(params->bwe_hz);
    out->lowrate = (unsigned char)(params->quality == 0 ? 0 : params->lowrate);
    out->epb_fold = (unsigned char)params->epb_fold;
    out->epb_flags = (unsigned char)params->epb_flags;
    out->epb_fill_db = (signed char)params->epb_fill_db;
    out->bwe_mode = (unsigned char)((getenv("MMX_BWE_MODE") ? atoi(getenv("MMX_BWE_MODE")) : 0) & 1);
    if (params->metadata)
        mmx_file_set_metadata(out, params->metadata);
    if (params->cover && params->cover_len)
        mmx_file_set_cover(out, params->cover_mime ? params->cover_mime : "image/jpeg", params->cover, params->cover_len);
}

/* Decides TNS per coded channel from the array that will be quantized and
   derives the thresholds for the filtered domain: the coded noise gets the
   temporal envelope E(t) of the synthesis filter, so sub-window w tolerates
   filtered-domain noise sub_thr[w] / mean_w(E) and the whole window
   frame_thr / G (G = mean power gain). Without TNS this reduces to the plain
   sub-window minimum. psy[c] is the L/R psy of the channel used as reference
   for coded channel c (for M/S: L for M, R for S — conservative). */
static void apply_tns(MMXCodec *codec, MMXFrameSyntax *syn, float *const *to_quantize,
                      const MMXFramePsy *const *psy, const float *const *thr_in,
                      float (*thr_buf)[MMX_MAX_BANDS], const float **thr_out,
                      float (*thr_lr_buf)[MMX_MAX_BANDS], const float **thr_lr_out, const float *const *thr_lr_in,
                      int enabled)
{
    unsigned int c, b, w, nb = codec->bands.band_count;
    double env[MMX_WIN], mean_e[5], gmax = 1.0;
    float relax_min[MMX_MAX_BANDS]; /* smallest effective/plain ratio over channels, for the L/R check */

    for (b = 0; b < MMX_MAX_BANDS; b++) relax_min[b] = 1.0f;
    for (c = 0; c < codec->channels; c++)
    {
        double g = 1.0;
        memset(&syn->tns[c], 0, sizeof(syn->tns[c]));
        if (enabled)
        {
            static double min_gain = -1.0;   /* MMX_TNS_MIN_GAIN: experiment, the prediction gain a frame needs to carry a filter */
            if (min_gain < 0.0) { const char *e = getenv("MMX_TNS_MIN_GAIN"); min_gain = e ? atof(e) : TNS_MIN_GAIN; }
            mmx_tns_analyze(to_quantize[c], codec->tns_k0, codec->tns_k1, MMX_TNS_MAX_ORDER, min_gain, &syn->tns[c]);
        }
        if (!syn->tns[c].active)
        {
            for (b = 0; b < MMX_MAX_BANDS; b++) thr_buf[c][b] = thr_in[c][b];
            thr_out[c] = thr_buf[c];
            continue;
        }
        mmx_tns_filter(&syn->tns[c], to_quantize[c], codec->tns_k0, codec->tns_k1);
        mmx_tns_time_envelope(&syn->tns[c], MMX_WIN, env);
        /* mean envelope per sub-window, weighted by the synthesis window */
        for (w = 0; w < 5; w++)
        {
            long long lo = (long long)w * 512 - 512, hi = lo + 1024, i;
            double num = 0.0, den = 0.0;
            for (i = lo < 0 ? 0 : lo; i < hi && i < MMX_WIN; i++)
            {
                double ws = codec->mdct.window[i] * codec->mdct.window[i];
                num += env[i] * ws;
                den += ws;
            }
            mean_e[w] = den > 0.0 ? num / den : 1.0;
        }
        g = mmx_tns_noise_gain(&syn->tns[c]);
        if (g > gmax) gmax = g;
        {
            MMXFramePsy tmp;
            memset(&tmp, 0, sizeof(tmp));
            for (b = 0; b < nb; b++)
            {
                float t = thr_in[c][b];
                int shaped = codec->bands.band_start[b] >= codec->tns_k0 && codec->bands.band_start[b] < codec->tns_k1;
                if (t >= 1e29f || !shaped)
                {
                    tmp.thr[b] = t;
                    continue;
                }
                t = (float)(psy[c]->frame_thr[b] / g);
                for (w = 0; w < 5; w++)
                {
                    float st = (float)(psy[c]->sub_thr[w][b] / mean_e[w]);
                    if (psy[c]->sub_thr[w][b] < 1e29f && st < t) t = st;
                }
                /* never exceed the plain threshold by more than the shaping can justify */
                if (t > thr_in[c][b] * (float)g) t = thr_in[c][b] * (float)g;
                tmp.thr[b] = t;
            }
            mmx_psy_limit_cliffs(&tmp, nb);
            for (b = 0; b < nb; b++)
            {
                thr_buf[c][b] = tmp.thr[b];
                if (thr_in[c][b] > 0.0f && thr_in[c][b] < 1e29f)
                {
                    float r = tmp.thr[b] / thr_in[c][b];
                    if (r < relax_min[b]) relax_min[b] = r;
                }
            }
            for (b = nb; b < MMX_MAX_BANDS; b++) thr_buf[c][b] = thr_in[c][b];
        }
        thr_out[c] = thr_buf[c];
    }
    if (thr_lr_in && thr_lr_out)
        for (c = 0; c < 2; c++)
        {
            for (b = 0; b < MMX_MAX_BANDS; b++)
            {
                float t = thr_lr_in[c][b];
                thr_lr_buf[c][b] = (gmax > 1.0 && t < 1e29f) ? t * relax_min[b] : t;
            }
            thr_lr_out[c] = thr_lr_buf[c];
        }
}

#define SHORT_ATTACK_DB 15.0
#define BT_TRIAL_LO_DB 10.0        /* attacks from here ... */
#define BT_TRIAL_HI_DB 20.0        /* ... to here get their block type by trial coding (bt_trial); above: short by rule */

/* START/STOP window shapes around the short frames of [lo, hi]: a frame before
   a short frame is START, after one STOP, between two of them SHORT (three
   passes; the neighbours outside the range as they are). */
static void derive_block_types(unsigned char *bt, const unsigned char *is_short, unsigned long lo, unsigned long hi, unsigned long nf)
{
    unsigned long f, pass;
    for (f = lo; f <= hi; f++) bt[f] = is_short[f] ? MMX_BT_SHORT : MMX_BT_LONG;
    for (pass = 0; pass < 3; pass++)
        for (f = lo; f <= hi; f++)
        {
            int before_short = f + 1 < nf && bt[f + 1] == MMX_BT_SHORT;
            int after_short = f > 0 && bt[f - 1] == MMX_BT_SHORT;
            if (bt[f] == MMX_BT_SHORT)
                continue;
            bt[f] = before_short && after_short ? MMX_BT_SHORT : before_short ? MMX_BT_START : after_short ? MMX_BT_STOP : MMX_BT_LONG;
        }
}

/* Block type per frame: frames with a strong attack inside the window get
   eight short transforms; their neighbours get the START/STOP window shapes
   (LONG -> START -> SHORT... -> STOP -> LONG). 15 dB measured on title A
   (quality 7, analysis 5): 196 short frames, -0.5 % size, pre-echo count and
   share of bands over the threshold unchanged; 10 dB: 2786 short frames,
   -2.6 %, share over 3.2 instead of 3.7 %, but the mean NMR 2 dB worse and the
   synthetic roundtrip track fails its 8 dB barrier (8.08 dB: START windows
   leak at -34 dB 250 Hz away, tonal LF material next to a kick). Frames with
   an attack between BT_TRIAL_LO_DB and BT_TRIAL_HI_DB are marked in `cand`:
   the closed loop decides them by coding (bt_trial), the rule's answer is
   their provisional type until then. */
static void decide_block_types(const MMXAnalysis *analysis, unsigned char *bt, unsigned char *is_short, unsigned char *cand,
                               int enabled, int trials)
{
    unsigned long f, nf = analysis->frame_count;
    unsigned int c;
    const char *e = getenv("MMX_SHORT_ATTACK_DB"); /* experiment */
    double short_db = e ? atof(e) : SHORT_ATTACK_DB;
    double lo_db = (e = getenv("MMX_BT_TRIAL_LO_DB")) ? atof(e) : BT_TRIAL_LO_DB;   /* experiments */
    double hi_db = (e = getenv("MMX_BT_TRIAL_HI_DB")) ? atof(e) : BT_TRIAL_HI_DB;
    for (f = 0; f < nf; f++)
    {
        float attack = 0.0f;
        int inner = enabled && f > 0 && f + 1 < nf;
        for (c = 0; c < analysis->channels; c++)
        {
            const MMXFramePsy *p = mmx_analysis_psy(analysis, f, c);
            if (p->attack_db > attack) attack = p->attack_db;
        }
        is_short[f] = (unsigned char)(inner && attack >= (float)short_db);
        cand[f] = (unsigned char)(inner && trials && attack >= (float)lo_db && attack < (float)hi_db);
    }
    if (nf)
        derive_block_types(bt, is_short, 0, nf - 1, nf);
}

/* Thresholds of a START/STOP frame. The cached thresholds of the analysis
   apply the sub-window rule over all five sub-windows and the temporal rule
   against both neighbours, so next to a short frame with an attack a START or
   STOP frame would be as tight as the attack itself. Its window however is
   zero in the last (START) or first (STOP) 448 samples: only the sub-windows
   its noise reaches count, and the neighbour rule only towards the LONG side
   (the short side is covered by those sub-windows). Same order as the
   analyzer: sub-window minimum, temporal rule, file edges, cliffs. */
static void start_stop_thresholds(const MMXAnalysis *analysis, unsigned long f, unsigned int c, unsigned int bt,
                                  unsigned int band_count, const double *bscale, MMXFramePsy *out)
{
    const MMXFramePsy *p = mmx_analysis_psy(analysis, f, c);
    const MMXFramePsy *prev = f > 0 ? mmx_analysis_psy(analysis, f - 1, c) : NULL;
    const MMXFramePsy *next = f + 1 < analysis->frame_count ? mmx_analysis_psy(analysis, f + 1, c) : NULL;
    unsigned int b, w, w0 = bt == MMX_BT_STOP ? 1 : 0, w1 = bt == MMX_BT_START ? MMX_SUB_WINDOWS - 1 : MMX_SUB_WINDOWS;
    static int no_subwin = -1, no_cliff = -1;
    if (no_subwin < 0) no_subwin = getenv("MMX_NO_SUBWIN") != NULL;
    if (no_cliff < 0) no_cliff = getenv("MMX_NO_CLIFF") != NULL;

    *out = *p;
    for (b = 0; b < band_count; b++)
    {
        float t = p->frame_thr[b];
        if (t >= 1e29f)
            continue;
        if (!no_subwin)
            for (w = w0; w < w1; w++)
                if (p->sub_thr[w][b] < t) t = p->sub_thr[w][b];
        out->thr[b] = (float)(t * bscale[b]);   /* same (tilted) bitrate offset as the cached thr[] */
    }
    mmx_psy_temporal(bt == MMX_BT_STOP ? NULL : prev, out, bt == MMX_BT_START ? NULL : next);
    if (f == 0 || f + 1 == analysis->frame_count)
        for (b = 0; b < band_count; b++)
            if (out->thr[b] < 1e29f) out->thr[b] *= 0.01f;
    if (!no_cliff)
        mmx_psy_limit_cliffs(out, band_count);
}

/* Masking thresholds of the eight short blocks of a frame (per channel L/R):
   the masking analysis of each 5.8 ms window, capped by the sub-window rule of
   the frame analysis. A group's quantization noise must stay under the
   thresholds of every 23 ms analysis sub-window it overlaps by at least half
   its length (in 256-MDCT units: a quarter of the 1024-MDCT power for the same
   time-domain noise, all groups at the limit), so the groups just before an
   attack are held to the pre-echo threshold of the sub-window that contains
   the attack, exactly like a LONG frame. Without the cap they sit at their own
   stationary threshold, 10-20 dB above what the model allows there (measured:
   worst NMR 32 dB in the roundtrip test). */
static void short_thresholds(MMXCodec *codec, MMXSpectrum *spec, const MMXAudioBuffer *audio, long long start,
                             unsigned int quality, const MMXFramePsy *const *psy, const double *sbscale, double clip_scale,
                             float (*thr)[MMX_SHORT_GROUPS][MMX_SHORT_BANDS], float (*cap)[MMX_SHORT_GROUPS][MMX_SHORT_BANDS])
{
    const MMXBandLayout *L = &codec->bands, *Ls = &codec->short_bands;
    unsigned int c, g, b, w, j, j0[MMX_SHORT_BANDS], j1[MMX_SHORT_BANDS];
    unsigned long ratio = L->m / Ls->m;                 /* long bins per short bin */
    float win[2 * MMX_SHORT_M], amp[MMX_SHORT_M];
    MMXFramePsy p;

    /* long bands overlapping each short band */
    for (b = 0; b < Ls->band_count; b++)
    {
        unsigned long lo = ratio * Ls->band_start[b], hi = ratio * Ls->band_start[b + 1];
        j0[b] = L->band_count;
        j1[b] = 0;
        for (j = 0; j < L->band_count; j++)
            if (L->band_start[j + 1] > lo && L->band_start[j] < hi)
            {
                if (j < j0[b]) j0[b] = j;
                j1[b] = j;
            }
    }
    for (c = 0; c < codec->channels; c++)
        for (g = 0; g < MMX_SHORT_GROUPS; g++)
        {
            long long g0 = (long long)codec->short_offset + (long long)g * MMX_SHORT_M, g1 = g0 + 2 * MMX_SHORT_M;
            mmx_codec_short_window(codec, audio, start, c, g, win);
            mmx_spectrum_analyze(spec, win, amp);
            mmx_psy_analyze(Ls, amp, win, quality, &p);
            for (b = 0; b < MMX_SHORT_BANDS; b++)
            {
                float t_cap = 1e30f;
                cap[c][g][b] = 1e30f;
                if (b >= Ls->band_count || p.thr[b] >= 1e29f)
                    continue;
                for (w = 0; w < MMX_SUB_WINDOWS; w++)
                {
                    long long w0 = (long long)w * MMX_SUB_STEP - MMX_SUB_STEP, w1 = w0 + MMX_SUB_WIN;
                    long long overlap = (g1 < w1 ? g1 : w1) - (g0 > w0 ? g0 : w0);
                    float units = 4.0f * MMX_SUB_WEIGHT(w);
                    if (overlap < MMX_SHORT_M)
                        continue;
                    for (j = j0[b]; j <= j1[b] && j < L->band_count; j++)
                    {
                        float t = psy[c]->sub_thr[w][j];
                        if (t < 1e29f && t / units < t_cap) t_cap = t / units;
                    }
                }
                cap[c][g][b] = t_cap;
                if (t_cap < p.thr[b]) p.thr[b] = t_cap;
            }
            mmx_psy_limit_cliffs(&p, Ls->band_count);
            for (b = 0; b < MMX_SHORT_BANDS; b++)
            {
                double scale = sbscale[b] * clip_scale;
                thr[c][g][b] = b < Ls->band_count && p.thr[b] < 1e29f ? (float)(p.thr[b] * scale) : 1e30f;
                if (cap[c][g][b] < 1e29f) cap[c][g][b] = (float)(cap[c][g][b] * scale);
            }
        }
}

/* ------------------------------------------------------ tracker patterns */

#define TRACKER_PURE_DB 10.0   /* band played from the source without residual: residual <= 10 % ("90 % the same") */
#define TRACKER_RELAX_DB 6.0   /* residual thresholds of pattern frames */
#define FUTURE_PURE_DB 6.0
#define FUTURE_RELAX_DB 12.0
#define PURE_SLACK_DB 4.0      /* inside a pattern that is good on average, a band may be this much worse and still be pure */
#define TRANSIENT_PURE_EXTRA_DB 10.0 /* frames with an attack: pure only from pure_db + this (near copies) */
#define TRACKER_RELAX_SUB_DB 3.0     /* the sub-window (pre-echo) part of the thresholds is relaxed only this much */

/* Mean prediction gain per EQ band over every run of frames with the same
   lineage (a "pattern"): a pattern that is good on average plays as a whole,
   single weaker frames inside it do not fall back to residual coding. */
static void pattern_gains(const MMXAnalysis *a, float *run_gain)
{
    unsigned long f = 0, nf = a->frame_count;
    while (f < nf)
    {
        unsigned long g = f + 1, i;
        unsigned int e;
        double sum[MMX_EQ_BANDS];
        if (!a->plan[f].n_sources) { f++; continue; }
        while (g < nf && a->plan[g].n_sources && same_lineage(&a->plan[g - 1], &a->plan[g])) g++;
        for (e = 0; e < MMX_EQ_BANDS; e++) sum[e] = 0.0;
        for (i = f; i < g; i++)
            for (e = 0; e < MMX_EQ_BANDS; e++) sum[e] += a->plan[i].band_gain_db[e];
        for (i = f; i < g; i++)
            for (e = 0; e < MMX_EQ_BANDS; e++) run_gain[i * MMX_EQ_BANDS + e] = (float)(sum[e] / (double)(g - f));
        f = g;
    }
}

#define PURE_SUBWIN_DB 6.0     /* in every 23 ms sub-window the residual must stay this far under the target */
#define PURE_FLOOR_DB 35.0     /* ... unless it is this far under the band's frame energy (nothing to hear) */
#define SUB_WIN 1024
#define SUB_STEP 512

/* Time-resolved check of a candidate "pure" band. The frame-level gain does
   not see a hit that the source plays a few milliseconds before the target's
   hit: it lands in the quiet sub-window before the attack, where the ear hears
   it as a ghost note or pre-echo. So the target and the prediction are also
   compared in five overlapping 23 ms sub-windows (Hann-windowed FFT, same
   grid as the psychoacoustic model): per EQ band and sub-window the residual
   spectrum T - sum(g_e * S) must stay PURE_SUBWIN_DB under the target, unless
   it is PURE_FLOOR_DB under the band's frame energy anyway. */
typedef struct
{
    MMXFft fft;
    MMXBandLayout bands;          /* half-bark bands of the 1024 sub-window */
    double win[SUB_WIN];
    double re[3][2][SUB_WIN], im[3][2][SUB_WIN];   /* [target/source 0/source 1][L/R] */
    int ok;
} PureTemporal;

static int pure_temporal_init(PureTemporal *t, unsigned long sample_rate)
{
    unsigned long k;
    memset(t, 0, sizeof(*t));
    if (mmx_fft_init(&t->fft, SUB_WIN) != 0 || mmx_bands_init(&t->bands, sample_rate, SUB_WIN / 2) != 0)
        return -1;
    for (k = 0; k < SUB_WIN; k++)
        t->win[k] = 0.5 - 0.5 * cos(2.0 * 3.14159265358979323846 * (k + 0.5) / SUB_WIN);
    t->ok = 1;
    return 0;
}

static void pure_temporal_free(PureTemporal *t)
{
    if (t->ok) mmx_fft_free(&t->fft);
    t->ok = 0;
}

static void sub_spectrum(PureTemporal *t, const MMXAudioBuffer *buf, long long start, unsigned int c, double *re, double *im)
{
    unsigned long k;
    for (k = 0; k < SUB_WIN; k++)
    {
        long long pos = start + (long long)k;
        re[k] = (pos >= 0 && pos < (long long)buf->frame_count) ? buf->samples[(size_t)pos * buf->channels + c] * t->win[k] : 0.0;
        im[k] = 0.0;
    }
    mmx_fft_forward(&t->fft, re, im);
}

/* ok[c][e] = 1 when band e of coded channel c may be played from the source */
static void pure_temporal_check(PureTemporal *t, const MMXCodec *codec, const MMXFrameSyntax *syn, unsigned int n_sources,
                                const MMXAudioBuffer *audio, const MMXAudioBuffer *decoded, long long start,
                                const long long *src_start, int use_ms, unsigned int channels,
                                const double (*frame_energy)[MMX_EQ_BANDS], int (*ok)[MMX_EQ_BANDS])
{
    unsigned int w, c, e, s, b;
    static double subwin_db = -1.0;   /* MMX_PURE_SUBWIN_DB overrides the margin (experiment) */
    double ratio, floor_ratio = pow(10.0, -PURE_FLOOR_DB / 10.0);
    if (subwin_db < 0.0)
    {
        const char *env = getenv("MMX_PURE_SUBWIN_DB");
        subwin_db = env ? atof(env) : PURE_SUBWIN_DB;
    }
    ratio = pow(10.0, -subwin_db / 10.0);
    (void)codec;
    for (c = 0; c < channels; c++)
        for (e = 0; e < MMX_EQ_BANDS; e++) ok[c][e] = 1;
    for (w = 0; w < 5; w++)
    {
        long long off = -(long long)SUB_STEP + (long long)w * SUB_STEP;
        double et[MMX_MAX_CH][MMX_EQ_BANDS], er[MMX_MAX_CH][MMX_EQ_BANDS];
        unsigned int nch = channels < 2 ? 1 : 2;
        for (c = 0; c < nch; c++)
        {
            sub_spectrum(t, audio, start + off, c, t->re[0][c], t->im[0][c]);
            for (s = 0; s < n_sources; s++)
                sub_spectrum(t, decoded, src_start[s] + off, c, t->re[1 + s][c], t->im[1 + s][c]);
        }
        for (c = 0; c < channels; c++)
            for (e = 0; e < MMX_EQ_BANDS; e++) { et[c][e] = 0.0; er[c][e] = 0.0; }
        for (b = 0; b < t->bands.band_count; b++)
        {
            unsigned long k;
            e = t->bands.eq_band[b];
            for (k = t->bands.band_start[b]; k < t->bands.band_start[b + 1]; k++)
                for (c = 0; c < channels; c++)
                {
                    /* coded channel c in the sub-window spectrum: L/R, or M = (L+R)/2, S = (L-R)/2 */
                    double tr, ti, rr, ri;
                    if (use_ms && channels == 2)
                    {
                        double sgn = c == 0 ? 1.0 : -1.0;
                        tr = 0.5 * (t->re[0][0][k] + sgn * t->re[0][1][k]);
                        ti = 0.5 * (t->im[0][0][k] + sgn * t->im[0][1][k]);
                        rr = tr; ri = ti;
                        for (s = 0; s < n_sources; s++)
                        {
                            double g = mmx_gain_value(syn->gain[s][c][e], syn->polarity[s][c]);
                            rr -= g * 0.5 * (t->re[1 + s][0][k] + sgn * t->re[1 + s][1][k]);
                            ri -= g * 0.5 * (t->im[1 + s][0][k] + sgn * t->im[1 + s][1][k]);
                        }
                    }
                    else
                    {
                        unsigned int cc = c < 2 ? c : 1;
                        tr = t->re[0][cc][k]; ti = t->im[0][cc][k];
                        rr = tr; ri = ti;
                        for (s = 0; s < n_sources; s++)
                        {
                            double g = mmx_gain_value(syn->gain[s][c][e], syn->polarity[s][c]);
                            rr -= g * t->re[1 + s][cc][k];
                            ri -= g * t->im[1 + s][cc][k];
                        }
                    }
                    et[c][e] += tr * tr + ti * ti;
                    er[c][e] += rr * rr + ri * ri;
                }
        }
        for (c = 0; c < channels; c++)
            for (e = 0; e < MMX_EQ_BANDS; e++)
                if (er[c][e] > et[c][e] * ratio && er[c][e] > frame_energy[c][e] * floor_ratio * t->bands.m / (double)MMX_HOP)
                    ok[c][e] = 0;
    }
}

typedef struct
{
    unsigned long pure_bands, coded_bands, vetoed;   /* channel bands played pure / of referenced frames / vetoed by the sub-window check */
    int all_pure;                                    /* the whole frame plays from the source */
    unsigned char eq_pure[MMX_MAX_CH][MMX_EQ_BANDS]; /* which EQ bands play pure (the structured-error guard leaves them) */
} PureCounts;

/* Tracker decision for one frame in the coded domain: zeroes the residual of
   every channel band whose prediction gain (this frame, or the pattern on
   average with some slack) reaches pure_db and which passes the time-resolved
   check. Returns 1 when the whole frame is played from the source; the counts
   go to pc (the closed loop books them for the candidate it keeps). */
static int tracker_pure_bands(const MMXCodec *codec, float *const *resid, float *const *t_cod, unsigned int channels,
                              const float *rg, double pure_db, int transient, PureCounts *pc,
                              PureTemporal *tmp, const MMXFrameSyntax *syn, unsigned int n_sources,
                              const MMXAudioBuffer *audio, const MMXAudioBuffer *decoded, long long start,
                              const long long *src_start, int use_ms)
{
    double te[MMX_MAX_CH][MMX_EQ_BANDS], re[MMX_MAX_CH][MMX_EQ_BANDS];
    int tok[MMX_MAX_CH][MMX_EQ_BANDS];
    unsigned int c, b, e;
    unsigned long k;
    int all_pure = 1, any = 0;
    for (c = 0; c < channels; c++)
        for (e = 0; e < MMX_EQ_BANDS; e++) { te[c][e] = 0.0; re[c][e] = 0.0; }
    for (c = 0; c < channels; c++)
        for (b = 0; b < codec->cutoff_band; b++)
        {
            e = codec->bands.eq_band[b];
            for (k = codec->bands.band_start[b]; k < codec->bands.band_start[b + 1]; k++)
            {
                te[c][e] += (double)t_cod[c][k] * t_cod[c][k];
                re[c][e] += (double)resid[c][k] * resid[c][k];
            }
        }
    /* candidates by the frame-level gain first; the sub-window FFTs only when there are any */
    for (c = 0; c < channels; c++)
        for (e = 0; e < MMX_EQ_BANDS; e++)
        {
            double g = re[c][e] > 1e-20 ? 10.0 * log10((te[c][e] + 1e-20) / re[c][e]) : 100.0;
            tok[c][e] = transient ? g >= pure_db + TRANSIENT_PURE_EXTRA_DB
                                  : (g >= pure_db || (rg && rg[e] >= pure_db && g >= pure_db - PURE_SLACK_DB));
            any |= tok[c][e];
        }
    if (any)
    {
        int ok[MMX_MAX_CH][MMX_EQ_BANDS];
        pure_temporal_check(tmp, codec, syn, n_sources, audio, decoded, start, src_start, use_ms, channels,
                            (const double (*)[MMX_EQ_BANDS])te, ok);
        for (c = 0; c < channels; c++)
            for (e = 0; e < MMX_EQ_BANDS; e++)
                if (tok[c][e] && !ok[c][e]) { tok[c][e] = 0; pc->vetoed++; }
    }
    for (c = 0; c < channels; c++)
        for (e = 0; e < MMX_EQ_BANDS; e++) pc->eq_pure[c][e] = (unsigned char)(tok[c][e] != 0);
    for (c = 0; c < channels; c++)
        for (b = 0; b < codec->cutoff_band; b++)
        {
            int pure;
            e = codec->bands.eq_band[b];
            pure = tok[c][e];
            pc->coded_bands++;
            if (pure)
            {
                for (k = codec->bands.band_start[b]; k < codec->bands.band_start[b + 1]; k++) resid[c][k] = 0.0f;
                pc->pure_bands++;
            }
            else
                all_pure = 0;
        }
    pc->all_pure = all_pure;
    return all_pure;
}

/* ------------------------------------------------ noise substitution */

#define PNS_FLAT_MIN 0.40      /* within-band flatness of the leak-free spectrum (white noise ~0.56, tone far below) */
#define PNS_CREST_MAX 2.0      /* temporal: peak 23 ms sub-window energy over the mean, 3 dB */
#define PNS_NEIGHBOUR_DB 6.0   /* the band's energy in the neighbouring frames stays within this */

typedef struct
{
    unsigned int first_band;   /* 0 = off */
    double flat_min, crest_max, neighbour;
    double hyst;               /* Q: flatness margin of the hysteresis (MMX_PNS_HYST, 0 = none): a noise band stays
                                  noise down to flat_min - hyst, a coded band becomes noise only above flat_min + hyst */
    double prom_max;           /* Q: a band whose strongest line stands this far (dB) above its neighbours is never
                                  replaced by noise (MMX_PNS_PROM, 0 = off) */
} PnsParams;

/* The last band the coder spends coefficients on. Above the band-replication
   crossover a band carries only its level and a mix index, so every rule that
   costs bits (quantizer, noise substitution, intensity stereo, the energy
   rules, the bit estimates) stops there. */
static unsigned int coded_cutoff(const MMXCodec *c)
{
    return c->bands.bwe_band < c->cutoff_band ? c->bands.bwe_band : c->cutoff_band;
}

/* MMX_BWE_MIX: force the noise share of every replicated band (0..3) instead
   of deriving it from the flatness of source and target (-1 = derive). */
static int bwe_mix_force = -2;

/* Band replication on the encoder side: every band of a long frame from the
   crossover on is replaced by its energy (the noise-band level grid, 1.5 dB
   steps) and a noise-mix index derived from the spectral flatness of the band
   and of the patch that will be copied into it. `target` is the frame's
   signal in the coded domain (never the residual: the replication replaces the
   band as a whole, prediction included, like a noise band). A band whose
   energy stays under the absolute threshold of hearing is written as a zero
   band - a master that ends at 16 kHz gets no hiss on top. */
static void bwe_apply(const MMXCodec *codec, MMXFrameSyntax *s, float *const *target)
{
    const MMXBandLayout *L = &codec->bands;
    unsigned int c, b;
    static double zero_rel = -1.0;
    if (zero_rel < 0.0)
    {
        /* How far under the absolute threshold of hearing a band is written as a
           zero band instead of being regenerated. The default is -100 dB, i.e.
           only digital silence: measured on title I at 128 kbit/s,
           a gate at the threshold itself (0 dB) leaves the top octave 3.45 dB
           too quiet, -20 dB 2.06 dB, -40 dB 1.18 dB and -100 dB 0.34 dB, at
           0.8 % more bytes - and the coder without replication sits at
           2.69 dB. Above the threshold of hearing the ear does not hear the
           band, but the level meter does, and the cheap levels are worth it. */
        const char *e = getenv("MMX_BWE_ZERO_DB");
        zero_rel = pow(10.0, (e ? atof(e) : -100.0) / 10.0);
    }
    if (bwe_mix_force == -2)
    {
        const char *e = getenv("MMX_BWE_MIX");
        bwe_mix_force = e ? atoi(e) : -1;
    }
    if (L->bwe_band >= L->band_count || s->block_type == MMX_BT_SHORT)
        return;
    for (c = 0; c < s->channels; c++)
        for (b = L->bwe_band; b < L->band_count; b++)
        {
            unsigned long k0 = L->band_start[b], k1 = L->band_start[b + 1], k, n = k1 - k0;
            double e = 0.0;
            for (k = k0; k < k1; k++)
            {
                e += (double)target[c][k] * target[c][k];
                s->q[c][k] = 0;
            }
            s->band_noise[c][b] = 0;
            s->band_bwe[c][b] = 1;
            if (c == 0) s->band_is[b] = 0;
            if (b >= codec->cutoff_band || e <= (double)L->abs_thr[b] * (double)n * zero_rel)
            {
                s->band_zero[c][b] = 1;
                s->sf[c][b] = 0;
                s->bwe_mix[c][b] = 0;
                continue;
            }
            s->band_zero[c][b] = 0;
            s->sf[c][b] = (unsigned char)mmx_pns_level_index(e, n);
            s->bwe_mix[c][b] = (unsigned char)(bwe_mix_force >= 0 ? bwe_mix_force : (int)mmx_bwe_mix_index(L, target[c], b));
        }
}

/* The three rules can be moved for listening experiments: MMX_PNS_FLAT,
   MMX_PNS_CREST, MMX_PNS_NEIGH (dB). Measured on title A. */
static void pns_params_init(PnsParams *pp, const MMXCodec *codec, unsigned int pns_hz)
{
    const char *e;
    unsigned int b, first = mmx_pns_first_band(&codec->bands);
    memset(pp, 0, sizeof(*pp));
    pp->flat_min = (e = getenv("MMX_PNS_FLAT")) ? atof(e) : PNS_FLAT_MIN;
    pp->crest_max = (e = getenv("MMX_PNS_CREST")) ? atof(e) : PNS_CREST_MAX;
    pp->neighbour = pow(10.0, ((e = getenv("MMX_PNS_NEIGH")) ? atof(e) : PNS_NEIGHBOUR_DB) / 10.0);
    pp->hyst = (e = getenv("MMX_PNS_HYST")) ? atof(e) : 0.0;
    pp->prom_max = (e = getenv("MMX_PNS_PROM")) ? atof(e) : 0.0;
    if (!pns_hz)
        return;
    pp->first_band = 0;
    for (b = first; b < codec->bands.band_count; b++)
        if (codec->bands.band_hz[b] >= (float)pns_hz) { pp->first_band = b ? b : 1; break; }
}

/* Noise substitution for one long frame: a coded band (from first_band on)
   whose signal is noise-like (flat leak-free spectrum) and stationary (no
   transient near the window, low temporal crest, neighbouring frames within a
   few dB) carries only its level; the decoder fills it with noise of that
   energy. The decision looks at the target, not at the residual: a noise band
   replaces the band as a whole, prediction included. For M/S frames both L
   and R must be noise-like. The transient flag is broadband on purpose: the
   model's pre-echo rule tightens every band next to an attack, a per-band
   check let noise through there (measured: worst NMR 50 dB, 37 attacks). */
static void pns_substitute(const MMXCodec *codec, MMXFrameSyntax *syn, float *const *target,
                           const MMXAnalysis *analysis, unsigned long f, const PnsParams *pp,
                           const unsigned char (*prev_noise)[MMX_MAX_BANDS])
{
    const MMXBandLayout *L = &codec->bands;
    unsigned int c, b, nch = codec->channels;

    if (!pp->first_band || f == 0 || f + 1 >= analysis->frame_count || analysis->transient[f])
        return;
    for (c = 0; c < nch; c++)
    {
        unsigned int p0 = c, p1 = c, p;   /* psy channels the coded channel depends on */
        if (syn->stereo_ms && nch == 2) { p0 = 0; p1 = 1; }
        if (syn->tns[c].active)
            continue;
        for (b = pp->first_band; b < coded_cutoff(codec); b++)
        {
            unsigned long k0 = L->band_start[b], k1 = L->band_start[b + 1], k;
            double e = 0.0;
            int ok = 1;
            if (syn->band_zero[c][b] || syn->band_is[b])
                continue;
            /* hysteresis: the flatness a band needs depends on what it was in the previous long frame */
            double flat_need = pp->flat_min + (prev_noise && prev_noise[c][b] ? -pp->hyst : pp->hyst);
            for (p = p0; p <= p1 && ok; p++)
            {
                const MMXFramePsy *cur = mmx_analysis_psy(analysis, f, p);
                const MMXFramePsy *prev = mmx_analysis_psy(analysis, f - 1, p);
                const MMXFramePsy *next = mmx_analysis_psy(analysis, f + 1, p);
                if (cur->flat[b] < flat_need || cur->crest[b] > pp->crest_max)
                    ok = 0;
                else if (pp->prom_max > 0.0 && cur->prom[b] >= pp->prom_max)
                    ok = 0;                                     /* a line stands out: never noise */
                else if (prev->energy[b] > cur->energy[b] * pp->neighbour || prev->energy[b] * pp->neighbour < cur->energy[b] ||
                         next->energy[b] > cur->energy[b] * pp->neighbour || next->energy[b] * pp->neighbour < cur->energy[b])
                    ok = 0;
            }
            if (!ok)
                continue;
            for (k = k0; k < k1; k++)
                e += (double)target[c][k] * target[c][k];
            if (e <= 0.0)
                continue;
            if (codec->bands.lowrate)
            {
                /* EPB: the band is dropped instead; mmx_epb_finish fills it from its energy index */
                syn->band_zero[c][b] = 1;
                syn->band_noise[c][b] = 0;
                syn->sf[c][b] = 0;
                for (k = k0; k < k1; k++) syn->q[c][k] = 0;
                continue;
            }
            syn->band_noise[c][b] = 1;
            syn->sf[c][b] = (unsigned char)mmx_pns_level_index(e, k1 - k0);
            for (k = k0; k < k1; k++) syn->q[c][k] = 0;
        }
    }
}

/* ------------------------------------------------------- noise filling */

/* The rate loop lifts every threshold by one offset; below the quality's own
   rate the coded bands above a few kHz round most coefficients to zero and
   whole bands fall under the lifted threshold and become holes: dull highs
   and birdies, where Opus keeps every band's energy (PVQ, folding). Noise
   filling gives the energy back without coding the waveform: (1) per channel
   and EQ region the zeros of the coded bands carry the rms they lost, in
   units of the band's step (mmx_frame_noise_fill_levels; the decoder fills
   them with deterministic noise), (2) a band that quantized to zero as a
   whole although the quality's own model would have coded it (E * crest >
   allowed / thr_scale, i.e. a hole made by the offset, not by masking) keeps
   its energy as a noise band (capped at the lifted threshold). Bands whose EQ region is
   predicted by a source are not holes and their zeros are not filled
   (MMX_NF_RESID=1 fills the zeros of predicted regions as well). The fill is
   not part of the quantizer's noise budget: it is the signal's own energy,
   not an error; the masking model of mmx compare counts it as error all the
   same (an energy-matched different waveform), like the noise bands of
   --pns. Knobs: MMX_NF_HZ (first EQ region, default MMX_NF_DEFAULT_HZ),
   MMX_NF_DB (fill energy relative to the lost energy), MMX_NF_HOLE_DB (fill
   holes down to this far below the lifted threshold instead of down to the
   quality's own threshold), MMX_NF_RESID. */
#define NF_GAIN_DB 0.0
#define NF_HOLE_FIXED_DB 12.0  /* --nf at a fixed quality: holes within this far below the threshold are filled */
#define NF_ZERO_ALPHA 1.0      /* distortion weight of a zero the decoder fills (MMXNfZeroWeight) */

typedef struct
{
    int on;                    /* the encoder applies noise filling */
    unsigned int first_band;   /* first band of the first filled EQ region */
    unsigned int first_region; /* first EQ region whose zeros and holes are filled */
    double gain;               /* rms factor of the fill */
    int resid;                 /* fill predicted regions as well */
    double hole_rel;           /* a hole is filled when E * crest > allowed * hole_rel; < 0 = 1 / thr_scale */
    MMXNfZeroWeight zero;      /* distortion weight of the zeros per filled region (MMX_NF_ZERO, 1 = plain quantizer) */
    int zero_on;               /* the weight is below 1 somewhere: pass it to the quantizer */
} NfParams;

/* The zero weight the quantizer calls take (NULL = the plain quantizer). */
static const MMXNfZeroWeight *nf_zero_of(const NfParams *np)
{
    return np->on && np->zero_on ? &np->zero : NULL;
}

static void nf_params_init(NfParams *np, const MMXCodec *codec, const MMXEncoderParams *params)
{
    const char *e;
    unsigned int b, hz = (e = getenv("MMX_NF_HZ")) ? (unsigned int)atoi(e) : MMX_NF_DEFAULT_HZ, first = mmx_pns_first_band(&codec->bands);
    memset(np, 0, sizeof(*np));
    np->on = params->noise_fill == MMX_NF_ON || (params->noise_fill == MMX_NF_AUTO && params->target_kbps > 0);
    np->gain = pow(10.0, ((e = getenv("MMX_NF_DB")) ? atof(e) : NF_GAIN_DB) / 20.0);
    np->resid = (e = getenv("MMX_NF_RESID")) ? atoi(e) != 0 : 0;
    /* MMX_NF_ZERO=<alpha> for every region, or "a,b,c,d" for the regions 2-4k, 4-8k, 8-12k, >12k */
    {
        double a[4] = { NF_ZERO_ALPHA, NF_ZERO_ALPHA, NF_ZERO_ALPHA, NF_ZERO_ALPHA };
        unsigned int i, n = 0;
        if ((e = getenv("MMX_NF_ZERO")) != NULL)
        {
            n = (unsigned int)sscanf(e, "%lf,%lf,%lf,%lf", &a[0], &a[1], &a[2], &a[3]);
            for (i = n > 0 ? n : 1; i < 4; i++) a[i] = a[i - 1];
        }
        for (i = 0; i < MMX_EQ_BANDS; i++)
        {
            np->zero.alpha[i] = i >= 4 ? a[i - 4] : 1.0;
            if (np->zero.alpha[i] < 1.0) np->zero_on = 1;
        }
    }
    /* holes: down to the quality's own threshold (the rate offset); at a fixed
       quality (--nf) there is no offset, then 12 dB below the threshold */
    np->hole_rel = (e = getenv("MMX_NF_HOLE_DB")) ? pow(10.0, -atof(e) / 10.0) : params->target_kbps > 0 ? -1.0 : pow(10.0, -NF_HOLE_FIXED_DB / 10.0);
    np->first_band = codec->bands.band_count;
    np->first_region = MMX_EQ_BANDS;
    for (b = first; b < codec->bands.band_count; b++)
        if (codec->bands.band_hz[b] >= (float)hz) { np->first_region = codec->bands.eq_band[b]; break; }
    for (b = first; b < codec->bands.band_count; b++)
        if (codec->bands.eq_band[b] == np->first_region) { np->first_band = b; break; }
    if (np->first_band >= codec->cutoff_band)
        np->on = 0;
    np->zero.first_band = np->first_band;
    np->zero.resid = np->resid;
}

/* Holes of a quantized long-frame candidate (see above): x are the
   coded-domain coefficients the quantizer saw, thr and crest its thresholds. */
static void nf_fill_holes(const MMXCodec *codec, MMXFrameSyntax *syn, float *const *x, const float *const *thr,
                          const float *const *crest, const double *bscale, unsigned int n_sources, const NfParams *np)
{
    const MMXBandLayout *L = &codec->bands;
    unsigned int c, b, src;
    for (c = 0; c < syn->channels; c++)
        for (b = np->first_band; b < coded_cutoff(codec); b++)
        {
            double rel = np->hole_rel >= 0.0 ? np->hole_rel : 1.0 / bscale[b];
            unsigned long k0 = L->band_start[b], k1 = L->band_start[b + 1], k, n = k1 - k0;
            double allowed, e = 0.0, cr, fill;
            int predicted = 0;
            if (!syn->band_zero[c][b] || thr[c][b] >= 1e29f)
                continue;
            if (syn->channels == 2 && syn->band_is[b])
                continue;   /* intensity band: channel 1 carries nothing, channel 0 the mid */
            for (src = 0; src < n_sources; src++)
                if (syn->gain[src][c][L->eq_band[b]] != MMX_GAIN_OFF)
                    predicted = 1;
            if (predicted)
                continue;
            for (k = k0; k < k1; k++)
                e += (double)x[c][k] * x[c][k];
            allowed = (double)thr[c][b] * (double)n;
            cr = crest ? (double)crest[c][b] : 1.0;
            if (rel >= 1.0 || e * cr <= allowed * rel)
                continue;   /* rel >= 1: no offset in this band, the zero is the model's own */
            fill = e * np->gain * np->gain;
            if (fill > allowed)
                fill = allowed;   /* the fill alone never exceeds the lifted threshold */
            if (fill <= 0.0)
                continue;
            syn->band_zero[c][b] = 0;
            syn->band_noise[c][b] = 1;
            syn->sf[c][b] = (unsigned char)mmx_pns_level_index(fill, n);
            for (k = k0; k < k1; k++) syn->q[c][k] = 0;
        }
}

/* -------------------------------------------- energy preservation (holes) */

/* Where the gurgling comes from. The rate loop lifts every masking threshold
   by one offset (analysis->thr_scale); a band whose energy sits near that
   lifted threshold is dropped in one frame and coded in the next, so a hole in
   that band opens and closes several times per second. Measured on title B
   at 128 kbit/s: 3-17 such changes per second and band ("toggles",
   `mmx compare` reports them). That is the bubbling the ear hears. Opus cannot
   do it: PVQ codes at least one pulse per band and the band energy is always
   transmitted.

   Three rules, encoder-side, on in every mode (no syntax change):

   (1) Minimum allocation. A band the quantizer zeroed although the model's own
       threshold - the one *without* the rate offset - would have coded it must
       keep its energy. Below KEEP_NOISE_HZ as at least one coded coefficient
       (the largest, at the step closest to it, refined while the band keeps
       less than KEEP_ENERGY_DB of its energy: mmx_frame_keep_band), above it
       as an energy-only noise band, the mechanism --pns and the noise filling
       already use. The offset therefore only makes the coding coarser; it no
       longer decides *whether* a band sounds.
   (2) Hysteresis. The decision uses two thresholds: a band that carried energy
       in the previous frame keeps it until its energy falls KEEP_HYST_DB below
       the model threshold, a band that did not must rise KEEP_HYST_DB above
       it. Near-threshold bands stop flapping at the frame rate. The state is
       the previous frame's syntax (keep_commit), so every trial of a frame
       sees the same decisions and the trial bits are the real ones.
   (3) Predicted bands. A zeroed residual is not a hole where a source predicts
       the band - the prediction carries it. Only when the prediction is more
       than KEEP_PRED_DB below the target does the band count as a hole, and
       then it is topped up with a coefficient (never with a noise band: the
       decoder drops the prediction in a noise band).

   Knobs for experiments: MMX_KEEP=0 switches all of it off (the coder before
   the rule), MMX_KEEP_HZ the crossover, MMX_KEEP_HYST_DB the hysteresis,
   MMX_KEEP_DB the energy a restored band keeps, MMX_KEEP_STEPS how far its
   step may be refined, MMX_KEEP_PRED=0 leaves predicted bands alone,
   MMX_KEEP_PRED_DB their hole margin. */
#define KEEP_NOISE_HZ 3000         /* crossover: coded coefficient below, energy-only noise band above.
                                      Measured on title B 20 s at 128 / 96 kbit/s (band toggles
                                      per second, cells over the threshold): no noise bands at all 1.50 /
                                      12.75 toggles and 60.6 / 70.7 % over, 4 kHz 1.10 / 3.50 and 58.6 /
                                      67.7 %, 3 kHz 0.80 / 3.35 and 58.4 / 67.7 %, 2 kHz 0.70 / 3.40 and
                                      59.0 / 67.9 %. Below 3 kHz a coefficient keeps the partial the ear
                                      follows, above it the energy is what counts. */
#define KEEP_HYST_DB 3.0
#define KEEP_ENERGY_DB 3.0         /* a restored band keeps its energy within this */
#define KEEP_STEPS 6               /* at most this many 1.5 dB refinements of its step */
#define KEEP_PRED_DB 10.0          /* a predicted band is a hole when the prediction is this far below the target */

typedef struct
{
    int on;
    unsigned int noise_band;   /* first band restored as a noise band (never below mmx_pns_first_band) */
    double hyst_lo, hyst_hi;   /* factors on the model threshold (previously coded / previously zeroed) */
    double keep_rel;           /* share of the band energy a restored band keeps */
    int steps;
    int pred;                  /* protect the bands a source predicts as well */
    double pred_rel;           /* ... when the prediction is below this share of the target */
    double max_rel;            /* 0, or the deepest the protection reaches below the *lifted* threshold:
                                  with a cap the rate loop keeps control of the size, without it the
                                  minimum allocation is a floor the offset cannot lower (MMX_KEEP_MAX_DB) */
} KeepParams;

static void keep_params_init(KeepParams *kp, const MMXCodec *codec)
{
    const char *e;
    unsigned int b, hz = (e = getenv("MMX_KEEP_HZ")) ? (unsigned int)atoi(e) : KEEP_NOISE_HZ;
    double hyst = (e = getenv("MMX_KEEP_HYST_DB")) ? atof(e) : KEEP_HYST_DB;
    memset(kp, 0, sizeof(*kp));
    kp->on = (e = getenv("MMX_KEEP")) ? atoi(e) != 0 : !codec->bands.lowrate;   /* EPB folds an uncoded band at its energy; the keep coefficient would become a pure tone at band energy */
    kp->hyst_lo = pow(10.0, -hyst / 10.0);
    kp->hyst_hi = pow(10.0, hyst / 10.0);
    kp->keep_rel = pow(10.0, -((e = getenv("MMX_KEEP_DB")) ? atof(e) : KEEP_ENERGY_DB) / 10.0);
    kp->steps = (e = getenv("MMX_KEEP_STEPS")) ? atoi(e) : KEEP_STEPS;
    kp->pred = (e = getenv("MMX_KEEP_PRED")) ? atoi(e) != 0 : 1;
    kp->pred_rel = pow(10.0, -((e = getenv("MMX_KEEP_PRED_DB")) ? atof(e) : KEEP_PRED_DB) / 10.0);
    kp->max_rel = (e = getenv("MMX_KEEP_MAX_DB")) ? pow(10.0, -atof(e) / 10.0) : 0.0;
    kp->noise_band = codec->bands.band_count;
    if (codec->bands.lowrate) hz = 100000;   /* EPB: a restored band is always a coefficient, never a noise band */
    for (b = mmx_pns_first_band(&codec->bands); b < codec->bands.band_count; b++)
        if (codec->bands.band_hz[b] >= (float)hz) { kp->noise_band = b; break; }
}

/* ------------------------------------------------------------ clip guard */

#define CLIP_GUARD_STEPS 3       /* at most this many re-quantizations per frame */
#define CLIP_GUARD_STEP_DB 3.0   /* every step tightens the selected thresholds by this much */
#define CLIP_GUARD_TOP 0.9       /* share of the frame's allowed noise energy whose bands are tightened (1 = all bands) */
#define CLIP_GUARD_MARGIN_DB 1.0 /* a band counts when the clamp lifts its NMR above 0 dB and by more than this */
#define CLIP_SUB_WIN 1024        /* the 23 ms analysis windows of mmx compare */

/* The reconstruction is float and unbounded, but every sink clamps it to full
   scale (the WAV writer, the player's 16/24-bit output). On a hot master the
   coding noise on top of samples near full scale leaves the range in
   thousands of places, and the clamp turns each into an error the masking
   model never allowed for: a flat top, broadband, in the loudest moments
   (title A strict: 8,900 samples, the worst band NMR of the file is such
   a place). The encoder knows its reconstruction, so after every frame the
   samples that are complete (the first half of its window; the second half
   still waits for the next frame) are checked. When one of them leaves the
   range, the two 23 ms windows that are complete are analysed exactly like
   mmx compare does (leak-free FFT, masking model of the original): if the
   clamped error exceeds a band's threshold where the unclamped does not (by
   more than the margin), the frame is quantized again with its thresholds
   tightened by a step, at most CLIP_GUARD_STEPS times. Overshoots the model
   cannot hear are left alone: a sample the original holds at full scale
   cannot be kept inside by any finite noise anyway. Only hot frames pay.
   MMX_CLIP_GUARD=0 turns the retries off (the counts are still reported),
   MMX_CLIP_STEPS, MMX_CLIP_STEP_DB, MMX_CLIP_MARGIN_DB, MMX_CLIP_LIMIT_DB
   (range as dBFS instead of the sink's clamp) and MMX_CLIP_TOP (tighten only
   the bands holding this share of the allowed noise, loudest thresholds
   first) move the knobs; MMX_CLIP_ALL=1 retries on every overshoot. */
typedef struct
{
    int steps, all;
    double step_scale;       /* threshold factor per step (< 1) */
    double top, margin_db;
    double lim_pos, lim_neg; /* range of the sink: the writer rounds and clamps at the source bit depth */
    MMXSpectrum spec;
    MMXBandLayout bands;     /* 512-bin layout of the analysis windows */
    unsigned int quality, cutoff;
    float *win, *amp_o, *amp_u, *amp_c;
    float *y, *fin;          /* a frame's own contribution (MMX_WIN); the final signal of its window as far as
                                it is known: [start - 512, start + 2048), the second half predicted (CLIP_SPAN) */
    int ok, debug;           /* MMX_DEBUG_CLIP: prints every retry and what the model still hears */
} ClipGuard;
#define CLIP_SPAN (MMX_WIN + CLIP_SUB_WIN / 2)

static int clip_guard_init(ClipGuard *g, unsigned long sample_rate, unsigned int source_bits, unsigned int quality)
{
    const char *e;
    double scale = pow(2.0, (double)(source_bits ? source_bits : 16) - 1.0);
    memset(g, 0, sizeof(*g));
    g->steps = (e = getenv("MMX_CLIP_STEPS")) ? atoi(e) : CLIP_GUARD_STEPS;
    if ((e = getenv("MMX_CLIP_GUARD")) && atoi(e) == 0)
        g->steps = 0;
    g->all = (e = getenv("MMX_CLIP_ALL")) && atoi(e) != 0;
    g->step_scale = pow(10.0, -((e = getenv("MMX_CLIP_STEP_DB")) ? atof(e) : CLIP_GUARD_STEP_DB) / 10.0);
    g->top = (e = getenv("MMX_CLIP_TOP")) ? atof(e) : CLIP_GUARD_TOP;
    g->margin_db = (e = getenv("MMX_CLIP_MARGIN_DB")) ? atof(e) : CLIP_GUARD_MARGIN_DB;
    g->lim_pos = (scale - 0.5) / scale;    /* x * scale + 0.5 >= scale rounds to a value the sink cannot hold */
    g->lim_neg = -(scale + 0.5) / scale;
    if ((e = getenv("MMX_CLIP_LIMIT_DB")))
    {
        double lim = pow(10.0, atof(e) / 20.0);
        g->lim_pos = lim;
        g->lim_neg = -lim;
    }
    g->quality = quality;
    g->debug = getenv("MMX_DEBUG_CLIP") != NULL;
    if (!g->steps)
        return 0;
    if (mmx_spectrum_init(&g->spec, CLIP_SUB_WIN) != 0)
        return -1;
    g->ok = 1;
    if (mmx_bands_init(&g->bands, sample_rate, CLIP_SUB_WIN / 2) != 0)
        return -1;
    g->cutoff = g->bands.cutoff_band[quality > MMX_QUALITY_MAX ? MMX_QUALITY_MAX : quality];
    g->win = (float *)malloc(sizeof(float) * CLIP_SUB_WIN);
    g->amp_o = (float *)malloc(sizeof(float) * CLIP_SUB_WIN / 2);
    g->amp_u = (float *)malloc(sizeof(float) * CLIP_SUB_WIN / 2);
    g->amp_c = (float *)malloc(sizeof(float) * CLIP_SUB_WIN / 2);
    g->y = (float *)malloc(sizeof(float) * MMX_WIN);
    g->fin = (float *)malloc(sizeof(float) * CLIP_SPAN * MMX_MAX_CH);
    return g->win && g->amp_o && g->amp_u && g->amp_c && g->y && g->fin ? 0 : -1;
}

static void clip_guard_free(ClipGuard *g)
{
    if (g->ok) mmx_spectrum_free(&g->spec);
    free(g->win); free(g->amp_o); free(g->amp_u); free(g->amp_c); free(g->y); free(g->fin);
    memset(g, 0, sizeof(*g));
}

/* A frame's own contribution to the output: the windowed inverse transform
   of `coefs` (L/R, the layout of the block type), as mmx_codec_reconstruct
   overlap-adds it, into out[MMX_WIN]. */
static void frame_contribution(MMXCodec *codec, const float *coefs, unsigned int block_type, float *out)
{
    unsigned int g;
    unsigned long k;
    if (block_type != MMX_BT_SHORT)
    {
        mmx_mdct_inverse_w(&codec->mdct, coefs,
                           block_type == MMX_BT_START ? codec->win_start : block_type == MMX_BT_STOP ? codec->win_stop : codec->mdct.window, out);
        return;
    }
    memset(out, 0, sizeof(float) * MMX_WIN);
    for (g = 0; g < MMX_SHORT_GROUPS; g++)
    {
        mmx_mdct_inverse(&codec->short_mdct, coefs + g * MMX_SHORT_M, codec->sy);
        for (k = 0; k < 2 * MMX_SHORT_M; k++)
            out[codec->short_offset + g * MMX_SHORT_M + k] += codec->sy[k];
    }
}

/* Samples of `dec` in [start, start + n) outside the range; *peak tracks the largest |x|. */
static unsigned long clip_guard_count(const ClipGuard *g, const MMXAudioBuffer *dec, long long start, unsigned long n, double *peak)
{
    long long i, lo = start < 0 ? 0 : start, hi = start + (long long)n;
    unsigned int c;
    unsigned long over = 0;
    if (hi > (long long)dec->frame_count)
        hi = (long long)dec->frame_count;
    for (i = lo; i < hi; i++)
        for (c = 0; c < dec->channels; c++)
        {
            double x = dec->samples[(size_t)i * dec->channels + c];
            if (x > g->lim_pos || x < g->lim_neg)
                over++;
            if (x > *peak) *peak = x;
            else if (-x > *peak) *peak = -x;
        }
    return over;
}

/* One 23 ms window of channel c at offset `off` of the frame's span (see
   clip_guard_audible): does the clamped error exceed the masking threshold of
   the original in a band where the unclamped error does not (by more than
   the margin)? The same analysis as mmx compare. */
static int clip_guard_window(ClipGuard *g, const MMXAudioBuffer *audio, long long start, unsigned long off, unsigned int c)
{
    MMXFramePsy psy;
    const float *fin = g->fin + (size_t)c * CLIP_SPAN + off;
    unsigned long k;
    unsigned int b, nch = audio->channels;
    long long first = start - CLIP_SUB_WIN / 2 + (long long)off;
    for (k = 0; k < CLIP_SUB_WIN; k++)
    {
        long long p = first + (long long)k;
        g->win[k] = (p >= 0 && p < (long long)audio->frame_count) ? audio->samples[(size_t)p * nch + c] : 0.0f;
    }
    mmx_spectrum_analyze(&g->spec, g->win, g->amp_o);
    mmx_psy_analyze(&g->bands, g->amp_o, g->win, g->quality, &psy);
    for (k = 0; k < CLIP_SUB_WIN; k++)
    {
        long long p = first + (long long)k;
        g->win[k] = fin[k] - ((p >= 0 && p < (long long)audio->frame_count) ? audio->samples[(size_t)p * nch + c] : 0.0f);
    }
    mmx_spectrum_analyze(&g->spec, g->win, g->amp_u);
    for (k = 0; k < CLIP_SUB_WIN; k++)
    {
        long long p = first + (long long)k;
        double x = fin[k];
        if (x > g->lim_pos) x = g->lim_pos;
        if (x < g->lim_neg) x = g->lim_neg;
        g->win[k] = (float)(x - ((p >= 0 && p < (long long)audio->frame_count) ? audio->samples[(size_t)p * nch + c] : 0.0f));
    }
    mmx_spectrum_analyze(&g->spec, g->win, g->amp_c);
    for (b = 0; b < g->cutoff; b++)
    {
        double nu = 0.0, nc = 0.0, n = (double)(g->bands.band_start[b + 1] - g->bands.band_start[b]), nmr_u, nmr_c;
        if (psy.thr[b] >= 1e29f || psy.energy[b] <= psy.thr[b] * n * 0.5)
            continue;
        for (k = g->bands.band_start[b]; k < g->bands.band_start[b + 1]; k++)
        {
            nu += (double)g->amp_u[k] * g->amp_u[k];
            nc += (double)g->amp_c[k] * g->amp_c[k];
        }
        nmr_c = mmx_psy_nmr_db(&g->bands, &psy, b, nc);
        if (nmr_c <= 0.0)
            continue;
        nmr_u = mmx_psy_nmr_db(&g->bands, &psy, b, nu);
        if (nmr_c > nmr_u + g->margin_db)
        {
            if (g->debug)
                fprintf(stderr, "    window %lld (%.3f s) ch %u band %u (%.0f Hz): NMR unclamped %.1f clamped %.1f dB\n", first,
                        (double)first / audio->sample_rate, c, b, g->bands.band_hz[b], nmr_u, nmr_c);
            return 1;
        }
    }
    return 0;
}

/* Does the clamp of the sink make an audible difference in the span this
   frame is responsible for? The span [start - 512, start + 2048) holds the
   samples before the frame as the sink will play them (committed, clamped),
   the first half of the window as it is (final), and the second half as it
   will be when the next frame adds a perfect contribution: x + e_f, with e_f
   the frame's own error (its contribution minus that of the original, one
   inverse transform of the L/R targets). So a frame answers for its own
   noise on both halves, not for what the previous frame left behind, and
   the previous frame could not shift onto the next one what only it caused.
   The four 23 ms windows of the span are judged like mmx compare does. */
static int clip_guard_audible(ClipGuard *g, MMXCodec *codec, const MMXAudioBuffer *audio, const MMXAudioBuffer *dec,
                              float *const *t_lr, unsigned int block_type, long long start)
{
    unsigned int c, nch = audio->channels;
    unsigned long k, w, over = 0;
    for (c = 0; c < nch; c++)
    {
        float *fin = g->fin + (size_t)c * CLIP_SPAN;
        frame_contribution(codec, t_lr[c], block_type, g->y);
        for (k = 0; k < CLIP_SPAN; k++)
        {
            long long p = start - CLIP_SUB_WIN / 2 + (long long)k;
            double x = (p >= 0 && p < (long long)dec->frame_count) ? dec->samples[(size_t)p * nch + c] : 0.0;
            if (k < CLIP_SUB_WIN / 2)
            {
                if (x > g->lim_pos) x = g->lim_pos;
                if (x < g->lim_neg) x = g->lim_neg;
            }
            else
            {
                if (k >= CLIP_SUB_WIN / 2 + MMX_HOP && p >= 0 && p < (long long)audio->frame_count)
                    x += audio->samples[(size_t)p * nch + c] - g->y[k - CLIP_SUB_WIN / 2];
                if (x > g->lim_pos || x < g->lim_neg) over++;
            }
            fin[k] = (float)x;
        }
    }
    if (!over)
        return 0;
    for (c = 0; c < nch; c++)
        for (w = 0; w < 4; w++)
            if (clip_guard_window(g, audio, start, w * (CLIP_SUB_WIN / 2), c))
                return 1;
    return 0;
}

/* Copy of the samples a frame's window covers (to undo its overlap-add). */
static void window_save(float *save, const MMXAudioBuffer *dec, long long start, int restore)
{
    long long lo = start < 0 ? 0 : start, hi = start + MMX_WIN;
    size_t n;
    if (hi > (long long)dec->frame_count)
        hi = (long long)dec->frame_count;
    if (hi <= lo)
        return;
    n = (size_t)(hi - lo) * dec->channels;
    if (restore)
        memcpy(dec->samples + (size_t)lo * dec->channels, save, sizeof(float) * n);
    else
        memcpy(save, dec->samples + (size_t)lo * dec->channels, sizeof(float) * n);
}

/* Thresholds of a frame analysis scaled by s (frame, sub-window and final):
   all bands, or (top < 1) the bands with the highest allowed noise per
   coefficient that together hold the share `top` of the frame's allowed noise
   energy - the cheapest bits against a time-domain overshoot, because the
   noise of a band costs bits per coefficient but adds amplitude per energy. */
static void psy_scale(MMXFramePsy *p, const MMXBandLayout *L, unsigned int cutoff, double s, double top)
{
    unsigned int b, w, n = 0, order[MMX_MAX_BANDS];
    unsigned char pick[MMX_MAX_BANDS];
    double total = 0.0, acc = 0.0;
    memset(pick, 1, sizeof(pick));
    if (top < 1.0)
    {
        memset(pick, 0, sizeof(pick));
        for (b = 0; b < cutoff && b < L->band_count; b++)
            if (p->thr[b] < 1e29f)
            {
                unsigned int i = n, j;
                total += (double)p->thr[b] * (L->band_start[b + 1] - L->band_start[b]);
                while (i > 0 && p->thr[order[i - 1]] < p->thr[b]) i--;   /* insertion sort, loudest threshold first */
                for (j = n; j > i; j--) order[j] = order[j - 1];
                order[i] = b;
                n++;
            }
        for (b = 0; b < n && acc < top * total; b++)
        {
            pick[order[b]] = 1;
            acc += (double)p->thr[order[b]] * (L->band_start[order[b] + 1] - L->band_start[order[b]]);
        }
    }
    for (b = 0; b < MMX_MAX_BANDS; b++)
    {
        if (!pick[b])
            continue;
        if (p->thr[b] < 1e29f) p->thr[b] = (float)(p->thr[b] * s);
        if (p->frame_thr[b] < 1e29f) p->frame_thr[b] = (float)(p->frame_thr[b] * s);
        for (w = 0; w < MMX_SUB_WINDOWS; w++)
            if (p->sub_thr[w][b] < 1e29f) p->sub_thr[w][b] = (float)(p->sub_thr[w][b] * s);
    }
}

/* ------------------------------------------------------ lossy closed loop */

/* Every rate decision of the closed loop is taken by coding, not by an
   estimate or a rule: the coded domain (L/R or M/S) and the prediction (on,
   or all gains off) are quantized under their own thresholds and trial-coded,
   the cheapest wins (frame_code); a noise band plays as noise only when its
   flag and level really cost less than its coefficients (pns_and_bits); the
   block type of a frame with a medium attack is decided by coding it and its
   neighbours both ways (bt_trial); the planner's runner-up lineage coded
   against the plan at the start of a block (plan_trial) is an experiment
   that does not pay. Single-frame
   trials code into copies of the coder state; multi-frame trials also
   reconstruct into the decoded signal, record in the reference graph and
   take everything back afterwards (trial_save / trial_restore). Decoder and
   encoder reconstruction stay bit-identical: only what the final pass writes
   is reconstructed for good. The experiment switches MMX_TRIAL_MS=0 and
   MMX_TRIAL_BT=0 fall back to the estimate and the rule, MMX_TRIAL_PLAN=1
   switches the runner-up trial on. */

#define BT_TRIAL_CELLS 0           /* the cheaper variant may have this many more (sub-window, band) cells over the threshold */
#define BT_TRIAL_DB 1.0            /* ... and a worst band this much above the other variant's (when above 0 dB) */
#define BT_REPAIR_EXTRA 0.10       /* the other block type may cost this much more when it repairs a smeared attack */
#define TRIAL_FRAMES 4             /* block-type trial: frames coded per variant (START, SHORT, STOP and the long frame after,
                                      so that the STOP window's whole reach is final and checked) */
#define PLAN_TRIAL_FRAMES 64       /* runner-up trial: at most this many frames of the block coded per lineage */
#define PLAN_DEPTH_PENALTY 0.10    /* as the planner (DEPTH_PENALTY in analyzer.c): a deeper chain costs later references */
#define QSUB_WIN 1024              /* time-domain check of multi-frame trials: the sub-window grid of mmx compare */
#define QSUB_HOP 512
#define QSUB_MAX 8

/* --------------------------------------- structured error ("echo") guard */

/* A referenced frame is target = prediction + residual, and the decoder plays
   prediction + the *quantized* residual. Where the residual of a band is
   quantized away - at a bitrate target every threshold carries the rate loop's
   dB offset (11 title B at 128 kbit/s: +6.6 dB, at 96 kbit/s about
   +10 dB), in the tracker modes they are relaxed by design - what is left is
   not noise: it is the earlier bar itself, mixed from up to two sources and
   carried through chains up to depth 5. The masking model judges it by its
   energy, as if it were noise of the same size; the ear hears music from
   another place in the song, i.e. an echo.

   The guard measures, per referenced channel band, the component of the
   reconstruction error e = rec - target along the prediction p,

       coef = <e,p> / <p,p>,   S = coef^2 <p,p> = <e,p>^2 / <p,p>,

   the energy of the prediction that survives in the error, and keeps it
   ECHO_MARGIN_DB under the *model's* threshold of that band - the threshold
   before the rate offset and before the clean-samples scaling
   (mmx_analysis_model_thr) - because tonal material from another bar is not
   the noise a masking threshold allows. A band above that is quantized again,
   finer, in bounded steps (as the clip guard does); when that costs more than
   the frame's own "no prediction" candidate, that candidate wins on bits, and
   it has no structured error at all because it carries no prediction. Both
   candidates are judged by the same trial coding as before, so the comparison
   stays like for like.

   Knobs: MMX_ECHO_GUARD=0 measures only (the file is then bit-identical to one
   coded without this guard), MMX_ECHO_MARGIN_DB=<dB>, MMX_ECHO_STEPS=<n>;
   MMX_DEBUG_ECHO=1 prints the summary even under --quiet, =2 also the seconds
   that leak, so the ear can be pointed at them. */

#define ECHO_MARGIN_DB 3.0      /* the leak must stay this far under the model threshold (6 dB measured: costs 40 more smeared attacks at 96 kbit/s and leaks more, the demand no longer fits the bounded steps) */
#define ECHO_STEPS 6            /* bounded re-quantizations of a frame */
#define ECHO_STEP_DB 3.0        /* ... each at least this much finer */
#define ECHO_STEP_MAX_DB 12.0   /* ... and at most this much per step */
#define ECHO_SCALE_MIN 1e-5f    /* a band is never tightened by more than 50 dB */

typedef struct
{
    int measured;
    double over_db[MMX_MAX_CH][MMX_MAX_BANDS];  /* how far the leak is above (model threshold - margin); <= 0 = fine */
    double worst_db;                            /* worst leak against the model threshold itself */
    unsigned long bands, pure_bands, pure_over, zero_bands, over_thr, over_guard;
    double sum_over_db;
} EchoFrame;

typedef struct
{
    int on;                   /* re-quantize leaking bands (default), else measure only */
    int measure, debug, steps;
    double margin_db;
    /* statistics over the file */
    unsigned long bands, pure_bands, pure_over, zero_bands, over_thr, over_guard;
    unsigned long frames, frames_over, frames_guarded, frames_left, retries;
    double sum_over_db, worst_db, worst_at;
    double *sec_worst;        /* worst leak per second of the timeline */
    unsigned long *sec_count; /* leaking channel bands per second */
    unsigned long seconds;
} EchoGuard;

static int echo_guard_init(EchoGuard *G, double duration_seconds)
{
    const char *e;
    memset(G, 0, sizeof(*G));
    G->measure = 1;
    G->on = (e = getenv("MMX_ECHO_GUARD")) ? atoi(e) != 0 : 1;
    G->margin_db = (e = getenv("MMX_ECHO_MARGIN_DB")) ? atof(e) : ECHO_MARGIN_DB;
    G->steps = (e = getenv("MMX_ECHO_STEPS")) ? atoi(e) : ECHO_STEPS;
    G->debug = (e = getenv("MMX_DEBUG_ECHO")) ? atoi(e) : 0;   /* 1 = the summary on stderr, 2 = also the leaking seconds */
    G->worst_db = -200.0;
    G->seconds = (unsigned long)(duration_seconds > 0.0 ? duration_seconds : 0.0) + 2;
    G->sec_worst = (double *)malloc(sizeof(double) * G->seconds);
    G->sec_count = (unsigned long *)calloc(G->seconds, sizeof(unsigned long));
    if (!G->sec_worst || !G->sec_count)
        return -1;
    {
        unsigned long i;
        for (i = 0; i < G->seconds; i++) G->sec_worst[i] = -200.0;
    }
    return 0;
}

static void echo_guard_free(EchoGuard *G)
{
    free(G->sec_worst);
    free(G->sec_count);
    G->sec_worst = NULL;
    G->sec_count = NULL;
}

/* One leak value against the model threshold of a band, in dB. */
static double echo_band_db(const float *pred, const float *rec, const float *tgt, unsigned long k0, unsigned long k1,
                           double allowed)
{
    double pp = 0.0, ep = 0.0;
    unsigned long k;
    for (k = k0; k < k1; k++)
    {
        double pv = pred[k], ev = (double)rec[k] - (double)tgt[k];
        pp += pv * pv;
        ep += ev * pv;
    }
    if (pp <= 0.0 || allowed <= 0.0)
        return -200.0;
    return 10.0 * log10(ep * ep / pp / allowed + 1e-300);
}

typedef struct
{
    const MMXAudioBuffer *audio;
    const MMXEncoderParams *params;
    MMXCodec *codec;
    MMXAnalysis *analysis;
    MMXFile *out;
    MMXAudioBuffer decoded;
    MMXReferenceGraph graph;
    BlockWriter block, base;
    MMXFrameSyntax syn, alt, best, coded;   /* the frame; a candidate; the cheapest candidate; a candidate before noise substitution */
    float *t_lr[MMX_MAX_CH], *t_cod[MMX_MAX_CH], *resid[MMX_MAX_CH];
    unsigned char *bt;               /* block type per frame */
    unsigned char *is_short;         /* short by the attack rule or by trial (START/STOP derive from it) */
    unsigned char *cand;             /* block type still to be decided by trial */
    float *run_gain;                 /* tracker: pattern gains [f * MMX_EQ_BANDS + e] */
    PureTemporal *temporal;          /* tracker: sub-window check of pure bands */
    PnsParams pns;
    /* intensity stereo (params->is_hz): the candidate bands of the frame, see is_prepare */
    unsigned int is_first_band;      /* 0 = off */
    double is_rho_min, is_bias, is_slack;
    float *t_is, *x_is;              /* the mid target of the frame's candidate bands; the residual the plan quantizes */
    MMXIntensityPlan is_plan;
    int debug_is, debug_bandms;      /* MMX_DEBUG_IS: why bands are not intensity; MMX_DEBUG_BANDMS: per-band M/S oracle */
    unsigned long is_dbg[5];         /* in range, rejected by coherence, by the slack guard, by bits, chosen */
    double bandms_bits[2][MMX_MAX_CH][MMX_MAX_BANDS], bandms_frame, bandms_oracle;
    unsigned long bandms_frames;
    NfParams nf;
    /* the rate loop's offset shaped over frequency (see "tilted rate offset"):
       the factor actually applied per band of each layout */
    TiltProfile tilt;
    double bscale[MMX_MAX_BANDS];    /* long bands (codec->bands) */
    double sbscale[MMX_SHORT_BANDS]; /* short bands (codec->short_bands) */
    double qbscale[MMX_MAX_BANDS];   /* the compare/quality layout of region_nmr */
    /* energy preservation (the gurgling rules, see "energy preservation") */
    KeepParams keep;
    unsigned char keep_prev[MMX_MAX_CH][MMX_MAX_BANDS];      /* the band carried energy in the previous long frame */
    int keep_prev_ms;                                        /* ... in that channel domain */
    unsigned char pns_prev[MMX_MAX_CH][MMX_MAX_BANDS];       /* the band was a noise band in the previous long frame (PNS hysteresis) */
    double elem_real_last[MMX_ELEM_COUNT];                   /* M0: bits per syntax element of the frame just written */
    unsigned char keep_mark[MMX_MAX_CH][MMX_MAX_BANDS];      /* what the rule restored in the candidate (1 = coefficient, 2 = noise) */
    unsigned char keep_mark_best[MMX_MAX_CH][MMX_MAX_BANDS]; /* ... in the cheapest candidate */
    MMXSpectrum short_spec;
    MMXFramePsy psy_bt[MMX_MAX_CH];  /* START/STOP frames: thresholds of the window shape */
    MMXFramePsy psy_clip[MMX_MAX_CH];/* clip guard: the frame's thresholds tightened for a retry */
    ClipGuard guard;                 /* the reconstruction against the full-scale range of the sink (final pass only) */
    EchoGuard echo;                  /* the prediction leaking into the reconstruction of referenced frames */
    float *rec_cod[MMX_MAX_CH], *tgt_cod[MMX_MAX_CH];  /* reconstruction and target in the coded domain (echo guard) */
    double ref_offset_factor;        /* experiment MMX_REF_OFFSET_SHARE: the rate loop's offset applied only in part to referenced frames */
    float *clip_save;                /* clip guard: the decoded window before the frame's overlap-add */
    double pcm_scale;                /* 2^(source bits - 1): the rounding of the sink (pcm_clamp) */
    double pure_db, relax, relax_sub;
    int calibrate;                   /* book the coded estimate of every long frame (second-pass calibration, MMX_DEBUG_EST) */
    /* time-domain check of multi-frame trials: the yardstick of mmx compare
       (masking model per 23 ms sub-window of the leak-free spectrum) */
    MMXSpectrum qspec, qpre_spec;
    MMXBandLayout qbands, qpre_bands;
    float *qwin, *qorig, *qerr;
    MMXFramePsy *qpsy;               /* [sub-window * channels + c] */
    float *snap;                     /* decoded samples of a trial region, taken back afterwards */
    double nmr_sum;
    unsigned long nmr_count;
    int debug_bits, debug_bt, trial_ms, trial_bt, trial_plan;
    unsigned long bt_cells;
    double bt_db;
    unsigned long bt_trials, bt_trials_short, bt_trials_quality, plan_trials, plan_trials_alt, pns_reverted;
} Enc;

typedef struct
{
    long long lo, hi;                /* decoded samples [lo, hi) saved */
    unsigned long f0, n;             /* frames whose plan entries are saved */
    MMXFramePlan plan[PLAN_TRIAL_FRAMES + 1];
    unsigned char sf[MMX_MAX_CH][MMX_MAX_BANDS], band_zero[MMX_MAX_CH][MMX_MAX_BANDS], band_noise[MMX_MAX_CH][MMX_MAX_BANDS];
                                     /* the long-frame arrays of the frame syntax (inter-frame state, see frame_code) */
    unsigned char keep_prev[MMX_MAX_CH][MMX_MAX_BANDS];   /* ... and the hysteresis state of the energy rules */
    int keep_prev_ms;
    unsigned char pns_prev[MMX_MAX_CH][MMX_MAX_BANDS];
} TrialState;

/* Trial-encodes the syntax with copies of the coder state; returns its bits. */
static double trial_bits(const BlockWriter *w, const MMXBandLayout *L, const MMXBandLayout *Ls, const MMXFrameSyntax *s, unsigned int n_sources)
{
    MMXRangeEncoder rc;
    MMXCodecContexts ctx = w->ctx;
    double bits;
    mmx_rc_enc_init(&rc);
    bits = mmx_rc_enc_bits(&rc);
    mmx_frame_encode(&rc, &ctx, L, Ls, s, n_sources);
    bits = mmx_rc_enc_bits(&rc) - bits;
    mmx_rc_enc_free(&rc);
    return bits;
}

/* The hysteresis state of the energy rules for one band of a coded channel:
   whether it carried energy in the previous long frame. When the channel
   domain flipped between the frames (L/R vs M/S) the two states are read
   together - the safe reading, the band stays protected. */
static int keep_was_coded(const Enc *E, const MMXFrameSyntax *s, unsigned int c, unsigned int b)
{
    if (s->channels == 2 && s->stereo_ms != E->keep_prev_ms)
        return E->keep_prev[0][b] || E->keep_prev[1][b];
    return E->keep_prev[c][b];
}

/* EXPERIMENT MMX_TONAL_RELAX=<alpha> (default 0 = off): a band whose crest marks a partial takes
   only a share of the rate loop's lift, thr_eff = thr_model * lift^(1 - alpha). The full lift drops a
   sustained partial into the quantizer's dead zone in one frame and not in the next: on title A
   5:30 at 96 kbit/s a partial between 6 and 11 kHz is gone in 23-47 % of its frames ("chopped
   synths"). The noise-like bands pay the bits. MMX_TONAL_CREST (default 4) is the crest from which a
   band counts as tonal (MMX_TONAL_FLAT moves the flatness from which the relief fades in). */
static void tonal_relax(const Enc *E, float (*thr)[MMX_MAX_BANDS], const MMXFramePsy *const *psy, unsigned int nch, unsigned int nb)
{
    /* tonality from the model's within-band spectral flatness: noise sits near 0.56 (geometric over
       arithmetic mean of exponentially distributed bin powers), a partial far below it */
    static double alpha = -1.0, flat_on = 0.35, flat_full = 0.12, fixed = 1.0, prom_on = 0.0;
    static int dbg = 0;
    unsigned int c, b;
    if (alpha < 0.0)
    {
        /* MMX_TONAL_DB=<dB>: a relief against the MODEL itself, not only against the lift - a sustained
           partial is audible below the threshold the noise-based model allows it (title A 5:30 at
           128 kbit/s: no lift at all, still 13 % of the partials' frames 8-11 kHz gone, 3 % with the
           thresholds 12 dB lower) */
        const char *e = getenv("MMX_TONAL_RELAX"), *k = getenv("MMX_TONAL_FLAT"), *d = getenv("MMX_TONAL_DB");
        alpha = e ? atof(e) : 0.0;
        if (k) flat_on = atof(k);
        if (flat_full > flat_on * 0.5) flat_full = flat_on * 0.5;
        if (d) fixed = pow(10.0, -atof(d) / 10.0);
        /* MMX_TONAL_PROM=<dB>: judge tonality by the band's line prominence instead of its flatness
           (full relief from that prominence on, fading in from half of it) */
        if ((d = getenv("MMX_TONAL_PROM")) != NULL) prom_on = atof(d);
        dbg = getenv("MMX_DEBUG_TONAL") != NULL;
    }
    if ((alpha <= 0.0 && fixed >= 1.0) || !psy)
        return;
    for (c = 0; c < nch; c++)
        for (b = 0; b < nb; b++)
        {
            double lift = E->bscale[b], fl = psy[c]->flat[b], w, f = 1.0;
            if (dbg && c == 0 && b >= 25 && b < 39 && lift > 1.0)
                fprintf(stderr, "tonal b%u lift %.1f dB flat %.3f\n", b, 10.0 * log10(lift), fl);
            if (prom_on > 0.0)
            {
                double pr = psy[c]->prom[b];
                if (pr <= prom_on * 0.5) continue;
                w = (pr - prom_on * 0.5) / (prom_on * 0.5);  /* 0 at half the prominence, 1 from prom_on on */
            }
            else
            {
                if (fl >= flat_on)
                    continue;
                w = (flat_on - fl) / (flat_on - flat_full);   /* 0 at flat_on, 1 at flat_full and below */
            }
            if (w > 1.0) w = 1.0;
            if (alpha > 0.0 && lift > 1.0) f *= pow(lift, -alpha * w);
            if (fixed < 1.0) f *= pow(fixed, w);
            thr[c][b] = (float)((double)thr[c][b] * f);
        }
}

/* Energy preservation of a quantized long-frame candidate (see "energy
   preservation" above): every band the quantizer zeroed although the model's
   own threshold would have coded it gets its energy back, as a coded
   coefficient in the lows and mids and as an energy-only noise band above the
   crossover. Runs for every candidate of the frame, so the trial bits are the
   bits of the frame that is written. */
static void keep_energy(Enc *E, MMXFrameSyntax *s, float *const *x, const float *const *thr,
                        const float *const *crest, unsigned int n_sources)
{
    MMXCodec *codec = E->codec;
    const MMXBandLayout *L = &codec->bands;
    const KeepParams *kp = &E->keep;
    const double *bscale = E->bscale;
    unsigned int c, b, src;

    memset(E->keep_mark, 0, sizeof(E->keep_mark));
    if (!kp->on || s->block_type == MMX_BT_SHORT)
        return;
    for (c = 0; c < s->channels; c++)
        for (b = 0; b < coded_cutoff(codec); b++)
        {
            unsigned long k0 = L->band_start[b], k1 = L->band_start[b + 1], k, n = k1 - k0;
            double e = 0.0, model, cr, limit;
            int predicted = 0, noise;
            if (!s->band_zero[c][b] || thr[c][b] >= 1e29f)
                continue;
            if (s->channels == 2 && s->band_is[b])
                continue;   /* intensity band: channel 0 carries the mid for both */
            for (src = 0; src < n_sources; src++)
                if (s->gain[src][c][L->eq_band[b]] != MMX_GAIN_OFF)
                    predicted = 1;
            for (k = k0; k < k1; k++)
                e += (double)x[c][k] * x[c][k];
            if (e <= 0.0)
                continue;
            if (predicted)
            {
                /* the prediction carries the band unless it is far below the target */
                double ep = 0.0, et = 0.0;
                if (!kp->pred || s->tns[c].active)
                    continue;   /* TNS filtered x, the prediction is not filtered: no comparable target here */
                for (k = k0; k < k1; k++)
                {
                    double pr = codec->pred[c][k];
                    ep += pr * pr;
                    et += (pr + (double)x[c][k]) * (pr + (double)x[c][k]);
                }
                if (ep > et * kp->pred_rel)
                    continue;
            }
            cr = crest ? (double)crest[c][b] : 1.0;
            model = (double)thr[c][b] * (double)n / (bscale[b] > 0.0 ? bscale[b] : 1.0);   /* the threshold without the rate offset */
            if (kp->max_rel > 0.0)
            {
                double capped = (double)thr[c][b] * (double)n * kp->max_rel;
                if (capped > model) model = capped;   /* protection capped below the lifted threshold */
            }
            limit = model * (keep_was_coded(E, s, c, b) ? kp->hyst_lo : kp->hyst_hi);
            if (e * cr <= limit)
                continue;
            noise = !predicted && b >= kp->noise_band;
            if (noise)
            {
                s->band_zero[c][b] = 0;
                s->band_noise[c][b] = 1;
                s->sf[c][b] = (unsigned char)mmx_pns_level_index(e, n);
                for (k = k0; k < k1; k++) s->q[c][k] = 0;
            }
            else
            {
                unsigned int sf = mmx_frame_keep_band(L, b, x[c], s->q[c], kp->keep_rel, kp->steps);
                int any = 0;
                for (k = k0; k < k1; k++)
                    if (s->q[c][k]) { any = 1; break; }
                if (!any)
                    continue;
                s->band_zero[c][b] = 0;
                s->band_noise[c][b] = 0;
                s->sf[c][b] = (unsigned char)sf;
            }
            E->keep_mark[c][b] = (unsigned char)(noise ? 2 : 1);
        }
}

/* The same for a short frame, per group and short band: the eight 5.8 ms
   transforms have no noise bands and no crest in their quantizer, so a hole
   the offset made is always filled with a coefficient, and the decision is the
   model's own threshold alone (the hysteresis state of the long frames does
   not map onto the groups; a band toggling between two groups of the same
   frame is 5.8 ms long). Measured: with block switching off, so that the long
   rule covers every frame, the toggles of title B 20 s at 96 kbit/s
   fall from 3.50 to 1.30 per second - the short frames were two thirds of what
   was left. */
static void keep_energy_short(Enc *E, MMXFrameSyntax *s, float *const *x,
                              const float (*const *thr)[MMX_SHORT_BANDS], unsigned int n_sources)
{
    MMXCodec *codec = E->codec;
    const MMXBandLayout *Ls = &codec->short_bands;
    const KeepParams *kp = &E->keep;
    const double *sbscale = E->sbscale;
    unsigned int c, g, b, src;

    if (!kp->on || s->block_type != MMX_BT_SHORT)
        return;
    for (c = 0; c < s->channels; c++)
        for (g = 0; g < MMX_SHORT_GROUPS; g++)
            for (b = 0; b < codec->short_cutoff_band && b < MMX_SHORT_BANDS; b++)
            {
                unsigned long k0 = Ls->band_start[b], k1 = Ls->band_start[b + 1], k, n = k1 - k0;
                const float *xg = x[c] + (size_t)g * MMX_SHORT_M;
                double e = 0.0, model;
                int predicted = 0, any = 0;
                unsigned int sf;
                if (!s->band_zero_s[c][g][b] || thr[c][g][b] >= 1e29f)
                    continue;
                for (src = 0; src < n_sources; src++)
                    if (s->gain[src][c][Ls->eq_band[b]] != MMX_GAIN_OFF)
                        predicted = 1;
                for (k = k0; k < k1; k++)
                    e += (double)xg[k] * xg[k];
                if (e <= 0.0)
                    continue;
                if (predicted)
                {
                    const float *pg = codec->pred[c] + (size_t)g * MMX_SHORT_M;
                    double ep = 0.0, et = 0.0;
                    if (!kp->pred)
                        continue;
                    for (k = k0; k < k1; k++)
                    {
                        ep += (double)pg[k] * pg[k];
                        et += ((double)pg[k] + xg[k]) * ((double)pg[k] + xg[k]);
                    }
                    if (ep > et * kp->pred_rel)
                        continue;
                }
                model = (double)thr[c][g][b] * (double)n / (sbscale[b] > 0.0 ? sbscale[b] : 1.0);
                if (kp->max_rel > 0.0)
                {
                    double capped = (double)thr[c][g][b] * (double)n * kp->max_rel;
                    if (capped > model) model = capped;
                }
                if (e <= model)
                    continue;
                sf = mmx_frame_keep_band(Ls, b, xg, s->q[c] + (size_t)g * MMX_SHORT_M, kp->keep_rel, kp->steps);
                for (k = k0; k < k1; k++)
                    if (s->q[c][(size_t)g * MMX_SHORT_M + k]) { any = 1; break; }
                if (!any)
                    continue;
                s->band_zero_s[c][g][b] = 0;
                s->sf_s[c][g][b] = (unsigned char)sf;
            }
}

/* Noise filling of a quantized long-frame candidate: the holes, then the
   fill levels of the regions (a region predicted by a source stays off
   unless MMX_NF_RESID). */
static void nf_apply(Enc *E, MMXFrameSyntax *s, float *const *x, const float *const *thr, const float *const *crest, unsigned int n_sources)
{
    unsigned char skip[MMX_MAX_CH][MMX_EQ_BANDS];
    unsigned int c, e, src;
    if (!E->nf.on || s->block_type == MMX_BT_SHORT)
        return;
    nf_fill_holes(E->codec, s, x, thr, crest, E->bscale, n_sources, &E->nf);
    memset(skip, 0, sizeof(skip));
    if (!E->nf.resid)
        for (src = 0; src < n_sources; src++)
            for (c = 0; c < s->channels; c++)
                for (e = 0; e < MMX_EQ_BANDS; e++)
                    if (s->gain[src][c][e] != MMX_GAIN_OFF)
                        skip[c][e] = 1;
    mmx_frame_noise_fill_levels(&E->codec->bands, s, x, E->nf.first_region, E->nf.gain, skip);
}

/* Everything that follows the quantization of a long-frame candidate: the
   energy preservation rules and, with --nf, the noise filling levels. Every
   candidate goes through it before it is trial-coded. */
/* EXPERIMENT MMX_PEAK_GUARD=<prominence dB> (0 = off): a band that carries a spectral peak at least
   this far above its mean bin power - a sustained partial - never loses that peak. A zero or noise
   band is coded again around its peak (keep_band at the peak's own step); in a coded band a peak the
   dead zone dropped is written as +-1. What the ear hears as chopped synths and strings is exactly
   these peaks coming and going frame by frame (title A 5:30 at 96 kbit/s: 47 % of the frames of a
   partial 8-11 kHz gone, Opus 3 %); a partial 3 dB off is nothing next to one that is missing every
   other frame. MMX_PEAK_GUARD_HZ (default 2000) is the frequency from which the guard applies. */
static unsigned long peak_guard_restored = 0, peak_guard_pinned = 0;
static void peak_guard(Enc *E, MMXFrameSyntax *s, float *const *x)
{
    static double prom = -1.0;
    static unsigned int from_hz = 2000;
    const MMXCodec *codec = E->codec;
    const MMXBandLayout *L = &codec->bands;
    unsigned int c, b, first = 0;
    if (prom < 0.0)
    {
        const char *e = getenv("MMX_PEAK_GUARD"), *h = getenv("MMX_PEAK_GUARD_HZ");
        prom = e ? pow(10.0, atof(e) / 10.0) : 0.0;
        if (h) from_hz = (unsigned int)atoi(h);
    }
    if (prom <= 0.0 || s->block_type == MMX_BT_SHORT)
        return;
    while (first + 1 < L->band_count && L->band_hz[first] < (float)from_hz) first++;
    for (c = 0; c < s->channels; c++)
        for (b = first; b < coded_cutoff(codec); b++)
        {
            unsigned long k0 = L->band_start[b], k1 = L->band_start[b + 1], k, kmax = k0, n = k1 - k0;
            double e = 0.0, pmax = 0.0;
            if (s->band_bwe[c][b] || (s->channels == 2 && s->band_is[b]))
                continue;
            for (k = k0; k < k1; k++)
            {
                double pw = (double)x[c][k] * x[c][k];
                e += pw;
                if (pw > pmax) { pmax = pw; kmax = k; }
            }
            if (e <= 0.0 || pmax < prom * e / (double)n || pmax < 4.0 * (double)L->abs_thr[b])
                continue;            /* no prominent peak, or one the ear cannot hear anyway */
            if (s->band_noise[c][b] || s->band_zero[c][b])
            {
                unsigned int sf;
                int any = 0;
                for (k = k0; k < k1; k++) s->q[c][k] = 0;
                sf = mmx_frame_keep_band(L, b, x[c], s->q[c], E->keep.keep_rel, E->keep.steps);
                for (k = k0; k < k1; k++)
                    if (s->q[c][k]) { any = 1; break; }
                if (!any)
                    continue;
                s->band_zero[c][b] = 0;
                s->band_noise[c][b] = 0;
                s->sf[c][b] = (unsigned char)sf;
                E->keep_mark[c][b] = 1;
                peak_guard_restored++;
            }
            else
            {
                /* MMX_PEAK_STEP=1: the band's step was chosen for its noise floor; a peak coded as 0 or as
                   a single level at a coarse step reconstructs at the wrong amplitude and jitters from
                   frame to frame. Re-quantise the band at the step nearest the peak (the same rule that
                   restores zero bands), so the peak carries its own amplitude on the 1.5 dB grid. */
                static int peak_step = -1;
                double step = mmx_sf_step(s->sf[c][b]), pk = sqrt(pmax);
                if (peak_step < 0) { const char *e = getenv("MMX_PEAK_STEP"); peak_step = e ? atoi(e) : 0; }
                if (peak_step && pk < 1.5 * step)
                {
                    unsigned int sf;
                    for (k = k0; k < k1; k++) s->q[c][k] = 0;
                    sf = mmx_frame_keep_band(L, b, x[c], s->q[c], 0.5, 0);
                    s->sf[c][b] = (unsigned char)sf;
                    peak_guard_pinned++;
                }
                else if (s->q[c][kmax] == 0)
                {
                    s->q[c][kmax] = x[c][kmax] < 0.0f ? -1 : 1;
                    peak_guard_pinned++;
                }
            }
        }
}

/* EXPERIMENT MMX_ENERGY_MATCH=<max dB> (0 = off): a coded band keeps its energy. The dead zone throws
   away the small coefficients of a band and what is left carries less energy than the original; a
   partial that shares its energy between two bins loses the smaller one in one frame and not in the
   next. After the quantizer the band's step is raised (never lowered) so that the coded coefficients
   carry the original band energy, at most by the given dB. Costs no bits - only the step index
   changes - and is what a gain-shape quantizer (Opus) does by construction. */
static unsigned long energy_match_bands = 0;
static void energy_match(Enc *E, MMXFrameSyntax *s, float *const *x)
{
    static double cap = -1.0;
    const MMXCodec *codec = E->codec;
    const MMXBandLayout *L = &codec->bands;
    unsigned int c, b;
    if (cap < 0.0) { const char *e = getenv("MMX_ENERGY_MATCH"); cap = e ? pow(10.0, atof(e) / 10.0) : 0.0; }
    if (cap <= 1.0 || s->block_type == MMX_BT_SHORT)
        return;
    for (c = 0; c < s->channels; c++)
        for (b = 0; b < coded_cutoff(codec); b++)
        {
            unsigned long k0 = L->band_start[b], k1 = L->band_start[b + 1], k;
            double e = 0.0, er = 0.0, step, ratio, v;
            int dsf;
            if (s->band_zero[c][b] || s->band_noise[c][b] || s->band_bwe[c][b] || (s->channels == 2 && s->band_is[b]))
                continue;
            step = mmx_sf_step(s->sf[c][b]);
            for (k = k0; k < k1; k++)
            {
                e += (double)x[c][k] * x[c][k];
                er += (double)s->q[c][k] * s->q[c][k];
            }
            er *= step * step;
            if (er <= 0.0 || e <= er)
                continue;
            ratio = e / er;
            if (ratio > cap) ratio = cap;
            dsf = (int)floor(2.0 * log(ratio) / log(2.0) + 0.5);   /* sf grid: 1.5 dB per index in amplitude */
            if (dsf <= 0)
                continue;
            v = (double)s->sf[c][b] + dsf;
            if (v > MMX_SF_COUNT - 1) v = MMX_SF_COUNT - 1;
            s->sf[c][b] = (unsigned char)v;
            energy_match_bands++;
        }
}

/* EPB: the energy index of every band of the frame's own signal in the coded domain (M/S if the frame is
   M/S), from the original, never from the residual: the decoder normalises the finished band to it. */
static void epb_set_energies(const MMXCodec *codec, MMXFrameSyntax *s, float *const *t_lr, int ms)
{
    const MMXBandLayout *L = &codec->bands;
    float buf[2][MMX_HOP];
    unsigned int c, b;
    if (!L->lowrate || s->block_type == MMX_BT_SHORT)
        return;
    /* from the untouched L/R target (apply_tns filters the quantiser's copy in place), M/S like the frame */
    for (c = 0; c < s->channels; c++) memcpy(buf[c], t_lr[c], sizeof(float) * MMX_HOP);
    if (ms && s->channels == 2) mmx_codec_lr_to_ms(buf[0], buf[1], MMX_HOP);
    for (c = 0; c < s->channels; c++)
        for (b = 0; b < L->band_count; b++)
        {
            unsigned long k0 = L->band_start[b], k1 = L->band_start[b + 1], k;
            double e = 0.0;
            for (k = k0; k < k1; k++) e += (double)buf[c][k] * buf[c][k];
            s->nrg[c][b] = (unsigned char)(e > 0.0 ? mmx_epb_level(e, k1 - k0) : 0);
        }
}

static void post_quantize(Enc *E, MMXFrameSyntax *s, float *const *x, const float *const *thr,
                          const float *const *crest, unsigned int n_sources)
{
    keep_energy(E, s, x, thr, crest, n_sources);
    nf_apply(E, s, x, thr, crest, n_sources);
    peak_guard(E, s, x);
    energy_match(E, s, x);
}

/* The hysteresis state for the next frame: which bands of the frame that was
   chosen carry energy. Short frames do not touch the long-frame bands and
   leave the state as it is. */
static void keep_commit(Enc *E, const MMXFrameSyntax *s, unsigned int bt)
{
    unsigned int c, b;
    if (bt == MMX_BT_SHORT)
        return;
    for (c = 0; c < s->channels; c++)
        for (b = 0; b < MMX_MAX_BANDS; b++)
            E->keep_prev[c][b] = (unsigned char)(!s->band_zero[c][b] || (s->channels == 2 && s->band_is[b]));
    E->keep_prev_ms = s->stereo_ms;
    if (bt != MMX_BT_SHORT)
        for (c = 0; c < s->channels; c++)
            for (b = 0; b < MMX_MAX_BANDS; b++)
                E->pns_prev[c][b] = s->band_noise[c][b];
}

/* Noise substitution with the closed-loop cost check, then the trial coding
   of the candidate; returns its bits. A substituted band whose flag and level
   do not cost less than its coefficients (booked per band from the arithmetic
   code, with and without the substitution) gets its coefficients back. The
   reverts are counted for the final pass only (`trial` = 0). */
static double pns_and_bits(Enc *E, const BlockWriter *w, MMXFrameSyntax *s, float *const *target, unsigned long f, unsigned int n_sources,
                           int trial)
{
    MMXCodec *codec = E->codec;
    const MMXBandLayout *L = &codec->bands, *Ls = &codec->short_bands;
    unsigned int c, b, any = 0, reverted = 0;
    double bits;
    bwe_apply(codec, s, target);
    if (!E->pns.first_band || E->pns.first_band >= coded_cutoff(codec))
        return trial_bits(w, L, Ls, s, n_sources);
    mmx_frame_syntax_copy(&E->coded, s);
    pns_substitute(codec, s, target, E->analysis, f, &E->pns, E->pns_prev);
    for (c = 0; c < s->channels; c++)
        for (b = E->pns.first_band; b < coded_cutoff(codec); b++)
            any |= s->band_noise[c][b];
    if (!any)
        return trial_bits(w, L, Ls, s, n_sources);
    {
        double noise_bits[MMX_MAX_CH][MMX_MAX_BANDS], coded_bits[MMX_MAX_CH][MMX_MAX_BANDS];
        mmx_frame_encode_book_bands(coded_bits);
        trial_bits(w, L, Ls, &E->coded, n_sources);
        mmx_frame_encode_book_bands(noise_bits);
        bits = trial_bits(w, L, Ls, s, n_sources);
        for (c = 0; c < s->channels; c++)
            for (b = E->pns.first_band; b < coded_cutoff(codec); b++)
                if (s->band_noise[c][b] && !E->coded.band_noise[c][b] && noise_bits[c][b] >= coded_bits[c][b])
                {
                    unsigned long k;
                    s->band_noise[c][b] = 0;
                    s->band_zero[c][b] = E->coded.band_zero[c][b];
                    s->sf[c][b] = E->coded.sf[c][b];
                    for (k = L->band_start[b]; k < L->band_start[b + 1]; k++) s->q[c][k] = E->coded.q[c][k];
                    reverted++;
                }
        if (reverted)
        {
            if (!trial) E->pns_reverted += reverted;
            bits = trial_bits(w, L, Ls, s, n_sources);
        }
    }
    return bits;
}

/* M/S or L/R by the bit estimate (experiment MMX_TRIAL_MS=0, the old rule):
   the estimate of the residual when there are sources, of the target otherwise. */
static int long_ms_estimate(Enc *E, unsigned long f, unsigned int n_sources, const MMXFramePsy *const *psy, const float *thr_ms, unsigned int bt)
{
    MMXCodec *codec = E->codec;
    MMXFramePlan *plan = &E->analysis->plan[f];
    float *const *t_cod = E->t_cod, *const *resid = E->resid;
    double est_lr = 0.0, est_ms = 0.0;
    unsigned int c, ms;
    unsigned long k;
    for (ms = 0; ms < 2; ms++)
    {
        double est = 0.0;
        for (c = 0; c < 2; c++) memcpy(t_cod[c], E->t_lr[c], sizeof(float) * MMX_HOP);
        if (ms) mmx_codec_lr_to_ms(t_cod[0], t_cod[1], MMX_HOP);
        if (n_sources)
        {
            E->alt.block_type = (unsigned char)bt;
            E->alt.stereo_ms = (int)ms;
            mmx_codec_source_coefs_bt(codec, &E->decoded, n_sources, plan->src_start, (int)ms, bt);
            mmx_codec_fit_gains(codec, &E->alt, n_sources, t_cod);
            mmx_codec_predict(codec, &E->alt, n_sources);
            for (c = 0; c < 2; c++)
                for (k = 0; k < MMX_HOP; k++) resid[c][k] = t_cod[c][k] - codec->pred[c][k];
        }
        for (c = 0; c < 2; c++)
            est += mmx_frame_estimate_bits(&codec->bands, n_sources ? resid[c] : t_cod[c], ms ? thr_ms : psy[c]->thr, coded_cutoff(codec));
        if (ms) est_ms = est; else est_lr = est;
    }
    return est_ms < est_lr;
}

static int short_ms_estimate(Enc *E, const float (*const *sthr)[MMX_SHORT_BANDS], const float (*sthr_ms)[MMX_SHORT_BANDS])
{
    MMXCodec *codec = E->codec;
    float *const *t_cod = E->t_cod;
    double est_lr, est_ms;
    unsigned int c;
    for (c = 0; c < 2; c++) memcpy(t_cod[c], E->t_lr[c], sizeof(float) * MMX_HOP);
    est_lr = mmx_frame_estimate_bits_short(&codec->short_bands, t_cod[0], sthr[0], codec->short_cutoff_band) +
             mmx_frame_estimate_bits_short(&codec->short_bands, t_cod[1], sthr[1], codec->short_cutoff_band);
    mmx_codec_lr_to_ms(t_cod[0], t_cod[1], MMX_HOP);
    est_ms = mmx_frame_estimate_bits_short(&codec->short_bands, t_cod[0], sthr_ms, codec->short_cutoff_band) +
             mmx_frame_estimate_bits_short(&codec->short_bands, t_cod[1], sthr_ms, codec->short_cutoff_band);
    return est_ms < est_lr;
}

/* Tracker / future: the residual thresholds of a referenced long frame are
   relaxed by relax, but the sub-window rule (pre-echo protection) only by
   relax_sub: the difference between two takes may be generous, a ghost note
   before a hit may not. */
static void tracker_relax_long(Enc *E, const MMXFramePsy *const *psy, const float *const *thr, const float *const *thr_lr, int ms,
                               float (*out)[MMX_MAX_BANDS], float (*out_lr)[MMX_MAX_BANDS])
{
    unsigned int c, b, w, nch = E->audio->channels;
    for (c = 0; c < nch; c++)
        for (b = 0; b < MMX_MAX_BANDS; b++)
        {
            float t = thr[c][b], cap = 1e30f;
            if (t >= 1e29f) { out[c][b] = t; continue; }
            for (w = 0; w < 5; w++)
            {
                float sw = psy[c]->sub_thr[w][b];
                if (ms && nch == 2 && psy[1]->sub_thr[w][b] < sw) sw = psy[1]->sub_thr[w][b];
                if (ms && nch == 2 && psy[0]->sub_thr[w][b] < sw) sw = psy[0]->sub_thr[w][b];
                if (sw < cap) cap = sw;
            }
            cap = cap < 1e29f ? (float)(cap * E->relax_sub) : cap;
            t = (float)(t * E->relax);
            out[c][b] = t < cap ? t : cap;
        }
    for (c = 0; c < 2; c++)
    {
        const MMXFramePsy *pc = psy[nch > 1 ? c : 0];
        for (b = 0; b < MMX_MAX_BANDS; b++)
        {
            float t = thr_lr[c][b], cap = 1e30f;
            if (t >= 1e29f) { out_lr[c][b] = t; continue; }
            for (w = 0; w < 5; w++) if (pc->sub_thr[w][b] < cap) cap = pc->sub_thr[w][b];
            cap = cap < 1e29f ? (float)(cap * E->relax_sub) : cap;
            t = (float)(t * E->relax);
            out_lr[c][b] = t < cap ? t : cap;
        }
    }
}

/* ---------- intensity stereo ---------- */
#define IS_RHO_MIN 0.5       /* normalized L/R cross-correlation of the band's MDCT coefficients from which the difference
                                may be dropped: the ear localizes the highs by level, the width of a coherent source is kept.
                                title G excerpt, 96 kbit/s, 5 kHz, slack 10 dB: rho 0.8 changes nothing (over 50.7 %,
                                coherence error from 4 kHz up 0.159 like the file without), 0.6 -> 50.2 % / 0.18, 0.4 ->
                                46.7 % / 0.25, 0.3 -> 42.3 % / 0.30, 0.2 -> 40.2 % / 0.33, every band -> 35.5 % / 0.41
                                (Opus 96 on the whole track: over 36.6 %, coherence error 0.215); 0.5 keeps the image
                                deviation at Opus' level */
#define IS_BIAS 1.0          /* intensity wins when its bits times this are below the two bands' bits */
#define IS_SLACK_DB 10.0     /* the dropped difference of a band may exceed the channel's masking threshold by this much:
                                the model counts it as error by construction, the guard keeps the worst cells (tonal bands,
                                pre-echo-tightened thresholds) out: without it the worst band of the excerpt is 30.5 dB
                                (a cut before silence), with 10 dB 12.3 dB at the same over %; 0 dB = only what M/S
                                zeroing does anyway (nothing). MMX_IS_SLACK_DB moves it, a large value = coherence only */

/* Candidates of a long stereo frame: per band from the first intensity band
   on, the L/R energies and their normalized cross-correlation on the MDCT
   coefficients; a band is a candidate when the channels are coherent enough
   that the ear does not localize the dropped difference (rho >= rho_min;
   anti-phase content, rho < 0, never), its position index is the quantized
   energy ratio and its mid target (L + R) / 2 is scaled so that the decoded
   band keeps the energy of both channels (gl^2 + gr^2 = 2: mid energy =
   (EL + ER) / 2, the ratio from the position). The dropped difference per
   channel is measured here and checked against the thresholds in is_target.
   MMX_IS_RHO and MMX_IS_BIAS move the two knobs for listening experiments. */
static void is_prepare(Enc *E, float *const *t_lr, unsigned int cutoff)
{
    MMXIntensityPlan *p = &E->is_plan;
    const MMXBandLayout *L = &E->codec->bands;
    unsigned int b, any = 0;
    memset(p->cand, 0, sizeof(p->cand));
    p->first_band = 0;
    p->bias = E->is_bias;
    p->x = E->x_is;
    p->n_range = p->n_rho = p->n_slack = 0;
    if (!E->is_first_band)
        return;
    for (b = E->is_first_band; b < cutoff && b < L->band_count; b++)
    {
        unsigned long k0 = L->band_start[b], k1 = L->band_start[b + 1], k;
        double el = 0.0, er = 0.0, c = 0.0, em, s;
        float gl, gr;
        for (k = k0; k < k1; k++)
        {
            double l = t_lr[0][k], r = t_lr[1][k];
            el += l * l;
            er += r * r;
            c += l * r;
        }
        if (el <= 0.0 || er <= 0.0)
            continue;
        p->n_range++;
        if (c / sqrt(el * er) < E->is_rho_min)
        {
            p->n_rho++;
            continue;
        }
        em = 0.25 * (el + er + 2.0 * c);   /* energy of (L + R) / 2 */
        if (em <= 0.0)
            continue;
        s = sqrt(0.5 * (el + er) / em);
        p->cand[b] = 1;
        p->pos[b] = mmx_is_position(el, er);
        mmx_is_gains(p->pos[b], &gl, &gr);
        p->err[0][b] = p->err[1][b] = 0.0f;
        for (k = k0; k < k1; k++)
        {
            double m = s * 0.5 * ((double)t_lr[0][k] + t_lr[1][k]), dl = t_lr[0][k] - gl * m, dr = t_lr[1][k] - gr * m;
            E->t_is[k] = (float)m;
            p->err[0][b] += (float)(dl * dl);
            p->err[1][b] += (float)(dr * dr);
        }
        any = 1;
    }
    if (any)
        p->first_band = E->is_first_band;
}

/* The plan of one candidate coding: the allowed noise of every candidate's
   mid band (the L/R allowances through the gains, the noise of the mid lands
   in both channels scaled) and its target, the mid target less the mid of
   the two channels' predictions (pred[0] in M/S, their mean in L/R) when the
   prediction is on. Returns the plan, or NULL when the frame has none (or
   TNS is active: the filter would spread into the uncoded bands of channel 1). */
static const MMXIntensityPlan *is_target(Enc *E, const MMXFrameSyntax *syn, const float *const *thr_lr, int ms, int use_pred)
{
    MMXIntensityPlan *p = &E->is_plan;
    const MMXBandLayout *L = &E->codec->bands;
    float *const *pred = E->codec->pred;
    unsigned int b;
    if (!p->first_band || syn->tns[0].active || syn->tns[1].active)
        return NULL;
    p->n_slack = 0;
    for (b = p->first_band; b < L->band_count; b++)
    {
        unsigned long k0 = L->band_start[b], k1 = L->band_start[b + 1], k;
        float gl, gr, t;
        if (!p->cand[b])
            continue;
        if (thr_lr[0][b] >= 1e29f || thr_lr[1][b] >= 1e29f ||
            p->err[0][b] > thr_lr[0][b] * (float)(k1 - k0) * E->is_slack || p->err[1][b] > thr_lr[1][b] * (float)(k1 - k0) * E->is_slack)
        {
            p->cand[b] = 0;   /* no threshold, or the dropped difference would be heard by the model beyond the slack */
            p->n_slack++;
            continue;
        }
        mmx_is_gains(p->pos[b], &gl, &gr);
        t = thr_lr[0][b] / (gl * gl);
        if (thr_lr[1][b] / (gr * gr) < t) t = thr_lr[1][b] / (gr * gr);
        p->thr[b] = t;
        for (k = k0; k < k1; k++)
            E->x_is[k] = E->t_is[k] - (use_pred ? (ms ? pred[0][k] : 0.5f * (pred[0][k] + pred[1][k])) : 0.0f);
    }
    return p;
}

/* Measures the structured error of the frame that was just reconstructed
   (codec->rec, codec->pred, the chosen syntax). `pure` marks the EQ bands the
   tracker plays without residual: they are counted, never guarded - playing
   them is the mode's promise. */
static void echo_frame_measure(Enc *E, unsigned long f, const MMXFrameSyntax *syn, unsigned int n_sources,
                               int use_ms, const MMXFramePsy *const *psy_base, const PureCounts *pure,
                               const float (*sthr)[MMX_SHORT_GROUPS][MMX_SHORT_BANDS], double clip_scale,
                               EchoFrame *fs)
{
    MMXCodec *codec = E->codec;
    const MMXAnalysis *a = E->analysis;
    unsigned int nch = E->audio->channels, c, b, g;
    float mthr[MMX_MAX_CH][MMX_MAX_BANDS], mthr_ms[MMX_MAX_BANDS];
    const float *mt[MMX_MAX_CH];

    memset(fs, 0, sizeof(*fs));
    fs->worst_db = -200.0;
    if (!E->echo.measure || !n_sources)
        return;
    fs->measured = 1;
    for (c = 0; c < nch; c++)
    {
        memcpy(E->rec_cod[c], codec->rec[c], sizeof(float) * MMX_HOP);
        memcpy(E->tgt_cod[c], E->t_lr[c], sizeof(float) * MMX_HOP);
    }
    if (use_ms && nch == 2)
    {
        mmx_codec_lr_to_ms(E->rec_cod[0], E->rec_cod[1], MMX_HOP);
        mmx_codec_lr_to_ms(E->tgt_cod[0], E->tgt_cod[1], MMX_HOP);
    }
    if (syn->block_type == MMX_BT_SHORT)
    {
        /* the group thresholds carry the same (tilted) factor per band */
        for (c = 0; c < nch; c++)
            for (g = 0; g < MMX_SHORT_GROUPS; g++)
                for (b = 0; b < codec->short_cutoff_band; b++)
                {
                    unsigned long k0 = (unsigned long)g * MMX_SHORT_M + codec->short_bands.band_start[b];
                    unsigned long k1 = (unsigned long)g * MMX_SHORT_M + codec->short_bands.band_start[b + 1];
                    double t = sthr[c][g][b], allowed, db, sc = E->sbscale[b] * clip_scale;
                    if (use_ms && nch == 2)
                    {
                        t = sthr[0][g][b] < sthr[1][g][b] ? sthr[0][g][b] : sthr[1][g][b];
                        t = t >= 1e29 ? t : t * 0.5;
                    }
                    if (t >= 1e29)
                        continue;
                    allowed = t / sc * (double)(k1 - k0);
                    db = echo_band_db(codec->pred[c], E->rec_cod[c], E->tgt_cod[c], k0, k1, allowed);
                    if (db <= -199.0)
                        continue;
                    fs->bands++;
                    if (syn->band_zero_s[c][g][b]) fs->zero_bands++;
                    if (db > 0.0) { fs->over_thr++; fs->sum_over_db += db; }
                    if (db > -E->echo.margin_db && db + E->echo.margin_db > fs->over_db[c][b])
                    {
                        if (fs->over_db[c][b] <= 0.0) fs->over_guard++;
                        fs->over_db[c][b] = db + E->echo.margin_db;
                    }
                    if (db > fs->worst_db) fs->worst_db = db;
                }
        return;
    }
    for (c = 0; c < nch; c++)
    {
        const float *base = mmx_analysis_model_thr(a, f, c);
        for (b = 0; b < MMX_MAX_BANDS; b++)
        {
            /* START/STOP frames derive their thresholds from the unscaled frame/sub-window
               model and apply the rate offset themselves: take it back out again */
            float t = syn->block_type == MMX_BT_LONG ? base[b] : psy_base[c]->thr[b];
            if (syn->block_type != MMX_BT_LONG && t < 1e29f) t = (float)(t / (E->bscale[b] > 0.0 ? E->bscale[b] : 1.0));
            mthr[c][b] = t;
        }
        mt[c] = mthr[c];
    }
    if (use_ms && nch == 2)
    {
        for (b = 0; b < MMX_MAX_BANDS; b++)
        {
            float t = mthr[0][b] < mthr[1][b] ? mthr[0][b] : mthr[1][b];
            mthr_ms[b] = t >= 1e29f ? t : t * 0.5f;
        }
        mt[0] = mthr_ms;
        mt[1] = mthr_ms;
    }
    for (c = 0; c < nch; c++)
        for (b = 0; b < codec->cutoff_band; b++)
        {
            double allowed, db;
            unsigned int e = codec->bands.eq_band[b];
            if (mt[c][b] >= 1e29f || syn->band_noise[c][b] || syn->band_bwe[c][b])
                continue;   /* not coded, a noise band or a replicated band: the prediction is not played there */
            allowed = (double)mt[c][b] * (double)(codec->bands.band_start[b + 1] - codec->bands.band_start[b]);
            db = echo_band_db(codec->pred[c], E->rec_cod[c], E->tgt_cod[c], codec->bands.band_start[b],
                              codec->bands.band_start[b + 1], allowed);
            if (db <= -199.0)
                continue;   /* no prediction in this band */
            fs->bands++;
            if (syn->band_zero[c][b]) fs->zero_bands++;
            if (pure && pure->eq_pure[c][e])
            {
                fs->pure_bands++;
                if (db > 0.0) fs->pure_over++;
                continue;   /* the tracker's promise: this band plays from the source */
            }
            if (db > 0.0) { fs->over_thr++; fs->sum_over_db += db; }
            if (db > -E->echo.margin_db)
            {
                fs->over_guard++;
                fs->over_db[c][b] = db + E->echo.margin_db;
            }
            if (db > fs->worst_db) fs->worst_db = db;
        }
}

/* Tightens the residual thresholds of the bands that leaked, by the measured
   violation plus one step (bounded, as the clip guard's steps). */
static int echo_tighten(const EchoFrame *fs, float (*scale)[MMX_MAX_BANDS], unsigned int nch)
{
    unsigned int c, b;
    int any = 0;
    for (c = 0; c < nch; c++)
        for (b = 0; b < MMX_MAX_BANDS; b++)
            if (fs->over_db[c][b] > 0.0)
            {
                double d = fs->over_db[c][b] + ECHO_STEP_DB;
                if (d > ECHO_STEP_MAX_DB) d = ECHO_STEP_MAX_DB;
                scale[c][b] = (float)(scale[c][b] * pow(10.0, -d / 10.0));
                if (scale[c][b] < ECHO_SCALE_MIN) scale[c][b] = ECHO_SCALE_MIN;
                any = 1;
            }
    return any;
}

/* Books the final measurement of a frame (retries overwrite, they do not add). */
static void echo_book(EchoGuard *G, const EchoFrame *fs, long long start, unsigned long sample_rate)
{
    double t;
    if (!fs->measured)
        return;
    G->frames++;
    G->bands += fs->bands;
    G->pure_bands += fs->pure_bands;
    G->pure_over += fs->pure_over;
    G->zero_bands += fs->zero_bands;
    G->over_thr += fs->over_thr;
    G->over_guard += fs->over_guard;
    G->sum_over_db += fs->sum_over_db;
    if (fs->over_thr) G->frames_over++;
    t = (double)(start + MMX_HOP / 2) / (double)sample_rate;
    if (fs->worst_db > G->worst_db) { G->worst_db = fs->worst_db; G->worst_at = t; }
    if (G->sec_worst && fs->over_thr)
    {
        unsigned long s = t < 0.0 ? 0 : (unsigned long)t;
        if (s < G->seconds)
        {
            if (fs->worst_db > G->sec_worst[s]) G->sec_worst[s] = fs->worst_db;
            G->sec_count[s] += fs->over_thr;
        }
    }
}

/* One line for the encode log, and with MMX_DEBUG_ECHO the seconds that leak,
   in time order (what the ear should be pointed at). */
static void echo_report(const EchoGuard *G)
{
    char line[768];
    if (!G->frames)
        return;
    snprintf(line, sizeof(line),
             "Structured error (prediction leaking into the reconstruction): %lu of %lu referenced channel bands over the model threshold (%.2f %%) in %lu of %lu frames, mean %+.1f dB, worst %+.1f dB at %.2f s; %lu bands still over the guard, %lu frames re-quantized (%lu retries), %lu frames left over; %lu bands with zero residual, %lu pure bands (%lu of them over)",
             G->over_thr, G->bands, G->bands ? 100.0 * (double)G->over_thr / (double)G->bands : 0.0,
             G->frames_over, G->frames,
             G->over_thr ? G->sum_over_db / (double)G->over_thr : 0.0, G->worst_db, G->worst_at,
             G->over_guard, G->frames_guarded, G->retries, G->frames_left, G->zero_bands, G->pure_bands, G->pure_over);
    mmx_info("%s", line);
    if (G->debug)
        fprintf(stderr, "%s\n", line);   /* the yardstick of this run, also under --quiet */
    if (G->debug >= 2 && G->sec_worst)
    {
        unsigned long s;
        fprintf(stderr, "structured error per second (leak above the model threshold, worst band / leaking bands):\n");
        for (s = 0; s < G->seconds; s++)
            if (G->sec_count[s])
                fprintf(stderr, "  %3lu:%02lu  worst %+6.1f dB  %6lu bands\n", s / 60, s % 60, G->sec_worst[s], G->sec_count[s]);
    }
}

/* Codes frame f with block type E->bt[f] into writer w: the block boundary,
   all decisions of the frame by trial coding, the entropy coding, the
   reconstruction into the decoded signal and the reference graph record.
   `trial` = 1 codes into a scratch writer (a new block only resets its
   contexts) and skips the baseline coder, the clip guard, the barrier and
   the statistics. In the final pass the decisions are taken again under
   tightened thresholds while the clip guard hears the clamp of the frame's
   reconstruction (ClipGuard): a retry takes back the frame's overlap-add
   and its counters, nothing else, and only the last attempt is coded.
   Returns the exact bits of the frame in bits_out. */
static int frame_code(Enc *E, unsigned long f, BlockWriter *w, MMXStatistics *st, int trial, double *bits_out)
{
    const MMXAudioBuffer *audio = E->audio;
    const MMXEncoderParams *params = E->params;
    MMXCodec *codec = E->codec;
    MMXAnalysis *analysis = E->analysis;
    MMXFramePlan *plan = &analysis->plan[f];
    MMXFrameSyntax *syn = &E->syn, *alt = &E->alt, *best = &E->best;
    float *const *t_lr = E->t_lr, *const *t_cod = E->t_cod, *const *resid = E->resid;
    float thr_ms[MMX_MAX_BANDS], thr_buf[MMX_MAX_CH][MMX_MAX_BANDS], thr_lr_buf[2][MMX_MAX_BANDS];
    float reuse_thr[MMX_MAX_CH][MMX_MAX_BANDS], reuse_thr_lr[2][MMX_MAX_BANDS];
    float sthr[MMX_MAX_CH][MMX_SHORT_GROUPS][MMX_SHORT_BANDS], sthr_ms[MMX_SHORT_GROUPS][MMX_SHORT_BANDS];
    float sthr_q[MMX_MAX_CH][MMX_SHORT_GROUPS][MMX_SHORT_BANDS];
    float scap[MMX_MAX_CH][MMX_SHORT_GROUPS][MMX_SHORT_BANDS];   /* the sub-window (pre-echo) part of the group thresholds */
    const float *thr[MMX_MAX_CH], *thr_lr[2], *crest[MMX_MAX_CH], *thr_q[MMX_MAX_CH], *thr_lr_q[2];
    const MMXFramePsy *psy[MMX_MAX_CH], *psy_base[MMX_MAX_CH];
    unsigned char bt = E->bt[f];
    long long start = (long long)f * MMX_HOP - MMX_HOP;
    unsigned int n_sources = plan->n_sources, c, ms, ms_lo = 0, ms_hi = 0, nch = audio->channels, qmode;
    unsigned long k, nf = analysis->frame_count;
    int depth, use_ms = 0, best_pred = 0, transient, guard_on, attempt, clip_left = 0;
    int clip_attempt = 0, echo_attempt = 0, echo_on = 0, echo_left = 0, retry;
    double best_bits = 1e300, bits, pos, clip_scale = 1.0, clip_peak = 0.0;
    unsigned long clip_over = 0, clip_over_raw = 0, clip_quiet = 0, pns_reverted_before = 0;
    float echo_scale[MMX_MAX_CH][MMX_MAX_BANDS], echo_thr[MMX_MAX_CH][MMX_MAX_BANDS];
    EchoFrame efs;
    MMXStatistics st_before;
    PureCounts pure, best_pure;

    memset(&pure, 0, sizeof(pure));
    memset(&best_pure, 0, sizeof(best_pure));
    memset(&st_before, 0, sizeof(st_before));
    depth = mmx_refgraph_check(&E->graph, start, n_sources, plan->src_start);
    if (depth < 0)
    {
        n_sources = 0;
        plan->n_sources = 0;
        depth = 0;
    }
    plan->depth = (unsigned char)depth;

    if (!w->open || f == 0 || !same_lineage(&analysis->plan[f - 1], plan))
    {
        if (trial)
            mmx_contexts_init(&w->ctx);   /* a block starts here: fresh contexts (its table row costs the same for every variant) */
        else
        {
            if (block_end(w, E->out, st) != 0)
            {
                mmx_error("Block %lu could not be finished (frame %lu)", E->out->block_count, f);
                return -1;
            }
            block_begin(w, f, plan);
        }
    }
    if (!trial)
        baseline_segment(&E->base, st, f, 0);
    transient = analysis->transient[f] || (f + 1 < nf && analysis->transient[f + 1]);
    qmode = n_sources ? 1 : 0;   /* the coder's context set the frame is coded with (audio / residual); the quantizer's RDO reads it */
    syn->block_type = bt;
    alt->block_type = bt;

    /* clip guard (final pass only): the decoded window before the frame's
       overlap-add and the counters before the frame, taken back for a retry */
    echo_on = !trial && E->echo.on && E->echo.steps && n_sources;
    guard_on = !trial && (E->guard.steps || echo_on);
    memset(&efs, 0, sizeof(efs));
    for (c = 0; c < MMX_MAX_CH; c++)
        for (k = 0; k < MMX_MAX_BANDS; k++) echo_scale[c][k] = 1.0f;
    if (guard_on)
    {
        window_save(E->clip_save, &E->decoded, start, 0);
        st_before = *st;
        pns_reverted_before = E->pns_reverted;
    }
    for (attempt = 0;; attempt++)
    {
        best_bits = 1e300;
        use_ms = 0;
        best_pred = 0;
        memset(&best_pure, 0, sizeof(best_pure));
        ms_lo = 0;
        ms_hi = nch == 2 ? 1 : 0;
        for (c = 0; c < nch; c++)
        {
            /* targets: cached long-window coefficients, or the window shape of the block type */
            if (bt == MMX_BT_LONG)
                memcpy(t_lr[c], mmx_analysis_coefs(analysis, f, c), sizeof(float) * MMX_HOP);
            else
                mmx_codec_window_mdct_bt(codec, audio, start, c, bt, t_lr[c]);
            psy[c] = mmx_analysis_psy(analysis, f, c);
            if (bt == MMX_BT_START || bt == MMX_BT_STOP)
            {
                start_stop_thresholds(analysis, f, c, bt, codec->bands.band_count, E->bscale, &E->psy_bt[c]);
                psy[c] = &E->psy_bt[c];
            }
            psy_base[c] = psy[c];
            if (clip_scale < 1.0)
            {
                /* a retry of the clip guard: the model's thresholds tightened by the steps so far */
                E->psy_clip[c] = *psy[c];
                psy_scale(&E->psy_clip[c], &codec->bands, codec->cutoff_band, clip_scale, E->guard.top);
                psy[c] = &E->psy_clip[c];
            }
            crest[c] = psy[c]->crest;
        }
        thr_lr[0] = psy[0]->thr;
        thr_lr[1] = psy[nch > 1 ? 1 : 0]->thr;
        if (bt != MMX_BT_SHORT && nch == 2)
            is_prepare(E, t_lr, coded_cutoff(codec));
        else
            E->is_plan.first_band = 0;

        if (bt == MMX_BT_SHORT)
        {
            /* ---------- short frame: eight 5.8 ms transforms, thresholds per group ---------- */
            const float (*sthr_p[MMX_MAX_CH])[MMX_SHORT_BANDS];
            const float (*sthr_lr_p[2])[MMX_SHORT_BANDS];
            const float (*sq[MMX_MAX_CH])[MMX_SHORT_BANDS];
            unsigned int g, b;
            short_thresholds(codec, &E->short_spec, audio, start, codec->quality, psy, E->sbscale, clip_scale, sthr, scap);
            sthr_lr_p[0] = sthr[0];
            sthr_lr_p[1] = sthr[nch > 1 ? 1 : 0];
            for (c = 0; c < nch; c++) sq[c] = sthr_q[c];
            if (nch == 2)
            {
                for (g = 0; g < MMX_SHORT_GROUPS; g++)
                    for (b = 0; b < MMX_SHORT_BANDS; b++)
                    {
                        float t = sthr[0][g][b] < sthr[1][g][b] ? sthr[0][g][b] : sthr[1][g][b];
                        sthr_ms[g][b] = t >= 1e29f ? t : t * 0.5f;
                    }
                if (!E->trial_ms)
                    ms_lo = ms_hi = short_ms_estimate(E, sthr_lr_p, sthr_ms) ? 1 : 0;
            }
            if (!trial && attempt == 0)
            {
                /* baseline (plain audio, same block type, never written), once per frame */
                int bms = nch == 2 ? short_ms_estimate(E, sthr_lr_p, sthr_ms) : 0;
                for (c = 0; c < nch; c++)
                {
                    memcpy(t_cod[c], t_lr[c], sizeof(float) * MMX_HOP);
                    sthr_p[c] = bms ? sthr_ms : sthr[c];
                }
                if (bms) mmx_codec_lr_to_ms(t_cod[0], t_cod[1], MMX_HOP);
                alt->stereo_ms = bms;
                memset(alt->gain, MMX_GAIN_OFF, sizeof(alt->gain));
                memset(alt->polarity, 0, sizeof(alt->polarity));
                memset(alt->tns, 0, sizeof(alt->tns));
                mmx_frame_quantize_short(&codec->short_bands, alt, t_cod, sthr_p, codec->short_cutoff_band, bms ? sthr_lr_p : NULL, &E->base.ctx, 0);
                keep_energy_short(E, alt, t_cod, sthr_p, 0);
                mmx_frame_encode(&E->base.rc, &E->base.ctx, &codec->bands, &codec->short_bands, alt, 0);
            }
            for (ms = ms_lo; ms <= ms_hi; ms++)
            {
                for (c = 0; c < nch; c++)
                {
                    memcpy(t_cod[c], t_lr[c], sizeof(float) * MMX_HOP);
                    sthr_p[c] = ms ? sthr_ms : sthr[c];
                }
                if (ms) mmx_codec_lr_to_ms(t_cod[0], t_cod[1], MMX_HOP);
                if (n_sources)
                {
                    float scale = params->reuse ? (float)pow(10.0, 3.0 * params->reuse / 10.0) : 1.0f;
                    double te = 0.0, re = 0.0;
                    memset(&pure, 0, sizeof(pure));
                    alt->stereo_ms = (int)ms;
                    memset(alt->gain, MMX_GAIN_OFF, sizeof(alt->gain));
                    memset(alt->polarity, 0, sizeof(alt->polarity));
                    memset(alt->tns, 0, sizeof(alt->tns));
                    mmx_codec_source_coefs_bt(codec, &E->decoded, n_sources, plan->src_start, (int)ms, MMX_BT_SHORT);
                    mmx_codec_fit_gains(codec, alt, n_sources, t_cod);
                    mmx_codec_predict(codec, alt, n_sources);
                    for (c = 0; c < nch; c++)
                        for (k = 0; k < MMX_HOP; k++)
                        {
                            resid[c][k] = t_cod[c][k] - codec->pred[c][k];
                            te += (double)t_cod[c][k] * t_cod[c][k];
                            re += (double)resid[c][k] * resid[c][k];
                        }
                    for (c = 0; c < nch; c++)
                        for (g = 0; g < MMX_SHORT_GROUPS; g++)
                            for (b = 0; b < MMX_SHORT_BANDS; b++)
                                sthr_q[c][g][b] = sthr_p[c][g][b] < 1e29f ? sthr_p[c][g][b] * scale : sthr_p[c][g][b];
                    if (params->mode)
                    {
                        /* tracker / future: residual thresholds relaxed by relax, the sub-window
                           (pre-echo) part only by relax_sub, as for long frames */
                        for (c = 0; c < nch; c++)
                            for (g = 0; g < MMX_SHORT_GROUPS; g++)
                                for (b = 0; b < MMX_SHORT_BANDS; b++)
                                {
                                    float t = sthr_p[c][g][b], cap = scap[c][g][b];
                                    if (ms && nch == 2)
                                    {
                                        cap = scap[0][g][b] < scap[1][g][b] ? scap[0][g][b] : scap[1][g][b];
                                        cap = cap < 1e29f ? cap * 0.5f : cap;
                                    }
                                    if (t >= 1e29f) { sthr_q[c][g][b] = t; continue; }
                                    t = (float)(t * E->relax);
                                    cap = cap < 1e29f ? (float)(cap * E->relax_sub) : cap;
                                    sthr_q[c][g][b] = t < cap ? t : cap;
                                }
                    }
                    if (echo_attempt || E->ref_offset_factor != 1.0)
                        for (c = 0; c < nch; c++)
                            for (g = 0; g < MMX_SHORT_GROUPS; g++)
                                for (b = 0; b < MMX_SHORT_BANDS; b++)
                                    if (sthr_q[c][g][b] < 1e29f)
                                        sthr_q[c][g][b] = (float)(sthr_q[c][g][b] * echo_scale[c][b] * E->ref_offset_factor);
                    /* short frames are never played pure in tracker / future mode: they are
                       attack frames, and the time-resolved check that vetoes a source hit
                       landing a few milliseconds early (pure_temporal_check) exists for long
                       frames only; measured, the whole-frame shortcut left ghost hits at
                       0 dB SNR right before the real attack (title A 318.06 s) */
                    if (!params->mode && params->reuse >= 2 && re > 1e-20 && 10.0 * log10(te / re) >= (params->reuse >= 3 ? 7.0 : 10.0))
                    {
                        for (c = 0; c < nch; c++) for (k = 0; k < MMX_HOP; k++) resid[c][k] = 0.0f;
                        pure.all_pure = 1;
                    }
                    mmx_frame_quantize_short(&codec->short_bands, alt, resid, sq, codec->short_cutoff_band, ms ? sthr_lr_p : NULL, &w->ctx, qmode);
                    keep_energy_short(E, alt, resid, sq, n_sources);
                    bits = trial_bits(w, &codec->bands, &codec->short_bands, alt, n_sources);
                    if (bits < best_bits)
                    {
                        best_bits = bits;
                        mmx_frame_syntax_copy(best, alt);
                        use_ms = (int)ms;
                        best_pred = 1;
                        best_pure = pure;
                    }
                }
                /* no prediction (all gains off, plain audio inside the block) or an AUDIO frame */
                alt->stereo_ms = (int)ms;
                memset(alt->gain, MMX_GAIN_OFF, sizeof(alt->gain));
                memset(alt->polarity, 0, sizeof(alt->polarity));
                memset(alt->tns, 0, sizeof(alt->tns));
                mmx_frame_quantize_short(&codec->short_bands, alt, t_cod, sthr_p, codec->short_cutoff_band, ms ? sthr_lr_p : NULL, &w->ctx, qmode);
                keep_energy_short(E, alt, t_cod, sthr_p, n_sources);
                bits = trial_bits(w, &codec->bands, &codec->short_bands, alt, n_sources);
                if (bits < best_bits)
                {
                    best_bits = bits;
                    mmx_frame_syntax_copy(best, alt);
                    use_ms = (int)ms;
                    best_pred = 0;
                    memset(&best_pure, 0, sizeof(best_pure));
                }
            }
            if (!trial) st->short_frames++;
        }
        else
        {
            /* ---------- long frame (LONG, START or STOP window) ---------- */
            if (nch == 2)
            {
                mmx_psy_stereo_threshold(&codec->bands, psy[0], psy[1], thr_ms);
                /* START/STOP frames keep the estimate: their thresholds (start_stop_thresholds)
                   do not cover the sidelobes of the window shape, and coding up to them in
                   L/R leaks the allowed noise of loud low bands 500 Hz up onto quiet bands
                   (roundtrip track 2.46 s: 11.6 dB over in the sub-window model, with the
                   more conservative M/S thresholds 7.2 dB); the block-type trial checks its
                   START/STOP frames against that model, the plain M/S trial cannot */
                if (!E->trial_ms || bt != MMX_BT_LONG)
                    ms_lo = ms_hi = long_ms_estimate(E, f, n_sources, psy, thr_ms, bt) ? 1 : 0;
            }
            if (!trial && attempt == 0)
            {
                /* --- baseline: the same frame as plain audio, never written once per frame --- */
                int bms = nch == 2 ? long_ms_estimate(E, f, 0, psy, thr_ms, bt) : 0;
                for (c = 0; c < nch; c++)
                {
                    memcpy(t_cod[c], t_lr[c], sizeof(float) * MMX_HOP);
                    thr[c] = bms ? thr_ms : psy[c]->thr;
                }
                if (bms) mmx_codec_lr_to_ms(t_cod[0], t_cod[1], MMX_HOP);
                alt->stereo_ms = bms;
                memset(alt->gain, MMX_GAIN_OFF, sizeof(alt->gain));
                memset(alt->polarity, 0, sizeof(alt->polarity));
                memset(alt->tns, 0, sizeof(alt->tns));
                apply_tns(codec, alt, t_cod, psy, thr, thr_buf, thr_q, thr_lr_buf, thr_lr_q, thr_lr, params->tns);
                tonal_relax(E, thr_buf, psy, nch, codec->bands.band_count);
                mmx_frame_quantize_ext(&codec->bands, alt, t_cod, thr_q, coded_cutoff(codec), bms ? thr_lr_q : NULL, crest, &E->base.ctx, 0,
                                       NULL, nf_zero_of(&E->nf));
                post_quantize(E, alt, t_cod, thr_q, crest, 0);
                epb_set_energies(codec, alt, t_lr, bms);
                bwe_apply(codec, alt, t_cod);
                pns_substitute(codec, alt, t_cod, analysis, f, &E->pns, E->pns_prev);
                mmx_frame_encode(&E->base.rc, &E->base.ctx, &codec->bands, &codec->short_bands, alt, 0);
            }
            for (ms = ms_lo; ms <= ms_hi; ms++)
            {
                for (c = 0; c < nch; c++)
                {
                    memcpy(t_cod[c], t_lr[c], sizeof(float) * MMX_HOP);
                    thr[c] = ms ? thr_ms : psy[c]->thr;
                }
                if (ms) mmx_codec_lr_to_ms(t_cod[0], t_cod[1], MMX_HOP);
                if (n_sources)
                {
                    const float *thr_r[MMX_MAX_CH], *thr_r_lr[2];
                    double te = 0.0, re = 0.0;
                    memset(&pure, 0, sizeof(pure));
                    alt->stereo_ms = (int)ms;
                    memset(alt->gain, MMX_GAIN_OFF, sizeof(alt->gain));
                    memset(alt->polarity, 0, sizeof(alt->polarity));
                    memset(alt->tns, 0, sizeof(alt->tns));
                    mmx_codec_source_coefs_bt(codec, &E->decoded, n_sources, plan->src_start, (int)ms, bt);
                    mmx_codec_fit_gains(codec, alt, n_sources, t_cod);
                    mmx_codec_predict(codec, alt, n_sources);
                    for (c = 0; c < nch; c++)
                        for (k = 0; k < MMX_HOP; k++)
                        {
                            resid[c][k] = t_cod[c][k] - codec->pred[c][k];
                            te += (double)t_cod[c][k] * t_cod[c][k];
                            re += (double)resid[c][k] * resid[c][k];
                        }
                    for (c = 0; c < nch; c++) thr_r[c] = thr[c];
                    thr_r_lr[0] = thr_lr[0];
                    thr_r_lr[1] = thr_lr[1];
                    /* tracker / future: "90 % the same is played the same" per band (see encoder.h) */
                    if (params->mode)
                    {
                        tracker_pure_bands(codec, resid, t_cod, nch, E->run_gain + f * MMX_EQ_BANDS, E->pure_db, transient, &pure,
                                           E->temporal, alt, n_sources, audio, &E->decoded, start, plan->src_start, (int)ms);
                        tracker_relax_long(E, psy, thr, thr_lr, (int)ms, reuse_thr, reuse_thr_lr);
                        for (c = 0; c < nch; c++) thr_r[c] = reuse_thr[c];
                        thr_r_lr[0] = reuse_thr_lr[0];
                        thr_r_lr[1] = reuse_thr_lr[1];
                    }
                    /* legacy experiment (--reuse): frame-level decision by the broadband gain */
                    else if (params->reuse)
                    {
                        double gain_db = re > 1e-20 ? 10.0 * log10(te / re) : 100.0;
                        double pure_at = params->reuse >= 3 ? 7.0 : 10.0;
                        float scale = (float)pow(10.0, 3.0 * params->reuse / 10.0);
                        unsigned int b;
                        if (params->reuse >= 2 && gain_db >= pure_at)
                        {
                            for (c = 0; c < nch; c++)
                                for (k = 0; k < MMX_HOP; k++) resid[c][k] = 0.0f;
                            pure.all_pure = 1;
                        }
                        for (c = 0; c < nch; c++)
                        {
                            for (b = 0; b < MMX_MAX_BANDS; b++)
                                reuse_thr[c][b] = thr[c][b] < 1e29f ? thr[c][b] * scale : thr[c][b];
                            thr_r[c] = reuse_thr[c];
                        }
                        for (c = 0; c < 2; c++)
                        {
                            for (b = 0; b < MMX_MAX_BANDS; b++)
                                reuse_thr_lr[c][b] = thr_lr[c][b] < 1e29f ? thr_lr[c][b] * scale : thr_lr[c][b];
                            thr_r_lr[c] = reuse_thr_lr[c];
                        }
                    }
                    /* structured-error guard: bands that leaked the prediction in an earlier
                       attempt are quantized finer; MMX_REF_OFFSET_SHARE gives referenced
                       frames only part of the rate loop's offset (experiment) */
                    if (echo_attempt || E->ref_offset_factor != 1.0)
                    {
                        unsigned int eb;
                        for (c = 0; c < nch; c++)
                        {
                            for (eb = 0; eb < MMX_MAX_BANDS; eb++)
                                echo_thr[c][eb] = thr_r[c][eb] < 1e29f
                                                      ? (float)(thr_r[c][eb] * echo_scale[c][eb] * E->ref_offset_factor)
                                                      : thr_r[c][eb];
                            thr_r[c] = echo_thr[c];
                        }
                    }
                    apply_tns(codec, alt, resid, psy, thr_r, thr_buf, thr_q, thr_lr_buf, thr_lr_q, thr_r_lr, params->tns);
                    tonal_relax(E, thr_buf, psy, nch, codec->bands.band_count);
                    mmx_frame_quantize_ext(&codec->bands, alt, resid, thr_q, coded_cutoff(codec), ms ? thr_lr_q : NULL, crest, &w->ctx, qmode,
                                           is_target(E, alt, thr_lr_q, (int)ms, 1), nf_zero_of(&E->nf));
                    post_quantize(E, alt, resid, thr_q, crest, n_sources);
                    epb_set_energies(codec, alt, t_lr, (int)ms);
                    bits = pns_and_bits(E, w, alt, t_cod, f, n_sources, trial);
                    if (bits < best_bits)
                    {
                        best_bits = bits;
                        mmx_frame_syntax_copy(best, alt);
                        memcpy(E->keep_mark_best, E->keep_mark, sizeof(E->keep_mark_best));
                        use_ms = (int)ms;
                        best_pred = 1;
                        best_pure = pure;
                    }
                }
                /* no prediction for this frame (all gains off, same block, own TNS, strict
                   thresholds: it is plain audio) or an AUDIO frame */
                alt->stereo_ms = (int)ms;
                memset(alt->gain, MMX_GAIN_OFF, sizeof(alt->gain));
                memset(alt->polarity, 0, sizeof(alt->polarity));
                memset(alt->tns, 0, sizeof(alt->tns));
                apply_tns(codec, alt, t_cod, psy, thr, thr_buf, thr_q, thr_lr_buf, thr_lr_q, thr_lr, params->tns);
                tonal_relax(E, thr_buf, psy, nch, codec->bands.band_count);
                mmx_frame_quantize_ext(&codec->bands, alt, t_cod, thr_q, coded_cutoff(codec), ms ? thr_lr_q : NULL, crest, &w->ctx, qmode,
                                       is_target(E, alt, thr_lr_q, (int)ms, 0), nf_zero_of(&E->nf));
                post_quantize(E, alt, t_cod, thr_q, crest, n_sources);
                epb_set_energies(codec, alt, t_lr, (int)ms);
                bits = pns_and_bits(E, w, alt, t_cod, f, n_sources, trial);
                if (E->debug_bandms && !trial && nch == 2 && ms_lo != ms_hi && bt == MMX_BT_LONG)
                {
                    /* per-band M/S oracle: the exact bits of every band in both domains */
                    mmx_frame_encode_book_bands(E->bandms_bits[ms]);
                    trial_bits(w, &codec->bands, &codec->short_bands, alt, n_sources);
                    if (ms)
                    {
                        unsigned int b;
                        double sum[2] = { 0.0, 0.0 }, best = 0.0;
                        for (b = 0; b < codec->cutoff_band; b++)
                        {
                            double b0 = E->bandms_bits[0][0][b] + E->bandms_bits[0][1][b], b1 = E->bandms_bits[1][0][b] + E->bandms_bits[1][1][b];
                            sum[0] += b0; sum[1] += b1;
                            best += b0 < b1 ? b0 : b1;
                        }
                        E->bandms_frame += sum[0] < sum[1] ? sum[0] : sum[1];
                        E->bandms_oracle += best;
                        E->bandms_frames++;
                    }
                }
                if (bits < best_bits)
                {
                    best_bits = bits;
                    mmx_frame_syntax_copy(best, alt);
                    memcpy(E->keep_mark_best, E->keep_mark, sizeof(E->keep_mark_best));
                    use_ms = (int)ms;
                    best_pred = 0;
                    memset(&best_pure, 0, sizeof(best_pure));
                }
            }
        }

        /* the cheapest candidate is the frame; its prediction (zero when the gains are off).
           The syntax struct mirrors the decoder's: the coder's inter-frame state reads
           the long-frame scalefactor arrays of a short frame as the last long frame
           left them (remember() in framecodec.c) and sees its noise flags cleared, so
           a short frame keeps those arrays and clears the flags. */
        if (bt == MMX_BT_SHORT)
        {
            unsigned char sf_keep[MMX_MAX_CH][MMX_MAX_BANDS], zero_keep[MMX_MAX_CH][MMX_MAX_BANDS];
            memcpy(sf_keep, syn->sf, sizeof(sf_keep));
            memcpy(zero_keep, syn->band_zero, sizeof(zero_keep));
            mmx_frame_syntax_copy(syn, best);
            memcpy(syn->sf, sf_keep, sizeof(sf_keep));
            memcpy(syn->band_zero, zero_keep, sizeof(zero_keep));
            memset(syn->band_noise, 0, sizeof(syn->band_noise));
            memset(syn->band_is, 0, sizeof(syn->band_is));
            memset(syn->band_bwe, 0, sizeof(syn->band_bwe));
            memset(syn->bwe_mix, 0, sizeof(syn->bwe_mix));
            memset(syn->nf_level, 0, sizeof(syn->nf_level));
        }
        else
            mmx_frame_syntax_copy(syn, best);
        if (n_sources)
            mmx_codec_source_coefs_bt(codec, &E->decoded, n_sources, plan->src_start, use_ms, bt);
        mmx_codec_predict(codec, syn, n_sources);
        if (!trial)
        {
            if (n_sources && !best_pred) st->ref_frames_gains_off++;
            if (best_pred)
            {
                st->pure_bands += best_pure.pure_bands;
                st->coded_bands += best_pure.coded_bands;
                st->pure_vetoed += best_pure.vetoed;
                if (best_pure.all_pure) st->pure_ref_frames++;
            }
        }
        mmx_codec_reconstruct(codec, syn, &E->decoded, start);
        if (trial)
            break;
        /* what the prediction leaks into this reconstruction (the echo) */
        echo_frame_measure(E, f, syn, best_pred ? n_sources : 0, use_ms, psy_base, &best_pure,
                           (const float (*)[MMX_SHORT_GROUPS][MMX_SHORT_BANDS])sthr, clip_scale, &efs);
        /* clip guard: the samples this frame completes against the range of the sink;
           the counts are always kept, the retries only while the guard is on */
        clip_peak = 0.0;
        clip_over = clip_guard_count(&E->guard, &E->decoded, start, MMX_HOP, &clip_peak);
        if (attempt == 0)
            clip_over_raw = clip_over;
        clip_left = 0;
        if (E->guard.debug && clip_over)
        {
            long long i, lo = start < 0 ? 0 : start, hi = start + MMX_HOP < (long long)E->decoded.frame_count ? start + MMX_HOP : (long long)E->decoded.frame_count;
            double worst = 0.0; long long at = lo; unsigned int wc = 0;
            for (i = lo; i < hi; i++)
                for (c = 0; c < nch; c++)
                {
                    double x = E->decoded.samples[(size_t)i * nch + c], o = x > 0.0 ? x - E->guard.lim_pos : E->guard.lim_neg - x;
                    if (o > worst) { worst = o; at = i; wc = c; }
                }
            fprintf(stderr, "clip guard frame %lu (%.3f s) attempt %d: %lu samples over, worst +%.4f at %lld ch %u (original %.4f, %s)\n", f,
                    (double)start / audio->sample_rate, attempt, clip_over, worst, at, wc, audio->samples[(size_t)at * nch + wc],
                    at < start + MMX_HOP / 2 ? "first quarter of the window" : "second quarter");
        }
        if (!guard_on)
            break;
        retry = 0;
        /* the prediction above the model threshold: quantize the offending bands
           again, finer. The frame's own "no prediction" candidate is trial-coded
           against the tightened one in every attempt, so a frame whose leak costs
           more than plain audio falls back to plain audio on bits. */
        if (echo_on && efs.over_guard)
        {
            if (echo_attempt < E->echo.steps && echo_tighten(&efs, echo_scale, nch))
            {
                echo_attempt++;
                E->echo.retries++;
                retry = 1;
            }
            else
                echo_left = 1;                          /* still leaking after the last step */
        }
        if (!retry)
        {
            if (!E->guard.steps)
                break;                                      /* clip guard off: its buffers do not exist */
            if (E->guard.all ? !clip_over : !clip_guard_audible(&E->guard, codec, audio, &E->decoded, t_lr, bt, start))
            {
                if (clip_attempt == 0) clip_quiet = clip_over;   /* outside the range, but the clamp stays under the model */
                break;
            }
            if (!E->guard.steps || clip_attempt >= E->guard.steps)
            {
                clip_left = 1;                          /* still audible by the model after the last step */
                break;
            }
            clip_attempt++;
            clip_scale *= E->guard.step_scale;
            retry = 1;
        }
        /* quantize the frame again, finer: the overlap-add and the counters of this attempt taken back */
        window_save(E->clip_save, &E->decoded, start, 1);
        *st = st_before;
        E->pns_reverted = pns_reverted_before;
    }
    keep_commit(E, syn, bt);   /* the hysteresis state the next frame reads */
    if (trial)
    {
        pos = mmx_rc_enc_bits(&w->rc);
        mmx_frame_encode(&w->rc, &w->ctx, &codec->bands, &codec->short_bands, syn, n_sources);
        if (w->rc.failed)
            return -1;
        *bits_out = mmx_rc_enc_bits(&w->rc) - pos;
        mmx_refgraph_set(&E->graph, f, (unsigned char)depth, n_sources, plan->src_start);
        return 0;
    }
    st->clip_samples_raw += clip_over_raw;
    st->clip_samples += clip_over;
    st->clip_samples_quiet += clip_quiet;
    if (clip_peak > st->clip_peak) st->clip_peak = clip_peak;
    if (clip_attempt) st->clip_frames++;
    if (clip_left) st->clip_frames_left++;
    if (echo_attempt) E->echo.frames_guarded++;
    if (echo_left) E->echo.frames_left++;
    echo_book(&E->echo, &efs, start, audio->sample_rate);
    for (c = 0; c < nch; c++)
        psy[c] = psy_base[c];   /* the quality barrier and the calibration judge against the model's thresholds */
    thr_lr[0] = psy[0]->thr;
    thr_lr[1] = psy[nch > 1 ? 1 : 0]->thr;
    if (E->debug_bits)
        mmx_frame_bitdump_next((n_sources ? 2u : 0u) + (analysis->transient[f] ? 1u : 0u));
    {
        double elem_real[MMX_ELEM_COUNT] = { 0.0 };
        if (E->calibrate && est_debug.on) mmx_frame_encode_book_elems(elem_real);
    pos = mmx_rc_enc_bits(&w->rc);
    mmx_frame_encode(&w->rc, &w->ctx, &codec->bands, &codec->short_bands, syn, n_sources);
    if (E->calibrate && est_debug.on) memcpy(E->elem_real_last, elem_real, sizeof(elem_real));
    }
    if (w->rc.failed)
    {
        mmx_error("Range coder failed at frame %lu", f);
        return -1;
    }
    *bits_out = mmx_rc_enc_bits(&w->rc) - pos;
    mmx_refgraph_set(&E->graph, f, (unsigned char)depth, n_sources, plan->src_start);

    /* real bits against the estimates (second-pass calibration, MMX_DEBUG_EST):
       the estimate of the syntax that was written, taken on the arrays of the
       chosen candidate (its coded domain, its prediction, its TNS filter: apply_tns
       decides the same filter again from the same input). A frame whose gains the
       trial switched off is its own class and left out of the calibration (its
       estimate here is the audio's, not the residual's). */
    if (E->calibrate)
    {
        double est = 0.0, est2 = 0.0;
        int gains_off = n_sources && !best_pred;
        float *const *x = n_sources && best_pred ? resid : t_cod;
        for (c = 0; c < nch; c++)
            memcpy(t_cod[c], t_lr[c], sizeof(float) * MMX_HOP);
        if (use_ms && nch == 2)
            mmx_codec_lr_to_ms(t_cod[0], t_cod[1], MMX_HOP);
        if (n_sources && best_pred)
            for (c = 0; c < nch; c++)
                for (k = 0; k < MMX_HOP; k++) resid[c][k] = t_cod[c][k] - codec->pred[c][k];
        if (bt == MMX_BT_SHORT)
        {
            if (est_debug.on && !analysis->coded_estimate)
                for (c = 0; c < nch; c++)
                    est += mmx_frame_estimate_bits_short(&codec->short_bands, x[c], use_ms && nch == 2 ? sthr_ms : sthr[c], codec->short_cutoff_band);
        }
        else
        {
            if (nch == 2)
                mmx_psy_stereo_threshold(&codec->bands, psy[0], psy[1], thr_ms);
            for (c = 0; c < nch; c++)
                thr[c] = use_ms && nch == 2 ? thr_ms : psy[c]->thr;
            if (est_debug.on && !analysis->coded_estimate)
                for (c = 0; c < nch; c++)
                    est += mmx_frame_estimate_bits(&codec->bands, x[c], thr[c], codec->cutoff_band);
            if (bt == MMX_BT_LONG && !gains_off)
            {
                double elem_est[MMX_ELEM_COUNT] = { 0.0 };
                unsigned int e;
                apply_tns(codec, alt, x, psy, thr, thr_buf, thr_q, thr_lr_buf, thr_lr_q, thr_lr, params->tns);
                tonal_relax(E, thr_buf, psy, nch, codec->bands.band_count);
                for (c = 0; c < nch; c++)
                {
                    if (est_debug.on) mmx_frame_estimate_book_elems(elem_est);
                    est2 += mmx_frame_estimate_bits_coded(&codec->bands, x[c], thr_q[c], codec->cutoff_band, crest[c]);
                }
                if (est_debug.on)
                {
                    unsigned int cls = n_sources ? 1u : 0u;
                    for (e = 0; e < MMX_ELEM_COUNT; e++) { est_debug.elem_est[cls][e] += elem_est[e]; est_debug.elem_real[cls][e] += E->elem_real_last[e]; }
                    est_debug.elem_frames[cls]++;
                }
            }
        }
        /* the calibration reads the estimate in the analyzer's scale (side information of the gains included there) */
        if (analysis->coded_estimate)
            est = est2 > 0.0 ? est2 + (n_sources ? MMX_GAIN_SIDE_BITS * n_sources * nch : 0.0) : 0.0;
        else if (n_sources)
            est += 8.0 * n_sources;
        est_debug_add(n_sources ? (gains_off ? 2u : 1u) : 0u, w->entry.frame_count, analysis->transient[f], est,
                      n_sources ? plan->ref_bits : plan->audio_bits, *bits_out, est2);
    }
    w->entry.frame_count++;

    /* quality barrier: NMR of the reconstructed L/R coefficients against the original */
    {
        double worst = -200.0;
        unsigned int over_total = 0;
        static long long debug_frame = -2;
        static double debug_time = -1.0;
        int dbg;
        if (debug_frame == -2)
        {
            const char *e = getenv("MMX_DEBUG_FRAME"), *t = getenv("MMX_DEBUG_TIME");
            debug_frame = e ? atol(e) : -1;
            debug_time = t ? atof(t) : -1.0;
        }
        dbg = (long long)f == debug_frame ||
              (debug_time >= 0.0 && start <= (long long)(debug_time * audio->sample_rate) && start + MMX_WIN > (long long)(debug_time * audio->sample_rate));
        /* short frames: the reconstructed groups against their own (L/R) group thresholds */
        for (c = 0; c < nch && bt == MMX_BT_SHORT; c++)
        {
            unsigned int g, b;
            if (dbg)
                fprintf(stderr, "encoder frame %lu (window %lld = %.3f s) ch %u SHORT ms=%d src=%u\n", f, start, (double)start / audio->sample_rate, c, syn->stereo_ms, n_sources);
            for (g = 0; g < MMX_SHORT_GROUPS; g++)
                for (b = 0; b < codec->short_cutoff_band; b++)
                {
                    double noise = 0.0, energy = 0.0, nmr, n = (double)(codec->short_bands.band_start[b + 1] - codec->short_bands.band_start[b]);
                    double allowed = (double)sthr[c][g][b] * n;
                    for (k = g * MMX_SHORT_M + codec->short_bands.band_start[b]; k < g * MMX_SHORT_M + codec->short_bands.band_start[b + 1]; k++)
                    {
                        noise += (double)(codec->rec[c][k] - t_lr[c][k]) * (codec->rec[c][k] - t_lr[c][k]);
                        energy += (double)t_lr[c][k] * t_lr[c][k];
                    }
                    nmr = noise > 0.0 && allowed > 0.0 ? 10.0 * log10(noise / allowed) : -200.0;
                    if (dbg)
                        fprintf(stderr, "  group %u (%lld) band %2u (%5.0f Hz) energy %10.4g thr*n %10.4g noise %10.4g nmr %7.2f sf %3u zero %u\n", g,
                                start + (long long)codec->short_offset + (long long)g * MMX_SHORT_M, b, codec->short_bands.band_hz[b], energy,
                                allowed, noise, nmr, syn->sf_s[c][g][b], syn->band_zero_s[c][g][b]);
                    if (energy <= allowed * 0.5)
                        continue;
                    if (nmr > 0.0) over_total++;
                    if (nmr > worst) worst = nmr;
                }
        }
        for (c = 0; c < nch && bt != MMX_BT_SHORT; c++)
        {
            unsigned int over = 0, b;
            unsigned char skip[MMX_MAX_BANDS];
            double wq;
            /* noise bands are counted, not measured: their error is the signal itself */
            for (b = 0; b < MMX_MAX_BANDS; b++)
            {
                skip[b] = syn->band_noise[c][b];
                if (syn->stereo_ms && nch == 2) skip[b] |= syn->band_noise[c ^ 1][b];
                if (nch == 2) skip[b] |= syn->band_is[b];   /* the dropped difference is the error, by construction */
                if (b < codec->cutoff_band)
                {
                    if (E->keep_mark_best[c][b] == 1) st->keep_bands++;
                    else if (E->keep_mark_best[c][b] == 2) st->keep_noise_bands++;
                    if (syn->band_bwe[c][b])
                    {
                        st->bwe_bands++;
                        if (syn->band_zero[c][b]) st->bwe_bands_zero++;
                    }
                }
                if (!syn->band_zero[c][b] && b < codec->cutoff_band)
                {
                    st->coded_bands_total++;
                    if (b >= E->pns.first_band && E->pns.first_band) st->pns_candidates++;
                    if (syn->band_noise[c][b]) st->pns_bands++;
                    if (syn->band_noise[c][b] && E->nf.on && !E->pns.first_band) st->nf_hole_bands++;
                }
                if (c == 0 && nch == 2 && E->is_first_band && b >= E->is_first_band && b < codec->cutoff_band)
                {
                    if (syn->band_is[b]) { st->is_bands++; st->is_candidates++; E->is_dbg[4]++; }
                    else if (!syn->band_zero[0][b] || !syn->band_zero[1][b]) st->is_candidates++;
                    if (E->debug_is && !syn->band_is[b] && E->is_plan.cand[b]) E->is_dbg[3]++;   /* a candidate the bits rejected */
                }
            }
            if (E->nf.on)
                for (b = E->nf.first_region; b < MMX_EQ_BANDS; b++)
                {
                    st->nf_regions_total++;
                    if (syn->nf_level[c][b]) st->nf_regions++;
                }
            wq = mmx_quality_frame_nmr(codec, psy[c], t_lr[c], codec->rec[c], skip, &over);
            if (dbg)
            {
                const MMXFramePsy *p = psy[c];
                fprintf(stderr, "encoder frame %lu (window %lld = %.3f s) ch %u bt=%u ms=%d src=%u\n", f, start, (double)start / audio->sample_rate, c, bt, syn->stereo_ms, n_sources);
                for (b = 0; b < codec->cutoff_band; b++)
                {
                    double noise = 0.0, n = (double)(codec->bands.band_start[b + 1] - codec->bands.band_start[b]);
                    for (k = codec->bands.band_start[b]; k < codec->bands.band_start[b + 1]; k++)
                        noise += (double)(codec->rec[c][k] - t_lr[c][k]) * (codec->rec[c][k] - t_lr[c][k]);
                    fprintf(stderr, "  band %2u (%5.0f Hz) energy %10.4g thr*n %10.4g noise %10.4g nmr %7.2f sf %3u zero %u crest %.2f flat %.2f%s\n", b, codec->bands.band_hz[b], p->energy[b],
                            p->thr[b] * n, noise, mmx_psy_nmr_db(&codec->bands, p, b, noise), syn->sf[c][b], syn->band_zero[c][b], p->crest[b], p->flat[b],
                            syn->band_noise[c][b] ? " NOISE" : nch == 2 && syn->band_is[b] ? (c ? " IS" : " IS(mid)") : "");
                }
            }
            over_total += over;
            if (wq > worst) worst = wq;
        }
        if (worst > -150.0)
        {
            E->nmr_sum += worst;
            E->nmr_count++;
        }
        if (worst > st->worst_nmr_db) st->worst_nmr_db = worst;
        if (over_total) st->frames_over_threshold++;
    }

    if (E->debug_is && bt != MMX_BT_SHORT && nch == 2)
    {
        E->is_dbg[0] += E->is_plan.n_range;
        E->is_dbg[1] += E->is_plan.n_rho;
        E->is_dbg[2] += E->is_plan.n_slack;
    }
    st->frames_by_sources[n_sources]++;
    if (syn->stereo_ms) st->stereo_ms_frames++;
    for (c = 0; c < nch; c++)
        if (syn->tns[c].active) { st->tns_frames++; break; }
    if ((unsigned)depth > st->max_depth_used) st->max_depth_used = (unsigned)depth;
    if (depth < 8) st->depth_histogram[depth]++;
    return 0;
}

/* ---------- multi-frame trials ---------- */
/* Saves what coding frames f0 .. f0+n-1 changes: their plan entries and the
   decoded samples their windows reach. */
static void trial_save(Enc *E, unsigned long f0, unsigned long n, TrialState *t)
{
    unsigned long nf = E->analysis->frame_count, nch = E->audio->channels;
    if (f0 + n > nf) n = nf - f0;
    t->f0 = f0;
    t->n = n;
    t->lo = (long long)f0 * MMX_HOP - MMX_HOP;
    t->hi = (long long)(f0 + n) * MMX_HOP;
    if (t->lo < 0) t->lo = 0;
    if (t->hi > (long long)E->decoded.frame_count) t->hi = (long long)E->decoded.frame_count;
    if (t->hi > t->lo)
        memcpy(E->snap, E->decoded.samples + (size_t)t->lo * nch, sizeof(float) * (size_t)(t->hi - t->lo) * nch);
    memcpy(t->plan, &E->analysis->plan[f0], sizeof(MMXFramePlan) * n);
    memcpy(t->sf, E->syn.sf, sizeof(t->sf));
    memcpy(t->band_zero, E->syn.band_zero, sizeof(t->band_zero));
    memcpy(t->band_noise, E->syn.band_noise, sizeof(t->band_noise));
    memcpy(t->keep_prev, E->keep_prev, sizeof(t->keep_prev));
    t->keep_prev_ms = E->keep_prev_ms;
    memcpy(t->pns_prev, E->pns_prev, sizeof(t->pns_prev));
}

/* Takes a trial back: the graph records of the `coded` frames (with the plan
   entries as the trial left them), then the plan entries and the samples. */
static void trial_restore(Enc *E, const TrialState *t, unsigned long coded)
{
    unsigned long k, nch = E->audio->channels;
    for (k = t->f0 + coded; k-- > t->f0;)
    {
        const MMXFramePlan *p = &E->analysis->plan[k];
        mmx_refgraph_unset(&E->graph, k, p->n_sources, p->src_start);
    }
    memcpy(&E->analysis->plan[t->f0], t->plan, sizeof(MMXFramePlan) * t->n);
    if (t->hi > t->lo)
        memcpy(E->decoded.samples + (size_t)t->lo * nch, E->snap, sizeof(float) * (size_t)(t->hi - t->lo) * nch);
    memcpy(E->syn.sf, t->sf, sizeof(t->sf));
    memcpy(E->syn.band_zero, t->band_zero, sizeof(t->band_zero));
    memcpy(E->syn.band_noise, t->band_noise, sizeof(t->band_noise));
    memcpy(E->keep_prev, t->keep_prev, sizeof(t->keep_prev));
    E->keep_prev_ms = t->keep_prev_ms;
    memcpy(E->pns_prev, t->pns_prev, sizeof(t->pns_prev));
}

/* A scratch writer continuing the real block's contexts. */
static void trial_writer_begin(BlockWriter *tw, const BlockWriter *w)
{
    memset(tw, 0, sizeof(*tw));
    tw->ctx = w->ctx;
    mmx_rc_enc_init(&tw->rc);
    tw->open = w->open;
}

static void trial_writer_end(BlockWriter *tw)
{
    mmx_rc_enc_free(&tw->rc);
    tw->open = 0;
}

/* Codes frames f0 .. f0+n-1 into a scratch writer; returns their bits, or a
   negative value when the coder failed. The trial's decoded samples are left
   in place for a time-domain check: the caller takes the state back with
   trial_restore(ts, coded). */
static double trial_frames(Enc *E, unsigned long f0, unsigned long n, TrialState *ts, unsigned long *coded_out)
{
    BlockWriter tw;
    MMXStatistics scratch;
    unsigned long k, coded = 0;
    double bits = 0.0, b;
    int ok = 1;
    trial_save(E, f0, n, ts);
    trial_writer_begin(&tw, &E->block);
    mmx_statistics_init(&scratch);
    for (k = f0; k < f0 + ts->n; k++)
    {
        if (frame_code(E, k, &tw, &scratch, 1, &b) != 0) { ok = 0; break; }
        bits += b;
        coded++;
    }
    trial_writer_end(&tw);
    *coded_out = coded;
    return ok ? bits : -1.0;
}

/* The decoded signal as the listener gets it: PCM at the source depth. The
   reconstruction may overshoot full scale where the recording is close to it
   (a loud passage of title A: 11 samples up to 1.04 in one 23 ms window),
   and the WAV the decoder writes clamps them; mmx compare then sees that as
   broadband noise (13.3 dB over at 10 kHz where the float reconstruction had
   1 dB). The trials must judge the same signal: rounded and clamped at the
   source bit depth, as the WAV writer does (clamp_round in wav_writer.c). */
static float pcm_clamp(const Enc *E, float v)
{
    double s = E->pcm_scale, q = floor((double)v * s + 0.5);
    if (q > s - 1.0) q = s - 1.0;
    if (q < -s) q = -s;
    return (float)(q / s);
}

/* The masking model of the original per 23 ms sub-window of [lo, hi) and
   channel (the yardstick of mmx compare); returns the number of sub-windows. */
static unsigned int region_psy(Enc *E, long long lo, long long hi)
{
    const MMXAudioBuffer *audio = E->audio;
    unsigned int n = 0, c, nch = audio->channels;
    double pre_db = mmx_psy_pre_attack_db();
    long long s;
    for (s = lo; s + QSUB_WIN <= hi && n < QSUB_MAX; s += QSUB_HOP, n++)
        for (c = 0; c < nch; c++)
        {
            MMXFramePsy *p = &E->qpsy[n * nch + c];
            long long i;
            for (i = 0; i < QSUB_WIN; i++)
            {
                long long q = s + i;
                E->qwin[i] = (q >= 0 && q < (long long)audio->frame_count) ? audio->samples[(size_t)q * nch + c] : 0.0f;
            }
            mmx_spectrum_analyze(&E->qspec, E->qwin, E->qorig);
            mmx_psy_analyze(&E->qbands, E->qorig, E->qwin, E->codec->quality, p);
            if (p->transient && pre_db >= 0.0)
            {
                float pre_win[MMX_PRE_ATTACK];
                long long end = s + p->attack_pos, j;
                for (j = 0; j < MMX_PRE_ATTACK; j++)
                {
                    long long q = end - MMX_PRE_ATTACK + j;
                    pre_win[j] = (q >= 0 && q < (long long)audio->frame_count) ? audio->samples[(size_t)q * nch + c] : 0.0f;
                }
                mmx_psy_pre_attack(&E->qbands, &E->qpre_bands, &E->qpre_spec, pre_win, E->codec->quality, pre_db, p);
            }
        }
    return n;
}

/* Error of the decoded signal (as PCM, see pcm_clamp) against that model:
   cells (sub-window, channel, band) over the (bitrate-scaled) threshold and
   the worst band NMR. */
static void region_nmr(Enc *E, long long lo, unsigned int nsub, double *worst, unsigned long *over)
{
    const MMXAudioBuffer *audio = E->audio;
    unsigned int n, c, b, nch = audio->channels, cutoff = E->qbands.cutoff_band[E->codec->quality];
    const double *scale = E->qbscale;
    *worst = -200.0;
    *over = 0;
    for (n = 0; n < nsub; n++)
        for (c = 0; c < nch; c++)
        {
            const MMXFramePsy *p = &E->qpsy[n * nch + c];
            long long s = lo + (long long)n * QSUB_HOP, i;
            for (i = 0; i < QSUB_WIN; i++)
            {
                long long q = s + i;
                E->qwin[i] = (q >= 0 && q < (long long)audio->frame_count) ? pcm_clamp(E, E->decoded.samples[(size_t)q * nch + c]) - audio->samples[(size_t)q * nch + c] : 0.0f;
            }
            mmx_spectrum_analyze(&E->qspec, E->qwin, E->qerr);
            for (b = 0; b < cutoff; b++)
            {
                unsigned long k;
                double noise = 0.0, nb = (double)(E->qbands.band_start[b + 1] - E->qbands.band_start[b]), allowed, nmr;
                if (p->thr[b] >= 1e29f || p->energy[b] <= p->thr[b] * nb * 0.5)
                    continue;
                for (k = E->qbands.band_start[b]; k < E->qbands.band_start[b + 1]; k++)
                    noise += (double)E->qerr[k] * E->qerr[k];
                allowed = (double)p->thr[b] * scale[b] * nb;
                nmr = noise > 0.0 ? 10.0 * log10(noise / allowed) : -200.0;
                if (nmr > 0.0) (*over)++;
                if (nmr > *worst) *worst = nmr;
            }
        }
}

/* Pre-echo as mmx compare measures it: attacks are 128-sample blocks at least
   10 dB above the mean of the 8 blocks before them (all channels summed), the
   metric is the SNR of those 46 ms. Returns the worst SNR of the attacks whose
   pre-attack blocks lie inside [lo, hi) (200 when there is none). */
#define PRE_BLK 128
static double region_pre_echo(Enc *E, long long lo, long long hi)
{
    const MMXAudioBuffer *audio = E->audio;
    unsigned int nch = audio->channels;
    double e_sig[(PLAN_TRIAL_FRAMES + 2) * MMX_HOP / PRE_BLK], e_err[(PLAN_TRIAL_FRAMES + 2) * MMX_HOP / PRE_BLK];
    double floor_e = pow(10.0, -60.0 / 10.0) * PRE_BLK * nch, worst = 200.0;
    unsigned long nb, i, j;
    long long b0 = lo / PRE_BLK;
    if (hi > (long long)audio->frame_count) hi = (long long)audio->frame_count;
    if (hi <= lo) return worst;
    nb = (unsigned long)((hi - lo) / PRE_BLK);
    if (nb > sizeof(e_sig) / sizeof(e_sig[0])) nb = sizeof(e_sig) / sizeof(e_sig[0]);
    for (i = 0; i < nb; i++)
    {
        double s2 = 0.0, d2 = 0.0;
        for (j = (unsigned long)((b0 + (long long)i) * PRE_BLK) * nch; j < (unsigned long)((b0 + (long long)i + 1) * PRE_BLK) * nch; j++)
        {
            double d = (double)audio->samples[j] - pcm_clamp(E, E->decoded.samples[j]);
            s2 += (double)audio->samples[j] * audio->samples[j];
            d2 += d * d;
        }
        e_sig[i] = s2;
        e_err[i] = d2;
    }
    for (i = 8; i < nb; i++)
    {
        double pre = 0.0, pre_err = 0.0, snr;
        for (j = i - 8; j < i; j++) { pre += e_sig[j]; pre_err += e_err[j]; }
        if (e_sig[i] <= floor_e || e_sig[i] < 10.0 * pre / 8.0)
            continue;
        snr = 10.0 * log10((pre + 1e-30) / (pre_err + 1e-30));
        if (pre <= floor_e * 8.0)
            snr = pre_err <= floor_e * 8.0 ? 200.0 : -200.0;
        if (snr < worst) worst = snr;
    }
    return worst;
}

/* Block type of frame g + 1, an attack between BT_TRIAL_LO_DB and
   BT_TRIAL_HI_DB, decided when frame g is about to be coded (its own type
   follows: START before a short frame): frames g .. g+3 are coded with g+1
   short and with g+1 long (the long frame g+3 makes the STOP frame's whole
   window final: with three frames the STOP window's later half went
   unchecked and its leakage put a 13.3 dB cell at 25.83 s of title A where
   the plan had 11.75). The attack rule's answer (short from
   SHORT_ATTACK_DB) stays unless the other block type is cheaper and the
   yardstick of mmx compare does not find it worse in the samples both
   variants have finished: no more (23 ms, band) cells over the masking
   threshold (BT_TRIAL_CELLS), no worst band above 0 dB and more than
   BT_TRIAL_DB above the rule's, no attack with a pre-attack SNR under 10 dB
   that the rule's variant keeps cleaner; the other block type is also taken
   when the rule's variant smears an attack and it does not, for at most
   BT_REPAIR_EXTRA more bits (title A tracker 301.0 s: long 9.7 dB, short
   12.2 dB for 3.7 % more). The rule alone (short from 10 dB)
   saved 2.6 % but smeared tonal material next to a kick (START windows leak
   250 Hz away); a symmetric decision (the better variant when the cheaper is
   worse) bought model quality with bits (attacks from 6 dB: +9 % size for
   1.4 % cells over) - the promise is the threshold, size follows. */
static void bt_trial(Enc *E, unsigned long g)
{
    unsigned long f = g + 1, nf = E->analysis->frame_count, last = g + TRIAL_FRAMES - 1 < nf ? g + TRIAL_FRAMES - 1 : nf - 1, derive_hi = last + 1 < nf ? last + 1 : nf - 1;
    long long lo = (long long)g * MMX_HOP - MMX_HOP, hi = (long long)last * MMX_HOP;
    double bits[2], worst[2], pre[2];
    unsigned long over[2], coded;
    unsigned char rule = E->is_short[f];
    unsigned int nsub, v, pick, cheaper;
    TrialState ts;

    E->cand[f] = 0;
    if (lo < 0) lo = 0;
    nsub = region_psy(E, lo, hi);
    for (v = 0; v < 2; v++)
    {
        E->is_short[f] = (unsigned char)v;
        derive_block_types(E->bt, E->is_short, g, derive_hi, nf);
        bits[v] = trial_frames(E, g, last - g + 1, &ts, &coded);
        worst[v] = 200.0;
        pre[v] = -200.0;
        over[v] = (unsigned long)-1;
        if (bits[v] >= 0.0)
        {
            region_nmr(E, lo, nsub, &worst[v], &over[v]);
            pre[v] = region_pre_echo(E, lo, hi);
        }
        trial_restore(E, &ts, coded);
    }
    /* the rule's answer stays unless the other block type is cheaper and no
       worse - or repairs an attack the rule's variant smears (pre-attack SNR
       under 10 dB) for at most BT_REPAIR_EXTRA more bits */
    pick = cheaper = rule;
    if (bits[rule] < 0.0 || (bits[1 - rule] >= 0.0 && bits[1 - rule] < bits[rule]))
    {
        unsigned int a = 1 - rule;
        cheaper = a;
        if (bits[rule] < 0.0 ||
            !(over[a] > over[rule] + E->bt_cells ||
              (worst[a] > 0.0 && worst[a] > worst[rule] + E->bt_db) ||
              (pre[a] < 10.0 && pre[a] + 0.5 < pre[rule])))
            pick = a;
    }
    else if (bits[1 - rule] >= 0.0 && pre[rule] < 10.0 && pre[1 - rule] >= 10.0 &&
             bits[1 - rule] <= bits[rule] * (1.0 + BT_REPAIR_EXTRA) && over[1 - rule] <= over[rule] + E->bt_cells)
        pick = 1 - rule;
    if (E->debug_bt)
        fprintf(stderr, "bt trial frame %lu (%.3f s): long %.0f bits over %lu worst %.2f pre %.1f | short %.0f bits over %lu worst %.2f pre %.1f | rule %s -> %s\n",
                f, (double)((long long)f * MMX_HOP - MMX_HOP) / E->audio->sample_rate, bits[0], over[0], worst[0], pre[0], bits[1], over[1], worst[1], pre[1],
                rule ? "short" : "long", pick ? "short" : "long");
    E->is_short[f] = (unsigned char)pick;
    derive_block_types(E->bt, E->is_short, g, derive_hi, nf);
    E->bt_trials++;
    if (pick) E->bt_trials_short++;
    if (pick != cheaper) E->bt_trials_quality++;
}

/* The planner's runner-up at the start of a block (experiment, MMX_TRIAL_PLAN=1;
   measured, it does not pay): the block (its first PLAN_TRIAL_FRAMES frames
   when it is longer) is coded with the plan's lineage and with the
   runner-up's (source windows advancing by one hop per frame, as in a block),
   and the whole block takes the cheaper lineage, the coded bits weighted by
   the chain depth as the planner weights its estimates, and never a deeper
   chain than planned. title A, level 3, strict: a 3-frame trial without
   the depth rules swapped 744 blocks and cost 1.6 % (deeper chains blocked
   later references, 74 more AUDIO frames, the roundtrip track exceeded its
   depth limit); with them, whole blocks, 416 swaps: +0.17 %; blocks up to 8
   frames: +0.13 %; a 3 % margin, 136 swaps: -0.04 %. The trial is exact (the
   swapped blocks code to the bit the trial measured, 202 of 242 swaps
   continue the previous block), the loss is indirect: the swapped block is
   a worse source for the frames that play from it later. In the tracker
   modes the pattern gains of a swapped block are unknown, so the slack for
   bands of a good pattern is not granted there (fewer pure bands, never
   more). */
static void plan_trial(Enc *E, unsigned long g)
{
    MMXAnalysis *a = E->analysis;
    MMXFramePlan *p = &a->plan[g], alt, saved[PLAN_TRIAL_FRAMES];
    unsigned long nf = a->frame_count, len = 1, n, k, coded, s;
    long long start = (long long)g * MMX_HOP - MMX_HOP;
    double bits_plan, bits_alt;
    int depth_plan, depth_alt;
    TrialState ts;

    if (!p->n_sources || !p->alt_n_sources || (g > 0 && same_lineage(&a->plan[g - 1], p)))
        return;
    while (g + len < nf && a->plan[g + len].n_sources && same_lineage(&a->plan[g + len - 1], &a->plan[g + len])) len++;
    memset(&alt, 0, sizeof(alt));
    alt.n_sources = p->alt_n_sources;
    alt.src_start[0] = p->alt_src_start[0];
    alt.src_start[1] = p->alt_src_start[1];
    depth_plan = mmx_refgraph_check(&E->graph, start, p->n_sources, p->src_start);
    depth_alt = mmx_refgraph_check(&E->graph, start, alt.n_sources, alt.src_start);
    if (depth_plan < 0 || depth_alt < 0 || depth_alt > depth_plan)   /* never a deeper chain than planned (star topology) */
        return;
    /* the runner-up must be valid for every frame of the block */
    for (k = 1; k < len; k++)
    {
        long long src[MMX_MAX_SOURCES];
        for (s = 0; s < alt.n_sources; s++) src[s] = alt.src_start[s] + (long long)k * MMX_HOP;
        if (mmx_refgraph_check(&E->graph, start + (long long)k * MMX_HOP, alt.n_sources, src) < 0)
            return;
    }
    n = len < PLAN_TRIAL_FRAMES ? len : PLAN_TRIAL_FRAMES;
    E->plan_trials++;
    bits_plan = trial_frames(E, g, n, &ts, &coded);
    trial_restore(E, &ts, coded);
    if (bits_plan < 0.0)
        return;
    memcpy(saved, &a->plan[g], sizeof(MMXFramePlan) * n);   /* the plan entries the runner-up overwrites */
    for (k = 0; k < n; k++)
    {
        MMXFramePlan *q = &a->plan[g + k];
        q->n_sources = alt.n_sources;
        for (s = 0; s < alt.n_sources; s++) q->src_start[s] = alt.src_start[s] + (long long)k * MMX_HOP;
    }
    bits_alt = trial_frames(E, g, n, &ts, &coded);
    trial_restore(E, &ts, coded);
    memcpy(&a->plan[g], saved, sizeof(MMXFramePlan) * n);
    if (bits_alt < 0.0 ||
        bits_alt * (1.0 + PLAN_DEPTH_PENALTY * (depth_alt - 1)) >= bits_plan * (1.0 + PLAN_DEPTH_PENALTY * (depth_plan - 1)))
        return;
    for (k = 0; k < len; k++)
    {
        MMXFramePlan *q = &a->plan[g + k];
        unsigned int e;
        q->n_sources = alt.n_sources;
        for (s = 0; s < alt.n_sources; s++) q->src_start[s] = alt.src_start[s] + (long long)k * MMX_HOP;
        q->alt_n_sources = 0;
        for (e = 0; e < MMX_EQ_BANDS; e++) q->band_gain_db[e] = 0.0f;
        if (E->run_gain)
            for (e = 0; e < MMX_EQ_BANDS; e++) E->run_gain[(g + k) * MMX_EQ_BANDS + e] = 0.0f;
    }
    E->plan_trials_alt++;
}

static int enc_init(Enc *E, const MMXAudioBuffer *audio, const MMXEncoderParams *params, MMXCodec *codec,
                    MMXAnalysis *analysis, MMXFile *out)
{
    unsigned long nf = analysis->frame_count;
    unsigned int c;
    unsigned char max_depth;
    const char *e;
    memset(E, 0, sizeof(*E));
    E->audio = audio;
    E->params = params;
    E->codec = codec;
    E->analysis = analysis;
    E->out = out;
    /* the rate loop's factor per band: thr_scale is the one scalar the
       bisection searched, the tilt shapes where it lands (see above) */
    tilt_profile_init(&E->tilt);
    tilt_band_scales(&E->tilt, &codec->bands, analysis->thr_scale > 0.0 ? analysis->thr_scale : 1.0,
                     E->bscale, MMX_MAX_BANDS);
    tilt_band_scales(&E->tilt, &codec->short_bands, analysis->thr_scale > 0.0 ? analysis->thr_scale : 1.0,
                     E->sbscale, MMX_SHORT_BANDS);
    E->relax_sub = pow(10.0, TRACKER_RELAX_SUB_DB / 10.0);
    E->pcm_scale = pow(2.0, (double)(audio->source_bits ? audio->source_bits : 16) - 1.0);
    E->calibrate = est_debug.on || (analysis->coded_estimate && second_pass_wanted(params));
    E->pure_db = params->mode >= MMX_MODE_FUTURE ? FUTURE_PURE_DB : TRACKER_PURE_DB;
    E->relax = pow(10.0, (params->mode >= MMX_MODE_FUTURE ? FUTURE_RELAX_DB : TRACKER_RELAX_DB) / 10.0);
    E->debug_bits = getenv("MMX_DEBUG_BITS") != NULL;   /* bit share per EQ region and frame class */
    E->debug_bt = getenv("MMX_DEBUG_BT_TRIAL") != NULL; /* every block-type decision with its bits and model numbers */
    /* M/S or L/R by coding only in strict mode: under the tracker's relaxed
       thresholds the L/R candidate fills a budget the estimate's M/S choice
       left unused (title A tracker, level 5: -0.37 % for +1.2 dB worst band,
       +0.13 % cells over, one more smeared attack) */
    E->trial_ms = ((e = getenv("MMX_TRIAL_MS")) ? atoi(e) != 0 : 1) && !params->mode && !rate_fast_probe;
    E->trial_bt = ((e = getenv("MMX_TRIAL_BT")) ? atoi(e) != 0 : 1) && !rate_fast_probe;
    E->trial_plan = (e = getenv("MMX_TRIAL_PLAN")) ? atoi(e) != 0 : 0;   /* measured, does not pay (see plan_trial) */
    E->bt_cells = (e = getenv("MMX_BT_TRIAL_CELLS")) ? (unsigned long)atol(e) : BT_TRIAL_CELLS;
    E->bt_db = (e = getenv("MMX_BT_TRIAL_DB")) ? atof(e) : BT_TRIAL_DB;
    E->bt = (unsigned char *)calloc(nf ? nf : 1, 1);
    E->is_short = (unsigned char *)calloc(nf ? nf : 1, 1);
    E->cand = (unsigned char *)calloc(nf ? nf : 1, 1);
    E->qwin = (float *)malloc(sizeof(float) * QSUB_WIN);
    E->qorig = (float *)malloc(sizeof(float) * QSUB_WIN / 2);
    E->qerr = (float *)malloc(sizeof(float) * QSUB_WIN / 2);
    E->qpsy = (MMXFramePsy *)malloc(sizeof(MMXFramePsy) * QSUB_MAX * MMX_MAX_CH);
    E->snap = (float *)malloc(sizeof(float) * (size_t)(PLAN_TRIAL_FRAMES + 2) * MMX_HOP * audio->channels);
    E->clip_save = (float *)malloc(sizeof(float) * MMX_WIN * audio->channels);
    if (!E->bt || !E->is_short || !E->cand || !E->qwin || !E->qorig || !E->qerr || !E->qpsy || !E->snap || !E->clip_save)
        return -1;
    if (clip_guard_init(&E->guard, audio->sample_rate, audio->source_bits, codec->quality) != 0)
        return -1;
    if (echo_guard_init(&E->echo, audio->sample_rate ? (double)audio->frame_count / audio->sample_rate : 0.0) != 0)
        return -1;
    /* experiment: only a share of the rate loop's dB offset is given to referenced
       frames (their residual is what leaks), the rest stays with the audio frames */
    E->ref_offset_factor = 1.0;
    if ((e = getenv("MMX_REF_OFFSET_SHARE")) != NULL && analysis->thr_scale > 1.0)
    {
        double share = atof(e);
        if (share < 0.0) share = 0.0;
        if (share > 1.0) share = 1.0;
        E->ref_offset_factor = pow(analysis->thr_scale, share - 1.0);
    }
    if (mmx_spectrum_init(&E->short_spec, 2 * MMX_SHORT_M) != 0 || mmx_spectrum_init(&E->qspec, QSUB_WIN) != 0 ||
        mmx_spectrum_init(&E->qpre_spec, MMX_PRE_ATTACK) != 0 || mmx_bands_init(&E->qbands, audio->sample_rate, QSUB_WIN / 2) != 0 ||
        mmx_bands_init(&E->qpre_bands, audio->sample_rate, MMX_PRE_ATTACK / 2) != 0)
        return -1;
    tilt_band_scales(&E->tilt, &E->qbands, analysis->thr_scale > 0.0 ? analysis->thr_scale : 1.0,
                     E->qbscale, MMX_MAX_BANDS);
    {
        const char *b = getenv("MMX_BLOCK_SWITCH");     /* MMX_BLOCK_SWITCH=0 codes every frame LONG (experiment) */
        decide_block_types(analysis, E->bt, E->is_short, E->cand, b == NULL || atoi(b) != 0, E->trial_bt);
    }
    pns_params_init(&E->pns, codec, params->pns_hz);
    if (params->is_hz && audio->channels == 2)
    {
        unsigned int b;
        for (b = mmx_is_first_band(&codec->bands); b < codec->bands.band_count; b++)
            if (codec->bands.band_hz[b] >= (float)params->is_hz) { E->is_first_band = b; break; }
    }
    E->is_rho_min = (e = getenv("MMX_IS_RHO")) ? atof(e) : IS_RHO_MIN;
    E->is_bias = (e = getenv("MMX_IS_BIAS")) ? atof(e) : IS_BIAS;
    E->is_slack = pow(10.0, ((e = getenv("MMX_IS_SLACK_DB")) ? atof(e) : IS_SLACK_DB) / 10.0);
    E->debug_is = getenv("MMX_DEBUG_IS") != NULL;
    E->debug_bandms = getenv("MMX_DEBUG_BANDMS") != NULL;
    E->t_is = (float *)calloc(MMX_HOP, sizeof(float));
    E->x_is = (float *)calloc(MMX_HOP, sizeof(float));
    if (!E->t_is || !E->x_is)
        return -1;
    nf_params_init(&E->nf, codec, params);
    keep_params_init(&E->keep, codec);
    if (params->mode)
    {
        E->run_gain = (float *)calloc((size_t)(nf ? nf : 1) * MMX_EQ_BANDS, sizeof(float));
        E->temporal = (PureTemporal *)malloc(sizeof(PureTemporal));
        if (!E->run_gain || !E->temporal || pure_temporal_init(E->temporal, audio->sample_rate) != 0)
            return -1;
        pattern_gains(analysis, E->run_gain);
    }
    /* experiment: a shallower chain limit once the rate loop's offset is large (deep
       chains accumulate the leak of every generation), MMX_REF_DEPTH_AT_DB + MMX_REF_MAX_DEPTH */
    {
        const char *at = getenv("MMX_REF_DEPTH_AT_DB"), *md = getenv("MMX_REF_MAX_DEPTH");
        double off = 10.0 * log10(analysis->thr_scale > 0.0 ? analysis->thr_scale : 1.0);
        max_depth = params->max_ref_depth;
        if (at && md && off > atof(at)) max_depth = (unsigned char)atoi(md);
    }
    if (mmx_audio_buffer_init(&E->decoded, audio->sample_rate, audio->channels, audio->frame_count) != 0 ||
        mmx_refgraph_init(&E->graph, nf, max_depth) != 0 ||
        mmx_frame_syntax_init(&E->syn, audio->channels, MMX_HOP) != 0 ||
        mmx_frame_syntax_init(&E->alt, audio->channels, MMX_HOP) != 0 ||
        mmx_frame_syntax_init(&E->best, audio->channels, MMX_HOP) != 0 ||
        mmx_frame_syntax_init(&E->coded, audio->channels, MMX_HOP) != 0)
        return -1;
    E->decoded.source_bits = audio->source_bits;
    for (c = 0; c < audio->channels; c++)
    {
        E->t_lr[c] = (float *)malloc(sizeof(float) * MMX_HOP);
        E->t_cod[c] = (float *)malloc(sizeof(float) * MMX_HOP);
        E->resid[c] = (float *)malloc(sizeof(float) * MMX_HOP);
        E->rec_cod[c] = (float *)malloc(sizeof(float) * MMX_HOP);
        E->tgt_cod[c] = (float *)malloc(sizeof(float) * MMX_HOP);
        if (!E->t_lr[c] || !E->t_cod[c] || !E->resid[c] || !E->rec_cod[c] || !E->tgt_cod[c])
            return -1;
    }
    return 0;
}

static void enc_free(Enc *E)
{
    unsigned int c;
    if (E->block.open) mmx_rc_enc_free(&E->block.rc);
    if (E->base.open) mmx_rc_enc_free(&E->base.rc);
    for (c = 0; c < MMX_MAX_CH; c++) { free(E->t_lr[c]); free(E->t_cod[c]); free(E->resid[c]); }
    free(E->t_is);
    free(E->x_is);
    free(E->bt);
    free(E->is_short);
    free(E->cand);
    free(E->qwin);
    free(E->qorig);
    free(E->qerr);
    free(E->qpsy);
    free(E->snap);
    free(E->clip_save);
    for (c = 0; c < MMX_MAX_CH; c++) { free(E->rec_cod[c]); free(E->tgt_cod[c]); }
    clip_guard_free(&E->guard);
    echo_guard_free(&E->echo);
    free(E->run_gain);
    if (E->temporal) { pure_temporal_free(E->temporal); free(E->temporal); }
    if (E->short_spec.n) mmx_spectrum_free(&E->short_spec);
    if (E->qspec.n) mmx_spectrum_free(&E->qspec);
    if (E->qpre_spec.n) mmx_spectrum_free(&E->qpre_spec);
    mmx_frame_syntax_free(&E->syn);
    mmx_frame_syntax_free(&E->alt);
    mmx_frame_syntax_free(&E->best);
    mmx_frame_syntax_free(&E->coded);
    mmx_refgraph_free(&E->graph);
    mmx_audio_buffer_free(&E->decoded);
}

static int encode_lossy(const MMXAudioBuffer *audio, const MMXEncoderParams *params, MMXCodec *codec,
                        MMXAnalysis *analysis, MMXFile *out, MMXStatistics *stats, MMXAudioBuffer *decoded_out)
{
    Enc *E = (Enc *)malloc(sizeof(Enc));
    unsigned long f, nf = analysis->frame_count;
    double bits;
    int rc = -1;

    if (!E)
        return -1;
    {
        int print = getenv("MMX_DEBUG_EST") != NULL;
        memset(&est_debug, 0, sizeof(est_debug));
        est_debug.on = print;
    }
    if (enc_init(E, audio, params, codec, analysis, out) != 0)
        goto done;
    if (E->nf.on)
        mmx_debug("Noise filling from %.0f Hz (EQ regions %u-7, zero weight %.2f/%.2f/%.2f/%.2f, fill %+.1f dB, holes %s, predicted regions %s)",
                  codec->bands.band_hz[E->nf.first_band], E->nf.first_region, E->nf.zero.alpha[4], E->nf.zero.alpha[5], E->nf.zero.alpha[6], E->nf.zero.alpha[7],
                  20.0 * log10(E->nf.gain), E->nf.hole_rel < 0.0 ? "down to the quality's own threshold" : "down to a fixed depth", E->nf.resid ? "filled" : "left alone");

    for (f = 0; f < nf; f++)
    {
        if (f + 1 < nf && E->cand[f + 1])
            bt_trial(E, f);
        if (E->trial_plan)
            plan_trial(E, f);
        if (frame_code(E, f, &E->block, stats, 0, &bits) != 0)
            goto done;
    }
    if (block_end(&E->block, out, stats) != 0)
    {
        mmx_error("Last block could not be finished");
        goto done;
    }
    baseline_segment(&E->base, stats, 0, 1);
    stats->mean_nmr_db = E->nmr_count ? E->nmr_sum / E->nmr_count : -200.0;
    if (E->debug_bits)
        mmx_frame_bitdump_print();
    if (E->debug_is)
        fprintf(stderr, "intensity: %lu bands with signal in the range (long stereo frames): %lu rejected by coherence (%.1f %%), %lu by the slack guard (%.1f %%), %lu by bits (%.1f %%), %lu intensity (%.1f %%)\n",
                E->is_dbg[0], E->is_dbg[1], E->is_dbg[0] ? 100.0 * E->is_dbg[1] / E->is_dbg[0] : 0.0, E->is_dbg[2], E->is_dbg[0] ? 100.0 * E->is_dbg[2] / E->is_dbg[0] : 0.0,
                E->is_dbg[3], E->is_dbg[0] ? 100.0 * E->is_dbg[3] / E->is_dbg[0] : 0.0, E->is_dbg[4], E->is_dbg[0] ? 100.0 * E->is_dbg[4] / E->is_dbg[0] : 0.0);
    if (E->debug_bandms)
        fprintf(stderr, "per-band M/S oracle: %lu long stereo AUDIO frames, band bits of the frame-level choice %.0f, per-band minimum %.0f (%.2f %% less)\n",
                E->bandms_frames, E->bandms_frame, E->bandms_oracle, E->bandms_frame > 0.0 ? 100.0 * (1.0 - E->bandms_oracle / E->bandms_frame) : 0.0);
    if (est_debug.on)
        est_debug_print();
    echo_report(&E->echo);
    mmx_info("Closed loop: %lu block types by trial (%lu short, %lu decided by the model), %lu blocks tried against the runner-up (%lu swapped), %lu noise bands reverted",
             E->bt_trials, E->bt_trials_short, E->bt_trials_quality, E->plan_trials, E->plan_trials_alt, E->pns_reverted);
    if (params->tns)
        mmx_info("TNS: %lu of %lu frames carry a filter", stats->tns_frames, stats->frames_by_sources[0] + stats->frames_by_sources[1] + stats->frames_by_sources[2]);
    if (getenv("MMX_PEAK_GUARD"))
        mmx_info("Peak guard: %lu bands coded again around their peak, %lu dead-zoned peaks pinned to +-1", peak_guard_restored, peak_guard_pinned);
    if (getenv("MMX_ENERGY_MATCH"))
        mmx_info("Energy match: %lu coded bands had their step raised to keep the band energy", energy_match_bands);
    rc = 0;

    if (decoded_out)
    {
        *decoded_out = E->decoded;
        memset(&E->decoded, 0, sizeof(E->decoded));
    }

done:
    enc_free(E);
    free(E);
    return rc;
}

/* --------------------------------------------------- lossless closed loop */

static long long to_int(float x, double scale, long long lim)
{
    double v = floor((double)x * scale + 0.5);
    if (v > lim) v = lim;
    if (v < -lim - 1) v = -lim - 1;
    return (long long)v;
}

/* The shift the lossless coder applies to every sample: the trailing bits that are zero in all of
   them, plus the bits --drop-bits rounds away. One function, so the analysis (which estimates bits
   at this resolution) and the coder agree. MMX_LL_NOSHIFT=1 disables the zero-bit part. */
static unsigned int lossless_shift(const MMXAudioBuffer *audio, const MMXEncoderParams *params,
                                   unsigned int *wasted_out, unsigned int *drop_out)
{
    unsigned int bits = audio->source_bits ? audio->source_bits : 16, w = 0, drop, shift;
    double scale = pow(2.0, (double)bits - 1.0);
    long long lim = (long long)scale - 1, all = 0;
    unsigned long i, n = (unsigned long)audio->frame_count * audio->channels;
    if (getenv("MMX_LL_NOSHIFT") == NULL)
        for (i = 0; i < n; i++)
            all |= audio->isamples ? (long long)audio->isamples[i] : to_int(audio->samples[i], scale, lim);
    if (all != 0)
        while (w < 16 && ((all >> w) & 1) == 0)
            w++;
    drop = params->drop_bits > 8 ? 8 : params->drop_bits;
    shift = w + drop;
    if (shift > bits - 2)
        shift = bits - 2;
    if (wasted_out) *wasted_out = w;
    if (drop_out) *drop_out = drop;
    return shift;
}

/* Near-lossless noise shaping, mode 3 (MMX_NL_SHAPE=3): the rounding noise of the dropped bits follows the
   signal's own spectral envelope. Per frame and channel an LPC of order MMX_NL_ORDER (16) is fitted to the
   input; the error feedback gives the total error the transfer A(z/g2) / A(z/g1) (MMX_NL_G2 = 0, MMX_NL_G1 = 0.9
   by default: noise ~ 1/|A(z/g1)|, i.e. under the formants, weaker where the signal is weak). Encoder only:
   the decoder multiplies the coded integers back. */
#define NL_ORDER_MAX 32
typedef struct
{
    double a1[MMX_MAX_CH][NL_ORDER_MAX + 1], b2[MMX_MAX_CH][NL_ORDER_MAX + 1];   /* A(z/g1), A(z/g2) per channel */
    double eq[MMX_MAX_CH][NL_ORDER_MAX], et[MMX_MAX_CH][NL_ORDER_MAX];           /* quantiser error and total error history */
    unsigned int order;
    double g1, g2;
} NlShaper;

static void nl_lpc(const float *x, unsigned long n, unsigned int stride, unsigned int p, double *a)
{
    double r[NL_ORDER_MAX + 1], e, k, tmp[NL_ORDER_MAX + 1];
    unsigned int i, j;
    unsigned long t;
    for (i = 0; i <= p; i++)
    {
        double acc = 0.0;
        for (t = i; t < n; t++) acc += (double)x[t * stride] * x[(t - i) * stride];
        r[i] = acc;
    }
    for (i = 0; i <= p; i++) a[i] = 0.0;
    a[0] = 1.0;
    if (r[0] <= 0.0) return;
    r[0] *= 1.0001;                                       /* white-noise correction, keeps A minimum phase */
    for (i = 1; i <= p; i++) r[i] *= exp(-0.5 * pow(2.0 * 3.14159265358979 * 60.0 * i / 44100.0, 2.0));   /* lag window */
    e = r[0];
    for (i = 1; i <= p; i++)
    {
        double acc = r[i];
        for (j = 1; j < i; j++) acc += a[j] * r[i - j];
        k = -acc / e;
        for (j = 1; j < i; j++) tmp[j] = a[j] + k * a[i - j];
        for (j = 1; j < i; j++) a[j] = tmp[j];
        a[i] = k;
        e *= (1.0 - k * k);
        if (e <= 0.0) break;
    }
}

static void nl_shaper_frame(NlShaper *sh, const MMXAudioBuffer *audio, long long start, unsigned long count)
{
    unsigned int c, i;
    double a[NL_ORDER_MAX + 1];
    for (c = 0; c < audio->channels; c++)
    {
        double g = 1.0;
        nl_lpc(audio->samples + (size_t)start * audio->channels + c, count, audio->channels, sh->order, a);
        for (i = 0; i <= sh->order; i++) { sh->a1[c][i] = a[i] * g; g *= sh->g1; }
        g = 1.0;
        for (i = 0; i <= sh->order; i++) { sh->b2[c][i] = sh->g2 > 0.0 ? a[i] * g : (i ? 0.0 : 1.0); g *= sh->g2; }
    }
}

/* one sample through the error feedback: returns the quantised value in the coder domain */
static long long nl_shape_sample(NlShaper *sh, unsigned int c, double x, double step, long long lo, long long hi)
{
    unsigned int i;
    double corr = 0.0, u, y, eq, et;
    long long vq;
    for (i = 1; i <= sh->order; i++)
        corr += sh->b2[c][i] * sh->eq[c][i - 1] - sh->a1[c][i] * sh->et[c][i - 1];
    u = x + corr;
    vq = llround(u / step);
    if (vq > hi) vq = hi;
    if (vq < lo) vq = lo;
    y = (double)vq * step;
    eq = y - u;
    et = y - x;
    for (i = sh->order; i-- > 1;) { sh->eq[c][i] = sh->eq[c][i - 1]; sh->et[c][i] = sh->et[c][i - 1]; }
    sh->eq[c][0] = eq;
    sh->et[c][0] = et;
    return vq;
}

/* Lossless with the LL2 core (revision 10, src/ll2codec.c): the same global references and block structure as
   encode_lossless, the frames coded by the shared frame processor, each block's residuals by the bitplane coder. */
/* The extra OLS memories for a main memory (header bytes 90, 91), from a measured matrix (twelve 30-s
   excerpts, the best pair measured around each main memory). MMX_LL2_XOLS="a,b" fixes them for experiments ("0" = none). */
static void ll2_xols_for(unsigned int decay, unsigned char *x)
{
    static const unsigned char rule[5][2] = { { 6, 11 }, { 7, 12 }, { 7, 13 }, { 8, 13 }, { 8, 10 } };   /* main 8..12 */
    const char *e = getenv("MMX_LL2_XOLS");
    x[0] = x[1] = 0;
    if (e)
    {
        int a = 0, b = 0;
        if (sscanf(e, "%d,%d", &a, &b) >= 1)
        {
            x[0] = (unsigned char)(a >= 6 && a <= 16 ? a : 0);
            x[1] = (unsigned char)(b >= 6 && b <= 16 ? b : 0);
        }
        return;
    }
    if (decay >= 8 && decay <= 12) { x[0] = rule[decay - 8][0]; x[1] = rule[decay - 8][1]; }
}

static unsigned long long container_bytes(const MMXFile *f, const MMXStatistics *s);

/* ---------------------------------------------------------- LL2: candidates of the per-file search */

/* What every candidate shares, read-only while they run: the coded integers and the blocks. */
typedef struct
{
    const MMXAudioBuffer *audio;
    const MMXEncoderParams *params;
    const MMXAnalysis *analysis;
    long long *dec[MMX_MAX_CH];
    unsigned long len, nf, *blk_start, nblk;
    unsigned int bits, shift;
    unsigned int pld;                   /* entry points every pld seconds (header byte 95), 0 = none */
    unsigned long pld_frames;           /* ... = every pld_frames frames: blocks start there */
    MMXStatistics stats;                /* the caller's statistics plus the block pass */
    int shared;                         /* dec belongs to another preparation (the no-reference variant) */
    MMXFramePlan *own_plan;             /* the no-reference variant's plan (all frames without sources) */
    MMXAnalysis plan_copy;              /* ... and the analysis view that points at it */
} Ll2Prep;

/* One candidate configuration and the file it produced. */
typedef struct
{
    unsigned int decay, bank;
    int extra;
    unsigned int own;                   /* OLS own taps (header byte 94), 0 = the profile's */
    MMXFile file;
    MMXStatistics stats;
    unsigned long long bytes;
    int rc, kept;
} Ll2Job;

static void ll2_prepare_free(Ll2Prep *p)
{
    unsigned int c;
    if (!p->shared) for (c = 0; c < MMX_MAX_CH; c++) free(p->dec[c]);
    free(p->blk_start);
    free(p->own_plan);
    memset(p, 0, sizeof(*p));
}

/* The same samples without any reference: every frame plain, blocks of MMX_LL2_BLOCK_FRAMES. At high sample rates
   the analysis finds approximate repeats in nearly every 1024-sample frame; their blocks shrink to a frame or two
   and cost far more than the repeats save (title A upsampled to 192 kHz: 92 % referenced, 7 % larger). */
static int ll2_prepare_noref(Ll2Prep *q, const Ll2Prep *p, const MMXStatistics *stats0)
{
    unsigned long f;
    *q = *p;
    q->shared = 1;
    q->stats = *stats0;
    q->blk_start = NULL;
    q->own_plan = (MMXFramePlan *)malloc(sizeof(MMXFramePlan) * (p->nf ? p->nf : 1));
    q->blk_start = (unsigned long *)malloc(sizeof(unsigned long) * (p->nf + 1));
    if (!q->own_plan || !q->blk_start) { free(q->own_plan); free(q->blk_start); q->own_plan = NULL; q->blk_start = NULL; return -1; }
    q->plan_copy = *p->analysis;
    q->plan_copy.plan = q->own_plan;
    q->analysis = &q->plan_copy;
    q->nblk = 0;
    for (f = 0; f < p->nf; f++)
    {
        q->own_plan[f] = p->analysis->plan[f];
        q->own_plan[f].n_sources = 0;
        q->own_plan[f].depth = 0;
        if (f % MMX_LL2_BLOCK_FRAMES == 0 || (p->pld_frames && f % p->pld_frames == 0)) q->blk_start[q->nblk++] = f;
        q->stats.frames_by_sources[0]++;
        q->stats.depth_histogram[0]++;
    }
    q->blk_start[q->nblk] = p->nf;
    return 0;
}

/* The integers (after the shift every sample shares) and the blocks: a block is a run of frames with one reference
   lineage, at most MMX_LL2_BLOCK_FRAMES long; the reference checks go frame by frame. Channel 1's future taps stop at
   the block end, so the block ends must be known before any frame is coded. */
static int ll2_prepare(Ll2Prep *p, const MMXAudioBuffer *audio, const MMXEncoderParams *params, MMXAnalysis *analysis,
                       const MMXStatistics *stats0)
{
    MMXReferenceGraph graph;
    unsigned int c, drop = 0, w = 0;
    double scale;
    long long lim;
    unsigned long f, n;
    const MMXFramePlan *prev = NULL;
    const int pld_strict = params->pld_strict || (getenv("MMX_PLD_STRICT") && atoi(getenv("MMX_PLD_STRICT")));
    memset(p, 0, sizeof(*p));
    memset(&graph, 0, sizeof(graph));
    p->audio = audio; p->params = params; p->analysis = analysis; p->stats = *stats0;
    p->bits = audio->source_bits ? audio->source_bits : 16;
    {   /* entry points for patchwork landscape decoding: every 30 s (measured on three full CD titles:
           +0.017..+0.043 %, 1 kB per entry point; references across segments are kept - cutting them cost +4.8 %
           on title A); MMX_PLD=seconds, 0 = none */
        const char *e = getenv("MMX_PLD");
        int v = e ? atoi(e) : 30;
        p->pld = (unsigned int)(v < 0 ? 0 : v > 255 ? 255 : v);
        p->pld_frames = mmx_ll2_pld_frames(p->pld, audio->sample_rate);
    }
    p->len = (unsigned long)audio->frame_count;
    p->nf = analysis->frame_count;
    scale = pow(2.0, (double)p->bits - 1.0);
    lim = (long long)scale - 1;
    if (audio->channels > 2) return -1;
    p->shift = lossless_shift(audio, params, &w, &drop);
    if (drop || params->drop_step > 1)
    {
        mmx_error("The LL2 core is lossless only (no --drop-bits / --drop-step)");
        return -1;
    }
    for (c = 0; c < audio->channels; c++)
    {
        p->dec[c] = (long long *)malloc(sizeof(long long) * (p->len ? p->len : 1));
        if (!p->dec[c]) { ll2_prepare_free(p); return -1; }
        for (n = 0; n < p->len; n++)
            p->dec[c][n] = (audio->isamples ? (long long)audio->isamples[(size_t)n * audio->channels + c]      /* 32-bit: exact */
                                            : to_int(audio->samples[(size_t)n * audio->channels + c], scale, lim)) >> p->shift;
    }
    if (p->shift)
        mmx_info("Lossless coder works on %u of %u bits (%u were zero in every sample)", p->bits - p->shift, p->bits, w);
    p->blk_start = (unsigned long *)malloc(sizeof(unsigned long) * (p->nf + 1));
    if (!p->blk_start || mmx_refgraph_init(&graph, p->nf, params->max_ref_depth) != 0) { ll2_prepare_free(p); return -1; }
    for (f = 0; f < p->nf; f++)
    {
        MMXFramePlan *plan = &analysis->plan[f];
        long long start = (long long)f * MMX_HOP - MMX_HOP;
        unsigned int n_sources = plan->n_sources;
        int depth = mmx_refgraph_check(&graph, start, n_sources, plan->src_start), outside = 0;
        if (pld_strict && p->pld_frames && n_sources)
        {   /* --pld: a source (with the margin the source stage reads around it, LL2_SRC_MARGIN in src/decoder.c) must
               lie inside the frame's own segment */
            const long long seg0 = (long long)(f / p->pld_frames * p->pld_frames) * MMX_HOP - MMX_HOP;
            unsigned int k2;
            for (k2 = 0; k2 < n_sources && k2 < 2; k2++)
                if (seg0 > 0 && plan->src_start[k2] - 256 < seg0) outside = 1;
        }
        if (outside || depth < 0 || (n_sources && plan->src_start[0] < 0) || (n_sources > 1 && plan->src_start[1] < 0))
        {
            n_sources = 0;
            plan->n_sources = 0;
            depth = 0;
        }
        plan->depth = (unsigned char)depth;
        if (!prev || !same_lineage(prev, plan) || f - p->blk_start[p->nblk - 1] >= MMX_LL2_BLOCK_FRAMES
            || (p->pld_frames && f % p->pld_frames == 0))
            p->blk_start[p->nblk++] = f;
        prev = plan;
        mmx_refgraph_set(&graph, f, (unsigned char)depth, n_sources, plan->src_start);
        p->stats.frames_by_sources[n_sources]++;
        if ((unsigned)depth > p->stats.max_depth_used) p->stats.max_depth_used = (unsigned)depth;
        if (depth < 8) p->stats.depth_histogram[depth]++;
    }
    p->blk_start[p->nblk] = p->nf;
    mmx_refgraph_free(&graph);
    return 0;
}

/* Codes the whole file with one configuration into j->file; reads only the shared preparation. */
static int ll2_code(const Ll2Prep *p, Ll2Job *j)
{
    MMXLl2Stream ls;
    MMXBlockEntry entry;
    MMXRangeEncoder rc;
    MMXFile *out = &j->file;
    unsigned long b, f;
    int rc_open = 0, ret = -1;
    mmx_file_init(out);
    fill_header(out, p->audio, p->params);
    j->stats = p->stats;
    out->ll2_xols[0] = out->ll2_xols[1] = 0;
    if (j->extra) ll2_xols_for(j->decay, out->ll2_xols);
    out->ll2_profile = (unsigned char)mmx_ll2_profile_for(p->bits, p->audio->sample_rate, p->audio->channels);
    out->ll2_own = (unsigned char)j->own;
    out->ll2_pld = (unsigned char)p->pld;
    if (mmx_ll2s_init(&ls, p->audio->channels, j->decay, j->bank, out->ll2_xols, out->ll2_profile, j->own) != 0)
        goto done;
    out->ll2_decay = (unsigned char)j->decay;
    out->ll2_bank = (unsigned char)j->bank;
    {   /* four Huber experts in the mixer (MMX_LL2_MIX=0: the two-expert mixer, for comparisons) */
        const char *em = getenv("MMX_LL2_MIX");
        out->ll2_mix = (unsigned char)(em && atoi(em) == 0 ? 0 : 1);
    }
    out->ll_shift = (unsigned char)p->shift;
    out->ll_dropped = 0;
    out->ll_core = MMX_LL2_CORE;
    if (mmx_ll2s_block_begin(&ls, (unsigned long)MMX_HOP * MMX_LL2_BLOCK_FRAMES) != 0) goto done;
    mmx_ll2s_set_rails(&ls, p->bits, p->shift);
    mmx_ll2s_set_rate(&ls, p->audio->sample_rate);
    mmx_ll2s_set_mixer(&ls, out->ll2_mix, p->bits, p->shift);

    /* block by block; the residuals of a block follow it in its payload */
    for (b = 0; b < p->nblk; b++)
    {
        const MMXFramePlan *p0 = &p->analysis->plan[p->blk_start[b]];
        if (p->pld_frames && b > 0 && p->blk_start[b] % p->pld_frames == 0)
        {   /* an entry point: predictor and coder start afresh, the regressors see nothing before this block
               (references to earlier audio stay allowed) - what a decoder entering here needs */
            mmx_ll2s_free(&ls);
            if (mmx_ll2s_init(&ls, p->audio->channels, j->decay, j->bank, out->ll2_xols, out->ll2_profile, j->own) != 0
                || mmx_ll2s_block_begin(&ls, (unsigned long)MMX_HOP * MMX_LL2_BLOCK_FRAMES) != 0)
                goto done;
            mmx_ll2s_set_rails(&ls, p->bits, p->shift);
            mmx_ll2s_set_rate(&ls, p->audio->sample_rate);
            mmx_ll2s_set_mixer(&ls, out->ll2_mix, p->bits, p->shift);
            ls.seg_start = (long long)p->blk_start[b] * MMX_HOP - MMX_HOP;
        }
        long long front = (long long)p->blk_start[b + 1] * MMX_HOP - MMX_HOP;   /* the block end */
        if (front > (long long)p->len) front = (long long)p->len;
        memset(&entry, 0, sizeof(entry));
        entry.start_frame = p->blk_start[b];
        entry.frame_count = p->blk_start[b + 1] - p->blk_start[b];
        entry.n_sources = p0->n_sources;
        entry.src_start[0] = p0->src_start[0];
        entry.src_start[1] = p0->src_start[1];
        entry.depth = p0->depth;
        mmx_rc_enc_init(&rc);
        rc_open = 1;
        ls.used = 0;
        for (f = p->blk_start[b]; f < p->blk_start[b + 1]; f++)
        {
            const MMXFramePlan *plan = &p->analysis->plan[f];
            long long start = (long long)f * MMX_HOP - MMX_HOP, src = plan->n_sources ? plan->src_start[0] : -1;
            unsigned long count = start < 0 ? 0 : (start + MMX_HOP <= (long long)p->len ? MMX_HOP : (unsigned long)((long long)p->len - start));
            if (count)
                mmx_ll2s_frame(&ls, (long long *const *)p->dec, start, count, src, front, 0);
            else
                mmx_ll2s_skip_frame(&ls, src);
        }
        mmx_ll2s_finish_block(&ls);
        mmx_ll2s_encode_block(&ls, &rc);
        mmx_rc_enc_finish(&rc);
        if (rc.failed) goto done;
        entry.payload = rc.data; entry.payload_size = rc.size; entry.crc32 = mmx_crc32(rc.data, rc.size);
        rc.data = NULL; mmx_rc_enc_free(&rc);
        rc_open = 0;
        j->stats.payload_bytes[entry.n_sources] += entry.payload_size;
        j->stats.block_count++;
        if (mmx_file_add_block(out, &entry) < 0) goto done;
    }
    j->bytes = container_bytes(out, &j->stats);
    ret = 0;
done:
    if (rc_open) mmx_rc_enc_free(&rc);
    mmx_ll2s_free(&ls);
    j->rc = ret;
    return ret;
}

typedef struct { const Ll2Prep *p; Ll2Job *jobs; } Ll2Run;
static void ll2_job_task(void *ctx, unsigned long i, unsigned int worker)
{
    Ll2Run *r = (Ll2Run *)ctx;
    (void)worker;
    ll2_code(r->p, &r->jobs[i]);
}

/* Runs n candidates at once (the encoder's thread pool; every candidate writes only its own job). */
static int ll2_run_jobs(const Ll2Prep *p, Ll2Job *jobs, unsigned long n)
{
    Ll2Run r;
    unsigned long i;
    r.p = p; r.jobs = jobs;
    if (n == 1) ll2_code(p, &jobs[0]);
    else if (n > 1) mmx_parallel_for(n, ll2_job_task, &r);
    for (i = 0; i < n; i++) if (jobs[i].rc != 0) return -1;
    return 0;
}

/* The per-file search on one preparation (see the LL2 branch in mmx_encoder_encode for the candidates): round 1 =
   the memories with their extra OLS and the longest without extras, round 2 = the best without extras and the bank
   on the extras winner, the plain best and the plain longest; the smallest file wins (earlier candidates win ties).
   Returns the winner's index in jobs (their files stay allocated) or -1. */
static int ll2_search(const Ll2Prep *prep, const unsigned int *cand, unsigned int nc, int plain, int try_bank, int force_bank,
                      Ll2Job *jobs, unsigned int *njobs, const char *label)
{
    unsigned int k, n1, n2, longest = 0, pstar;
    int bext = -1, plong = -1, pbest = -1, cur, win, q, bestbank = -1;
    int bank_of[3] = { -1, -1, -1 };
    const char *only = getenv("MMX_LL2_ONLY");         /* lab: "decay,bank,extra[,own]" codes exactly this configuration */
    const char *eo = getenv("MMX_LL2_OWNS");
    unsigned int owns[4], no = 0;
    if (only)
    {
        int d = 11, bk = 0, ex = 1, ow = 0;
        sscanf(only, "%d,%d,%d,%d", &d, &bk, &ex, &ow);
        jobs[0].decay = (unsigned int)d; jobs[0].bank = (unsigned int)(bk != 0); jobs[0].extra = ex != 0;
        jobs[0].own = (unsigned int)(ow > 0 ? ow : 0);
        *njobs = 1;
        if (ll2_run_jobs(prep, jobs, 1) != 0 || jobs[0].rc != 0) return -1;
        mmx_info("  %sonly memory 2^%d, bank %d, extra %d, own %d: %llu bytes", label, d, bk, ex, ow, jobs[0].bytes);
        return 0;
    }
    /* (4) the hi-res profile: the winner with longer OLS on the own past (MMX_LL2_OWNS, default 20,24,32; 0 = skip) -
       at high rates 16 taps span little time (measured 384 kHz: 32 taps -1.0 %, 88.2 kHz: 20 taps -0.05 %) */
    if (mmx_ll2_profile_for(prep->bits, prep->audio->sample_rate, prep->audio->channels) == MMX_LL2_PROFILE_HIRES)
    {
        char ob[32];
        char *t;
        snprintf(ob, sizeof(ob), "%s", eo ? eo : "20,24,32");
        for (t = strtok(ob, ","); t && no < 4; t = strtok(NULL, ","))
            if (atoi(t) >= 4 && atoi(t) <= MMX_LL2_OWN_MAX) owns[no++] = (unsigned int)atoi(t);
    }
    for (k = 0; k < nc; k++) if (cand[k] > longest) longest = cand[k];
    for (k = 0; k < 16; k++) jobs[k].own = 0;
    for (k = 0; k < nc; k++) { jobs[k].decay = cand[k]; jobs[k].bank = 0; jobs[k].extra = 1; }
    n1 = nc;
    if (plain) { plong = (int)n1; jobs[n1].decay = longest; jobs[n1].bank = 0; jobs[n1].extra = 0; n1++; }
    *njobs = n1;
    if (ll2_run_jobs(prep, jobs, n1) != 0) return -1;
    for (k = 0; k < nc; k++)
    {
        unsigned char x_[2];
        ll2_xols_for(jobs[k].decay, x_);
        mmx_info("  %sOLS memory 2^%u (extra 2^%u, 2^%u): %llu bytes", label, jobs[k].decay, x_[0], x_[1], jobs[k].bytes);
        if (bext < 0 || jobs[k].bytes < jobs[bext].bytes) bext = (int)k;
    }
    pstar = jobs[bext].decay;
    n2 = n1;
    if (plain)
    {
        if (pstar == longest) pbest = plong;
        else { pbest = (int)n2; jobs[n2].decay = pstar; jobs[n2].bank = 0; jobs[n2].extra = 0; n2++; }
    }
    if (try_bank)
    {   /* the bank on every configuration that can be the best (the bank can reorder them) */
        bank_of[0] = (int)n2; jobs[n2].decay = pstar; jobs[n2].bank = 1; jobs[n2].extra = 1; n2++;
        if (plain) { bank_of[1] = (int)n2; jobs[n2].decay = pstar; jobs[n2].bank = 1; jobs[n2].extra = 0; n2++; }
        if (plain && longest != pstar) { bank_of[2] = (int)n2; jobs[n2].decay = longest; jobs[n2].bank = 1; jobs[n2].extra = 0; n2++; }
    }
    *njobs = n2;
    if (ll2_run_jobs(prep, jobs + n1, n2 - n1) != 0) return -1;
    cur = bext;
    if (plain)
    {
        mmx_info("  %swithout extra OLS: 2^%u: %llu bytes", label, jobs[pbest].decay, jobs[pbest].bytes);
        if (jobs[pbest].bytes < jobs[cur].bytes) cur = pbest;
        if (pbest != plong)
        {
            mmx_info("  %swithout extra OLS: 2^%u: %llu bytes", label, jobs[plong].decay, jobs[plong].bytes);
            if (jobs[plong].bytes < jobs[cur].bytes) cur = plong;
        }
    }
    win = cur;
    for (q = 0; q < 3; q++)
    {
        if (bank_of[q] < 0) continue;
        mmx_info("  %sfilter bank at 2^%u (extra %s): %llu bytes", label, jobs[bank_of[q]].decay,
                 jobs[bank_of[q]].extra ? "on" : "off", jobs[bank_of[q]].bytes);
        if (bestbank < 0 || jobs[bank_of[q]].bytes < jobs[bestbank].bytes) bestbank = bank_of[q];
    }
    if (bestbank >= 0 && (jobs[bestbank].bytes < jobs[cur].bytes || force_bank)) win = bestbank;
    if (no)
    {
        unsigned int n3 = n2;
        for (k = 0; k < no && n3 < 16; k++)
        {
            jobs[n3] = jobs[win];
            memset(&jobs[n3].file, 0, sizeof(jobs[n3].file));
            jobs[n3].kept = 0;
            jobs[n3].own = owns[k];
            n3++;
        }
        *njobs = n3;
        if (ll2_run_jobs(prep, jobs + n2, n3 - n2) != 0) return -1;
        for (k = n2; k < n3; k++)
        {
            mmx_info("  %sOLS own taps %u: %llu bytes", label, jobs[k].own, jobs[k].bytes);
            if (jobs[k].bytes < jobs[win].bytes) win = (int)k;
        }
    }
    return win;
}

static int encode_lossless(const MMXAudioBuffer *audio, const MMXEncoderParams *params,
                           MMXAnalysis *analysis, MMXFile *out, MMXStatistics *stats, MMXAudioBuffer *decoded_out)
{
    MMXReferenceGraph graph;
    BlockWriter block, base;
    long long *dec[MMX_MAX_CH], *x[MMX_MAX_CH], *srcp[MMX_MAX_CH], *srcp2[MMX_MAX_CH];
    unsigned int bits = audio->source_bits ? audio->source_bits : 16, c, shift = 0, drop = 0, qdrop = params->drop_step > 1 ? params->drop_step : 1;
    double scale = pow(2.0, (double)bits - 1.0), step;
    long long lim = (long long)scale - 1;
    unsigned long f, nf = analysis->frame_count, n, len = (unsigned long)audio->frame_count;
    MMXLosslessFrame fr;
    int rc = -1, nl_shape;
    NlShaper *shaper = NULL;
    {
        const char *e = getenv("MMX_NL_SHAPE");
        nl_shape = e ? atoi(e) : 0;
        if (nl_shape == 3)
        {
            shaper = (NlShaper *)calloc(1, sizeof(*shaper));
            if (!shaper) return -1;
            shaper->order = (e = getenv("MMX_NL_ORDER")) ? (unsigned int)atoi(e) : 16;
            if (shaper->order > NL_ORDER_MAX) shaper->order = NL_ORDER_MAX;
            shaper->g1 = (e = getenv("MMX_NL_G1")) ? atof(e) : 0.9;
            shaper->g2 = (e = getenv("MMX_NL_G2")) ? atof(e) : 0.0;
        }
    }

    memset(&graph, 0, sizeof(graph));
    memset(&block, 0, sizeof(block));
    memset(&base, 0, sizeof(base));
    memset(dec, 0, sizeof(dec));
    memset(x, 0, sizeof(x));
    if (mmx_refgraph_init(&graph, nf, params->max_ref_depth) != 0)
        goto done;
    for (c = 0; c < audio->channels; c++)
    {
        dec[c] = (long long *)malloc(sizeof(long long) * (len ? len : 1));
        x[c] = (long long *)malloc(sizeof(long long) * MMX_HOP);
        if (!dec[c] || !x[c])
            goto done;
    }
    /* the filter cascade of the lossless coder is carried across blocks */
    mmx_ll_state_init(&block.llctx);
    mmx_ll_state_init(&base.llctx);

    /* One shift for the whole file: the trailing bits every sample has clear, plus the bits
       --drop-bits rounds away. The coder works on x >> shift and the decoder shifts back
       (metadata ll_shift), so cascade and history never see a scale jump. The coder does not
       see zero bits on its own - measured on title A: 1/2/3 low bits cleared coded
       at 100.0/100.1/100.4 % of full size, the shifted signal at 79.2/69.1/59.2 % - and a
       24-bit signal in a 32-bit container paid eight dead bits. MMX_LL_NOSHIFT=1 disables it. */
    {
        unsigned int w = 0;
        shift = lossless_shift(audio, params, &w, &drop);
        out->ll_shift = (unsigned char)shift;
        out->ll_dropped = (unsigned char)drop;
        out->ll_qdrop = (unsigned char)(qdrop > 1 ? qdrop : 0);
        if (shift || qdrop > 1)
        {
            mmx_info("Lossless coder works on %u of %u bits (%u were zero in every sample%s)%s", bits - shift, bits, w,
                     drop ? ", the rest rounded away on request" : "", qdrop > 1 ? " with an extra step multiplier" : "");
            if (qdrop > 1) mmx_info("Near-lossless step %llu = 2^%.2f LSB (%u bits + x%u)%s", (unsigned long long)((1LL << shift) * qdrop) >> (shift - drop),
                                    log((double)(1LL << drop) * qdrop) / log(2.0), drop, qdrop, nl_shape == 3 ? ", LPC-shaped rounding" : nl_shape ? ", shaped rounding" : "");
        }
    }
    step = (double)(1LL << shift) * (double)qdrop;

    for (f = 0; f < nf; f++)
    {
        MMXFramePlan *plan = &analysis->plan[f];
        long long start = (long long)f * MMX_HOP - MMX_HOP;   /* first owned sample */
        unsigned int n_sources = plan->n_sources;
        unsigned long count;
        int depth, used = 0;

        depth = mmx_refgraph_check(&graph, start, n_sources, plan->src_start);
        if (depth < 0 || (n_sources && plan->src_start[0] < 0) || (n_sources > 1 && plan->src_start[1] < 0))
        {
            n_sources = 0;
            plan->n_sources = 0;
            depth = 0;
        }
        /* two sources reach the lossless coder since revision 7: a second broadband gain fitted on
           what the first leaves, and a second side stage (see mmx_ll_encode_frame) */
        plan->depth = (unsigned char)depth;
        {   /* lab: MMX_PLAN_DUMP=file writes the final reference plan (frame, first owned sample, sources) */
            static FILE *pd = NULL;
            static int pd_init = 0;
            if (!pd_init) { const char *e = getenv("MMX_PLAN_DUMP"); pd = e ? fopen(e, "w") : NULL; pd_init = 1; }
            if (pd)
            {
                fprintf(pd, "%lu %lld %u %lld %lld\n", f, start, n_sources, n_sources > 0 ? plan->src_start[0] : -1LL,
                        n_sources > 1 ? plan->src_start[1] : -1LL);
                if (f + 1 == nf) fflush(pd);
            }
        }

        if (!block.open || f == 0 || !same_lineage(&analysis->plan[f - 1], plan))
        {
            if (block_end(&block, out, stats) != 0)
                goto done;
            block_begin(&block, f, plan);
        }
        baseline_segment(&base, stats, f, 0);

        count = start < 0 ? 0 : (start + MMX_HOP <= (long long)len ? MMX_HOP : (unsigned long)((long long)len - start));
        block.entry.frame_count++;
        if (count == 0)
        {
            mmx_refgraph_set(&graph, f, (unsigned char)depth, n_sources, plan->src_start);
            stats->frames_by_sources[n_sources]++;
            continue;
        }
        if (shaper && (drop || qdrop > 1))
            nl_shaper_frame(shaper, audio, start, count);
        for (c = 0; c < audio->channels; c++)
        {
            for (n = 0; n < count; n++)
            {
                long long v = to_int(audio->samples[(size_t)(start + n) * audio->channels + c], scale, lim);
                if (shift || qdrop > 1)
                {
                    long long hi = ((lim + 1) >> shift) / (long long)qdrop, lo = -hi;   /* +2^(bits-1-shift) is fine: the sink saturates */
                    /* MMX_NL_SHAPE=1|2: noise-shaped rounding of the dropped bits (error feedback, 1 = first order
                       1 - z^-1, 2 = second order (1 - z^-1)^2); 3 = LPC-weighted, see NlShaper. The rounding noise
                       moves where the ear is least sensitive; same size, the decoder only scales back. The
                       "max error +-2^(drop-1) LSB" label no longer holds for a shaped file. */
                    static double e1[MMX_MAX_CH], e2[MMX_MAX_CH];
                    if ((drop || qdrop > 1) && nl_shape == 3)
                        v = nl_shape_sample(shaper, c, (double)v, step, lo, hi);
                    else if ((drop || qdrop > 1) && nl_shape > 0)
                    {
                        double u = (double)v - (nl_shape >= 2 ? 2.0 * e1[c] - e2[c] : e1[c]);
                        long long vq = llround(u / step);
                        if (vq > hi) vq = hi;
                        if (vq < lo) vq = lo;
                        e2[c] = e1[c];
                        e1[c] = (double)vq * step - u;
                        v = vq;
                    }
                    else if (qdrop > 1)
                    {
                        long long vq = llround((double)v / step);
                        if (vq > hi) vq = hi;
                        if (vq < lo) vq = lo;
                        v = vq;
                    }
                    else
                    {
                        v = drop ? (v + (1LL << (shift - 1))) >> shift : v >> shift;   /* round only what is dropped */
                        if (v > hi) v = hi;
                        if (v < lo) v = lo;
                    }
                }
                x[c][n] = v;
            }
            srcp[c] = n_sources ? dec[c] + plan->src_start[0] : NULL;
            srcp2[c] = n_sources > 1 ? dec[c] + plan->src_start[1] : NULL;
        }
        if (mmx_ll_encode_frame(&block.rc, &block.llctx, audio->channels, x, n_sources ? srcp : NULL,
                                n_sources > 1 ? srcp2 : NULL, count, &fr) != 0)
        {
            mmx_error("Lossless coder failed at frame %lu", f);
            goto done;
        }
        {
            MMXLosslessFrame bfr;
            mmx_ll_encode_frame(&base.rc, &base.llctx, audio->channels, x, NULL, NULL, count, &bfr);
        }
        for (c = 0; c < audio->channels; c++)
        {
            memcpy(dec[c] + start, x[c], sizeof(long long) * count);
            used |= fr.use_pred[c];
        }
        mmx_refgraph_set(&graph, f, (unsigned char)depth, n_sources, plan->src_start);
        stats->frames_by_sources[n_sources]++;
        if (n_sources && !used) stats->ref_frames_gains_off++;
        if (fr.stereo_ms) stats->stereo_ms_frames++;
        if ((unsigned)depth > stats->max_depth_used) stats->max_depth_used = (unsigned)depth;
        if (depth < 8) stats->depth_histogram[depth]++;
    }
    if (block_end(&block, out, stats) != 0)
        goto done;
    baseline_segment(&base, stats, 0, 1);
    stats->worst_nmr_db = -200.0;
    stats->mean_nmr_db = -200.0;
    rc = 0;

    if (decoded_out)
    {
        if (mmx_audio_buffer_init(decoded_out, audio->sample_rate, audio->channels, audio->frame_count) == 0)
        {
            unsigned long i;
            decoded_out->source_bits = audio->source_bits;
            for (c = 0; c < audio->channels; c++)
                for (i = 0; i < len; i++)
                    decoded_out->samples[(size_t)i * audio->channels + c] = (float)((double)(dec[c][i] * (1LL << shift)) / scale);
        }
    }

done:
    free(shaper);
    if (block.open) mmx_rc_enc_free(&block.rc);
    if (base.open) mmx_rc_enc_free(&base.rc);
    for (c = 0; c < MMX_MAX_CH; c++) { free(dec[c]); free(x[c]); }
    mmx_refgraph_free(&graph);
    return rc;
}

/* ---------------------------------------------------------- clean samples */

#define CLEAN_ALPHA 2.0   /* thresholds of a source band scale by (1 + importance)^-CLEAN_ALPHA */

/* "Clean samples": a MOD file stores every sample once and cleanly, the
   patterns only play it. Here the material that later parts play from is
   coded finer, so that the repeats predict better and play pure - at a low
   bitrate the residual of a repeat is mostly the quantization noise of its
   source, not the difference between the takes. Per EQ band the residual of
   a referencing frame g is modelled as D + N: D = what stays after a clean
   source (band energy / plan gain), N = the noise the source windows carry
   (sum of min(threshold, energy) over the frames under the window, weighted
   by w^2 of the synthesis window, i.e. by the share of each frame's noise
   inside the window). The importance of source frame f in band e is the sum
   over the frames g reading it of the share of g's residual that is f's
   noise, times 1 + the importance of g (a source of a source counts through
   the chain). Sources always lie before their target, so the chain sums are
   complete when the frames are visited from the end. The masking thresholds
   of band e of frame f are then scaled by (1 + importance)^-alpha. The
   thresholds enter the noise model, so this runs after the rate loop's
   offset: the coarser the coding, the more the sources matter. */
static void clean_samples(MMXAnalysis *a, const MMXCodec *codec, double alpha)
{
    unsigned long nf = a->frame_count, f, i;
    unsigned int c, b, e, s;
    double *imp, *noise, cum[MMX_WIN + 1];
    if (alpha <= 0.0 || !nf)
        return;
    imp = (double *)calloc((size_t)nf * MMX_EQ_BANDS, sizeof(double));
    noise = (double *)calloc((size_t)nf * MMX_EQ_BANDS, sizeof(double));
    if (!imp || !noise)
    {
        free(imp);
        free(noise);
        return;
    }
    cum[0] = 0.0;
    for (i = 0; i < MMX_WIN; i++)
        cum[i + 1] = cum[i] + codec->mdct.window[i] * codec->mdct.window[i];
    /* quantization noise of every frame per EQ band: the threshold, or the band itself when it is dropped */
    for (f = 0; f < nf; f++)
        for (c = 0; c < a->channels; c++)
        {
            const MMXFramePsy *p = &a->psy[f * a->channels + c];
            for (b = 0; b < codec->cutoff_band; b++)
            {
                double n = (double)(codec->bands.band_start[b + 1] - codec->bands.band_start[b]) * p->thr[b];
                if (p->thr[b] >= 1e29f)
                    continue;
                noise[f * MMX_EQ_BANDS + codec->bands.eq_band[b]] += n < p->energy[b] ? n : p->energy[b];
            }
        }
    for (f = nf; f-- > 0;)
    {
        const MMXFramePlan *p = &a->plan[f];
        double resid[MMX_EQ_BANDS], share[MMX_MAX_SOURCES][4];
        long long first[MMX_MAX_SOURCES], g;
        if (!p->n_sources)
            continue;
        /* D per band, and the share of every frame under the source windows */
        for (e = 0; e < MMX_EQ_BANDS; e++)
        {
            double energy = 0.0;
            for (c = 0; c < a->channels; c++)
                for (b = 0; b < codec->cutoff_band; b++)
                    if (codec->bands.eq_band[b] == e)
                        energy += a->psy[f * a->channels + c].energy[b];
            resid[e] = energy * pow(10.0, -(double)p->band_gain_db[e] / 10.0);
        }
        for (s = 0; s < p->n_sources; s++)
        {
            long long src = p->src_start[s];
            if (src < -(long long)MMX_HOP)
                src = -(long long)MMX_HOP;
            first[s] = (long long)((src + MMX_HOP) / MMX_HOP) - 1;   /* first g with g*HOP+HOP > src; at most 4 frames overlap */
            for (i = 0; i < 4; i++)
            {
                long long ws = (first[s] + (long long)i) * (long long)MMX_HOP - MMX_HOP;
                long long lo = ws > src ? ws : src, hi = ws + MMX_WIN < src + MMX_WIN ? ws + MMX_WIN : src + MMX_WIN;
                share[s][i] = hi > lo ? (cum[hi - ws] - cum[lo - ws]) / (double)MMX_HOP : 0.0;
            }
            for (i = 0; i < 4; i++)
            {
                g = first[s] + (long long)i;
                if (g < 0 || (unsigned long)g >= nf)
                    continue;
                for (e = 0; e < MMX_EQ_BANDS; e++)
                    resid[e] += share[s][i] * noise[(unsigned long)g * MMX_EQ_BANDS + e] / p->n_sources;
            }
        }
        /* the share of the residual that is the source frame's noise, weighted by the chain */
        for (s = 0; s < p->n_sources; s++)
            for (i = 0; i < 4; i++)
            {
                g = first[s] + (long long)i;
                if (g < 0 || (unsigned long)g >= nf)
                    continue;
                for (e = 0; e < MMX_EQ_BANDS; e++)
                    imp[(unsigned long)g * MMX_EQ_BANDS + e] += (1.0 + imp[f * MMX_EQ_BANDS + e]) * share[s][i] *
                                                               noise[(unsigned long)g * MMX_EQ_BANDS + e] / p->n_sources / (resid[e] + 1e-20);
            }
    }
    if (getenv("MMX_DEBUG_CLEAN"))
    {
        double mx = 0.0, sum = 0.0;
        unsigned long used = 0, hist[6] = {0, 0, 0, 0, 0, 0};
        for (i = 0; i < (unsigned long)nf * MMX_EQ_BANDS; i++)
        {
            if (imp[i] > mx) mx = imp[i];
            if (imp[i] > 0.0) { used++; sum += imp[i]; }
            hist[imp[i] <= 0.0 ? 0 : imp[i] < 1.0 ? 1 : imp[i] < 2.0 ? 2 : imp[i] < 4.0 ? 3 : imp[i] < 8.0 ? 4 : 5]++;
        }
        fprintf(stderr, "clean samples: %lu of %lu frame bands are sources, importance mean %.2f max %.2f, histogram 0/<1/<2/<4/<8/>=8: %lu %lu %lu %lu %lu %lu, alpha %.2f\n",
                used, nf * MMX_EQ_BANDS, used ? sum / used : 0.0, mx, hist[0], hist[1], hist[2], hist[3], hist[4], hist[5], alpha);
    }
    for (f = 0; f < nf; f++)
        for (c = 0; c < a->channels; c++)
        {
            MMXFramePsy *p = &a->psy[f * a->channels + c];
            for (b = 0; b < codec->bands.band_count; b++)
            {
                double v = imp[f * MMX_EQ_BANDS + codec->bands.eq_band[b]];
                if (v > 0.0 && p->thr[b] < 1e29f)
                    p->thr[b] = (float)(p->thr[b] * pow(1.0 + v, -alpha));
            }
        }
    free(imp);
    free(noise);
}

/* ------------------------------------------------------------- top level */

/* The rate loop's offset, shaped over frequency (see "tilted rate offset"):
   `factor` is the one scalar the bisection searches, thr_scale keeps it (the
   reporting, the metadata and the experiments that reason about "the offset"
   use it), and every band is scaled by factor^tilt(band_hz). */
static void scale_thresholds(MMXAnalysis *a, const MMXCodec *codec, const float *base, double factor)
{
    size_t i, n = (size_t)a->frame_count * a->channels;
    double bs[MMX_MAX_BANDS];
    TiltProfile tp;
    unsigned int b;
    tilt_profile_init(&tp);
    tilt_band_scales(&tp, &codec->bands, factor, bs, MMX_MAX_BANDS);
    a->thr_scale = factor;
    for (i = 0; i < n; i++)
        for (b = 0; b < MMX_MAX_BANDS; b++)
        {
            float t = base[i * MMX_MAX_BANDS + b];
            a->psy[i].thr[b] = t >= 1e29f ? t : (float)(t * bs[b]);
        }
}

/* Clean samples with the current thresholds (after a rate-loop offset): always
   in the tracker modes, in strict mode only with a bitrate target (at a fixed
   quality the strict thresholds are the promise, finer sources would only add
   bits; with a target the offset rebalances and the error at equal size drops).
   MMX_CLEAN=<alpha> overrides the strength for experiments (0 = off). */
static void clean_samples_apply(MMXAnalysis *a, const MMXCodec *codec, const MMXEncoderParams *params)
{
    const char *ea = getenv("MMX_CLEAN");
    if (params->mode || params->target_kbps)
    {
        size_t i, n = (size_t)a->frame_count * a->channels;
        clean_samples(a, codec, ea ? atof(ea) : CLEAN_ALPHA);
        /* the per-band scaling can open gaps of more than the cliff limit between
           neighbour bands; the leak of the coarse neighbour would then waste the
           finer band's extra bits, so the limiter runs again */
        if (!getenv("MMX_NO_CLIFF"))
            for (i = 0; i < n; i++)
                mmx_psy_limit_cliffs(&a->psy[i], codec->bands.band_count);
    }
}

static unsigned long long container_bytes(const MMXFile *f, const MMXStatistics *s)
{
    unsigned char *table = NULL;
    unsigned long table_size = mmx_table_encode(f, &table);
    free(table);
    return 96 + f->metadata_len + f->cover_len + table_size +
           s->payload_bytes[0] + s->payload_bytes[1] + s->payload_bytes[2];
}

/* ---------------------------------------------------------- second pass */

/* Iterated planning (analysis levels >= MMX_SECOND_PASS_LEVEL): the first
   closed-loop pass shows what the plan really costs - the references read
   the decoded signal, not the input, and the range coder charges residuals
   about 10 % more than audio frames relative to the estimate (title A:
   residuals 1.27x the estimate, audio 1.13x, transient frames less). The
   analysis runs again with the decoded output as the source signal and the
   estimates calibrated per class from the first pass, the file is encoded
   again, and the smaller of the two files is kept. `analysis`, `out`,
   `stats` and `decoded` hold pass one on entry and the winner on return. */
/* The second pass replans against the DECODED signal, so it can only find something where frames
   actually predict from reconstructions. Measured at quality 7, analysis 7, four performance cores
   (encode time in brackets): title G, 0.0 % of frames referenced -> byte-identical output (3.94x
   the time); title E, 5.3 % -> 0.085 % smaller (3.74x); title A, 76.2 % -> 0.842 % smaller (2.96x).
   Below the threshold the pass is paid for in full and buys nothing at all. The share is known for
   free once pass one has run. MMX_SECOND_PASS_MIN_REF=<percent> overrides it; 0 always runs it. */
static int second_pass_worthwhile(const MMXStatistics *stats)
{
    const char *e = getenv("MMX_SECOND_PASS_MIN_REF");
    double min_pct = e ? atof(e) : 2.0;
    unsigned long total = stats->frames_by_sources[0] + stats->frames_by_sources[1] + stats->frames_by_sources[2];
    double ref_pct;
    if (min_pct <= 0.0 || total == 0)
        return 1;
    ref_pct = 100.0 * (double)(stats->frames_by_sources[1] + stats->frames_by_sources[2]) / (double)total;
    if (ref_pct >= min_pct)
        return 1;
    mmx_info("Second pass skipped: %.1f %% of frames are referenced, below %.1f %% - it replans against the "
             "decoded signal and has nothing to find here", ref_pct, min_pct);
    return 0;
}

static int second_pass(const MMXAudioBuffer *audio, const MMXEncoderParams *params, MMXCodec *codec,
                       const MMXAnalysisParams *ap, MMXAnalysis *analysis, MMXFile *out, MMXStatistics *stats,
                       MMXAudioBuffer *decoded)
{
    MMXAnalysisParams ap2 = *ap;
    MMXAnalysis a2;
    MMXFile f2;
    MMXStatistics s2;
    MMXAudioBuffer d2;
    unsigned long long bytes1, bytes2;

    memset(&a2, 0, sizeof(a2));
    memset(&d2, 0, sizeof(d2));
    ap2.sources = decoded;
    /* The calibration holds where the coder does what the estimate models:
       the strict mode (the caller runs the second pass there only). Pure
       bands (tracker, future, reuse) and noise bands (pns) drop whole bands
       of the good candidates only; the mean factor (0.56-0.77 measured)
       would flatter every candidate. */
    if (params->mode == MMX_MODE_NORMAL && !params->pns_hz && !params->reuse)
        est_calibration(ap2.cal_audio, ap2.cal_ref);
    mmx_info("Second pass: planning against the decoded signal (estimates calibrated: audio x%.3f/%.3f, residual x%.3f/%.3f stationary/transient)...",
             ap2.cal_audio[0] > 0.0 ? ap2.cal_audio[0] : 1.0, ap2.cal_audio[1] > 0.0 ? ap2.cal_audio[1] : 1.0,
             ap2.cal_ref[0] > 0.0 ? ap2.cal_ref[0] : 1.0, ap2.cal_ref[1] > 0.0 ? ap2.cal_ref[1] : 1.0);
    /* the first pass's coefficients and thresholds are done with (the encoder
       has run, the report reads the plan only): release them before the
       second analysis holds its own, ~200 MB on a 6-minute track */
    free(analysis->coefs);
    analysis->coefs = NULL;
    free(analysis->psy);
    analysis->psy = NULL;
    if (mmx_analyzer_run(audio, codec, &ap2, &a2) != 0)
        return 0;   /* keep pass one */
    stats->analysis_seconds += a2.seconds;
    clean_samples_apply(&a2, codec, params);
    mmx_file_init(&f2);
    fill_header(&f2, audio, params);
    mmx_statistics_init(&s2);
    s2.input_bytes = stats->input_bytes;
    s2.duration_seconds = stats->duration_seconds;
    s2.analysis_seconds = stats->analysis_seconds;
    s2.total_frames = stats->total_frames;
    s2.transient_frames = stats->transient_frames;
    mmx_info("Encoding %lu frames closed-loop, second pass...", a2.frame_count);
    if (encode_lossy(audio, params, codec, &a2, &f2, &s2, &d2) != 0)
    {
        mmx_file_free(&f2);
        mmx_analysis_free(&a2);
        return 0;
    }
    bytes1 = container_bytes(out, stats);
    bytes2 = container_bytes(&f2, &s2);
    mmx_info("Second pass: %llu -> %llu bytes (%+.2f %%), keeping the %s", bytes1, bytes2,
             100.0 * ((double)bytes2 / (double)bytes1 - 1.0), bytes2 < bytes1 ? "second" : "first");
    if (bytes2 < bytes1)
    {
        mmx_file_free(out);
        *out = f2;
        *stats = s2;
        mmx_analysis_free(analysis);
        *analysis = a2;
        mmx_audio_buffer_free(decoded);
        *decoded = d2;
    }
    else
    {
        stats->analysis_seconds = s2.analysis_seconds;
        mmx_file_free(&f2);
        mmx_analysis_free(&a2);
        mmx_audio_buffer_free(&d2);
    }
    return 0;
}

/* ------------------------------------------------------ the rate loop ----

   A bitrate target is ONE scalar: the dB offset that scales every masking
   threshold (scale_thresholds). Size falls monotonically with that offset and
   is nearly straight in log size, so the loop looks for the root of
   f(x) = ln(size(x)) - ln(target).

   Two things used to make that search cost the whole encode several times
   over. It bisected [-20, +30] dB, which needs 6-8 evaluations to reach 1.5 %,
   and every evaluation was a FULL encode - trellis quantizer, block-type trial
   and M/S trial included, and those three stages are about six sevenths of the
   encode time (title I, 20 s excerpt, at 320 kbit/s: 36.8 s, of which 4.6 s remain with all
   three off).

   What a probe is actually for is the SIZE, and the three expensive stages
   move the size by a factor that is nearly constant WHERE IT MATTERS:
   full/cheap was 0.9726-0.9953 over 25 probe points on three excerpts at 128
   and 320 kbit/s and offsets from -17 to +15 dB - it differs by up to 1.2 %
   between tracks, but only by 0.05-0.3 % between two offsets a few tenths of
   a dB apart, which is exactly the neighbourhood the search ends in. So:

     - the search runs CHEAP probes (rate_fast_probe) and multiplies their size
       by that factor;
     - the root finder is a bracketing secant (regula falsi, bisection whenever
       the secant step would leave the bracket, the bracket is stale or the
       function goes flat), which needs 3-4 probes instead of 6-8;
     - the full-cost encode at the end IS the final file, and its size is
       checked against the target. Only if it misses RATE_VERIFY_TOLERANCE is
       the factor re-measured from that very encode (where it is now known
       exactly), the cheap probes already paid for re-read with the corrected
       factor - no encoding at all - and one more cheap probe plus one more
       full encode made. Never more than RATE_FULL_ENCODES full encodes, and
       if the second one comes out worse than the first, the first is put back;
     - and if after all that the file is still outside the loop's 1.5 %
       tolerance, the estimate is dropped and the search is redone with
       FULL-COST probes. That is the corner where the factor is not constant:
       at the saturating low rates the replication crossover and the automatic
       noise substitution move with the offset, and full/cheap then runs from
       0.87 to 1.01 over one search (measured on the round-trip test's
       synthetic track at 16 and 24 kbit/s).

   The cost is therefore 3-6 cheap probes plus one or two full encodes, and the
   last full encode is the file itself - the bisection's 6-8 full probes plus a
   separate final encode are gone. The file that comes out is the same file: at
   a given offset the encode is byte-identical to the old one (MMX_RATE_OFFSET
   was added to check exactly that), and the offset lands inside the tolerance
   the loop always had - in fact closer to the target than the bisection's.

   MMX_RATE_RATIO=<f> overrides the factor, MMX_RATE_CHEAP=0 goes back to
   full-cost probes (the root finder stays), MMX_RATE_RATIO_DEBUG=1 encodes
   every probe both ways and prints the measured factor, MMX_RATE_OFFSET=<dB>
   skips the search and encodes at that offset. */

/* Undo a full encode that is about to be replaced by one at another offset:
   the file, the statistics, the plan (the closed loop rewrites it) and the
   decoded signal all go back to what they were before it ran. */
static void rate_reset_output(const MMXAudioBuffer *audio, const MMXEncoderParams *params,
                              MMXFile *out, MMXStatistics *stats, const MMXStatistics *stats0,
                              MMXAnalysis *analysis, const MMXFramePlan *plan0, MMXAudioBuffer *decoded_out)
{
    mmx_file_free(out);
    mmx_file_init(out);
    fill_header(out, audio, params);
    *stats = *stats0;
    memcpy(analysis->plan, plan0, sizeof(MMXFramePlan) * analysis->frame_count);
    if (decoded_out)
    {
        mmx_audio_buffer_free(decoded_out);
        memset(decoded_out, 0, sizeof(*decoded_out));
    }
}

static double rate_probe_ratio(void)
{
    const char *e = getenv("MMX_RATE_RATIO");
    double r = e ? atof(e) : RATE_PROBE_RATIO;
    return r > 0.5 && r < 2.0 ? r : RATE_PROBE_RATIO;
}

static int rate_cheap_probes(void)
{
    const char *e = getenv("MMX_RATE_CHEAP");
    return e == NULL || atoi(e) != 0;
}

/* One probe encode at the thresholds currently in `analysis`. `fast` switches
   the trellis quantizer and the two closed-loop trials off for its duration.
   Returns the container size, or 0 on failure. */
static unsigned long long rate_probe_bytes(const MMXAudioBuffer *audio, const MMXEncoderParams *params,
                                           MMXCodec *codec, const MMXAnalysis *analysis, int fast)
{
    MMXFile tmp;
    MMXStatistics ts;
    MMXAnalysis plan_copy;
    unsigned long long bytes = 0;
    mmx_file_init(&tmp);
    fill_header(&tmp, audio, params);
    mmx_statistics_init(&ts);
    /* the plan is adjusted by the closed loop (depth fallbacks); work on a copy of it */
    plan_copy = *analysis;
    plan_copy.plan = (MMXFramePlan *)malloc(sizeof(MMXFramePlan) * analysis->frame_count);
    if (!plan_copy.plan)
    {
        mmx_file_free(&tmp);
        return 0;
    }
    memcpy(plan_copy.plan, analysis->plan, sizeof(MMXFramePlan) * analysis->frame_count);
    rate_fast_probe = fast;
    mmx_frame_rdoq_set_enabled(!fast);
    if (encode_lossy(audio, params, codec, &plan_copy, &tmp, &ts, NULL) == 0)
        bytes = container_bytes(&tmp, &ts);
    rate_fast_probe = 0;
    mmx_frame_rdoq_set_enabled(1);
    free(plan_copy.plan);
    mmx_file_free(&tmp);
    return bytes;
}

/* The root finder. `px`/`pb` hold every probe made so far (offset and its
   probe size) and survive a change of `ratio`, so a corrected factor costs no
   encode at all - the points are simply read again. Returns the best offset;
   *err_out is its estimated relative size error, *slope_out the local
   d ln(size)/d offset (used to extrapolate and to correct). */
static double rate_search(const MMXAudioBuffer *audio, const MMXEncoderParams *params,
                          MMXCodec *codec, MMXAnalysis *analysis, const float *base_thr,
                          unsigned long long target, double ratio, int max_probes, int cheap,
                          double *px, unsigned long long *pb, int *pn,
                          double *err_out, double *slope_out)
{
    const double lo_bound = -20.0, hi_bound = 30.0;
    static int ratio_debug = -1;
    double best_off = 0.0, best_f = 1e30, slope = RATE_SLOPE0;
    double x = 0.0;
    int i;
    if (ratio_debug < 0)
        ratio_debug = getenv("MMX_RATE_RATIO_DEBUG") != NULL;
    for (;;)
    {
        double a = 0.0, fa = 0.0, b = 0.0, fb = 0.0, nx;
        unsigned long long ba = 0, bb = 0;
        int have_a = 0, have_b = 0, repeat = 0;
        unsigned long long bytes;
        /* read every probe we have with the current factor */
        best_f = 1e30;
        for (i = 0; i < *pn; i++)
        {
            double f = log((double)pb[i] * ratio / (double)target);
            if (fabs(f) < fabs(best_f)) { best_f = f; best_off = px[i]; }
            if (f > 0.0) { if (!have_a || px[i] > a) { a = px[i]; fa = f; ba = pb[i]; have_a = 1; } }
            else         { if (!have_b || px[i] < b) { b = px[i]; fb = f; bb = pb[i]; have_b = 1; } }
        }
        /* the local slope: from the bracket if there is one, else from the last
           two probes; kept in a sane range so a flat or noisy pair cannot throw
           the next step across the map */
        {
            double dx = 0.0, df = 0.0;
            if (have_a && have_b && b != a) { dx = b - a; df = log((double)bb / (double)ba); }
            else if (*pn >= 2 && px[*pn - 1] != px[*pn - 2])
            { dx = px[*pn - 1] - px[*pn - 2]; df = log((double)pb[*pn - 1] / (double)pb[*pn - 2]); }
            if (dx != 0.0 && df / dx <= -0.004 && df / dx >= -0.5)
                slope = df / dx;
        }
        if (*pn > 0 && fabs(expm1(best_f)) < RATE_PROBE_TOLERANCE)
            break;
        if (*pn >= max_probes)
            break;
        /* where to look next */
        if (*pn == 0)
            nx = 0.0;                                   /* the first probe is the model's own thresholds */
        else if (have_a && have_b)
        {
            /* regula falsi inside the bracket; bisect when the step would leave
               it or when the bracket is stale (the last two probes fell on the
               same side, so one end has not moved) */
            int stale = (*pn >= 2) &&
                        ((pb[*pn - 1] * ratio > (double)target) == (pb[*pn - 2] * ratio > (double)target));
            if (b - a < 0.01)
                break;
            nx = (fabs(fa - fb) > 1e-12) ? a + (b - a) * fa / (fa - fb) : 0.5 * (a + b);
            if (stale || !(nx > a + 0.02 * (b - a) && nx < b - 0.02 * (b - a)))
                nx = 0.5 * (a + b);
        }
        else
        {
            /* no bracket yet: extrapolate along the slope, capped and clamped */
            double f0 = have_a ? fa : fb, x0 = have_a ? a : b, step = -f0 / slope;
            if (step > 20.0) step = 20.0;
            if (step < -20.0) step = -20.0;
            nx = x0 + step;
            if (nx > hi_bound) nx = hi_bound;
            if (nx < lo_bound) nx = lo_bound;
            if (fabs(nx - x0) < 0.01)
                break;                                  /* saturated at a bound */
        }
        for (i = 0; i < *pn; i++)
            if (fabs(px[i] - nx) < 1e-9)
                repeat = 1;
        if (repeat)
            break;                                      /* nothing new to learn */
        x = nx;
        scale_thresholds(analysis, codec, base_thr, pow(10.0, x / 10.0));
        clean_samples_apply(analysis, codec, params);
        bytes = rate_probe_bytes(audio, params, codec, analysis, cheap);
        if (bytes == 0)
            break;
        if (ratio_debug && cheap)
        {
            unsigned long long full = rate_probe_bytes(audio, params, codec, analysis, 0);
            mmx_info("  [ratio] offset %+.2f dB: cheap %llu, full %llu, full/cheap %.4f",
                     x, bytes, full, full ? (double)full / (double)bytes : 0.0);
        }
        px[*pn] = x;
        pb[*pn] = bytes;
        (*pn)++;
        mmx_info("  offset %+.2f dB -> %llu bytes%s (%+.1f %%)", x, bytes,
                 cheap ? " probe" : "",
                 100.0 * expm1(log((double)bytes * ratio / (double)target)));
    }
    *err_out = *pn > 0 ? expm1(best_f) : 1e30;
    *slope_out = slope;
    return best_off;
}

int mmx_encoder_encode(const MMXAudioBuffer *audio, const MMXEncoderParams *params,
                       MMXFile *out, MMXStatistics *stats, MMXAudioBuffer *decoded_out, MMXAnalysis *analysis_out)
{
    MMXCodec codec;
    MMXAnalysis analysis;
    MMXAnalysisParams ap;
    unsigned int codec_quality = params->quality == 0 ? MMX_QUALITY_MAX : params->quality;
    clock_t t0;
    int rc = -1;
    double offset_db = 0.0;

    memset(&codec, 0, sizeof(codec));
    memset(&analysis, 0, sizeof(analysis));
    tilt_off_override = 0;
    /* the fixed profile belongs to the rate loop: at a fixed quality the
       thresholds are never scaled and bscale must stay 1 */
    /* The fixed profile is OFF by default: coupling it to the replication was measured to push the
       ear's region above the masking threshold on files whose rate loop still lands on a large offset
       (title A 128 bass 0.1/0.2/0.2/0.1/0.1 -> 1.9/2.8/3.6/3.6/5.2, pre-echo 14 -> 29; title B
       128 20.74 -> 22.18 % in a 0.8 % larger file). With the multiplied tilt instead, replication
       is better AND smaller on every point measured. MMX_TILT_MIRROR=1 restores the fixed profile. */
    tilt_mirror_on = 0;
    (void)0;

    mmx_statistics_init(stats);
    stats->input_bytes = audio->frame_count * audio->channels * ((audio->source_bits ? audio->source_bits : 16) / 8);
    stats->duration_seconds = mmx_audio_buffer_duration_seconds(audio);

    if (mmx_codec_init(&codec, audio->sample_rate, audio->channels, codec_quality) != 0)
    {
        mmx_error("Codec init failed (channels %u)", audio->channels);
        return -1;
    }
    mmx_bands_set_epb(&codec.bands, params->quality == 0 ? 0 : params->lowrate, params->epb_fold, params->epb_flags, params->epb_fill_db);
    if (params->quality != 0 && params->bwe_hz)
        mmx_bands_set_bwe(&codec.bands, (double)bwe_header_hz(params->bwe_hz),
                          (unsigned int)((getenv("MMX_BWE_MODE") ? atoi(getenv("MMX_BWE_MODE")) : 0) & 1));
    mmx_analysis_params_default(&ap);
    ap.level = params->analysis;
    ap.quality = codec_quality;
    ap.max_depth = params->max_ref_depth;
    ap.verbose = params->verbose;
    ap.lossless = params->quality == 0;
    ap.seg_frames = 0;
    if (ap.lossless && params->pld_strict)          /* --pld: the analysis picks sources inside each segment */
    {
        const char *e = getenv("MMX_PLD");
        int v = e ? atoi(e) : 30;
        ap.seg_frames = mmx_ll2_pld_frames((unsigned int)(v < 0 ? 0 : v > 255 ? 255 : v), audio->sample_rate);
    }
    ap.tns = params->tns;
    ap.source_bits = audio->source_bits ? audio->source_bits : 16;
    if (ap.lossless)
        ap.source_bits -= lossless_shift(audio, params, NULL, NULL);   /* the coder codes x >> shift */
    {
        const char *e = getenv("MMX_CAL");   /* experiment: fixed calibration "audio_stat,audio_trans,resid_stat,resid_trans" for the first pass */
        if (e)
            sscanf(e, "%lf,%lf,%lf,%lf", &ap.cal_audio[0], &ap.cal_audio[1], &ap.cal_ref[0], &ap.cal_ref[1]);
    }
    if (mmx_analyzer_run(audio, &codec, &ap, &analysis) != 0)
    {
        mmx_error("Analysis failed");
        goto done;
    }
    stats->analysis_seconds = analysis.seconds;
    stats->total_frames = analysis.frame_count;
    stats->transient_frames = analysis.transient_frames;
    fill_header(out, audio, params);

    t0 = clock();
    if (params->quality == 0 && params->target_kbps > 0)
    {
        /* near-lossless rate loop: the rounding step 2^d * M is searched so that the file lands at the
           target. The lossless coder's size is close to linear in log2(step): every halving of the step
           costs one bit per sample and channel, so one full encode plus one or two corrections hit the
           target within half a percent. */
        MMXEncoderParams p = *params;
        MMXStatistics stats0 = *stats;
        unsigned long long target = (unsigned long long)(params->target_kbps * 1000.0 / 8.0 * stats->duration_seconds);
        double s = params->drop_bits ? params->drop_bits + log((double)(params->drop_step > 1 ? params->drop_step : 1)) / log(2.0) : 4.0;
        double per_bit = (double)audio->frame_count * audio->channels / 8.0;   /* bytes per bit of step */
        int attempt;
        mmx_info("Encoding %lu frames near-lossless with a size target of %llu bytes (%.0f kbit/s)...", analysis.frame_count, target, params->target_kbps);
        for (attempt = 0; attempt < 4; attempt++)
        {
            unsigned int d, M;
            unsigned long long bytes;
            if (s < 0.0) s = 0.0;
            if (s > 12.0) s = 12.0;
            if (s >= 4.0) { d = (unsigned int)floor(s) - 4; M = (unsigned int)llround(pow(2.0, s - d)); }
            else { d = 0; M = (unsigned int)llround(pow(2.0, s)); }
            if (M < 1) M = 1;
            if (M > 255) M = 255;
            p.drop_bits = d;
            p.drop_step = M;
            if (attempt)
            {
                mmx_file_free(out);
                mmx_file_init(out);
                fill_header(out, audio, &p);
                *stats = stats0;
            }
            rc = encode_lossless(audio, &p, &analysis, out, stats, decoded_out);
            if (rc != 0)
                break;
            bytes = container_bytes(out, stats);
            mmx_info("  step 2^%.2f (%u bits x%u) -> %llu bytes (%+.2f %% of the target)", log((double)(1u << d) * M) / log(2.0), d, M, bytes,
                     100.0 * ((double)bytes / (double)target - 1.0));
            if (bytes <= target && (double)bytes >= 0.995 * (double)target)
                break;
            s += ((double)bytes - (double)target) / per_bit;
            if (attempt == 2 && bytes > target) s += 0.02;   /* last correction lands under the target */
        }
    }
    else if (params->quality == 0 && !params->drop_bits && params->drop_step <= 1 && !(getenv("MMX_LL2") && atoi(getenv("MMX_LL2")) == 0))
        /* lossless: the LL2 core (revision 10) by default; MMX_LL2=0 writes the revision 8 core, near-lossless stays there */
    {
        /* The configuration per file; every candidate is a full encode, the smallest file wins (encoder time is free):
           (1) the main OLS memory with its two extra OLS (ll2_xols_for), MMX_LL2_DECAYS lists the memories (default
           8..12); (2) without the extra OLS at the best memory and at the longest (MMX_LL2_PLAIN=0 skips); (3) the
           mixed filter bank on the extras winner, the plain best and the plain longest (MMX_LL2_BANK=0 skips, 1 forces);
           the smallest file wins. The candidates run in parallel in two rounds on the shared, read-only preparation:
           round 1 = (1) and the longest without extras, round 2 = the best without extras and the three bank trials;
           the winner's file is kept instead of being coded once more. */
        const char *dl = getenv("MMX_LL2_DECAYS"), *ep = getenv("MMX_LL2_PLAIN"), *eb = getenv("MMX_LL2_BANK"), *en = getenv("MMX_LL2_NOREF");
        const int plain = !(ep && atoi(ep) == 0), try_bank = !(eb && atoi(eb) == 0), force_bank = eb && atoi(eb) == 1;
        unsigned int cand[8], nc = 0, nj1 = 0, nj2 = 0, i;
        int win = -1, win2 = -1, noref = 0;
        Ll2Prep prep, prep2;
        Ll2Job jobs[16], jobs2[16];
        Ll2Job *best = NULL;
        char buf[64];
        char *tok;
        snprintf(buf, sizeof(buf), "%s", dl ? dl : "8,9,10,11,12");
        for (tok = strtok(buf, ","); tok && nc < 8; tok = strtok(NULL, ",")) cand[nc++] = (unsigned int)atoi(tok);
        memset(jobs, 0, sizeof(jobs));
        memset(jobs2, 0, sizeof(jobs2));
        memset(&prep2, 0, sizeof(prep2));
        mmx_info("Encoding %lu frames lossless with the LL2 core (revision 10), %u memory candidates...", analysis.frame_count, nc);
        rc = nc ? ll2_prepare(&prep, audio, params, &analysis, stats) : -1;
        if (rc == 0)
        {
            mmx_ll2_statics();                  /* shared statics set once, before the candidates run in parallel */
            (void)mmx_ll2_profile(0);           /* ... and the profile table (lab overrides are read on first use) */
            mmx_ll2c_free(mmx_ll2c_new());
            win = ll2_search(&prep, cand, nc, plain, try_bank, force_bank, jobs, &nj1, "");
            if (win < 0) rc = -1;
            else best = &jobs[win];
        }
        /* The hi-res profile also tries the file without references (MMX_LL2_NOREF=1: every profile, 0: never) */
        noref = en ? atoi(en) != 0
                   : rc == 0 && mmx_ll2_profile_for(prep.bits, audio->sample_rate, audio->channels) == MMX_LL2_PROFILE_HIRES;
        if (rc == 0 && noref && ll2_prepare_noref(&prep2, &prep, stats) == 0)
        {
            win2 = ll2_search(&prep2, cand, nc, plain, try_bank, force_bank, jobs2, &nj2, "without references, ");
            if (win2 >= 0 && jobs2[win2].bytes < best->bytes) best = &jobs2[win2];
        }
        if (rc == 0 && best)
        {
            unsigned char x[2] = { 0, 0 };
            if (best->extra) ll2_xols_for(best->decay, x);
            char ow[32] = "";
            if (best->own) snprintf(ow, sizeof(ow), ", %u OLS own taps", best->own);
            mmx_info("  chosen: memory 2^%u, extra OLS 2^%u 2^%u, %s bank%s%s (%u candidates, %u at once)", best->decay,
                     x[0], x[1], best->bank ? "mixed" : "sign-LMS", best == &jobs2[win2 >= 0 ? win2 : 0] && win2 >= 0 ? ", without references" : "",
                     ow, nj1 + nj2, mmx_threads());
            mmx_file_free(out);
            *out = best->file;                  /* the winner's file is the result: no second encode */
            best->kept = 1;
            *stats = best->stats;
            if (decoded_out)                    /* lossless: the decoded signal is the input */
            {
                double scale = pow(2.0, (double)prep.bits - 1.0);
                unsigned long n;
                unsigned int c;
                for (n = 0; n < prep.len; n++)
                    for (c = 0; c < audio->channels; c++)
                        decoded_out->samples[(size_t)n * audio->channels + c] =
                            (float)((double)(prep.dec[c][n] * (1LL << prep.shift)) / scale);
            }
        }
        for (i = 0; i < 16; i++) { if (!jobs[i].kept) mmx_file_free(&jobs[i].file); if (!jobs2[i].kept) mmx_file_free(&jobs2[i].file); }
        if (prep2.own_plan) ll2_prepare_free(&prep2);
        if (nc) ll2_prepare_free(&prep);
    }
    else if (params->quality == 0 && audio->isamples)
    {   /* the revision-8 core and near-lossless read the float buffer, which cannot hold 32 bits */
        mmx_error("32-bit integer input needs the LL2 lossless core (plain --lossless, without MMX_LL2=0, --drop-bits or --drop-step)");
        rc = -1;
    }
    else if (params->quality == 0)
    {
        mmx_info("Encoding %lu frames lossless (integer residuals, global references)...", analysis.frame_count);
        rc = encode_lossless(audio, params, &analysis, out, stats, decoded_out);
    }
    else if (params->target_kbps > 0)
    {
        /* bitrate target: scale the masking thresholds until the size matches
           (cheap probes + bracketing secant, then the full encode that IS the
           file; the analysis is done only once - see "the rate loop" above) */
        size_t nthr = (size_t)analysis.frame_count * analysis.channels * MMX_MAX_BANDS, i;
        float *base_thr = (float *)malloc(sizeof(float) * nthr);
        unsigned long long target = (unsigned long long)(params->target_kbps * 1000.0 / 8.0 * stats->duration_seconds);
        double best_off = 0.0, best_err = 1e30, est_err = 1e30, slope = RATE_SLOPE0;
        double px[RATE_ITERATIONS + 2];
        unsigned long long pb[RATE_ITERATIONS + 2];
        int pn = 0, attempt, cheap = rate_cheap_probes();
        double ratio0 = cheap ? rate_probe_ratio() : 1.0;
        MMXFramePlan *plan_pristine = NULL;
        MMXStatistics stats_pristine;
        if (!base_thr)
            goto done;
        plan_pristine = (MMXFramePlan *)malloc(sizeof(MMXFramePlan) * analysis.frame_count);
        if (!plan_pristine) { free(base_thr); goto done; }
        memcpy(plan_pristine, analysis.plan, sizeof(MMXFramePlan) * analysis.frame_count);
        stats_pristine = *stats;
        for (i = 0; i < (size_t)analysis.frame_count * analysis.channels; i++)
            memcpy(base_thr + i * MMX_MAX_BANDS, analysis.psy[i].thr, sizeof(float) * MMX_MAX_BANDS);
        mmx_info("Bitrate target %u kbit/s (%llu bytes): searching the threshold offset...", params->target_kbps, target);
        for (attempt = 0; attempt < 2; attempt++)
        {
            double ratio = ratio0, cur_off = 0.0, cur_err = 1e30;
            int full_encodes = 0, have_encoded = 0;
            pn = 0;
            for (;;)
            {
                unsigned long long bytes, probe_at = 0;
                const char *forced = getenv("MMX_RATE_OFFSET");   /* experiments: skip the search */
                double try_off, err;
                int restored = 0;
                if (forced)
                {
                    try_off = atof(forced);
                    est_err = 0.0;
                    px[0] = try_off; pb[0] = 0; pn = 1;
                }
                else
                {
                    try_off = rate_search(audio, params, &codec, &analysis, base_thr, target, ratio,
                                          RATE_ITERATIONS, cheap, px, pb, &pn, &est_err, &slope);
                    if (pn == 0) { rc = -1; break; }
                }
                if (have_encoded && (try_off == cur_off || fabs(est_err) >= 0.6 * fabs(cur_err)))
                    break;                     /* the corrected factor promises no real improvement:
                                                  it has to predict under 0.6 of the error we already
                                                  have, or another full encode is not worth it */
                if (have_encoded)
                    rate_reset_output(audio, params, out, stats, &stats_pristine, &analysis, plan_pristine, decoded_out);
                scale_thresholds(&analysis, &codec, base_thr, pow(10.0, try_off / 10.0));
                clean_samples_apply(&analysis, &codec, params);
                mmx_info("Encoding with threshold offset %+.2f dB", try_off);
                rc = encode_lossy(audio, params, &codec, &analysis, out, stats, decoded_out);
                if (rc != 0) break;
                full_encodes++;
                bytes = container_bytes(out, stats);
                err = (double)bytes / (double)target - 1.0;
                for (i = 0; i < (size_t)pn; i++)
                    if (px[i] == try_off) probe_at = pb[i];
                if (cheap && probe_at)
                    mmx_info("  full encode at %+.2f dB -> %llu bytes (%+.2f %%, probe said %+.2f %%, full/cheap %.4f)",
                             try_off, bytes, 100.0 * err, 100.0 * est_err, (double)bytes / (double)probe_at);
                else
                    mmx_info("  full encode at %+.2f dB -> %llu bytes (%+.2f %%)", try_off, bytes, 100.0 * err);
                if (have_encoded && fabs(err) > fabs(cur_err))
                {
                    /* the correction made it worse: put the earlier file back */
                    mmx_info("  keeping the encode at %+.2f dB (%+.2f %%)", cur_off, 100.0 * cur_err);
                    rate_reset_output(audio, params, out, stats, &stats_pristine, &analysis, plan_pristine, decoded_out);
                    scale_thresholds(&analysis, &codec, base_thr, pow(10.0, cur_off / 10.0));
                    clean_samples_apply(&analysis, &codec, params);
                    rc = encode_lossy(audio, params, &codec, &analysis, out, stats, decoded_out);
                    if (rc != 0) break;
                    restored = 1;              /* `out` holds the earlier encode again */
                }
                else
                {
                    cur_err = err;
                    cur_off = try_off;
                    have_encoded = 1;
                }
                if (forced || !cheap || fabs(cur_err) <= RATE_VERIFY_TOLERANCE)
                    break;
                if (!restored && probe_at && full_encodes < RATE_FULL_ENCODES)
                {
                    /* the file missed the tight band: the cheap/full factor is
                       now known exactly at this offset, so re-read the probes
                       we have already paid for with it (no encode) and let the
                       search pick again - one more cheap probe, then one more
                       full encode */
                    ratio = (double)bytes / (double)probe_at;
                    mmx_info("  factor re-measured as %.4f, re-reading the %d probes", ratio, pn);
                    continue;
                }
                if (fabs(cur_err) <= RATE_TOLERANCE)
                    break;                     /* inside the loop's tolerance: good enough */
                /* The cheap probes have not landed this file. That happens
                   where the factor is NOT nearly constant: at the saturating
                   low rates the replication crossover and the noise
                   substitution move with the offset, and on the synthetic test
                   track full/cheap runs from 0.87 to 1.01 over one search
                   instead of the 0.97-0.995 of real material at 96 kbit/s and
                   up. Give up on the estimate and redo the search with
                   full-cost probes - the exact behaviour, the secant keeps it
                   cheaper than the old bisection anyway. */
                mmx_info("  the cheap probes did not land the file (%+.2f %%): redoing the search with full-cost probes",
                         100.0 * cur_err);
                cheap = 0;
                ratio0 = 1.0;
                ratio = 1.0;
                pn = 0;
                full_encodes = 0;
            }
            best_off = cur_off;
            best_err = cur_err;
            if (rc != 0)
                break;
            /* The fallback. The tilt protects the low bands by coding them BELOW
               the model (tilt < 0 there), and that part of the budget no longer
               shrinks when the loop raises the offset. On normal material the
               highs carry enough of the bits for the loop to keep its authority;
               on a file whose protected bands alone are already over budget the
               search saturates, and then the tilt has made the rate target
               worse instead of better. Detect exactly that case (the same 5 %
               the warning below uses) and redo the search with the uniform
               offset - the coder before run 6. */
            {
                TiltProfile probe;
                tilt_profile_init(&probe);
                if (getenv("MMX_RATE_OFFSET") || fabs(best_err) <= 0.05 || !probe.on)
                    break;
                mmx_info("  the tilted offset saturated at %+.1f %%: retrying with the uniform offset", 100.0 * best_err);
                tilt_off_override = 1;
                rate_reset_output(audio, params, out, stats, &stats_pristine, &analysis, plan_pristine, decoded_out);
            }
        }
        free(plan_pristine);
        free(base_thr);
        offset_db = best_off;
        if (rc == 0 && fabs(best_err) > 0.05)   /* the search saturated: the energy rules keep a floor under the size */
            fprintf(stderr, "warning: bitrate target %u kbit/s not reached, closest is %+.1f %% at offset %+.2f dB "
                            "(the energy floor of the low-rate rules; MMX_KEEP=0 removes it)\n",
                    params->target_kbps, 100.0 * best_err, best_off);
        if (rc == 0)
        {
            char line[128];
            snprintf(line, sizeof(line), "bitrate_target=%u\nthreshold_offset_db=%.2f\n", params->target_kbps, offset_db);
            metadata_append(out, line);
        }
    }
    else
    {
        MMXAudioBuffer dec;
        memset(&dec, 0, sizeof(dec));
        mmx_info("Encoding %lu frames closed-loop (references read the decoded signal)...", analysis.frame_count);
        clean_samples_apply(&analysis, &codec, params);
        rc = encode_lossy(audio, params, &codec, &analysis, out, stats, &dec);
        /* strict mode only: with pure bands (tracker, future, reuse) or noise
           bands (pns) the replanned file came out 5-16 % larger on the synthetic
           track and the first pass was kept every time, so the time is saved */
        if (rc == 0 && second_pass_wanted(params) && second_pass_worthwhile(stats))
            rc = second_pass(audio, params, &codec, &ap, &analysis, out, stats, &dec);
        if (rc == 0 && decoded_out)
        {
            *decoded_out = dec;
            memset(&dec, 0, sizeof(dec));
        }
        mmx_audio_buffer_free(&dec);
    }
    stats->encoding_seconds = (double)(clock() - t0) / CLOCKS_PER_SEC;

    if (rc == 0 && analysis_out)
    {
        *analysis_out = analysis;
        memset(&analysis, 0, sizeof(analysis));
    }

done:
    mmx_analysis_free(&analysis);
    mmx_codec_free(&codec);
    return rc;
}
