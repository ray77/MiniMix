#ifndef MMX_LL2CODEC_H
#define MMX_LL2CODEC_H
/* LL2 lossless core (bitstream revision 10, header byte 87 = 2): the frame processor shared by encoder and decoder.
 *
 * Per block (a run of at most MMX_LL2_BLOCK_FRAMES frames of the 1024-sample grid with one reference lineage)
 * channel 0 is processed first, then channel 1: channel 0 predicts from its own past only, channel 1 from its own
 * past, channel 0 at t..t-7 and channel 0 at t+1..t+4 (zero past the block end, so a block needs only its own
 * channel 0). A referenced frame adds the source stage (src/ll2.c): the source window,
 * whitened with the channel's current OLS weights on both channels' source samples, predicts the OLS residual.
 * Every read of the decoded signal goes through one getter with the decoder's frontier (own channel: before t;
 * channel 1 read by channel 0: before the frame; channel 0 read by channel 1: before the frame end), so encoder and
 * decoder see the same numbers by construction. Residuals of a block (the container's unit) are coded after the
 * block with the bitplane coder (src/ll2coder.c), channel 0 then channel 1, the model carried across blocks. Short
 * blocks keep the decoder's first output, its memory and its seek unit small and let the three pipeline stages
 * (residuals, channel 0, channel 1) run on different blocks at once; they cost nothing measurable (2^16 samples:
 * +0.003 %). Earlier builds stopped channel 1's future taps at the frame end, which cost 0.2-0.4 % on stereo files.
 */
#include "ll2.h"
#include "ll2coder.h"

#define MMX_LL2_CORE 2                  /* header byte 87 */
#define MMX_LL2_OWN 16
#define MMX_LL2_CROSS 8
#define MMX_LL2_FUT 4                   /* channel 1's future taps on channel 0 (CD profile; the profile sets them) */
#define MMX_LL2_FUT_MAX 8
#define MMX_LL2_BLOCK_FRAMES 64         /* the encoder closes a block after this many frames (1.5 s at 44.1 kHz) */

/* Format profiles (header byte 93). Every format-dependent setting of the LL2 core lives in one profile, so a new
 * format gets its own settings instead of a copy of the core: the encoder picks the profile from the source
 * (mmx_ll2_profile_for), the decoder takes it from the header and refuses a profile it does not know (a file from a
 * newer encoder). The CD profile is frozen: 16-bit files stay byte-identical (their MD5s are checked before every change). */
#define MMX_LL2_PROFILE_CD 0            /* up to 16 bits and 48 kHz */
#define MMX_LL2_PROFILE_HIRES 1         /* more than 16 bits or more than 48 kHz */
#define MMX_LL2_PROFILES 2
typedef struct
{
    unsigned int id;
    const char *name;
    unsigned int pred_bits;             /* the predictor sees at most this many bits: x >> (bits - pred_bits) */
    int cov_shift;                      /* the OLS covariance scale (mmx_ll2_set_cov_shift) */
    unsigned int own_taps;              /* OLS taps on the channel's own past (at most MMX_LL2_OWN_MAX) */
    int keep_on_fail;                   /* ill-conditioned OLS solves keep the previous weights (mmx_ll2_set_keep_on_fail) */
    int dpred;                          /* the OLS predicts with double weights (mmx_ll2_set_dpred) */
    int stage_target;                   /* 0: fixed stage scale; else adaptive to about 2^stage_target (mmx_ll2_set_stage_target) */
    int raw_depth;                      /* 0: every residual bit modelled; else deep refinement bits raw (mmx_ll2c_set_raw_depth) */
    int fnorm;                          /* the normalised LMS stages in float (mmx_ll2_set_float_norm) */
    int dcov;                           /* the OLS covariance in double (mmx_ll2_set_dcov) */
    int solve_every;                    /* samples between two OLS solves (0 = MMX_LL2_SOLVE; double path only) */
    unsigned int fut;                   /* channel 1's future taps on channel 0 (at most MMX_LL2_FUT_MAX) */
} MMXLl2Profile;
unsigned int mmx_ll2_profile_for(unsigned int bits, unsigned long sample_rate, unsigned int channels);
const MMXLl2Profile *mmx_ll2_profile(unsigned int id);   /* NULL: unknown profile */

typedef struct
{
    unsigned int channels;
    MMXLl2Chan ch[2];
    MMXLl2Coder *coder;
    long long *res[2];                  /* the residuals of the open block */
    unsigned long cap, used;
    long long last_src[2];              /* per channel: the previous frame's source start (-1 none); a repeat continues when src = last + 1024 */
    long long lo[2], hi[2];             /* the clamp bounds of the open block per channel (encoder) */
    long long rail_lo, rail_hi;         /* the full-scale range after the shift: the bounds a block uses by default */
    int pshift;                         /* the predictor sees x >> pshift (profile: pred_bits) */
    int wrap32;                         /* 32 coded bits: residuals modulo 2^32 in (-2^31, 2^31], samples wrap back */
    unsigned int nfut;                  /* channel 1's future taps (profile) */
    long long seg_start;                /* the regressors read nothing before this sample (the segment's first) */
    long long store_from;               /* decoding: samples before this are final already (a segment decoded again
                                           after its context was reclaimed) - computed, not stored again */
    const MMXLl2Profile *prof;          /* the format profile (mmx_ll2s_set_profile; the CD profile until then) */
    long long *praw[2];                 /* encoder: the unclamped predictions of the open block */
    int expl[2];                        /* encoder: the block carries explicit bounds for channel c */
} MMXLl2Stream;

/* decay: the OLS memory, R -= R >> decay (8..12; 0 = the default 9); the file carries it in header byte 88 */
/* bank: 0 = four integer sign-LMS stages 1024/256/32/16; 1 = six stages 1024s/1280n/256s/256n/32n/16s with integer
   normalised LMS (header byte 89) */
/* xols: the memories of up to two extra OLS per channel (header bytes 90, 91; 0 = none), NULL = none */
/* profile: header byte 93 (returns -1 for a profile this build does not know, see mmx_ll2_profile) */
/* own: OLS taps on the channel's own past (header byte 94; 0 = the profile's, at most MMX_LL2_OWN_MAX) */
int mmx_ll2s_init(MMXLl2Stream *s, unsigned int channels, unsigned int decay, unsigned int bank, const unsigned char *xols,
                  unsigned int profile, unsigned int own);
void mmx_ll2s_free(MMXLl2Stream *s);
/* makes room for a block of n samples per channel and empties the residual buffers */
int mmx_ll2s_block_begin(MMXLl2Stream *s, unsigned long n);

/* One frame. dec[c]: the whole signal (integers after the shift), start: first owned sample, count: samples, src: the
 * source start of this frame (-1 = no reference), front: the end of the frame's block (channel 1 reads channel 0
 * before it). Every prediction is clamped (sac does the same per frame): a clipped master's samples at the rail cost
 * nothing when the prediction overshoots, and clamping to bounds the true sample respects never makes a residual
 * larger. mmx_ll2s_frame (the encoder) clamps to the rails and keeps the unclamped prediction; mmx_ll2s_finish_block
 * then decides the block's bounds. Encoder (decode = 0): dec holds the original, residuals are
 * appended to the block buffers. Decoder (decode = 1): the residuals are taken from the block buffers at the current
 * position and dec is written. */
void mmx_ll2s_frame(MMXLl2Stream *s, long long *const *dec, long long start, unsigned long count, long long src,
                    long long front, int decode);
/* One channel of one frame, residuals at res (count values): the unit the threaded decoder schedules. Channel 1 of a
   block needs channel 0 of the whole block first; channel 0 needs only its own past. */
void mmx_ll2s_frame_ch(MMXLl2Stream *s, unsigned int c, long long *const *dec, long long start, unsigned long count,
                       long long src, long long front, int decode, long long *res, long long lo, long long hi);
/* a frame without samples (the first one) still moves the repeat bookkeeping */
void mmx_ll2s_skip_frame(MMXLl2Stream *s, long long src);

/* The mixer (header byte 92): 1 = four Huber experts with thresholds 32 and 512 for 16-bit material, scaled with the
   bits the coded integers carry beyond 16; 0 = the two-expert mixer. Call after init. */
void mmx_ll2s_set_mixer(MMXLl2Stream *s, unsigned int mix, unsigned int bits, unsigned int shift);
/* The full-scale range of the coded integers (source bits after the shift): the default clamp bounds. Also sets
   the predictor's input scale: the OLS clamps its regressor at 2^20 and the LMS stages saturate at 16 bits, so a
   source with more than 20 bits (24-bit hi-res) is predicted from x >> (bits - 20) and the prediction scaled back
   (cell centre); the residual and everything coded stay at full resolution. Must be called before mmx_ll2s_set_mixer. */
void mmx_ll2s_set_rails(MMXLl2Stream *s, unsigned int bits, unsigned int shift);
/* The sample rate (header): profiles with the double OLS path solve less often at high rates - every 16 samples up
   to 96 kHz, 32 from 176.4 kHz, 64 from 352.8 kHz, so the solve work per second of audio stays the same (measured
   +0.011 % at 192 and 384 kHz, the largest decoder share there). After mmx_ll2s_set_rails. */
void mmx_ll2s_set_rate(MMXLl2Stream *s, unsigned long sample_rate);

/* Patchwork landscape decoding (header byte 95 = seconds, 0 = none): the encoder starts a block at every multiple of
   mmx_ll2_pld_frames frames and there predictor and coder start afresh, the regressors read nothing before the
   segment (seg_start); references may still read any earlier audio. A decoder can so enter at every segment and
   decode segments side by side. 0 when seconds is 0. */
unsigned long mmx_ll2_pld_frames(unsigned int seconds, unsigned long sample_rate);
/* Encoder, after the frames of a block (coded with the rails as bounds): per channel, explicit bounds (the block's
   smallest and largest sample) replace the rails when the residuals they shrink save more than they cost to send;
   the residuals are recomputed from the stored unclamped predictions. */
void mmx_ll2s_finish_block(MMXLl2Stream *s);
/* block payload: per channel one bit (0 = the rails, 1 = explicit [lo, hi], bypass coded), then the residuals,
   channel 0 then channel 1. The decoder reads the bounds (mmx_ll2s_decode_bounds) and then n residuals per channel
   with the coder (mmx_ll2c_decode_cont), or both into the stream's own buffers (mmx_ll2s_decode_block). */
void mmx_ll2s_encode_block(MMXLl2Stream *s, MMXRangeEncoder *rc);
void mmx_ll2s_decode_bounds(const MMXLl2Stream *s, MMXRangeDecoder *rc, long long *lo, long long *hi);
int mmx_ll2s_decode_block(MMXLl2Stream *s, MMXRangeDecoder *rc, unsigned long n);

#endif
