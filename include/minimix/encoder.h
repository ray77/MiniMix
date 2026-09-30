#ifndef MMX_ENCODER_H
#define MMX_ENCODER_H

#include "minimix/audio_buffer.h"
#include "minimix/mmx_format.h"
#include "minimix/analyzer.h"
#include "../../src/statistics.h"

typedef struct
{
    unsigned int quality;          /* 1..10, 0 = lossless */
    unsigned int drop_step;        /* lossless only: extra integer step multiplier M (1..255) on top of drop_bits; the rate loop (target_kbps) sets both */
    int pld_strict;                /* lossless (LL2) only, --pld: no reference reads before its own segment - every
                                      segment decodes on its own, a seek never waits for earlier ones */
    unsigned int drop_bits;        /* lossless only: round the input by this many bits first (near-lossless,
                                      max error +-2^(drop_bits-1) LSB); 0 = bit-exact */
    unsigned int analysis;         /* 1..9 */
    unsigned int target_kbps;      /* 0 = quality fixed; otherwise the thresholds are scaled until this bitrate is met */
    int tns;                       /* experimental: temporal noise shaping */
    unsigned int pns_hz;           /* perceptual noise substitution from this frequency on (0 = off): noise-like,
                                      temporally flat bands carry only their energy */
    unsigned int lowrate;          /* EPB (revision 9): energy-preserving bands, see mmx_epb_finish */
    unsigned int epb_fold;         /* EPB fill policy, written to the header: fold floor in 250 Hz units (255 = never) ... */
    unsigned int epb_flags;        /* ... bit0 noise under the floor, bit1 fill inside coded bands, bit2 fill kept out of the reference ... */
    int epb_fill_db;               /* ... fill level relative to the missing energy, dB */
    unsigned int bwe_hz;           /* band replication (SBR) from this frequency on (0 = off): the bands above it
                                      carry only an energy level and a noise-mix index, the decoder regenerates
                                      them from the octave below the crossover */
    unsigned int is_hz;            /* intensity stereo from this frequency on (0 = off): a band whose channels are
                                      coherent enough is coded once with a position when that costs fewer bits */
    unsigned int noise_fill;       /* noise filling (energy preservation): MMX_NF_OFF (the default), MMX_NF_AUTO
                                      (rate-matched files only) or MMX_NF_ON (at every rate) */
    unsigned int reuse;            /* 0 = residuals strictly under the masking threshold; 1..3 = residuals of
                                      referenced frames 3/6/9 dB more generous, from 2 on parts that are
                                      predicted well enough are played as pure references (no residual) */
    unsigned int mode;             /* MMX_MODE_NORMAL, MMX_MODE_TRACKER or MMX_MODE_FUTURE (see below) */
    unsigned char max_ref_depth;
    const char *metadata;          /* "key=value\n" lines or NULL */
    const char *cover_mime;
    const unsigned char *cover;
    unsigned long cover_len;
    int verbose;
} MMXEncoderParams;

/* Encoder modes. NORMAL codes every residual strictly under the masking
   threshold. TRACKER works like a tracker/MOD file: runs of frames that play
   from earlier material are "patterns"; every band of a pattern whose
   prediction gain is at least 10 dB ("90 % the same") is played from the
   source without any residual, the remaining bands keep a residual with a
   6 dB more generous threshold (the difference between two similar takes is
   masked by the take itself). FUTURE pushes that to 6 dB / 12 dB and codes the
   unique material at quality 4: maximum summarization, smallest file. */
#define MMX_MODE_NORMAL 0
#define MMX_MODE_TRACKER 1
#define MMX_MODE_FUTURE 2

/* Noise substitution: default start frequency, whether the CLI enables it at
   every rate without --pns (no), and the highest bitrate target at which it is
   switched on automatically. Below about 104 kbit/s the plain coder can only
   keep the ear's region clean by damaging it: the rate loop's offset reaches
   +3 dB and, even tilted, 2-8 kHz falls apart. Noise substitution buys that
   region back - it is what Opus does at every rate. Measured on the 20 s
   excerpts (analysis 5, tilted rate loop, cells over the threshold; the model
   counts every substituted band as an error by construction, so the numbers
   below 4 kHz are the trustworthy half):
     title I 128: 19.2 -> 30.0 % (below 4 kHz 1.9/2.5/3.1/4.4/7.1
       -> 0.0 everywhere) - a model loss, PNS stays off
     112: 27.1 -> 30.0 %   104: 31.4 -> 30.1 %   96: 33.3 -> 30.8 %
     title G 112: 26.8 -> 29.5 %   104: 29.6 -> 29.7 %   96: 32.0 -> 30.3 %
   104 kbit/s is where the two lines cross. Above it the plain coder is ahead
   by the model and by the ear's own region; below it substitution wins both.
   MMX_PNS_DEFAULT_MAX_KBPS (env, same name) moves the threshold, --pns forces
   it on at any rate, --no-pns off. */
#define MMX_PNS_DEFAULT_HZ 5000
#define MMX_PNS_DEFAULT_ON 0
#define MMX_PNS_DEFAULT_MAX_KBPS 104

/* Intensity stereo: default start frequency of --is, and the highest bitrate
   target at which the CLI switches it on WITHOUT --is. That is 0: intensity
   stereo is opt-in, nothing gets it by default. MMX_IS_DEFAULT_MAX_KBPS
   (env: MMX_IS_DEFAULT_MAX_KBPS) can turn it back into an automatic tool.
   Measured on the title G excerpt: at 96 kbit/s the model's cells over
   the threshold fall (the saved bits lower the rate loop's offset), at
   128 kbit/s they rise (over 18.5 -> 19.8 %: the offset is near 0 dB there
   and every dropped difference counts); on title A it gains nothing at
   equal size. The model cannot judge a dropped stereo difference - only the
   ear can, so it is switched on by hand. */
#define MMX_IS_DEFAULT_HZ 5000
#define MMX_IS_DEFAULT_MAX_KBPS 0
/* Noise filling: the zeros of the coded bands above MMX_NF_DEFAULT_HZ and the
   bands the rate loop's offset zeroed as a whole keep their energy as
   deterministic noise (see framecodec.h). OFF never fills, ON fills at every
   rate (--nf), AUTO would fill rate-matched files (--bitrate) only, where the
   thresholds sit above the quality's own masking model. The default is OFF:
   pure filling is a loss by the model (it hears an energy-matched different
   waveform as error) and the trade it makes - bits from the highs to the
   lows, steered by MMX_NF_ZERO - has not been measured on whole tracks, only
   on excerpts. --nf switches it on. MMX_NF_HZ, MMX_NF_DB, MMX_NF_ZERO,
   MMX_NF_RESID and MMX_NF_HOLE_DB move the knobs (see encoder.c). */
/* Band replication (--bwe / --no-bwe, bitstream revision 6). Above the
   crossover the encoder stops coding coefficients and transmits per band and
   frame only the energy and a two-bit noise share; the decoder copies the
   octave below the crossover up, mixes in noise and scales the band to the
   transmitted energy. 65 % of the bits of a 96 kbit/s file sit above 4 kHz,
   and they buy the region the ear lives in nothing - this is the tool that
   moves them down. It is off at a fixed quality and above
   MMX_BWE_DEFAULT_MAX_KBPS, so the strict operating point and the 320 kbit/s
   class are byte-identical to the coder before revision 6. The crossover
   falls with the rate (mmx_bwe_default_hz): the lower the budget, the more of
   the spectrum is described instead of coded. */
#define MMX_BWE_DEFAULT_MAX_KBPS 144
#define MMX_BWE_DEFAULT_HZ 10000
unsigned int mmx_bwe_default_hz(unsigned int target_kbps);

#define MMX_NF_OFF 0
#define MMX_NF_AUTO 1
#define MMX_NF_ON 2
#define MMX_NF_DEFAULT MMX_NF_OFF
#define MMX_NF_DEFAULT_HZ 4000

void mmx_encoder_params_default(MMXEncoderParams *p);

/* Global analysis followed by closed-loop encoding. Fills `out` (an initialized
   MMXFile) and `stats`. `decoded` (optional) receives the encoder-side
   reconstruction, `analysis_out` (optional) the analysis results. */
int mmx_encoder_encode(const MMXAudioBuffer *audio, const MMXEncoderParams *params,
                       MMXFile *out, MMXStatistics *stats, MMXAudioBuffer *decoded, MMXAnalysis *analysis_out);

#endif
