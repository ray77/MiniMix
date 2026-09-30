#ifndef MMX_CODEC_H
#define MMX_CODEC_H

/* Shared MDCT codec engine used by analyzer, encoder and decoder. The same
   prediction and reconstruction code runs on both sides so references are
   reproduced exactly. */

#include "minimix/audio_buffer.h"
#include "../../src/framecodec.h"
#include "../../src/mdct.h"
#include "../../src/psymodel.h"

typedef struct
{
    unsigned long sample_rate;
    unsigned int channels;
    unsigned int quality;
    unsigned int cutoff_band;
    unsigned long tns_k0, tns_k1;                /* coefficient range shaped by TNS */
    MMXBandLayout bands;
    MMXMdct mdct;
    /* block switching */
    MMXMdct short_mdct;                          /* 256-sample transforms */
    MMXBandLayout short_bands;                   /* 128-bin layout */
    unsigned int short_cutoff_band;
    double *win_start, *win_stop;                /* MMX_WIN each */
    unsigned long short_offset;                  /* first short window inside the long window */
    float *swin, *sy;                            /* 256-sample work buffers */
    float *win;                                  /* MMX_WIN work samples */
    float *y;                                    /* MMX_WIN inverse output */
    float *src[MMX_MAX_SOURCES][MMX_MAX_CH];     /* source coefficients, coded domain */
    float *pred[MMX_MAX_CH];                     /* prediction, coded domain */
    float *rec[MMX_MAX_CH];                      /* reconstruction, coded domain */
    float *tmp[MMX_MAX_CH];
} MMXCodec;

int mmx_codec_init(MMXCodec *c, unsigned long sample_rate, unsigned int channels, unsigned int quality);
void mmx_codec_free(MMXCodec *c);

/* Number of frames needed to cover `samples` (frame f windows [f*HOP-HOP, f*HOP+HOP)). */
unsigned long mmx_codec_frame_count(unsigned long long samples);

/* MDCT of channel ch of buf, window starting at sample `start` (zero padded outside). */
void mmx_codec_window_mdct(MMXCodec *c, const MMXAudioBuffer *buf, long long start, unsigned int ch, float *out);
/* Same for a given block type: START/STOP use their window shapes, SHORT
   produces 8 x 128 group-interleaved coefficients. */
void mmx_codec_window_mdct_bt(MMXCodec *c, const MMXAudioBuffer *buf, long long start, unsigned int ch,
                              unsigned int block_type, float *out);
/* Short-window samples of group g (256 samples, zero padded) for psychoacoustic analysis. */
void mmx_codec_short_window(const MMXCodec *c, const MMXAudioBuffer *buf, long long start, unsigned int ch,
                            unsigned int group, float *out);

/* In-place L/R <-> M/S on two coefficient arrays. */
void mmx_codec_lr_to_ms(float *l, float *r, unsigned long m);
void mmx_codec_ms_to_lr(float *m_, float *s, unsigned long m);

/* Fills c->src[s][ch] for each source from `decoded` (window start per source),
   converted to M/S when stereo_ms. */
void mmx_codec_source_coefs(MMXCodec *c, const MMXAudioBuffer *decoded, unsigned int n_sources,
                            const long long *src_start, int stereo_ms);
void mmx_codec_source_coefs_bt(MMXCodec *c, const MMXAudioBuffer *decoded, unsigned int n_sources,
                               const long long *src_start, int stereo_ms, unsigned int block_type);

/* c->pred = sum of gained sources (zero when n_sources == 0). */
void mmx_codec_predict(MMXCodec *c, const MMXFrameSyntax *s, unsigned int n_sources);

/* c->rec = c->pred + dequantized syntax; converts to L/R, IMDCTs and overlap-adds
   the frame into out at window start `start`. */
void mmx_codec_reconstruct(MMXCodec *c, const MMXFrameSyntax *s, MMXAudioBuffer *out, long long start);
/* decoder: ref = the reconstruction later frames predict from, out = the output; identical unless EPB reference mode 1 */
void mmx_codec_reconstruct_out(MMXCodec *c, const MMXFrameSyntax *s, MMXAudioBuffer *ref, MMXAudioBuffer *out, long long start);

/* Overlap-add helper: adds n samples of channel ch at `start` (clipped to buffer). */
void mmx_codec_overlap_add(const MMXAudioBuffer *out, long long start, unsigned int ch, const float *y, unsigned long n);

/* Least-squares gains per EQ band for `n_sources` sources against target t[ch][m]
   (all in the coded domain); writes s->gain and s->polarity. */
void mmx_codec_fit_gains(MMXCodec *c, MMXFrameSyntax *s, unsigned int n_sources, float *const *target);

#endif
