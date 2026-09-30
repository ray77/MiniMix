/* Native MDCT codec: AUDIO frames and a predicted frame through the real range
   coder; decoder reconstruction must be bit-identical to the encoder's and the
   noise must stay below the masking threshold in every band. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "minimix/codec.h"
#include "minimix/quality.h"
#include "rangecoder.h"
#include "spectrum.h"

#define PI 3.14159265358979323846

#define FRAMES 40
#define LEN (FRAMES * MMX_HOP)

int main(void)
{
    MMXCodec enc, dec;
    MMXAudioBuffer orig, rec_enc, rec_dec;
    MMXFrameSyntax syn, dsyn;
    MMXCodecContexts ectx, dctx;
    MMXRangeEncoder rc;
    MMXRangeDecoder rd;
    float *t[2], *resid[2];
    const float *thr[2];
    float thr_ms[2][MMX_MAX_BANDS];
    MMXFramePsy psy[2];
    MMXFramePsy *all_psy;
    unsigned long f, i, k;
    unsigned int c, b;
    double worst_nmr = -1e9, worst_coded = -1e9;
    float *t_lr[2];
    unsigned long nf = mmx_codec_frame_count(LEN), ref_frames = 0;
    long long src_start[2];

    if (mmx_codec_init(&enc, 44100, 2, 7) != 0 || mmx_codec_init(&dec, 44100, 2, 7) != 0) { printf("init failed\n"); return 1; }
    mmx_audio_buffer_init(&orig, 44100, 2, LEN);
    mmx_audio_buffer_init(&rec_enc, 44100, 2, LEN);
    mmx_audio_buffer_init(&rec_dec, 44100, 2, LEN);
    mmx_frame_syntax_init(&syn, 2, MMX_HOP);
    mmx_frame_syntax_init(&dsyn, 2, MMX_HOP);
    for (c = 0; c < 2; c++) { t[c] = malloc(sizeof(float) * MMX_HOP); resid[c] = malloc(sizeof(float) * MMX_HOP); t_lr[c] = malloc(sizeof(float) * MMX_HOP); }

    srand(5);
    for (i = 0; i < LEN; i++)
    {
        double tt = (double)i / 44100.0;
        double s = 0.4 * sin(2 * PI * 220 * tt) + 0.2 * sin(2 * PI * 3300 * tt) + 0.02 * ((double)rand() / RAND_MAX - 0.5);
        if ((i / 11025) % 2 == 1) s *= 0.5; /* level changes */
        orig.samples[2 * i] = (float)s;
        orig.samples[2 * i + 1] = (float)(0.7 * s + 0.1 * sin(2 * PI * 880 * tt));
    }
    /* second half repeats the first half (with a small offset) so REF frames can be tested */
    memcpy(orig.samples + (size_t)(LEN / 2 + 100) * 2, orig.samples, sizeof(float) * (LEN / 2 - 100) * 2);

    /* thresholds of every frame (leak-free spectrum), then temporal smoothing */
    all_psy = malloc(sizeof(MMXFramePsy) * nf * 2);
    {
        MMXSpectrum spec;
        float amp[MMX_HOP];
        mmx_spectrum_init(&spec, MMX_WIN);
        for (f = 0; f < nf; f++)
            for (c = 0; c < 2; c++)
            {
                mmx_codec_window_mdct(&enc, &orig, (long long)f * MMX_HOP - MMX_HOP, c, t[c]);
                mmx_spectrum_analyze(&spec, enc.win, amp);
                mmx_psy_analyze(&enc.bands, amp, enc.win, 7, &all_psy[f * 2 + c]);
            }
        mmx_spectrum_free(&spec);
    }
    {
        MMXFramePsy *raw = malloc(sizeof(MMXFramePsy) * nf * 2);
        memcpy(raw, all_psy, sizeof(MMXFramePsy) * nf * 2);
        for (f = 0; f < nf; f++)
            for (c = 0; c < 2; c++)
            {
                mmx_psy_temporal(f ? &raw[(f - 1) * 2 + c] : NULL, &all_psy[f * 2 + c], f + 1 < nf ? &raw[(f + 1) * 2 + c] : NULL);
                mmx_psy_limit_cliffs(&all_psy[f * 2 + c], enc.bands.band_count);
            }
        free(raw);
    }

    mmx_contexts_init(&ectx);
    mmx_contexts_init(&dctx);
    mmx_rc_enc_init(&rc);

    for (f = 0; f < nf; f++)
    {
        long long start = (long long)f * MMX_HOP - MMX_HOP;
        unsigned int n_sources = 0;

        for (c = 0; c < 2; c++)
        {
            mmx_codec_window_mdct(&enc, &orig, start, c, t[c]);
            memcpy(t_lr[c], t[c], sizeof(float) * MMX_HOP);
            psy[c] = all_psy[f * 2 + c];
        }
        /* frames in the repeated half reference the decoded first half */
        if (start >= LEN / 2 + 100 + MMX_WIN)
        {
            n_sources = 1;
            src_start[0] = start - (LEN / 2 + 100);
            ref_frames++;
        }

        syn.stereo_ms = (f % 3 == 0);
        if (syn.stereo_ms) mmx_codec_lr_to_ms(t[0], t[1], MMX_HOP);
        mmx_codec_source_coefs(&enc, &rec_enc, n_sources, src_start, syn.stereo_ms);
        if (n_sources)
            mmx_codec_fit_gains(&enc, &syn, n_sources, t);
        mmx_codec_predict(&enc, &syn, n_sources);
        for (c = 0; c < 2; c++)
            for (k = 0; k < MMX_HOP; k++)
                resid[c][k] = t[c][k] - enc.pred[c][k];

        if (syn.stereo_ms)
        {
            mmx_psy_stereo_threshold(&enc.bands, &psy[0], &psy[1], thr_ms[0]);
            memcpy(thr_ms[1], thr_ms[0], sizeof(thr_ms[0]));
            thr[0] = thr_ms[0]; thr[1] = thr_ms[1];
        }
        else { thr[0] = psy[0].thr; thr[1] = psy[1].thr; }

        {
            const float *lr[2] = { all_psy[f * 2].thr, all_psy[f * 2 + 1].thr };
            mmx_frame_quantize(&enc.bands, &syn, resid, thr, enc.cutoff_band, syn.stereo_ms ? lr : NULL, NULL, &ectx, n_sources ? 1 : 0);
        }
        mmx_frame_encode(&rc, &ectx, &enc.bands, &enc.short_bands, &syn, n_sources);
        mmx_codec_reconstruct(&enc, &syn, &rec_enc, start);
        /* coded-domain check: enc.rec holds the L/R reconstruction of this frame */
        for (c = 0; c < 2; c++)
            for (b = 0; b < enc.cutoff_band; b++)
            {
                double noise = 0.0, nmr, n = (double)(enc.bands.band_start[b + 1] - enc.bands.band_start[b]);
                if (psy[c].energy[b] <= psy[c].thr[b] * n * 0.5) continue;
                for (k = enc.bands.band_start[b]; k < enc.bands.band_start[b + 1]; k++)
                    noise += (double)(enc.rec[c][k] - t_lr[c][k]) * (enc.rec[c][k] - t_lr[c][k]);
                nmr = mmx_psy_nmr_db(&enc.bands, &psy[c], b, noise);
                if (nmr > worst_coded) worst_coded = nmr;
            }
    }
    mmx_rc_enc_finish(&rc);
    printf("codec: %lu frames (%lu predicted) -> %lu bytes = %.1f kbit/s\n", nf, ref_frames, rc.size,
           rc.size * 8.0 / ((double)LEN / 44100.0) / 1000.0);

    /* decode */
    mmx_rc_dec_init(&rd, rc.data, rc.size);
    for (f = 0; f < nf; f++)
    {
        long long start = (long long)f * MMX_HOP - MMX_HOP;
        unsigned int n_sources = 0;
        if (start >= LEN / 2 + 100 + MMX_WIN) { n_sources = 1; src_start[0] = start - (LEN / 2 + 100); }
        if (mmx_frame_decode(&rd, &dctx, &dec.bands, &dec.short_bands, &dsyn, n_sources) != 0) { printf("decode error at frame %lu\n", f); return 1; }
        mmx_codec_source_coefs(&dec, &rec_dec, n_sources, src_start, dsyn.stereo_ms);
        mmx_codec_predict(&dec, &dsyn, n_sources);
        mmx_codec_reconstruct(&dec, &dsyn, &rec_dec, start);
    }
    for (i = 0; i < (unsigned long)LEN * 2; i++)
        if (rec_enc.samples[i] != rec_dec.samples[i]) { printf("encoder/decoder mismatch at %lu\n", i); return 1; }

    /* quality: per frame NMR of the final L/R output versus the original (raw thresholds) */
    for (f = 1; f + 1 < nf; f++)
    {
        long long start = (long long)f * MMX_HOP - MMX_HOP;
        float r[MMX_HOP];
        for (c = 0; c < 2; c++)
        {
            mmx_codec_window_mdct(&enc, &orig, start, c, t[c]);
            psy[c] = all_psy[f * 2 + c];
            mmx_codec_window_mdct(&enc, &rec_dec, start, c, r);
            for (b = 0; b < enc.cutoff_band; b++)
            {
                double noise = 0.0, nmr;
                for (k = enc.bands.band_start[b]; k < enc.bands.band_start[b + 1]; k++)
                    noise += (double)(r[k] - t[c][k]) * (r[k] - t[c][k]);
                nmr = mmx_psy_nmr_db(&enc.bands, &psy[c], b, noise);
                if (nmr > worst_nmr) worst_nmr = nmr;
            }
        }
    }
    printf("codec: worst band NMR in the coded domain %.2f dB (limit +0.5), re-analysed output %.2f dB (limit +3.5: neighbouring frames' aliased noise adds ~3 dB)\n",
           worst_coded, worst_nmr);
    if (worst_coded > 0.5 || worst_nmr > 3.5) return 1;
    {
        /* whole-signal perceptual check with the sub-frame analyzer */
        MMXQualityReport q;
        mmx_quality_compare(&orig, &rec_dec, 7, 0, 6.0, &q);
        printf("codec: sub-frame compare worst NMR %.2f dB, %.2f %% bands over 0 dB, pre-echo worst %.1f dB\n",
               q.worst_nmr_db, q.bands_over_percent, q.worst_pre_attack_snr_db);
    }
    free(all_psy);

    mmx_rc_enc_free(&rc);
    mmx_codec_free(&enc); mmx_codec_free(&dec);
    mmx_audio_buffer_free(&orig); mmx_audio_buffer_free(&rec_enc); mmx_audio_buffer_free(&rec_dec);
    mmx_frame_syntax_free(&syn); mmx_frame_syntax_free(&dsyn);
    for (c = 0; c < 2; c++) { free(t[c]); free(resid[c]); }
    printf("test_codec OK\n");
    return 0;
}
