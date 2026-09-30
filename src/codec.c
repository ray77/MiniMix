#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "lab.h"
#include "minimix/codec.h"

int mmx_codec_init(MMXCodec *c, unsigned long sample_rate, unsigned int channels, unsigned int quality)
{
    unsigned int s, ch;

    memset(c, 0, sizeof(*c));
    if (channels == 0 || channels > MMX_MAX_CH)
        return -1;
    c->sample_rate = sample_rate;
    c->channels = channels;
    c->quality = quality > MMX_QUALITY_MAX ? MMX_QUALITY_MAX : (quality < 1 ? 1 : quality);
    if (mmx_bands_init(&c->bands, sample_rate, MMX_HOP) != 0 || mmx_mdct_init(&c->mdct, MMX_WIN) != 0)
        return -1;
    c->cutoff_band = c->bands.cutoff_band[c->quality];
    {
        unsigned int b;
        c->tns_k0 = MMX_HOP;
        for (b = 0; b < c->bands.band_count; b++)
            if (c->bands.band_hz[b] >= 1500.0f) { c->tns_k0 = c->bands.band_start[b]; break; }
        c->tns_k1 = c->bands.band_start[c->cutoff_band];
        if (c->tns_k1 <= c->tns_k0) c->tns_k1 = c->tns_k0;
    }
    c->win = (float *)calloc(MMX_WIN, sizeof(float));
    c->y = (float *)calloc(MMX_WIN, sizeof(float));
    if (!c->win || !c->y)
        return -1;
    if (mmx_mdct_init(&c->short_mdct, 2 * MMX_SHORT_M) != 0 || mmx_bands_init(&c->short_bands, sample_rate, MMX_SHORT_M) != 0)
        return -1;
    c->short_cutoff_band = c->short_bands.cutoff_band[c->quality];
    c->win_start = (double *)malloc(sizeof(double) * MMX_WIN);
    c->win_stop = (double *)malloc(sizeof(double) * MMX_WIN);
    c->swin = (float *)calloc(2 * MMX_SHORT_M, sizeof(float));
    c->sy = (float *)calloc(2 * MMX_SHORT_M, sizeof(float));
    if (!c->win_start || !c->win_stop || !c->swin || !c->sy)
        return -1;
    if (mmx_mdct_window_start(&c->mdct, &c->short_mdct, c->win_start) != 0 ||
        mmx_mdct_window_stop(&c->mdct, &c->short_mdct, c->win_stop) != 0)
        return -1;
    c->short_offset = mmx_mdct_short_offset(&c->mdct, &c->short_mdct);
    for (ch = 0; ch < channels; ch++)
    {
        c->pred[ch] = (float *)calloc(MMX_HOP, sizeof(float));
        c->rec[ch] = (float *)calloc(MMX_HOP, sizeof(float));
        c->tmp[ch] = (float *)calloc(MMX_HOP, sizeof(float));
        if (!c->pred[ch] || !c->rec[ch] || !c->tmp[ch])
            return -1;
        for (s = 0; s < MMX_MAX_SOURCES; s++)
        {
            c->src[s][ch] = (float *)calloc(MMX_HOP, sizeof(float));
            if (!c->src[s][ch])
                return -1;
        }
    }
    return 0;
}

void mmx_codec_free(MMXCodec *c)
{
    unsigned int s, ch;
    if (!c)
        return;
    mmx_mdct_free(&c->mdct);
    mmx_mdct_free(&c->short_mdct);
    free(c->win_start);
    free(c->win_stop);
    free(c->swin);
    free(c->sy);
    free(c->win);
    free(c->y);
    for (ch = 0; ch < MMX_MAX_CH; ch++)
    {
        free(c->pred[ch]);
        free(c->rec[ch]);
        free(c->tmp[ch]);
        for (s = 0; s < MMX_MAX_SOURCES; s++)
            free(c->src[s][ch]);
    }
    memset(c, 0, sizeof(*c));
}

unsigned long mmx_codec_frame_count(unsigned long long samples)
{
    return (unsigned long)((samples + MMX_HOP - 1) / MMX_HOP) + 1;
}

void mmx_codec_window_mdct(MMXCodec *c, const MMXAudioBuffer *buf, long long start, unsigned int ch, float *out)
{
    long long i;
    unsigned int nch = buf->channels;
    for (i = 0; i < MMX_WIN; i++)
    {
        long long p = start + i;
        c->win[i] = (p >= 0 && p < (long long)buf->frame_count) ? buf->samples[(size_t)p * nch + ch] : 0.0f;
    }
    mmx_mdct_forward(&c->mdct, c->win, out);
}

void mmx_codec_short_window(const MMXCodec *c, const MMXAudioBuffer *buf, long long start, unsigned int ch,
                            unsigned int group, float *out)
{
    long long i, base = start + (long long)c->short_offset + (long long)group * MMX_SHORT_M;
    unsigned int nch = buf->channels;
    for (i = 0; i < 2 * MMX_SHORT_M; i++)
    {
        long long p = base + i;
        out[i] = (p >= 0 && p < (long long)buf->frame_count) ? buf->samples[(size_t)p * nch + ch] : 0.0f;
    }
}

void mmx_codec_window_mdct_bt(MMXCodec *c, const MMXAudioBuffer *buf, long long start, unsigned int ch,
                              unsigned int block_type, float *out)
{
    long long i;
    unsigned int nch = buf->channels, g;
    if (block_type == MMX_BT_SHORT)
    {
        for (g = 0; g < MMX_SHORT_GROUPS; g++)
        {
            mmx_codec_short_window(c, buf, start, ch, g, c->swin);
            mmx_mdct_forward(&c->short_mdct, c->swin, out + g * MMX_SHORT_M);
        }
        return;
    }
    for (i = 0; i < MMX_WIN; i++)
    {
        long long p = start + i;
        c->win[i] = (p >= 0 && p < (long long)buf->frame_count) ? buf->samples[(size_t)p * nch + ch] : 0.0f;
    }
    mmx_mdct_forward_w(&c->mdct, c->win, block_type == MMX_BT_START ? c->win_start : block_type == MMX_BT_STOP ? c->win_stop : c->mdct.window, out);
}

void mmx_codec_lr_to_ms(float *l, float *r, unsigned long m)
{
    unsigned long k;
    for (k = 0; k < m; k++)
    {
        float mid = 0.5f * (l[k] + r[k]);
        float side = 0.5f * (l[k] - r[k]);
        l[k] = mid;
        r[k] = side;
    }
}

void mmx_codec_ms_to_lr(float *m_, float *s, unsigned long m)
{
    unsigned long k;
    for (k = 0; k < m; k++)
    {
        float l = m_[k] + s[k];
        float r = m_[k] - s[k];
        m_[k] = l;
        s[k] = r;
    }
}

void mmx_codec_source_coefs_bt(MMXCodec *c, const MMXAudioBuffer *decoded, unsigned int n_sources,
                               const long long *src_start, int stereo_ms, unsigned int block_type)
{
    unsigned int s, ch;
    for (s = 0; s < n_sources; s++)
    {
        for (ch = 0; ch < c->channels; ch++)
            mmx_codec_window_mdct_bt(c, decoded, src_start[s], ch, block_type, c->src[s][ch]);
        if (stereo_ms && c->channels == 2)
            mmx_codec_lr_to_ms(c->src[s][0], c->src[s][1], MMX_HOP);
    }
}

void mmx_codec_source_coefs(MMXCodec *c, const MMXAudioBuffer *decoded, unsigned int n_sources,
                            const long long *src_start, int stereo_ms)
{
    mmx_codec_source_coefs_bt(c, decoded, n_sources, src_start, stereo_ms, MMX_BT_LONG);
}

void mmx_codec_predict(MMXCodec *c, const MMXFrameSyntax *s, unsigned int n_sources)
{
    unsigned int ch, src, b, g;
    const MMXBandLayout *L = s->block_type == MMX_BT_SHORT ? &c->short_bands : &c->bands;
    unsigned int groups = s->block_type == MMX_BT_SHORT ? MMX_SHORT_GROUPS : 1;
    unsigned long gm = s->block_type == MMX_BT_SHORT ? MMX_SHORT_M : MMX_HOP;

    for (ch = 0; ch < c->channels; ch++)
    {
        memset(c->pred[ch], 0, sizeof(float) * MMX_HOP);
        for (src = 0; src < n_sources; src++)
            for (g = 0; g < groups; g++)
                for (b = 0; b < L->band_count; b++)
                {
                    double gain = mmx_gain_value(s->gain[src][ch][L->eq_band[b]], s->polarity[src][ch]);
                    unsigned long k;
                    if (gain == 0.0)
                        continue;
                    for (k = L->band_start[b]; k < L->band_start[b + 1]; k++)
                        c->pred[ch][g * gm + k] += (float)(gain * c->src[src][ch][g * gm + k]);
                }
    }
}

void mmx_codec_overlap_add(const MMXAudioBuffer *out, long long start, unsigned int ch, const float *y, unsigned long n)
{
    long long i, lo = start < 0 ? -start : 0;
    long long hi = (long long)n;
    if (start + hi > (long long)out->frame_count)
        hi = (long long)out->frame_count - start;
    for (i = lo; i < hi; i++)
        out->samples[(size_t)(start + i) * out->channels + ch] += y[i];
}

/* the reconstruction up to the finished coded-domain spectrum c->rec (prediction added, TNS undone,
   noise bands, band replication); no EPB fill */
static void reconstruct_core(MMXCodec *c, const MMXFrameSyntax *s, long long start)
{
    unsigned int ch;
    unsigned long k;

    if (s->block_type == MMX_BT_SHORT)
        mmx_frame_dequantize_short(&c->short_bands, s, c->tmp);
    else
        mmx_frame_dequantize(&c->bands, s, c->tmp, (unsigned long)((start + MMX_HOP) / MMX_HOP));
    for (ch = 0; ch < c->channels; ch++)
    {
        unsigned int b;
        if (s->block_type != MMX_BT_SHORT && s->tns[ch].active)
            mmx_tns_inverse(&s->tns[ch], c->tmp[ch], c->tns_k0, c->tns_k1);
        for (k = 0; k < MMX_HOP; k++)
            c->rec[ch][k] = c->pred[ch][k] + c->tmp[ch][k];
        /* noise bands replace the band as a whole, the prediction is not added */
        if (s->block_type != MMX_BT_SHORT)
            for (b = 0; b < c->bands.band_count; b++)
                if (s->band_noise[ch][b])
                    for (k = c->bands.band_start[b]; k < c->bands.band_start[b + 1]; k++)
                        c->rec[ch][k] = c->tmp[ch][k];
    }
    /* band replication: the region above the crossover is regenerated from the
       finished reconstruction below it (coded channel domain, so a M/S pair
       keeps its correlation). The source coefficients are all below the
       crossover and therefore final. */
    mmx_bwe_regenerate(&c->bands, s, c->rec, (unsigned long)((start + MMX_HOP) / MMX_HOP));
    /* EXPERIMENT (MMX_BWE_TNS=1): the frame's TNS filter, which the encoder fitted over the whole
       spectrum, is run over the replicated band too, so the regenerated coefficients carry the
       frame's temporal envelope (hi-hats) instead of a flat 23 ms wash. No extra bits: the filter
       is already in the frame. Symmetric: encoder and decoder both reconstruct through here. */
    {
        static int bwe_tns = -1;
        if (bwe_tns < 0) { const char *e = mmx_lab_getenv("MMX_BWE_TNS"); bwe_tns = e ? atoi(e) : 0; }
        if (bwe_tns && s->block_type != MMX_BT_SHORT && c->bands.bwe_band < c->bands.band_count)
            for (ch = 0; ch < c->channels; ch++)
                if (s->tns[ch].active)
                {
                    /* the all-pole synthesis raises the energy by the filter's gain: shape only, so every
                       replicated band is scaled back to the energy it carried before the filter */
                    double before[MMX_MAX_BANDS], after;
                    unsigned int b;
                    for (b = c->bands.bwe_band; b < c->bands.band_count; b++)
                    {
                        before[b] = 0.0;
                        for (k = c->bands.band_start[b]; k < c->bands.band_start[b + 1]; k++)
                            before[b] += (double)c->rec[ch][k] * c->rec[ch][k];
                    }
                    mmx_tns_inverse(&s->tns[ch], c->rec[ch], c->bands.band_start[c->bands.bwe_band], MMX_HOP);
                    for (b = c->bands.bwe_band; b < c->bands.band_count; b++)
                    {
                        after = 0.0;
                        for (k = c->bands.band_start[b]; k < c->bands.band_start[b + 1]; k++)
                            after += (double)c->rec[ch][k] * c->rec[ch][k];
                        if (after > 0.0)
                        {
                            float g = (float)sqrt(before[b] / after);
                            for (k = c->bands.band_start[b]; k < c->bands.band_start[b + 1]; k++)
                                c->rec[ch][k] *= g;
                        }
                    }
                }
    }
}

/* M/S and intensity back to L/R, inverse transform, overlap-add: consumes c->rec */
static void synth_out(MMXCodec *c, const MMXFrameSyntax *s, MMXAudioBuffer *out, long long start)
{
    unsigned int ch, g, b;
    unsigned long k;
    if (s->stereo_ms && c->channels == 2)
        mmx_codec_ms_to_lr(c->rec[0], c->rec[1], MMX_HOP);
    /* intensity bands: the coded mid (prediction of the mid included) scaled to L and R */
    if (s->block_type != MMX_BT_SHORT && c->channels == 2)
        for (b = 0; b < c->bands.band_count; b++)
            if (s->band_is[b])
            {
                float gl, gr;
                mmx_is_gains(s->is_pos[b], &gl, &gr);
                for (k = c->bands.band_start[b]; k < c->bands.band_start[b + 1]; k++)
                {
                    float m = (s->stereo_ms ? c->pred[0][k] : 0.5f * (c->pred[0][k] + c->pred[1][k])) + c->tmp[0][k];
                    c->rec[0][k] = gl * m;
                    c->rec[1][k] = gr * m;
                }
            }
    for (ch = 0; ch < c->channels; ch++)
    {
        if (s->block_type == MMX_BT_SHORT)
        {
            for (g = 0; g < MMX_SHORT_GROUPS; g++)
            {
                mmx_mdct_inverse(&c->short_mdct, c->rec[ch] + g * MMX_SHORT_M, c->sy);
                mmx_codec_overlap_add(out, start + (long long)c->short_offset + (long long)g * MMX_SHORT_M, ch, c->sy, 2 * MMX_SHORT_M);
            }
        }
        else
        {
            mmx_mdct_inverse_w(&c->mdct, c->rec[ch],
                               s->block_type == MMX_BT_START ? c->win_start : s->block_type == MMX_BT_STOP ? c->win_stop : c->mdct.window, c->y);
            mmx_codec_overlap_add(out, start, ch, c->y, MMX_WIN);
        }
    }
}

void mmx_codec_reconstruct(MMXCodec *c, const MMXFrameSyntax *s, MMXAudioBuffer *out, long long start)
{
    reconstruct_core(c, s, start);
    if (!c->bands.epb_ref)
        mmx_epb_finish(&c->bands, s, c->rec, (unsigned long)((start + MMX_HOP) / MMX_HOP));   /* EPB: every band to its energy */
    synth_out(c, s, out, start);
}

/* decoder: ref receives what later frames predict from, out what the listener gets; the two differ
   only in the EPB fill of reference mode 1 */
void mmx_codec_reconstruct_out(MMXCodec *c, const MMXFrameSyntax *s, MMXAudioBuffer *ref, MMXAudioBuffer *out, long long start)
{
    float keep[MMX_MAX_CH][MMX_HOP];
    unsigned int ch;
    if (ref == out)
    {
        mmx_codec_reconstruct(c, s, out, start);
        return;
    }
    reconstruct_core(c, s, start);
    for (ch = 0; ch < c->channels; ch++) memcpy(keep[ch], c->rec[ch], sizeof(float) * MMX_HOP);
    if (!c->bands.epb_ref)
        mmx_epb_finish(&c->bands, s, c->rec, (unsigned long)((start + MMX_HOP) / MMX_HOP));
    synth_out(c, s, ref, start);
    for (ch = 0; ch < c->channels; ch++) memcpy(c->rec[ch], keep[ch], sizeof(float) * MMX_HOP);
    mmx_epb_finish(&c->bands, s, c->rec, (unsigned long)((start + MMX_HOP) / MMX_HOP));
    synth_out(c, s, out, start);
}

void mmx_codec_fit_gains(MMXCodec *c, MMXFrameSyntax *s, unsigned int n_sources, float *const *target)
{
    const MMXBandLayout *L = s->block_type == MMX_BT_SHORT ? &c->short_bands : &c->bands;
    unsigned int groups = s->block_type == MMX_BT_SHORT ? MMX_SHORT_GROUPS : 1, grp;
    unsigned long gm = s->block_type == MMX_BT_SHORT ? MMX_SHORT_M : MMX_HOP;
    unsigned int cutoff = s->block_type == MMX_BT_SHORT ? c->short_cutoff_band : c->cutoff_band;
    unsigned int ch, e, src;

    for (ch = 0; ch < c->channels; ch++)
    {
        double pol_vote[MMX_MAX_SOURCES] = {0.0, 0.0};
        double g[MMX_MAX_SOURCES][MMX_EQ_BANDS];

        for (e = 0; e < MMX_EQ_BANDS; e++)
        {
            double a11 = 0.0, a12 = 0.0, a22 = 0.0, b1 = 0.0, b2 = 0.0;
            unsigned int b;
            for (grp = 0; grp < groups; grp++)
                for (b = 0; b < L->band_count && b < cutoff; b++)
                {
                    unsigned long k;
                    if (L->eq_band[b] != e)
                        continue;
                    for (k = grp * gm + L->band_start[b]; k < grp * gm + L->band_start[b + 1]; k++)
                    {
                        double t = target[ch][k], x1 = c->src[0][ch][k], x2 = n_sources > 1 ? c->src[1][ch][k] : 0.0;
                        a11 += x1 * x1; a12 += x1 * x2; a22 += x2 * x2;
                        b1 += x1 * t; b2 += x2 * t;
                    }
                }
            if (n_sources > 1)
            {
                double det = a11 * a22 - a12 * a12;
                if (det > 1e-12 * (a11 * a22 + 1e-30))
                {
                    g[0][e] = (b1 * a22 - b2 * a12) / det;
                    g[1][e] = (b2 * a11 - b1 * a12) / det;
                }
                else
                {
                    g[0][e] = a11 > 1e-20 ? b1 / a11 : 0.0;
                    g[1][e] = 0.0;
                }
            }
            else
            {
                g[0][e] = a11 > 1e-20 ? b1 / a11 : 0.0;
                g[1][e] = 0.0;
            }
            for (src = 0; src < n_sources; src++)
                pol_vote[src] += g[src][e] * (a11 + a22);
        }
        for (src = 0; src < n_sources; src++)
        {
            unsigned char pol = pol_vote[src] < 0.0;
            s->polarity[src][ch] = pol;
            for (e = 0; e < MMX_EQ_BANDS; e++)
            {
                double v = pol ? -g[src][e] : g[src][e];
                unsigned char dummy;
                if (v <= 0.0)
                    s->gain[src][ch][e] = MMX_GAIN_OFF;
                else
                    s->gain[src][ch][e] = mmx_gain_index(v, &dummy);
            }
        }
    }
}
