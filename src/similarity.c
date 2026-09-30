#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "minimix/similarity.h"

#define DECIMATE 8

void mmx_similarity_measure(const float *target, const float *source, unsigned long n,
                            unsigned int channels, double *ncc, double *gain)
{
    unsigned long total = n * channels, i;
    double xy = 0.0, xx = 0.0, yy = 0.0;
    for (i = 0; i < total; i++)
    {
        double x = target[i], y = source[i];
        xy += x * y;
        xx += x * x;
        yy += y * y;
    }
    *gain = (yy > 1e-20) ? xy / yy : 0.0;
    *ncc = (xx > 1e-20 && yy > 1e-20) ? xy / sqrt(xx * yy) : 0.0;
}

static void decimate_mono(const float *p, unsigned long frames, unsigned int channels, double *out, unsigned long out_len)
{
    unsigned long i, c, k;
    for (k = 0; k < out_len; k++)
    {
        double s = 0.0;
        unsigned long from = k * DECIMATE, to = from + DECIMATE;
        if (to > frames) to = frames;
        for (i = from; i < to; i++)
            for (c = 0; c < channels; c++)
                s += p[i * channels + c];
        out[k] = s;
    }
}

int mmx_similarity_best_match(const float *target, unsigned long n, unsigned int channels,
                              const float *source, unsigned long long source_frames,
                              unsigned long long candidate_frame, unsigned long max_delay,
                              unsigned long long limit_frame, MMXMatch *out)
{
    long long lo, hi, pos, best_coarse = 0, refine_lo, refine_hi;
    unsigned long tn, k;
    double *tdec, *sdec;
    double t_energy = 0.0, best_score = -2.0;
    long long win_start, win_len;

    memset(out, 0, sizeof(*out));
    if (n < DECIMATE * 2 || limit_frame < n || source_frames < n)
        return -1;

    lo = (long long)candidate_frame - (long long)max_delay;
    hi = (long long)candidate_frame + (long long)max_delay;
    if (lo < 0) lo = 0;
    if (hi > (long long)limit_frame - (long long)n) hi = (long long)limit_frame - (long long)n;
    if (hi > (long long)source_frames - (long long)n) hi = (long long)source_frames - (long long)n;
    if (hi < lo)
        return -1;

    /* coarse stage on a decimated mono downmix of the whole search window */
    tn = n / DECIMATE;
    win_start = lo;
    win_len = hi - lo + (long long)n;
    tdec = (double *)malloc(sizeof(double) * tn);
    sdec = (double *)malloc(sizeof(double) * ((unsigned long)win_len / DECIMATE + 2));
    if (!tdec || !sdec)
    {
        free(tdec); free(sdec);
        return -1;
    }
    decimate_mono(target, n, channels, tdec, tn);
    decimate_mono(source + (size_t)win_start * channels, (unsigned long)win_len, channels, sdec, (unsigned long)win_len / DECIMATE);
    for (k = 0; k < tn; k++)
        t_energy += tdec[k] * tdec[k];

    {
        unsigned long lags = (unsigned long)((hi - lo) / DECIMATE) + 1, lag;
        for (lag = 0; lag < lags; lag++)
        {
            double xy = 0.0, yy = 0.0, score;
            for (k = 0; k < tn; k++)
            {
                double y = sdec[lag + k];
                xy += tdec[k] * y;
                yy += y * y;
            }
            score = (yy > 1e-20 && t_energy > 1e-20) ? xy / sqrt(yy * t_energy) : 0.0;
            if (score > best_score)
            {
                best_score = score;
                best_coarse = lo + (long long)lag * DECIMATE;
            }
        }
    }
    free(tdec);
    free(sdec);

    /* fine stage: full-rate, all channels, +-DECIMATE around the coarse winner */
    refine_lo = best_coarse - DECIMATE;
    refine_hi = best_coarse + DECIMATE;
    if (refine_lo < lo) refine_lo = lo;
    if (refine_hi > hi) refine_hi = hi;

    best_score = -2.0;
    for (pos = refine_lo; pos <= refine_hi; pos++)
    {
        double ncc, gain;
        mmx_similarity_measure(target, source + (size_t)pos * channels, n, channels, &ncc, &gain);
        if (ncc > best_score)
        {
            best_score = ncc;
            out->source_frame = (unsigned long long)pos;
            out->ncc = ncc;
            out->gain = gain;
        }
    }
    return best_score > -2.0 ? 0 : -1;
}
