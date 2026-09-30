#include <stdlib.h>
#include <string.h>
#include "reference_graph.h"
#include "framecodec.h"

int mmx_refgraph_init(MMXReferenceGraph *g, unsigned long frame_count, unsigned char max_depth)
{
    memset(g, 0, sizeof(*g));
    g->frame_count = frame_count;
    g->max_depth = max_depth;
    g->depth = (unsigned char *)calloc(frame_count ? frame_count : 1, 1);
    g->used = (unsigned long *)calloc(frame_count ? frame_count : 1, sizeof(unsigned long));
    if (!g->depth || !g->used)
    {
        mmx_refgraph_free(g);
        return -1;
    }
    return 0;
}

void mmx_refgraph_free(MMXReferenceGraph *g)
{
    if (!g)
        return;
    free(g->depth);
    free(g->used);
    memset(g, 0, sizeof(*g));
}

/* Frames whose window overlaps [start, start+WIN): frame f covers [f*HOP-HOP, f*HOP+HOP). */
static void overlapping_frames(long long start, long long *first, long long *last)
{
    *first = (long long)((start + MMX_HOP) / MMX_HOP) - 1;          /* f*HOP+HOP > start */
    *last = (long long)((start + MMX_WIN - 1 + MMX_HOP) / MMX_HOP);  /* f*HOP-HOP < start+WIN */
    if (*first < 0) *first = 0;
}

int mmx_refgraph_window_depth(const MMXReferenceGraph *g, long long start)
{
    long long first, last, f;
    int d = 0;
    overlapping_frames(start, &first, &last);
    for (f = first; f <= last && f < (long long)g->frame_count; f++)
        if (g->depth[f] > d)
            d = g->depth[f];
    return d;
}

int mmx_refgraph_check_tentative(const MMXReferenceGraph *g, long long target_start, unsigned int n_sources,
                                 const long long *src_start, const unsigned char *tentative, unsigned long tentative_from)
{
    unsigned int s;
    int d = 0;
    if (n_sources == 0)
        return 0;
    for (s = 0; s < n_sources; s++)
    {
        long long first, last, f;
        if (src_start[s] < -(long long)MMX_HOP)
            return -1;
        /* source samples must be final: window ends at or before the target window start */
        if (src_start[s] + MMX_WIN > target_start)
            return -1;
        overlapping_frames(src_start[s], &first, &last);
        for (f = first; f <= last && f < (long long)g->frame_count; f++)
        {
            int df = (tentative && (unsigned long)f >= tentative_from) ? tentative[f] : g->depth[f];
            if (df > d)
                d = df;
        }
    }
    if (d + 1 > (int)g->max_depth)
        return -1;
    return d + 1;
}

int mmx_refgraph_check(const MMXReferenceGraph *g, long long target_start, unsigned int n_sources,
                       const long long *src_start)
{
    return mmx_refgraph_check_tentative(g, target_start, n_sources, src_start, NULL, 0);
}

void mmx_refgraph_set(MMXReferenceGraph *g, unsigned long f, unsigned char depth, unsigned int n_sources,
                      const long long *src_start)
{
    unsigned int s;
    if (f >= g->frame_count)
        return;
    g->depth[f] = depth;
    for (s = 0; s < n_sources; s++)
    {
        long long first, last, k;
        overlapping_frames(src_start[s], &first, &last);
        for (k = first; k <= last && k < (long long)g->frame_count; k++)
            g->used[k]++;
    }
}

void mmx_refgraph_unset(MMXReferenceGraph *g, unsigned long f, unsigned int n_sources, const long long *src_start)
{
    unsigned int s;
    if (f >= g->frame_count)
        return;
    g->depth[f] = 0;
    for (s = 0; s < n_sources; s++)
    {
        long long first, last, k;
        overlapping_frames(src_start[s], &first, &last);
        for (k = first; k <= last && k < (long long)g->frame_count; k++)
            if (g->used[k]) g->used[k]--;
    }
}
