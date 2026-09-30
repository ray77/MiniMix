#ifndef MMX_REFERENCE_GRAPH_H
#define MMX_REFERENCE_GRAPH_H

/* Reference relationships on the frame timeline: per-frame chain depth, usage
   counts and the validity rules for an edge (backwards only, source samples
   final, depth limit). Cycles are impossible by construction; the checks make
   sure of it anyway. */

typedef struct
{
    unsigned long frame_count;
    unsigned char *depth;       /* per frame */
    unsigned long *used;        /* how many frames reference this frame's region */
    unsigned char max_depth;
} MMXReferenceGraph;

int mmx_refgraph_init(MMXReferenceGraph *g, unsigned long frame_count, unsigned char max_depth);
void mmx_refgraph_free(MMXReferenceGraph *g);

/* Depth a frame at window start `target_start` would get from source windows
   src_start[0..n_sources). Returns -1 when an edge is invalid. */
int mmx_refgraph_check(const MMXReferenceGraph *g, long long target_start, unsigned int n_sources,
                       const long long *src_start);
/* Same, but frames from `tentative_from` on take their depth from `tentative`
   (a planner's not yet committed path) instead of the graph; NULL = graph only. */
int mmx_refgraph_check_tentative(const MMXReferenceGraph *g, long long target_start, unsigned int n_sources,
                                 const long long *src_start, const unsigned char *tentative, unsigned long tentative_from);

/* Records the decision for frame f. */
void mmx_refgraph_set(MMXReferenceGraph *g, unsigned long f, unsigned char depth, unsigned int n_sources,
                      const long long *src_start);
/* Takes a recorded decision back (same arguments; the encoder's trial coding). */
void mmx_refgraph_unset(MMXReferenceGraph *g, unsigned long f, unsigned int n_sources, const long long *src_start);

/* Maximum depth of the frames overlapping window [start, start + WIN). */
int mmx_refgraph_window_depth(const MMXReferenceGraph *g, long long start);

#endif
