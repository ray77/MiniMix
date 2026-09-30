#ifndef MMX_SIMILARITY_H
#define MMX_SIMILARITY_H

/* Result of matching a target region against a source region of the timeline. */
typedef struct
{
    unsigned long long source_frame;  /* best aligned source position */
    double gain;                      /* least-squares gain, target ~= gain * source */
    double ncc;                       /* normalized cross-correlation, -1..1 */
} MMXMatch;

/* Searches for the best alignment of target (n frames, interleaved) inside the
   signal `source` (source_frames frames) around `candidate_frame`, testing
   offsets in [-max_delay, +max_delay]. The source window must end at or before
   `limit_frame` (exclusive) so the decoder can resolve it. Coarse-to-fine:
   decimated mono correlation first, full-rate refinement afterwards. Returns 0
   on success, -1 if no valid alignment exists. */
int mmx_similarity_best_match(const float *target, unsigned long n, unsigned int channels,
                              const float *source, unsigned long long source_frames,
                              unsigned long long candidate_frame, unsigned long max_delay,
                              unsigned long long limit_frame, MMXMatch *out);

/* Full-rate NCC and gain for one exact alignment. */
void mmx_similarity_measure(const float *target, const float *source, unsigned long n,
                            unsigned int channels, double *ncc, double *gain);

#endif
