#ifndef MMX_LL2CODER_H
#define MMX_LL2CODER_H
/* LL2 residual coder: bitplane coding of a block of residuals with context mixing (lossless core, revision 10). Integer-only C port of the design of sac's BitplaneCoder (github.com/slmdev/sac, its files
 * coder/bpn.cpp and model/, MIT licence, (c) 2024 Sebastian Lehmann): residuals are mapped to unsigned
 * values and coded plane by plane from the highest set bit down, so every sample's context contains the higher
 * planes of its neighbours on BOTH sides. Per bit: significance contexts from the 16 nearest neighbours and a +-32
 * count, refinement contexts from neighbouring bits, a Laplace estimate from the +-32 local mean, logistic mixing
 * selected by context, two SSE stages and a final mix. Differences to sac: the logit tables are built from a
 * 33-point integer anchor table with linear interpolation instead of log()/exp() at start-up, and the Laplace
 * estimate is squash(-256 * plane / mean) in integer arithmetic instead of exp() - identical bits on every machine.
 */
#include "rangecoder.h"

typedef struct MMXLl2Coder MMXLl2Coder;

/* The model state is large (~0.5 MB); it is allocated once and reset per block. */
MMXLl2Coder *mmx_ll2c_new(void);
void mmx_ll2c_free(MMXLl2Coder *c);
/* 0 (default, CD profile): every bit through the model; d > 0: refinement bits d or more planes below the sample's
   top bit are coded raw (hi-res profile) */
void mmx_ll2c_set_raw_depth(MMXLl2Coder *c, int depth);

/* Codes n signed residuals as one block (the model restarts with every block, like sac's frames). */
void mmx_ll2c_encode(MMXLl2Coder *c, MMXRangeEncoder *rc, const int *res, int n);
void mmx_ll2c_decode(MMXLl2Coder *c, MMXRangeDecoder *rc, int *res, int n);
/* The same, but the model (counters, mixers, SSE) carries on from the previous block instead of restarting: for
   short blocks in file order (the decoder walks them in the same order). The first call after mmx_ll2c_new starts
   from the initial model. */
void mmx_ll2c_encode_cont(MMXLl2Coder *c, MMXRangeEncoder *rc, const int *res, int n);
void mmx_ll2c_decode_cont(MMXLl2Coder *c, MMXRangeDecoder *rc, int *res, int n);
/* The same for 64-bit residuals (the codec's own buffers): values in [-(2^31 - 1), 2^31]. */
void mmx_ll2c_encode_cont64(MMXLl2Coder *c, MMXRangeEncoder *rc, const long long *res, int n);
void mmx_ll2c_decode_cont64(MMXLl2Coder *c, MMXRangeDecoder *rc, long long *res, int n);

#endif
