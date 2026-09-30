#ifndef MMX_DECODER_H
#define MMX_DECODER_H
#include "minimix/audio_buffer.h"
#include "minimix/mmx_format.h"

/* Resolves the timeline into PCM in one sequential pass. The output buffer is
   the block cache: references read straight from already decoded samples. */
int mmx_decoder_decode(const MMXFile *f, MMXAudioBuffer *out);

/* The same pass in steps, for a player that wants to start before the file is done:
   `out` is sized to the whole file at open and filled front to back; after every step the
   first mmx_decoder_valid_frames() PCM frames of `out` are final (references only ever point
   backwards and the overlap-add reaches one hop back, so nothing already counted changes).
   mmx_decoder_step decodes up to max_hops frames of 1024 samples and returns 1 while more
   remain, 0 when the file is complete, -1 on a bitstream error (out is then unusable). */
typedef struct MMXDecoder MMXDecoder;
int mmx_decoder_open(const MMXFile *f, MMXAudioBuffer *out, MMXDecoder **d);
int mmx_decoder_step(MMXDecoder *d, unsigned long max_hops);
unsigned long long mmx_decoder_valid_frames(const MMXDecoder *d);
void mmx_decoder_close(MMXDecoder *d);

/* Patchwork landscape decoding (lossless files with entry points, header byte 95): the file is a row of segments that
   decode side by side (lanes on several cores), so "final" is no longer only a prefix. mmx_decoder_valid_frames stays
   the final prefix from the start. mmx_decoder_focus makes the segment containing `frame` the next one to decode (a
   player's seek target); mmx_decoder_final_from returns the end of the final run of frames starting at `frame`
   (`frame` itself when it is not final yet); mmx_decoder_segments copies each segment's first frame and the end of
   its final part into lo/final (up to max) and returns the number of segments. Files without entry points are one
   segment. Call these from the thread that calls mmx_decoder_step. */
void mmx_decoder_focus(MMXDecoder *d, unsigned long long frame);
unsigned long long mmx_decoder_final_from(const MMXDecoder *d, unsigned long long frame);
unsigned long mmx_decoder_segments(const MMXDecoder *d, unsigned long long *lo, unsigned long long *final,
                                   unsigned long max);
/* Where a seek to `frame` should land (decode where the seek aims; only where that is too dear - songs whose segments
   read many earlier ones - move to an easier neighbour, but always close to the aim): `frame` itself
   when it is final already or when at most `budget` frames still have to be decoded to reach it (its own segment up
   to it and the segments its references read); otherwise the nearest of the neighbouring entry points (this
   segment's start, the next, the previous) that fits the budget, else the cheapest of them. Files without entry
   points: `frame`. `budget`: what the decoder does in the wait a player accepts. Follow with mmx_decoder_focus. */
unsigned long long mmx_decoder_seek_point(const MMXDecoder *d, unsigned long long frame, unsigned long long budget);
#endif
