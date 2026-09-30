#include <stdlib.h>
#include <string.h>
#include "mmx_io.h"
#include "rangecoder.h"
#include "log.h"

/* Compact block table (table format 1): one adaptive range-coded stream
   instead of 48-byte rows. A six-minute track has ~5000 blocks, so the rows
   cost 230 KB - 3 % of a quality-7 file and 5 % of a --future file, and xz
   proved they are almost entirely redundant: start frames and payload offsets
   are cumulative, source windows of consecutive blocks continue each other or
   sit a typical lag before the block, sizes and depths are small numbers.
   Coded per block: start frame delta to the expected continuation, frame
   count, number of sources, depth, flags, per source the delta of its window
   start to the prediction (the previous block's source continued, otherwise
   one window before the block), payload size, CRC-32 as raw bits. */

typedef struct
{
    MMXProb start[24], frames[24], depth[24], flags[24], payload[24];
    MMXProb nsrc[4];
    MMXProb src[MMX_MAX_BLOCK_SOURCES][2][22];   /* [source][prediction is a continuation][seg contexts] */
} TableContexts;

static void contexts_init(TableContexts *c)
{
    MMXProb *p = (MMXProb *)c;
    size_t i, n = sizeof(*c) / sizeof(MMXProb);
    for (i = 0; i < n; i++)
        p[i] = MMX_PROB_INIT;
}

static void tree_enc(MMXRangeEncoder *e, MMXProb *p, unsigned int v)
{
    mmx_rc_enc_bit(e, &p[1], (v >> 1) & 1);
    mmx_rc_enc_bit(e, &p[2 + ((v >> 1) & 1)], v & 1);
}

static unsigned int tree_dec(MMXRangeDecoder *d, MMXProb *p)
{
    unsigned int hi = mmx_rc_dec_bit(d, &p[1]);
    return (hi << 1) | mmx_rc_dec_bit(d, &p[2 + hi]);
}

static long long source_prediction(const MMXBlockEntry *prev, const MMXBlockEntry *b, unsigned int s,
                                   unsigned long hop, int *continuation)
{
    if (prev && prev->n_sources > s)
    {
        *continuation = 1;
        return prev->src_start[s] + (long long)prev->frame_count * (long long)hop;
    }
    *continuation = 0;
    return (long long)b->start_frame * (long long)hop - (long long)hop - 2 * (long long)hop;
}

unsigned long mmx_table_encode(const MMXFile *f, unsigned char **data)
{
    MMXRangeEncoder e;
    TableContexts ctx;
    unsigned long i, expected = 0, size;
    const MMXBlockEntry *prev = NULL;

    *data = NULL;
    contexts_init(&ctx);
    mmx_rc_enc_init(&e);
    for (i = 0; i < f->block_count; i++)
    {
        const MMXBlockEntry *b = &f->blocks[i];
        unsigned int s;
        mmx_rc_enc_ueg(&e, ctx.start, b->start_frame >= expected ? b->start_frame - expected : 0);
        mmx_rc_enc_ueg(&e, ctx.frames, b->frame_count);
        tree_enc(&e, ctx.nsrc, b->n_sources);
        mmx_rc_enc_ueg(&e, ctx.depth, b->depth);
        mmx_rc_enc_ueg(&e, ctx.flags, b->flags);
        for (s = 0; s < b->n_sources; s++)
        {
            int cont;
            long long pred = source_prediction(prev, b, s, f->hop, &cont);
            mmx_rc_enc_seg(&e, ctx.src[s][cont], (long long)(b->src_start[s] - pred));
        }
        mmx_rc_enc_ueg(&e, ctx.payload, b->payload_size);
        mmx_rc_enc_bypass(&e, b->crc32 >> 16, 16);
        mmx_rc_enc_bypass(&e, b->crc32 & 0xFFFFUL, 16);
        expected = b->start_frame + b->frame_count;
        prev = b;
    }
    mmx_rc_enc_finish(&e);
    if (e.failed)
    {
        mmx_rc_enc_free(&e);
        return 0;
    }
    size = e.size;
    *data = e.data;
    e.data = NULL;
    mmx_rc_enc_free(&e);
    return size;
}

int mmx_table_decode(MMXFile *f, const unsigned char *data, unsigned long size, unsigned long count,
                     unsigned long long payload_offset)
{
    MMXRangeDecoder d;
    TableContexts ctx;
    unsigned long i, expected = 0;
    unsigned long long running = 0;
    MMXBlockEntry prev_entry;
    const MMXBlockEntry *prev = NULL;

    contexts_init(&ctx);
    mmx_rc_dec_init(&d, data, size);
    for (i = 0; i < count; i++)
    {
        MMXBlockEntry b;
        unsigned int s;
        memset(&b, 0, sizeof(b));
        b.start_frame = expected + mmx_rc_dec_ueg(&d, ctx.start);
        b.frame_count = mmx_rc_dec_ueg(&d, ctx.frames);
        b.n_sources = (unsigned char)tree_dec(&d, ctx.nsrc);
        b.depth = (unsigned char)mmx_rc_dec_ueg(&d, ctx.depth);
        b.flags = (unsigned short)mmx_rc_dec_ueg(&d, ctx.flags);
        if (b.n_sources > MMX_MAX_BLOCK_SOURCES || d.failed)
        {
            mmx_error("Corrupt block table entry %lu", i);
            return -1;
        }
        for (s = 0; s < b.n_sources; s++)
        {
            int cont;
            long long pred = source_prediction(prev, &b, s, f->hop, &cont);
            b.src_start[s] = pred + mmx_rc_dec_seg(&d, ctx.src[s][cont]);
        }
        b.payload_size = mmx_rc_dec_ueg(&d, ctx.payload);
        b.crc32 = (mmx_rc_dec_bypass(&d, 16) << 16) | mmx_rc_dec_bypass(&d, 16);
        b.payload_offset = payload_offset + running;
        running += b.payload_size;
        expected = b.start_frame + b.frame_count;
        if (d.failed)
        {
            mmx_error("Corrupt block table (entry %lu)", i);
            return -1;
        }
        if (mmx_file_add_block(f, &b) < 0)
            return -1;
        prev_entry = b;
        prev = &prev_entry;
    }
    return 0;
}
