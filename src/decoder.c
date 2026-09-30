#include "lab.h"
#include "workers.h"
#include "ll2codec.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include "minimix/decoder.h"
#include "minimix/codec.h"
#include <math.h>
#include "lossless.h"
#include "crc32.h"
#include "log.h"

/* Readout of what a finished file plays in its referenced frames
   (MMX_DEBUG_ECHO_DECODE=<from>:<to> in seconds, debug only, nothing is
   decoded differently): per coded band of a referenced frame, whether the
   residual is zero while the prediction carries audible energy - those bands
   play the earlier bar and nothing else. `abs_thr` is the absolute threshold
   of hearing per coefficient, the weakest yardstick there is. */
typedef struct
{
    int on;
    double from, to;
    unsigned long frames, bands, zero, zero_loud, gain_on;
    double worst_db;
    double worst_s;
} DecodeEchoProbe;

static void decode_echo_probe(DecodeEchoProbe *p, const MMXCodec *codec, const MMXFrameSyntax *s,
                              unsigned int n_sources, long long start, unsigned long sample_rate)
{
    double t = (double)start / (double)sample_rate;
    unsigned int c, b;
    unsigned long k;
    if (!p->on || !n_sources || s->block_type == MMX_BT_SHORT || t < p->from || t > p->to)
        return;
    p->frames++;
    for (c = 0; c < s->channels; c++)
        for (b = 0; b < codec->cutoff_band; b++)
        {
            double e = 0.0, n = (double)(codec->bands.band_start[b + 1] - codec->bands.band_start[b]), db;
            if (s->band_noise[c][b])
                continue;
            for (k = codec->bands.band_start[b]; k < codec->bands.band_start[b + 1]; k++)
                e += (double)codec->pred[c][k] * codec->pred[c][k];
            if (e <= 0.0)
                continue;
            p->bands++;
            p->gain_on++;
            if (!s->band_zero[c][b])
                continue;
            p->zero++;
            db = 10.0 * log10(e / (codec->bands.abs_thr[b] * n + 1e-300));
            if (db <= 0.0)
                continue;
            p->zero_loud++;
            if (db > p->worst_db) { p->worst_db = db; p->worst_s = t; }
        }
}

/* Everything the sequential pass carries from one frame to the next, so the pass can pause
   between any two frames. One block is open at a time: its contexts and range decoder live
   here from its first frame to its last. */
struct MMXDecoder
{
    const MMXFile *f;
    MMXAudioBuffer *out;
    MMXAudioBuffer *ref;                 /* what later frames predict from: out itself, or a fill-free copy (EPB reference mode 1) */
    MMXAudioBuffer ref_buf;
    int lossless;
    unsigned long block;             /* the block being decoded */
    unsigned long frame;             /* the next frame inside it */
    int block_open;
    unsigned long nf;                /* frames in the timeline */
    unsigned long next_frame;        /* lossy: the timeline must continue here */
    unsigned long long valid;        /* PCM frames of `out` that are final */
    int done;
    MMXRangeDecoder rd;
    /* the MDCT codec */
    MMXCodec codec;
    MMXFrameSyntax syn;
    MMXCodecContexts ctx;
    DecodeEchoProbe probe;
    /* the lossless codec */
    long long *dec[MMX_MAX_CH], *x[MMX_MAX_CH];
    MMXLosslessContexts *lctx;       /* heap: about 200 KB, the player decodes on a small stack */
    /* the LL2 core (header byte 87 = 2): segments decoded by lanes (patchwork landscape decoding); NULL for the
       revision 6-8 core */
    struct Ll2Lane *lanes;           /* decoding contexts: one segment's whole state each (nctx of them) */
    unsigned int nctx, nlanes;       /* contexts, and how many of them run side by side */
    unsigned int *run;               /* the contexts running in this round */
    MMXWorkers *lane_wk;             /* runs the lanes side by side */
    MMXWorkers **slot_wk;            /* the channel helpers of each running slot */
    unsigned char *inclose;          /* the focus segment and every segment it reads from, transitively */
    unsigned long slot_cap;
    unsigned long nseg;              /* segments: 1 without entry points (header byte 95 = 0) */
    unsigned long *seg_b0;           /* first block of each segment, seg_b0[nseg] = block_count */
    unsigned char *seg_state;        /* 0 waiting, 1 in a lane, 2 done */
    long long *seg_lo, *seg_hi;      /* the segment's samples [lo, hi) */
    long long *seg_final;            /* samples [lo, seg_final) are final (written by the segment's lane) */
    long long *seg_seen;             /* seg_final as of the start of the round: what lanes read about other segments */
    unsigned long *dep_off, *dep;    /* dep[dep_off[s] .. dep_off[s+1]): the other segments s reads references from */
    unsigned long focus;             /* the segment to decode first (mmx_decoder_focus) */
    MMXLosslessFrame fr;
    unsigned int shift;
    unsigned int qdrop;                  /* near-lossless step multiplier (revision 9 byte 86), 1 = none */
    double scale;
    unsigned long len;
};

unsigned long long mmx_decoder_valid_frames(const MMXDecoder *d) { return d->valid; }

static unsigned long ll2_block_samples(const MMXDecoder *d, const MMXBlockEntry *b);
static int finish(MMXDecoder *d);
static int ll2_open(MMXDecoder *d);
static void ll2_close(MMXDecoder *d);

int mmx_decoder_open(const MMXFile *f, MMXAudioBuffer *out, MMXDecoder **dp)
{
    MMXDecoder *d;
    unsigned int c;
    const char *e;

    *dp = NULL;
    memset(out, 0, sizeof(*out));
    if (f->hop != MMX_HOP || (f->codec_id != MMX_CODEC_LOSSLESS && f->codec_id != MMX_CODEC_MDCT))
    {
        mmx_error("Unsupported codec id %u / hop %lu", f->codec_id, f->hop);
        return -1;
    }
    d = (MMXDecoder *)calloc(1, sizeof(*d));
    if (!d)
        return -1;
    d->f = f;
    d->out = out;
    d->lossless = f->codec_id == MMX_CODEC_LOSSLESS;
    d->nf = mmx_codec_frame_count(f->frame_count);
    if (mmx_audio_buffer_init(out, f->sample_rate, f->channels, f->frame_count) != 0)
        goto fail;
    out->source_bits = f->source_bits;
    if (d->lossless && f->ll_core == MMX_LL2_CORE && f->source_bits > 24 && mmx_audio_buffer_alloc_int(out) != 0)
        goto fail;                                   /* 32-bit sources: the exact integers besides the float */
    d->ref = out;
    if (f->lowrate && (f->epb_flags & 4u))
    {
        if (mmx_audio_buffer_init(&d->ref_buf, f->sample_rate, f->channels, f->frame_count) != 0)
            goto fail;
        d->ref = &d->ref_buf;
    }

    if (d->lossless)
    {
        unsigned int bits = f->source_bits ? f->source_bits : 16;
        d->scale = pow(2.0, (double)bits - 1.0);
        d->len = (unsigned long)f->frame_count;
        d->shift = f->ll_shift > 16 ? 16 : f->ll_shift;   /* the encoder coded x >> shift; older files carry 0 */
        d->qdrop = f->ll_qdrop > 1 ? f->ll_qdrop : 1;
        if (f->ll_core == MMX_LL2_CORE)
        {
            mmx_ll2_statics();
            if (!mmx_ll2_profile(f->ll2_profile))
            {
                mmx_error("This file uses LL2 profile %u, which needs a newer decoder", f->ll2_profile);
                goto fail;
            }
            if (f->channels > 2 || ll2_open(d) != 0)
                goto fail;
        }
        d->lctx = (MMXLosslessContexts *)malloc(sizeof(*d->lctx));
        if (!d->lctx)
            goto fail;
        for (c = 0; c < f->channels; c++)
        {
            d->dec[c] = (long long *)calloc(d->len ? d->len : 1, sizeof(long long));
            d->x[c] = (long long *)malloc(sizeof(long long) * MMX_HOP);
            if (!d->dec[c] || !d->x[c])
                goto fail;
        }
        mmx_ll_state_init(d->lctx);   /* the filter cascade is carried across blocks, in file order */
        mmx_ll_state_set_revision(d->lctx, f->bitstream_rev);   /* revisions 6-7: sign-sign cascade */
    }
    else
    {
        if (mmx_codec_init(&d->codec, f->sample_rate, f->channels, f->quality ? f->quality : MMX_QUALITY_MAX) != 0 ||
            mmx_frame_syntax_init(&d->syn, f->channels, MMX_HOP) != 0)
        {
            mmx_error("Decoder init failed");
            goto fail;
        }
        mmx_bands_set_bwe(&d->codec.bands, (double)f->bwe_hz, f->bwe_mode);   /* band replication as the header asks for it */
        mmx_bands_set_epb(&d->codec.bands, f->lowrate, f->epb_fold, f->epb_flags, f->epb_fill_db);   /* energy-preserving bands (revision 9) */
        e = mmx_lab_getenv("MMX_DEBUG_ECHO_DECODE");
        if (e)
        {
            const char *colon = strchr(e, ':');
            d->probe.on = 1;
            d->probe.from = atof(e);
            d->probe.to = colon ? atof(colon + 1) : d->probe.from + 20.0;
            d->probe.worst_db = -200.0;
        }
    }
    if (f->block_count == 0)
        d->done = 1;
    *dp = d;
    return 0;

fail:
    mmx_decoder_close(d);
    mmx_audio_buffer_free(out);
    return -1;
}

void mmx_decoder_close(MMXDecoder *d)
{
    unsigned int c;
    if (!d)
        return;
    if (d->lossless)
    {
        for (c = 0; c < MMX_MAX_CH; c++) { free(d->dec[c]); free(d->x[c]); }
        free(d->lctx);
        ll2_close(d);
    }
    else
    {
        mmx_frame_syntax_free(&d->syn);
        mmx_codec_free(&d->codec);
    }
    if (d->ref == &d->ref_buf)
        mmx_audio_buffer_free(&d->ref_buf);
    free(d);
}

/* One lossless frame: integers into the block cache, floats into the output right away. */
static int lossless_frame(MMXDecoder *d, const MMXBlockEntry *b, unsigned long i, unsigned long j)
{
    const MMXFile *f = d->f;
    long long *srcp[MMX_MAX_CH], *srcp2[MMX_MAX_CH];
    unsigned long fr_idx = b->start_frame + j;
    long long start = (long long)fr_idx * MMX_HOP - MMX_HOP, s = 0, s2 = 0;
    unsigned long count = start < 0 ? 0 : (start + MMX_HOP <= (long long)d->len ? MMX_HOP : (unsigned long)((long long)d->len - start));
    unsigned int c;
    unsigned long n;

    if (count == 0)
        return 0;
    if (b->n_sources)
    {
        s = b->src_start[0] + (long long)j * MMX_HOP;
        if (s < 0 || s + (long long)count > start)
        {
            mmx_error("Block %lu frame %lu references samples that are not decoded yet", i, j);
            return -1;
        }
    }
    if (b->n_sources > 1)
    {
        s2 = b->src_start[1] + (long long)j * MMX_HOP;
        if (s2 < 0 || s2 + (long long)count > start)
        {
            mmx_error("Block %lu frame %lu: second source references samples that are not decoded yet", i, j);
            return -1;
        }
    }
    for (c = 0; c < f->channels; c++)
    {
        srcp[c] = b->n_sources ? d->dec[c] + s : NULL;
        srcp2[c] = b->n_sources > 1 ? d->dec[c] + s2 : NULL;
    }
    if (mmx_ll_decode_frame(&d->rd, d->lctx, f->channels, d->x, b->n_sources ? srcp : NULL,
                            b->n_sources > 1 ? srcp2 : NULL, count, &d->fr) != 0)
    {
        mmx_error("Lossless bitstream error in block %lu frame %lu", i, j);
        return -1;
    }
    for (c = 0; c < f->channels; c++)
    {
        memcpy(d->dec[c] + start, d->x[c], sizeof(long long) * count);
        for (n = 0; n < count; n++)
            d->out->samples[(size_t)(start + (long long)n) * f->channels + c] =
                (float)((double)(d->x[c][n] * (1LL << d->shift) * (long long)d->qdrop) / d->scale);
    }
    return 0;
}

static unsigned long ll2_block_samples(const MMXDecoder *d, const MMXBlockEntry *b)
{
    unsigned long j, total = 0;
    for (j = 0; j < b->frame_count; j++)
    {
        long long st = (long long)(b->start_frame + j) * MMX_HOP - MMX_HOP;
        total += st < 0 ? 0 : (st + MMX_HOP <= (long long)d->len ? MMX_HOP : (unsigned long)((long long)d->len - st));
    }
    return total;
}

/* ------------------------------------------------------------------ LL2: lanes and segments

   Patchwork landscape decoding. A file with entry points (header byte 95) is a row of segments that decode
   independently: at a segment's first block predictor and coder start afresh and the regressors read nothing before
   it (the encoder does the same, src/encoder.c ll2_code). Only references cross segments: they read already decoded
   audio of earlier segments. A lane decodes one segment at a time with its own stream and the three-task pipeline
   (iteration k: residuals of block k, channel 0 of block k-1, channel 1 of block k-2 - disjoint data, the result is
   the sequential one). Lanes run side by side in rounds; between rounds, on the calling thread, the decoder notes
   finished segments, hands waiting segments to idle lanes (the focus first, a segment only once every segment it
   reads from is at least under way - references point backwards, so no lane can wait on a segment nobody decodes)
   and publishes progress. Inside a round a lane reads other segments' progress only from the snapshot taken before
   the round, so the decoded samples never depend on timing. Without entry points there is one segment and one
   lane: the pass of before. */
typedef struct Ll2Lane
{
    MMXDecoder *d;
    MMXLl2Stream l2;
    int live;                           /* l2 initialised */
    MMXWorkers *wk;                     /* the running slot's helpers, set per round */
    unsigned int iters;                 /* pipeline iterations run in this round */
    long long *slot_res[3][2];
    long long slot_lo[3][2], slot_hi[3][2];
    long seg;                           /* the segment in progress, -1 = idle */
    unsigned long b0, b1, it;           /* its blocks [b0, b1) and the next pipeline iteration */
    int err[3];
    int finished;                       /* the segment completed during this round */
    unsigned long hops;                 /* frames made final during this round */
} Ll2Lane;

#define LL2_ROUND_ITERS 1               /* pipeline iterations per lane and round (a block is up to 64 frames) */
#define LL2_SRC_MARGIN 256              /* samples around a source the source stage may read (src/ll2codec.c) */

static long long seg_of_sample(const MMXDecoder *d, long long pos)
{
    unsigned long lo = 0, hi = d->nseg;
    while (hi - lo > 1)
    {
        unsigned long mid = (lo + hi) / 2;
        if (d->seg_lo[mid] <= pos) lo = mid; else hi = mid;
    }
    return (long long)lo;
}

/* the samples block b's references read: [lo, hi); none: lo == hi */
static void block_sources(const MMXDecoder *d, const MMXBlockEntry *b, long long *lo, long long *hi)
{
    long long span = (long long)b->frame_count * MMX_HOP;
    unsigned int k;
    *lo = *hi = 0;
    for (k = 0; k < b->n_sources && k < 2; k++)
    {
        long long a = b->src_start[k] - LL2_SRC_MARGIN, z = b->src_start[k] + span + LL2_SRC_MARGIN;
        if (a < 0) a = 0;
        if (z > (long long)d->len) z = (long long)d->len;
        if (*lo == *hi) { *lo = a; *hi = z; }
        else { if (a < *lo) *lo = a; if (z > *hi) *hi = z; }
    }
}

static int ll2_open(MMXDecoder *d)
{
    const MMXFile *f = d->f;
    const unsigned long rf = mmx_ll2_pld_frames(f->ll2_pld, f->sample_rate);
    unsigned long bi, s, mx = 1, ndep = 0, *mark;
    unsigned int k, c, want;
    const char *e = getenv("MMX_DECODE_LANES");
    (void)mmx_crc32(NULL, 0);                   /* its table is built on first use: here, before lanes run side by side */
    for (bi = 0; bi < f->block_count; bi++) { unsigned long v = ll2_block_samples(d, &f->blocks[bi]); if (v > mx) mx = v; }
    d->slot_cap = mx;
    /* segments: a new one at every block that starts on a multiple of rf frames */
    d->seg_b0 = (unsigned long *)malloc(sizeof(unsigned long) * (f->block_count + 2));
    if (!d->seg_b0) return -1;
    d->nseg = 0;
    d->seg_b0[d->nseg++] = 0;
    for (bi = 1; rf && bi < f->block_count; bi++)
        if (f->blocks[bi].start_frame % rf == 0) d->seg_b0[d->nseg++] = bi;
    d->seg_b0[d->nseg] = f->block_count;
    d->seg_state = (unsigned char *)calloc(d->nseg, 1);
    d->seg_lo = (long long *)calloc(d->nseg + 1, sizeof(long long));
    d->seg_hi = (long long *)calloc(d->nseg + 1, sizeof(long long));
    d->seg_final = (long long *)calloc(d->nseg + 1, sizeof(long long));
    d->seg_seen = (long long *)calloc(d->nseg + 1, sizeof(long long));
    d->dep_off = (unsigned long *)calloc(d->nseg + 1, sizeof(unsigned long));
    if (!d->seg_state || !d->seg_lo || !d->seg_hi || !d->seg_final || !d->seg_seen || !d->dep_off) return -1;
    for (s = 0; s < d->nseg; s++)
    {
        long long lo = f->block_count ? (long long)f->blocks[d->seg_b0[s]].start_frame * MMX_HOP - MMX_HOP : 0;
        long long hi = s + 1 < d->nseg ? (long long)f->blocks[d->seg_b0[s + 1]].start_frame * MMX_HOP - MMX_HOP : (long long)d->len;
        d->seg_lo[s] = lo < 0 ? 0 : lo > (long long)d->len ? (long long)d->len : lo;
        d->seg_hi[s] = hi < d->seg_lo[s] ? d->seg_lo[s] : hi > (long long)d->len ? (long long)d->len : hi;
        d->seg_final[s] = d->seg_seen[s] = d->seg_lo[s];
    }
    /* dependencies: the other segments each segment's references read (counted, then listed) */
    mark = (unsigned long *)calloc(d->nseg + 1, sizeof(unsigned long));
    if (!mark) return -1;
    for (k = 0; k < 2; k++)
    {
        ndep = 0;
        for (s = 0; s < d->nseg; s++)
        {
            unsigned long t, first = ndep;
            d->dep_off[s] = ndep;
            for (bi = d->seg_b0[s]; bi < d->seg_b0[s + 1]; bi++)
            {
                long long lo, hi;
                block_sources(d, &f->blocks[bi], &lo, &hi);
                if (lo == hi) continue;
                for (t = (unsigned long)seg_of_sample(d, lo); t <= (unsigned long)seg_of_sample(d, hi - 1); t++)
                {
                    unsigned long q;
                    int seen = t == s;
                    for (q = first; q < ndep && k && !seen; q++) if (d->dep[q] == t) seen = 1;
                    if (!k) { seen = seen || (ndep > first && mark[t] == s + 1); if (!seen) mark[t] = s + 1; }
                    if (!seen) { if (k) d->dep[ndep] = t; ndep++; }
                }
            }
        }
        d->dep_off[d->nseg] = ndep;
        if (!k && !(d->dep = (unsigned long *)malloc(sizeof(unsigned long) * (ndep ? ndep : 1)))) { free(mark); return -1; }
    }
    free(mark);
    if (mmx_lab_getenv("MMX_DEBUG_PLD"))                /* lab: the segments and what each one reads from */
        for (s = 0; s < d->nseg; s++)
        {
            unsigned long q;
            fprintf(stderr, "segment %lu (%.1f s):", s, (double)d->seg_lo[s] / f->sample_rate);
            for (q = d->dep_off[s]; q < d->dep_off[s + 1]; q++) fprintf(stderr, " %lu", d->dep[q]);
            fprintf(stderr, "\n");
        }
    /* lanes: one per (channels + 1) cores, at most one per segment (MMX_DECODE_LANES overrides) */
    want = (mmx_workers_cores() + f->channels) / (f->channels + 1u);   /* M1 (8 cores), stereo: 3 lanes (measured best) */
    if (e && atoi(e) >= 1) want = (unsigned int)atoi(e);
    if (want < 1) want = 1;
    if (want > 5) want = 5;
    if (want > d->nseg) want = (unsigned int)d->nseg;
    d->nlanes = want;
    /* contexts: a few more than lanes, so a seek can pause a segment instead of throwing its state away */
    d->nctx = d->nseg > 1 ? d->nlanes + 3 : 1;
    if (d->nctx > d->nseg) d->nctx = (unsigned int)d->nseg;
    d->lanes = (Ll2Lane *)calloc(d->nctx, sizeof(Ll2Lane));
    d->run = (unsigned int *)calloc(d->nlanes, sizeof(unsigned int));
    d->slot_wk = (MMXWorkers **)calloc(d->nlanes, sizeof(MMXWorkers *));
    d->inclose = (unsigned char *)calloc(d->nseg, 1);
    if (!d->lanes || !d->run || !d->slot_wk || !d->inclose) return -1;
    for (k = 0; k < d->nctx; k++)
    {
        Ll2Lane *L = &d->lanes[k];
        unsigned int k2;
        L->d = d;
        L->seg = -1;
        for (k2 = 0; k2 < 3; k2++)
            for (c = 0; c < f->channels; c++)
                if (!(L->slot_res[k2][c] = (long long *)malloc(sizeof(long long) * mx))) return -1;
    }
    for (k = 0; k < d->nlanes; k++)
        if (!(d->slot_wk[k] = mmx_workers_new(f->channels))) return -1;   /* channel predictors beside the lane */
    d->lane_wk = mmx_workers_new(d->nlanes - 1);
    return d->lane_wk ? 0 : -1;
}

static void ll2_close(MMXDecoder *d)
{
    unsigned int k, k2, c;
    for (k = 0; d->lanes && k < d->nctx; k++)
    {
        Ll2Lane *L = &d->lanes[k];
        if (L->live) mmx_ll2s_free(&L->l2);
        for (k2 = 0; k2 < 3; k2++) for (c = 0; c < 2; c++) free(L->slot_res[k2][c]);
    }
    for (k = 0; d->slot_wk && k < d->nlanes; k++) mmx_workers_free(d->slot_wk[k]);
    free(d->lanes); free(d->run); free(d->slot_wk); free(d->inclose);
    mmx_workers_free(d->lane_wk);
    free(d->seg_b0); free(d->seg_state); free(d->seg_lo); free(d->seg_hi); free(d->seg_final); free(d->seg_seen);
    free(d->dep_off); free(d->dep);
}

/* a fresh stream for segment s: what the encoder does at an entry point */
static int lane_start(Ll2Lane *L, unsigned long s)
{
    MMXDecoder *d = L->d;
    const MMXFile *f = d->f;
    unsigned int bits = f->source_bits ? f->source_bits : 16;
    if (L->live) { mmx_ll2s_free(&L->l2); L->live = 0; }
    if (mmx_ll2s_init(&L->l2, f->channels, f->ll2_decay, f->ll2_bank, f->ll2_xols, f->ll2_profile, f->ll2_own) != 0)
        return -1;
    L->live = 1;
    mmx_ll2s_set_rails(&L->l2, bits, d->shift);
    mmx_ll2s_set_rate(&L->l2, f->sample_rate);
    mmx_ll2s_set_mixer(&L->l2, f->ll2_mix, bits, d->shift);
    L->l2.seg_start = f->block_count ? (long long)f->blocks[d->seg_b0[s]].start_frame * MMX_HOP - MMX_HOP : 0;
    L->l2.store_from = d->seg_final[s];         /* a segment decoded again: its final part is not written twice */
    L->seg = (long)s;
    L->b0 = d->seg_b0[s];
    L->b1 = d->seg_b0[s + 1];
    L->it = 0;
    L->err[0] = L->err[1] = L->err[2] = 0;
    return 0;
}

static void ll2_task(void *ctx, unsigned int i)
{
    Ll2Lane *L = (Ll2Lane *)ctx;
    MMXDecoder *d = L->d;
    const MMXFile *f = d->f;
    long long kk = (long long)L->b0 + (long long)L->it - (long long)i;
    const MMXBlockEntry *b;
    long long **res;
    if (kk < (long long)L->b0 || kk >= (long long)L->b1 || L->err[i])
        return;
    b = &f->blocks[kk];
    res = L->slot_res[kk % 3];
    if (i == 0)
    {
        unsigned long n = ll2_block_samples(d, b);
        MMXRangeDecoder rd;
        unsigned int c;
        if (b->start_frame + b->frame_count > d->nf || !b->payload || mmx_crc32(b->payload, b->payload_size) != b->crc32
            || n > d->slot_cap)
        {
            L->err[0] = 1;
            return;
        }
        mmx_rc_dec_init(&rd, b->payload, b->payload_size);
        mmx_ll2s_decode_bounds(&L->l2, &rd, L->slot_lo[kk % 3], L->slot_hi[kk % 3]);
        for (c = 0; c < f->channels; c++)
            mmx_ll2c_decode_cont64(L->l2.coder, &rd, res[c], (int)n);
    }
    else
    {
        unsigned int c = i - 1;
        unsigned long j, off = 0, n;
        long long front = (long long)(b->start_frame + b->frame_count) * MMX_HOP - MMX_HOP;   /* the block end */
        if (front > (long long)d->len) front = (long long)d->len;
        for (j = 0; j < b->frame_count; j++)
        {
            unsigned long fr_idx = b->start_frame + j;
            long long start = (long long)fr_idx * MMX_HOP - MMX_HOP;
            unsigned long count = start < 0 ? 0 : (start + MMX_HOP <= (long long)d->len ? MMX_HOP : (unsigned long)((long long)d->len - start));
            long long src = b->n_sources ? b->src_start[0] + (long long)j * MMX_HOP : -1;
            if (count == 0)
            {
                L->l2.last_src[c] = src;
                continue;
            }
            if (src >= 0 && src + (long long)count > start)
            {
                L->err[i] = 1;
                return;
            }
            mmx_ll2s_frame_ch(&L->l2, c, d->dec, start, count, src, front, 1, res[c] + off, L->slot_lo[kk % 3][c],
                              L->slot_hi[kk % 3][c]);
            for (n = 0; n < count; n++)
            {
                const long long v = d->dec[c][start + (long long)n] * (1LL << d->shift);
                if (start + (long long)n < L->l2.store_from) continue;
                const size_t at = (size_t)(start + (long long)n) * f->channels + c;
                d->out->samples[at] = (float)((double)v / d->scale);
                if (d->out->isamples) d->out->isamples[at] = (int)v;
            }
            off += count;
        }
    }
}

/* may the lane run its next iteration: every source sample its channel tasks read in another segment is final
   (by the snapshot); sources inside the lane's own segment lie before the block, the pipeline order covers them */
static int lane_ready(const Ll2Lane *L)
{
    const MMXDecoder *d = L->d;
    unsigned int i;
    for (i = 1; i <= d->f->channels; i++)
    {
        long long kk = (long long)L->b0 + (long long)L->it - (long long)i, lo, hi, t;
        if (kk < (long long)L->b0 || kk >= (long long)L->b1) continue;
        block_sources(d, &d->f->blocks[kk], &lo, &hi);
        if (lo == hi) continue;
        for (t = seg_of_sample(d, lo); t < (long long)d->nseg && d->seg_lo[t] < hi; t++)
        {
            long long need = hi < d->seg_hi[t] ? hi : d->seg_hi[t];
            if (t == L->seg) continue;
            if (d->seg_seen[t] < need) return 0;
        }
    }
    return 1;
}

/* one round of one lane: up to LL2_ROUND_ITERS pipeline iterations, fewer when it has to wait for another segment */
static void lane_round(void *ctx, unsigned int k)
{
    MMXDecoder *d = (MMXDecoder *)ctx;
    Ll2Lane *L = &d->lanes[d->run[k]];
    const MMXFile *f = d->f;
    const unsigned long lag = f->channels == 2 ? 2 : 1;
    unsigned int m;
    L->finished = 0;
    L->hops = 0;
    L->iters = 0;
    L->wk = d->slot_wk[k];
    if (L->seg < 0) return;
    for (m = 0; m < LL2_ROUND_ITERS; m++)
    {
        long long done_block;
        if (!lane_ready(L)) return;
        L->iters++;
        mmx_workers_run(L->wk, ll2_task, L, f->channels + 1u);
        if (L->err[0] || L->err[1] || L->err[2]) return;
        done_block = (long long)L->b0 + (long long)L->it - (long long)lag;
        if (done_block >= (long long)L->b0)
        {
            const MMXBlockEntry *b = &f->blocks[done_block];
            long long fin = (long long)(b->start_frame + b->frame_count - 1) * MMX_HOP;
            if (fin > d->seg_hi[L->seg] || done_block + 1 == (long long)L->b1) fin = d->seg_hi[L->seg];
            if (fin > d->seg_final[L->seg]) d->seg_final[L->seg] = fin;
            L->hops += b->frame_count;
        }
        L->it++;
        if (L->it >= L->b1 - L->b0 + lag) { L->finished = 1; return; }
    }
}

/* the next segment for an idle lane: the focus, then the segments after it, then those before; a segment is taken
   only when every segment it reads from is under way or done - otherwise its earliest such dependency is taken */
static long pick_segment(const MMXDecoder *d)
{
    unsigned long n, s;
    for (n = 0; n < d->nseg; n++)
    {
        s = (d->focus + n) % d->nseg;
        while (d->seg_state[s] == 0)
        {
            unsigned long q, blocker = d->nseg;
            for (q = d->dep_off[s]; q < d->dep_off[s + 1]; q++)
                if (d->seg_state[d->dep[q]] == 0 && d->dep[q] < blocker) blocker = d->dep[q];
            if (blocker == d->nseg) return (long)s;
            s = blocker;                        /* references point backwards: this ends */
        }
    }
    return -1;
}

/* the focus segment and every segment it reads from, transitively: decoded before anything else */
static void mark_closure(MMXDecoder *d)
{
    unsigned long q, top = 0, t;
    unsigned long *stack = d->dep_off ? (unsigned long *)malloc(sizeof(unsigned long) * (d->nseg + 1)) : NULL;
    memset(d->inclose, 0, d->nseg);
    if (!stack) return;
    d->inclose[d->focus] = 1;
    stack[top++] = d->focus;
    while (top)
        for (t = stack[--top], q = d->dep_off[t]; q < d->dep_off[t + 1]; q++)
            if (!d->inclose[d->dep[q]]) { d->inclose[d->dep[q]] = 1; stack[top++] = d->dep[q]; }
    free(stack);
}

/* 0: the focus or what it reads from; otherwise 1 + how far after the focus in playback order */
static unsigned long seg_rank(const MMXDecoder *d, unsigned long s)
{
    return d->inclose[s] ? 0 : 1 + (s + d->nseg - d->focus) % d->nseg;
}

/* does any segment in a context read from s */
static int needed_by_others(const MMXDecoder *d, unsigned long s)
{
    unsigned int k;
    unsigned long q;
    for (k = 0; k < d->nctx; k++)
    {
        long o = d->lanes[k].seg;
        if (o < 0 || (unsigned long)o == s) continue;
        for (q = d->dep_off[o]; q < d->dep_off[o + 1]; q++) if (d->dep[q] == s) return 1;
    }
    return 0;
}

/* runs rounds until max_hops frames are final or the file is done. Every round, on the calling thread: segments
   to contexts (the focus and its sources may take a context from a later segment - paused, or reclaimed when every
   context is taken), then the nlanes most urgent contexts that can go on run side by side. */
static int ll2_step(MMXDecoder *d, unsigned long max_hops)
{
    unsigned long hops = 0, s;
    unsigned int k;
    for (;;)
    {
        unsigned int nrun = 0, inprog = 0, iters = 0;
        mark_closure(d);
        for (k = 0; k < d->nctx; k++) if (d->lanes[k].seg >= 0) inprog++;
        for (;;)
        {
            long ns = pick_segment(d);
            int slot = -1;
            if (ns < 0) break;
            if (seg_rank(d, (unsigned long)ns) != 0 && inprog >= d->nlanes) break;
            for (k = 0; k < d->nctx && slot < 0; k++) if (d->lanes[k].seg < 0) slot = (int)k;
            if (slot < 0)
            {   /* every context holds a segment: reclaim one that is less urgent and that no one reads from - the
                   one with the least progress, the cheapest to decode again later (its final part stays final and
                   is not written twice) */
                long long least = 0;
                for (k = 0; k < d->nctx; k++)
                {
                    unsigned long o = (unsigned long)d->lanes[k].seg, r = seg_rank(d, o);
                    long long done = d->seg_final[o] - d->seg_lo[o];
                    if (r > seg_rank(d, (unsigned long)ns) && !needed_by_others(d, o) && (slot < 0 || done < least))
                    { least = done; slot = (int)k; }
                }
                if (slot < 0) break;
                d->seg_state[d->lanes[slot].seg] = 0;
                d->lanes[slot].seg = -1;
                inprog--;
            }
            if (lane_start(&d->lanes[slot], (unsigned long)ns) != 0) return -1;
            d->seg_state[ns] = 1;
            inprog++;
        }
        for (s = 0; s < d->nseg; s++) d->seg_seen[s] = d->seg_final[s];
        /* the running set: the most urgent contexts that can go on now (a context waiting for a paused segment
           steps aside; the earliest segment in a context reads only finished ones, so one always can) */
        while (nrun < d->nlanes)
        {
            int best = -1;
            unsigned long br = 0;
            for (k = 0; k < d->nctx; k++)
            {
                Ll2Lane *L = &d->lanes[k];
                unsigned int q2;
                int taken = 0;
                if (L->seg < 0) continue;
                for (q2 = 0; q2 < nrun; q2++) if (d->run[q2] == k) taken = 1;
                if (taken || !lane_ready(L)) continue;
                if (best < 0 || seg_rank(d, (unsigned long)L->seg) < br
                    || (seg_rank(d, (unsigned long)L->seg) == br && L->seg < d->lanes[best].seg))
                { best = (int)k; br = seg_rank(d, (unsigned long)L->seg); }
            }
            if (best < 0) break;
            d->run[nrun++] = (unsigned int)best;
        }
        if (nrun == 0)
        {
            if (inprog == 0) break;                     /* done */
            mmx_error("Lossless segments wait on each other (damaged block table)");
            return -1;
        }
        mmx_workers_run(d->lane_wk, lane_round, d, nrun);
        for (k = 0; k < nrun; k++)
        {
            Ll2Lane *L = &d->lanes[d->run[k]];
            if (L->err[0] || L->err[1] || L->err[2])
            {
                mmx_error("Lossless block %lu is damaged or references samples that are not decoded yet",
                          L->b0 + L->it);
                return -1;
            }
            hops += L->hops;
            iters += L->iters;
            if (L->finished) { d->seg_state[L->seg] = 2; d->seg_final[L->seg] = d->seg_hi[L->seg]; L->seg = -1; }
        }
        /* the final prefix: every segment from the start that is done, then the part of the first one that is not */
        {
            unsigned long long v = 0;
            for (s = 0; s < d->nseg; s++)
            {
                v = (unsigned long long)d->seg_final[s];
                if (d->seg_state[s] != 2) break;
            }
            if (v > d->f->frame_count) v = d->f->frame_count;
            if (v > d->valid) d->valid = v;
        }
        if (iters == 0)
        {
            mmx_error("Lossless segments wait on each other (damaged block table)");
            return -1;
        }
        if (hops >= max_hops) break;
    }
    for (s = 0; s < d->nseg; s++) if (d->seg_state[s] != 2) return 1;
    return finish(d) != 0 ? -1 : 0;
}

void mmx_decoder_focus(MMXDecoder *d, unsigned long long frame)
{
    if (!d || !d->lanes || !d->nseg) return;
    d->focus = (unsigned long)seg_of_sample(d, (long long)(frame < d->len ? frame : d->len ? d->len - 1 : 0));
}

unsigned long long mmx_decoder_final_from(const MMXDecoder *d, unsigned long long frame)
{
    long long s, end;
    if (!d) return frame;
    if (d->done) return d->f->frame_count > frame ? d->f->frame_count : frame;
    if (!d->lanes) return d->valid > frame ? d->valid : frame;
    if (frame >= d->len) return frame;
    s = seg_of_sample(d, (long long)frame);
    end = (long long)frame;
    while (s < (long long)d->nseg && d->seg_final[s] > end)
    {
        end = d->seg_final[s];
        if (d->seg_final[s] < d->seg_hi[s]) break;
        s++;
    }
    return (unsigned long long)end;
}

/* the samples still to decode in the segments segment s reads from, transitively (whole segments, an upper bound) */
static unsigned long long closure_cost(const MMXDecoder *d, unsigned long s, unsigned char *seen, unsigned long *stack)
{
    unsigned long long cost = 0;
    unsigned long top = 0, q;
    memset(seen, 0, d->nseg);
    seen[s] = 1;
    stack[top++] = s;
    while (top)
    {
        unsigned long t = stack[--top];
        for (q = d->dep_off[t]; q < d->dep_off[t + 1]; q++)
        {
            unsigned long u = d->dep[q];
            if (seen[u]) continue;
            seen[u] = 1;
            if (d->seg_state[u] != 2) cost += (unsigned long long)(d->seg_hi[u] - d->seg_final[u]);
            stack[top++] = u;
        }
    }
    return cost;
}

unsigned long long mmx_decoder_seek_point(const MMXDecoder *d, unsigned long long frame, unsigned long long budget)
{
    unsigned char *seen;
    unsigned long *stack;
    long long s, from, cand[3];
    unsigned long long at = frame, cost, best_cost = 0;
    int k, n = 0, found = 0;
    if (!d || !d->lanes || d->nseg < 2 || d->done) return frame;
    if (frame >= d->len) return frame;
    if (mmx_decoder_final_from(d, frame) > frame) return frame;
    seen = (unsigned char *)malloc(d->nseg);
    stack = (unsigned long *)malloc(sizeof(unsigned long) * (d->nseg + 1));
    if (!seen || !stack) { free(seen); free(stack); return frame; }
    s = seg_of_sample(d, (long long)frame);
    /* the exact spot: what its own segment still has to decode up to it (from its final part when a context holds
       the segment, else from its start) plus the segments its references read */
    from = d->seg_state[s] == 1 ? d->seg_final[s] : d->seg_lo[s];
    cost = (unsigned long long)((long long)frame - from) + closure_cost(d, (unsigned long)s, seen, stack);
    if (cost <= budget) { free(seen); free(stack); return frame; }
    /* too dear: the neighbouring entry points - this segment's start, the next and the previous one, nearest first -
       the nearest that fits the budget, else the cheapest of them (a seek always lands close to where it aimed) */
    cand[n++] = s;
    if (s + 1 < (long long)d->nseg) cand[n++] = s + 1;
    if (s > 0) cand[n++] = s - 1;
    for (k = 0; k < n; k++)
    {
        int j;
        for (j = k + 1; j < n; j++)
        {
            long long dk = (long long)frame - d->seg_lo[cand[k]], dj = (long long)frame - d->seg_lo[cand[j]];
            if ((dj < 0 ? -dj : dj) < (dk < 0 ? -dk : dk)) { long long x = cand[k]; cand[k] = cand[j]; cand[j] = x; }
        }
    }
    for (k = 0; k < n; k++)
    {
        unsigned long long c = closure_cost(d, (unsigned long)cand[k], seen, stack);
        if (c <= budget) { at = (unsigned long long)d->seg_lo[cand[k]]; found = 1; break; }
        if (k == 0 || c < best_cost) { best_cost = c; at = (unsigned long long)d->seg_lo[cand[k]]; }
    }
    (void)found;
    free(seen);
    free(stack);
    return at;
}

unsigned long mmx_decoder_segments(const MMXDecoder *d, unsigned long long *lo, unsigned long long *final,
                                   unsigned long max)
{
    unsigned long s;
    if (!d || !d->lanes)
    {
        if (max && lo) { lo[0] = 0; final[0] = d ? (d->done ? d->f->frame_count : d->valid) : 0; }
        return 1;
    }
    for (s = 0; s < d->nseg && s < max; s++)
    {
        lo[s] = (unsigned long long)d->seg_lo[s];
        final[s] = d->seg_state[s] == 2 ? (unsigned long long)d->seg_hi[s] : (unsigned long long)d->seg_final[s];
    }
    return d->nseg;
}

static int lossy_frame(MMXDecoder *d, const MMXBlockEntry *b, unsigned long i, unsigned long j)
{
    unsigned long fr = b->start_frame + j;
    long long start = (long long)fr * MMX_HOP - MMX_HOP;
    long long src_start[MMX_MAX_SOURCES];
    unsigned int s;

    for (s = 0; s < b->n_sources; s++)
    {
        src_start[s] = b->src_start[s] + (long long)j * MMX_HOP;
        if (src_start[s] + MMX_WIN > start)
        {
            mmx_error("Block %lu frame %lu references samples that are not decoded yet", i, j);
            return -1;
        }
    }
    if (mmx_frame_decode(&d->rd, &d->ctx, &d->codec.bands, &d->codec.short_bands, &d->syn, b->n_sources) != 0)
    {
        mmx_error("Bitstream error in block %lu frame %lu", i, j);
        return -1;
    }
    mmx_codec_source_coefs_bt(&d->codec, d->ref, b->n_sources, src_start, d->syn.stereo_ms, d->syn.block_type);
    mmx_codec_predict(&d->codec, &d->syn, b->n_sources);
    decode_echo_probe(&d->probe, &d->codec, &d->syn, b->n_sources, start, d->f->sample_rate);
    mmx_codec_reconstruct_out(&d->codec, &d->syn, d->ref, d->out, start);
    return 0;
}

static int open_block(MMXDecoder *d)
{
    const MMXFile *f = d->f;
    const MMXBlockEntry *b = &f->blocks[d->block];
    if (d->lossless)
    {
        if (b->start_frame + b->frame_count > d->nf || !b->payload || mmx_crc32(b->payload, b->payload_size) != b->crc32)
        {
            mmx_error("Block %lu is damaged", d->block);
            return -1;
        }
        mmx_ll_contexts_init(d->lctx);
    }
    else
    {
        if (b->start_frame != d->next_frame || b->start_frame + b->frame_count > d->nf)
        {
            mmx_error("Block %lu does not continue the timeline", d->block);
            return -1;
        }
        if (!b->payload || mmx_crc32(b->payload, b->payload_size) != b->crc32)
        {
            mmx_error("Checksum mismatch in block %lu", d->block);
            return -1;
        }
        mmx_contexts_init(&d->ctx);
    }
    mmx_rc_dec_init(&d->rd, b->payload, b->payload_size);
    d->frame = 0;
    d->block_open = 1;
    return 0;
}

static int finish(MMXDecoder *d)
{
    const DecodeEchoProbe *p = &d->probe;
    if (!d->lossless && d->next_frame != d->nf)
    {
        mmx_error("Timeline incomplete: %lu of %lu frames", d->next_frame, d->nf);
        return -1;
    }
    if (p->on)
        fprintf(stderr, "decode probe %.1f-%.1f s: %lu referenced long frames, %lu channel bands with a prediction, %lu of them zero residual (%.1f %%), %lu of those with the prediction above the absolute threshold (%.1f %%), worst +%.1f dB at %.2f s\n",
                p->from, p->to, p->frames, p->bands, p->zero,
                p->bands ? 100.0 * (double)p->zero / (double)p->bands : 0.0, p->zero_loud,
                p->bands ? 100.0 * (double)p->zero_loud / (double)p->bands : 0.0, p->worst_db, p->worst_s);
    d->valid = d->f->frame_count;
    d->done = 1;
    return 0;
}

int mmx_decoder_step(MMXDecoder *d, unsigned long max_hops)
{
    const MMXFile *f = d->f;
    unsigned long hops = 0;

    if (d->done)
        return 0;
    if (d->lanes)
        return ll2_step(d, max_hops);
    while (hops < max_hops)
    {
        const MMXBlockEntry *b = &f->blocks[d->block];
        if (!d->block_open && open_block(d) != 0)
            return -1;
        if (d->frame < b->frame_count)
        {
            unsigned long fr = b->start_frame + d->frame;
            unsigned long long final;
            if ((d->lossless ? lossless_frame(d, b, d->block, d->frame) : lossy_frame(d, b, d->block, d->frame)) != 0)
                return -1;
            d->frame++;
            hops++;
            /* frame fr fills [fr*HOP - HOP, fr*HOP + HOP); the first half completes the overlap with
               its predecessor, the second half still waits for the next frame */
            final = (unsigned long long)fr * MMX_HOP;
            if (final > f->frame_count) final = f->frame_count;
            if (final > d->valid) d->valid = final;
        }
        if (d->frame >= b->frame_count)
        {
            d->next_frame = b->start_frame + b->frame_count;
            d->block_open = 0;
            d->block++;
            if (d->block >= f->block_count)
                return finish(d) != 0 ? -1 : 0;
        }
    }
    return 1;
}

int mmx_decoder_decode(const MMXFile *f, MMXAudioBuffer *out)
{
    MMXDecoder *d;
    int rc;
    if (mmx_decoder_open(f, out, &d) != 0)
        return -1;
    if (d->done)                 /* an empty timeline */
        rc = finish(d);
    else
        while ((rc = mmx_decoder_step(d, 4096)) > 0)
            ;
    mmx_decoder_close(d);
    if (rc != 0)
    {
        mmx_audio_buffer_free(out);
        return -1;
    }
    return 0;
}
