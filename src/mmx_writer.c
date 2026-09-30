#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "minimix/mmx_writer.h"
#include "mmx_io.h"
#include "log.h"

/* Layout (version 2):
   [header 96][metadata][cover][block table (compact, header byte 38 = 1)][payload area]
   Index and tables sit at the front so playback can start before the whole
   file is available. */

int mmx_writer_write(const char *path, MMXFile *f, unsigned long long *bytes_written)
{
    FILE *fp;
    unsigned char hdr[MMX_HEADER_BYTES];
    unsigned char *table = NULL;
    unsigned long table_size;
    unsigned long long table_offset, payload_offset, running = 0, total;
    unsigned long i;

    table_size = mmx_table_encode(f, &table);
    if (f->block_count && (!table || !table_size))
    {
        mmx_error("Block table could not be encoded");
        free(table);
        return -1;
    }
    table_offset = MMX_HEADER_BYTES + f->metadata_len + f->cover_len;
    payload_offset = table_offset + table_size;

    memset(hdr, 0, sizeof(hdr));
    memcpy(hdr, MMX_MAGIC, 4);
    mmx_put_u16(hdr + 4, MMX_FORMAT_VERSION);
    mmx_put_u16(hdr + 6, MMX_HEADER_BYTES);
    mmx_put_u32(hdr + 8, f->sample_rate);
    mmx_put_u16(hdr + 12, f->channels);
    mmx_put_u16(hdr + 14, f->source_bits);
    mmx_put_u64(hdr + 16, f->frame_count);
    mmx_put_u32(hdr + 24, f->hop);
    mmx_put_u32(hdr + 28, f->block_count);
    hdr[32] = f->quality;
    hdr[33] = f->analysis_level;
    hdr[34] = f->max_ref_depth;
    hdr[35] = f->codec_id;
    hdr[36] = (unsigned char)((f->bwe_hz + 125) / 250 > 255 ? 255 : (f->bwe_hz + 125) / 250);  /* band replication crossover, 0 = off */
    hdr[37] = f->bitstream_rev;
    hdr[39] = f->bwe_mode;
    hdr[38] = MMX_TABLE_FORMAT_COMPACT;
    mmx_put_u32(hdr + 40, f->metadata_len);
    mmx_put_u32(hdr + 44, f->cover_len);
    memcpy(hdr + 48, f->cover_mime, 16);
    mmx_put_u64(hdr + 64, table_offset);
    mmx_put_u64(hdr + 72, payload_offset);
    hdr[80] = f->ll_shift;
    hdr[81] = f->ll_dropped;
    hdr[86] = f->ll_qdrop;
    hdr[87] = f->ll_core;
    hdr[88] = f->ll2_decay;
    hdr[89] = f->ll2_bank;
    hdr[90] = f->ll2_xols[0];
    hdr[91] = f->ll2_xols[1];
    hdr[92] = f->ll2_mix;
    hdr[93] = f->ll2_profile;
    hdr[94] = f->ll2_own;
    hdr[95] = f->ll2_pld;
    hdr[82] = f->lowrate;
    hdr[83] = f->epb_fold;
    hdr[84] = f->epb_flags;
    hdr[85] = (unsigned char)f->epb_fill_db;

    fp = fopen(path, "wb");
    if (!fp)
    {
        mmx_error("Cannot create output file: %s", path);
        free(table);
        return -1;
    }
    if (fwrite(hdr, 1, MMX_HEADER_BYTES, fp) != MMX_HEADER_BYTES)
        goto fail;
    if (f->metadata_len && fwrite(f->metadata, 1, f->metadata_len, fp) != f->metadata_len)
        goto fail;
    if (f->cover_len && fwrite(f->cover, 1, f->cover_len, fp) != f->cover_len)
        goto fail;

    if (table_size && fwrite(table, 1, table_size, fp) != table_size)
        goto fail;
    for (i = 0; i < f->block_count; i++)
    {
        MMXBlockEntry *b = &f->blocks[i];
        b->payload_offset = payload_offset + running;
        running += b->payload_size;
    }
    for (i = 0; i < f->block_count; i++)
    {
        const MMXBlockEntry *b = &f->blocks[i];
        if (b->payload_size && fwrite(b->payload, 1, b->payload_size, fp) != b->payload_size)
            goto fail;
    }

    total = payload_offset + running;
    if (bytes_written)
        *bytes_written = total;
    fclose(fp);
    free(table);
    return 0;

fail:
    mmx_error("Write error on %s", path);
    fclose(fp);
    free(table);
    return -1;
}
