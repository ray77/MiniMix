#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "minimix/mmx_reader.h"
#include "mmx_io.h"
#include "log.h"

static size_t memory_read_at(void *ctx, unsigned long long pos, void *buf, size_t n)
{
    const MMXMemorySource *m = (const MMXMemorySource *)ctx;
    if (pos >= m->size)
        return 0;
    if (n > m->size - (size_t)pos)
        n = m->size - (size_t)pos;
    memcpy(buf, m->data + pos, n);
    return n;
}

void mmx_byte_source_memory(MMXByteSource *src, MMXMemorySource *mem, const void *data, size_t size)
{
    mem->data = (const unsigned char *)data;
    mem->size = size;
    src->ctx = mem;
    src->read_at = memory_read_at;
    src->size = size;
}

static size_t file_read_at(void *ctx, unsigned long long pos, void *buf, size_t n)
{
    FILE *fp = (FILE *)ctx;
    if (fseek(fp, (long long)pos, SEEK_SET) != 0)
        return 0;
    return fread(buf, 1, n, fp);
}

/* exact read: everything or nothing */
static int read_all(const MMXByteSource *src, unsigned long long pos, void *buf, size_t n)
{
    return src->read_at(src->ctx, pos, buf, n) == n ? 0 : -1;
}

int mmx_reader_read_source(const MMXByteSource *src, const char *name, MMXFile *f, int load_payloads)
{
    unsigned char hdr[MMX_HEADER_BYTES];
    unsigned char row[MMX_TABLE_ENTRY_BYTES];
    unsigned int version, header_bytes, table_format;
    unsigned long long table_offset, payload_offset, pos;
    unsigned long i, count;

    mmx_file_init(f);
    if (!name)
        name = "stream";

    if (read_all(src, 0, hdr, MMX_HEADER_BYTES) != 0 || memcmp(hdr, MMX_MAGIC, 4) != 0)
    {
        if (memcmp(hdr, "MMX1", 4) == 0)
            mmx_error("%s is a MiniMix version 1 prototype file; re-encode it with this version", name);
        else
            mmx_error("Not a MiniMix file (bad magic): %s", name);
        return -1;
    }
    version = mmx_get_u16(hdr + 4);
    header_bytes = mmx_get_u16(hdr + 6);
    if (version != MMX_FORMAT_VERSION || header_bytes < MMX_HEADER_BYTES)
    {
        mmx_error("Unsupported MMX version %u", version);
        return -1;
    }

    f->sample_rate = mmx_get_u32(hdr + 8);
    f->channels = (unsigned short)mmx_get_u16(hdr + 12);
    f->source_bits = (unsigned short)mmx_get_u16(hdr + 14);
    f->frame_count = mmx_get_u64(hdr + 16);
    f->hop = mmx_get_u32(hdr + 24);
    count = mmx_get_u32(hdr + 28);
    f->quality = hdr[32];
    f->analysis_level = hdr[33];
    f->max_ref_depth = hdr[34];
    f->codec_id = hdr[35];
    f->bwe_hz = (unsigned int)hdr[36] * 250u;
    f->bitstream_rev = hdr[37];
    f->bwe_mode = hdr[39];
    table_format = hdr[38];
    f->metadata_len = mmx_get_u32(hdr + 40);
    /* revisions 6 and 7 decode with this build: revision 6 lossless blocks never had two sources, both
       run the sign-sign filter cascade (the decoder selects it by revision), and the lossy syntax is
       the same in 6-8; anything older or newer is refused */
    if (f->bitstream_rev < 6 || f->bitstream_rev > MMX_BITSTREAM_REVISION)
    {
        mmx_error("%s uses bitstream revision %u, this build decodes revisions 6-%u: re-encode the file",
                  name, f->bitstream_rev, MMX_BITSTREAM_REVISION);
        mmx_file_free(f);
        return -1;
    }
    f->cover_len = mmx_get_u32(hdr + 44);
    memcpy(f->cover_mime, hdr + 48, 16);
    f->cover_mime[15] = '\0';
    table_offset = mmx_get_u64(hdr + 64);
    payload_offset = mmx_get_u64(hdr + 72);
    f->ll_shift = hdr[80];        /* zero in files from early encoders */
    f->ll_dropped = hdr[81];
    f->ll_qdrop = hdr[86];
    f->ll_core = hdr[87];
    f->ll2_decay = hdr[88];
    f->ll2_bank = hdr[89];
    f->ll2_xols[0] = hdr[90];
    f->ll2_xols[1] = hdr[91];
    f->ll2_mix = hdr[92];
    f->ll2_profile = hdr[93];
    f->ll2_own = hdr[94];
    f->ll2_pld = hdr[95];
    f->lowrate = hdr[82];
    f->epb_fold = hdr[83];
    f->epb_flags = hdr[84];
    f->epb_fill_db = (signed char)hdr[85];

    if (f->channels == 0 || f->hop == 0 || f->sample_rate == 0)
    {
        mmx_error("Corrupt MMX header");
        return -1;
    }

    pos = header_bytes;
    if (f->metadata_len)
    {
        f->metadata = (char *)malloc(f->metadata_len + 1);
        if (!f->metadata || read_all(src, pos, f->metadata, f->metadata_len) != 0)
            goto fail;
        f->metadata[f->metadata_len] = '\0';
        pos += f->metadata_len;
    }
    if (f->cover_len)
    {
        f->cover = (unsigned char *)malloc(f->cover_len);
        if (!f->cover || read_all(src, pos, f->cover, f->cover_len) != 0)
            goto fail;
    }

    if (table_format == MMX_TABLE_FORMAT_COMPACT)
    {
        unsigned long table_size = (unsigned long)(payload_offset - table_offset);
        unsigned char *table = (unsigned char *)malloc(table_size ? table_size : 1);
        if (!table || (table_size && read_all(src, table_offset, table, table_size) != 0))
        {
            mmx_error("Truncated block table");
            free(table);
            goto fail;
        }
        if (count && mmx_table_decode(f, table, table_size, count, payload_offset) != 0)
        {
            free(table);
            goto fail;
        }
        free(table);
    }
    else
    for (i = 0; i < count; i++)
    {
        MMXBlockEntry b;
        memset(&b, 0, sizeof(b));
        if (read_all(src, table_offset + (unsigned long long)i * MMX_TABLE_ENTRY_BYTES, row, MMX_TABLE_ENTRY_BYTES) != 0)
        {
            mmx_error("Truncated block table");
            goto fail;
        }
        b.start_frame = mmx_get_u32(row + 0);
        b.frame_count = mmx_get_u32(row + 4);
        b.n_sources = row[8];
        b.depth = row[9];
        b.flags = (unsigned short)mmx_get_u16(row + 10);
        b.src_start[0] = (long long)mmx_get_u64(row + 12);
        b.src_start[1] = (long long)mmx_get_u64(row + 20);
        b.payload_size = mmx_get_u32(row + 28);
        b.crc32 = mmx_get_u32(row + 32);
        b.payload_offset = payload_offset + mmx_get_u64(row + 36);
        if (b.n_sources > MMX_MAX_BLOCK_SOURCES)
        {
            mmx_error("Corrupt block table entry %lu", i);
            goto fail;
        }
        if (mmx_file_add_block(f, &b) < 0)
            goto fail;
    }

    if (load_payloads)
    {
        for (i = 0; i < f->block_count; i++)
        {
            MMXBlockEntry *b = &f->blocks[i];
            if (!b->payload_size)
                continue;
            b->payload = (unsigned char *)malloc(b->payload_size);
            if (!b->payload)
                goto fail;
            if (read_all(src, b->payload_offset, b->payload, b->payload_size) != 0)
            {
                mmx_error("Truncated payload for block %lu", i);
                goto fail;
            }
        }
    }
    return 0;

fail:
    mmx_file_free(f);
    return -1;
}

int mmx_reader_read(const char *path, MMXFile *f, int load_payloads)
{
    MMXByteSource src;
    FILE *fp = fopen(path, "rb");
    int rc;
    if (!fp)
    {
        mmx_file_init(f);
        mmx_error("Cannot open MMX file: %s", path);
        return -1;
    }
    src.ctx = fp;
    src.read_at = file_read_at;
    src.size = 0;
    rc = mmx_reader_read_source(&src, path, f, load_payloads);
    fclose(fp);
    return rc;
}
