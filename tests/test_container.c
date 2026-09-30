/* MMX v2 writer/reader round trip: header, metadata, cover, block table, payloads. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "minimix/mmx_writer.h"
#include "minimix/mmx_reader.h"
#include "crc32.h"

int main(void)
{
    MMXFile f, g;
    MMXBlockEntry e;
    const char *path = "bin/test_container.mmx";
    unsigned long long written;
    unsigned char *p, cover[5] = {1, 2, 3, 4, 5};
    char *v;

    mmx_file_init(&f);
    f.sample_rate = 48000; f.channels = 2; f.source_bits = 24; f.frame_count = 100000;
    f.quality = 7; f.analysis_level = 5; f.max_ref_depth = 3; f.codec_id = 2; f.bitstream_rev = MMX_BITSTREAM_REVISION;
    mmx_file_set_metadata(&f, "encoder=test\ntitle=Song\n");
    mmx_file_set_cover(&f, "image/png", cover, 5);

    memset(&e, 0, sizeof(e));
    e.start_frame = 0; e.frame_count = 40; e.n_sources = 0;
    p = malloc(5); memcpy(p, "hello", 5);
    e.payload = p; e.payload_size = 5; e.crc32 = mmx_crc32(p, 5);
    mmx_file_add_block(&f, &e);

    memset(&e, 0, sizeof(e));
    e.start_frame = 40; e.frame_count = 59; e.n_sources = 2; e.depth = 1;
    e.src_start[0] = 17; e.src_start[1] = -1024;
    p = malloc(3); memcpy(p, "abc", 3);
    e.payload = p; e.payload_size = 3; e.crc32 = mmx_crc32(p, 3);
    mmx_file_add_block(&f, &e);

    if (mmx_writer_write(path, &f, &written) != 0) { printf("write failed\n"); return 1; }
    if (mmx_reader_read(path, &g, 1) != 0) { printf("read failed\n"); return 1; }

    if (g.sample_rate != 48000 || g.channels != 2 || g.source_bits != 24 || g.frame_count != 100000 ||
        g.block_count != 2 || g.quality != 7 || g.codec_id != 2 || g.metadata_len != f.metadata_len ||
        strcmp(g.metadata, f.metadata) != 0 || g.cover_len != 5 || memcmp(g.cover, cover, 5) != 0 ||
        strcmp(g.cover_mime, "image/png") != 0)
    { printf("header mismatch\n"); return 1; }
    v = mmx_file_metadata_get(&g, "title");
    if (!v || strcmp(v, "Song") != 0) { printf("metadata lookup failed\n"); return 1; }
    free(v);
    if (g.blocks[0].payload_size != 5 || memcmp(g.blocks[0].payload, "hello", 5) != 0 || g.blocks[0].crc32 != f.blocks[0].crc32)
    { printf("payload mismatch\n"); return 1; }
    if (g.blocks[1].n_sources != 2 || g.blocks[1].src_start[0] != 17 || g.blocks[1].src_start[1] != -1024 || g.blocks[1].depth != 1 ||
        g.blocks[1].start_frame != 40 || g.blocks[1].frame_count != 59)
    { printf("ref entry mismatch\n"); return 1; }

    printf("container: %llu bytes written, all fields round-trip\n", written);
    mmx_file_free(&f); mmx_file_free(&g);
    remove(path);
    printf("test_container OK\n");
    return 0;
}
