#ifndef MMX_READER_H
#define MMX_READER_H
#include <stddef.h>
#include "minimix/mmx_format.h"

/* Where the bytes of a container come from: a file on disk, a buffer in memory, or a host's own stream
   (a player's file service). `read_at` copies up to n bytes from absolute offset pos and returns how
   many it got; `size` is the total length, 0 when unknown. */
typedef struct
{
    void *ctx;
    size_t (*read_at)(void *ctx, unsigned long long pos, void *buf, size_t n);
    unsigned long long size;
} MMXByteSource;

/* A byte source over a memory buffer; `mem` must outlive the source. */
typedef struct
{
    const unsigned char *data;
    size_t size;
} MMXMemorySource;
void mmx_byte_source_memory(MMXByteSource *src, MMXMemorySource *mem, const void *data, size_t size);

/* Reads the container structure (header, metadata, cover, block table) from a byte source. With
   load_payloads != 0 the payload bytes are loaded too (needed for decoding). `name` only labels
   error messages. */
int mmx_reader_read_source(const MMXByteSource *src, const char *name, MMXFile *f, int load_payloads);

/* The same from a file on disk. */
int mmx_reader_read(const char *path, MMXFile *f, int load_payloads);
#endif
