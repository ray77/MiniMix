#ifndef MMX_IO_H
#define MMX_IO_H

/* Little-endian field helpers shared by reader and writer. */

#include "minimix/mmx_format.h"

#define MMX_HEADER_BYTES 96
#define MMX_TABLE_ENTRY_BYTES 48        /* table format 0: fixed rows */
#define MMX_TABLE_FORMAT_ROWS 0
#define MMX_TABLE_FORMAT_COMPACT 1      /* table format 1: range-coded (src/mmx_table.c) */

/* Compact block table. Encode allocates *data (free it) and returns its size,
   0 on failure. Decode appends `count` blocks to f. */
unsigned long mmx_table_encode(const MMXFile *f, unsigned char **data);
int mmx_table_decode(MMXFile *f, const unsigned char *data, unsigned long size, unsigned long count,
                     unsigned long long payload_offset);

void mmx_put_u16(unsigned char *p, unsigned int v);
void mmx_put_u32(unsigned char *p, unsigned long v);
void mmx_put_u64(unsigned char *p, unsigned long long v);
unsigned int mmx_get_u16(const unsigned char *p);
unsigned long mmx_get_u32(const unsigned char *p);
unsigned long long mmx_get_u64(const unsigned char *p);

#endif
