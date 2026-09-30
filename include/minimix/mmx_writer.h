#ifndef MMX_WRITER_H
#define MMX_WRITER_H

#include "minimix/mmx_format.h"

/* Serializes header, metadata, block table and payloads. Fills payload_offset
   in each block entry. Returns 0 on success. */
int mmx_writer_write(const char *path, MMXFile *f, unsigned long long *bytes_written);

#endif
