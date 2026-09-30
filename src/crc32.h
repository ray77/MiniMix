#ifndef MMX_CRC32_H
#define MMX_CRC32_H

/* Standard CRC-32 (IEEE 802.3), used for per-block payload checksums. */
unsigned long mmx_crc32(const unsigned char *data, unsigned long size);

#endif
