#include "crc32.h"

static unsigned long g_table[256];
static int g_table_ready = 0;

static void build_table(void)
{
    unsigned long i, j, c;
    for (i = 0; i < 256; i++)
    {
        c = i;
        for (j = 0; j < 8; j++)
            c = (c & 1) ? (0xEDB88320UL ^ (c >> 1)) : (c >> 1);
        g_table[i] = c;
    }
    g_table_ready = 1;
}

unsigned long mmx_crc32(const unsigned char *data, unsigned long size)
{
    unsigned long crc = 0xFFFFFFFFUL;
    unsigned long i;

    if (!g_table_ready)
        build_table();

    for (i = 0; i < size; i++)
        crc = g_table[(crc ^ data[i]) & 0xFF] ^ (crc >> 8);

    return (crc ^ 0xFFFFFFFFUL) & 0xFFFFFFFFUL;
}
