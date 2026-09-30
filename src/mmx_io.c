#include "mmx_io.h"

void mmx_put_u16(unsigned char *p, unsigned int v)
{
    p[0] = (unsigned char)(v & 0xFF);
    p[1] = (unsigned char)((v >> 8) & 0xFF);
}

void mmx_put_u32(unsigned char *p, unsigned long v)
{
    int i;
    for (i = 0; i < 4; i++)
        p[i] = (unsigned char)((v >> (8 * i)) & 0xFF);
}

void mmx_put_u64(unsigned char *p, unsigned long long v)
{
    int i;
    for (i = 0; i < 8; i++)
        p[i] = (unsigned char)((v >> (8 * i)) & 0xFF);
}

unsigned int mmx_get_u16(const unsigned char *p)
{
    return (unsigned int)p[0] | ((unsigned int)p[1] << 8);
}

unsigned long mmx_get_u32(const unsigned char *p)
{
    unsigned long v = 0;
    int i;
    for (i = 3; i >= 0; i--)
        v = (v << 8) | p[i];
    return v;
}

unsigned long long mmx_get_u64(const unsigned char *p)
{
    unsigned long long v = 0;
    int i;
    for (i = 7; i >= 0; i--)
        v = (v << 8) | p[i];
    return v;
}
