#include <stdlib.h>
#include <string.h>
#include "minimix/mmx_format.h"

void mmx_file_init(MMXFile *f)
{
    memset(f, 0, sizeof(*f));
    f->hop = 1024;
}

void mmx_file_free(MMXFile *f)
{
    unsigned long i;
    if (!f)
        return;
    if (f->blocks)
    {
        for (i = 0; i < f->block_count; i++)
            free(f->blocks[i].payload);
        free(f->blocks);
    }
    free(f->metadata);
    free(f->cover);
    memset(f, 0, sizeof(*f));
}

long long mmx_file_add_block(MMXFile *f, const MMXBlockEntry *entry)
{
    if (f->block_count == f->block_capacity)
    {
        unsigned long new_cap = f->block_capacity ? f->block_capacity * 2 : 64;
        MMXBlockEntry *grown = (MMXBlockEntry *)realloc(f->blocks, new_cap * sizeof(MMXBlockEntry));
        if (!grown)
            return -1;
        f->blocks = grown;
        f->block_capacity = new_cap;
    }
    f->blocks[f->block_count] = *entry;
    f->block_count++;
    return (long long)(f->block_count - 1);
}

int mmx_file_set_metadata(MMXFile *f, const char *text)
{
    unsigned long len = text ? (unsigned long)strlen(text) : 0;
    char *copy = (char *)malloc(len + 1);
    if (!copy)
        return -1;
    memcpy(copy, text ? text : "", len + 1);
    free(f->metadata);
    f->metadata = copy;
    f->metadata_len = len;
    return 0;
}

int mmx_file_set_cover(MMXFile *f, const char *mime, const unsigned char *data, unsigned long len)
{
    unsigned char *copy = (unsigned char *)malloc(len ? len : 1);
    if (!copy)
        return -1;
    memcpy(copy, data, len);
    free(f->cover);
    f->cover = copy;
    f->cover_len = len;
    memset(f->cover_mime, 0, sizeof(f->cover_mime));
    strncpy(f->cover_mime, mime, sizeof(f->cover_mime) - 1);
    return 0;
}

char *mmx_file_metadata_get(const MMXFile *f, const char *key)
{
    const char *p = f->metadata;
    size_t klen = strlen(key);
    if (!p)
        return NULL;
    while (*p)
    {
        const char *eol = strchr(p, '\n');
        size_t line_len = eol ? (size_t)(eol - p) : strlen(p);
        if (line_len > klen && strncmp(p, key, klen) == 0 && p[klen] == '=')
        {
            char *v = (char *)malloc(line_len - klen);
            if (!v)
                return NULL;
            memcpy(v, p + klen + 1, line_len - klen - 1);
            v[line_len - klen - 1] = '\0';
            return v;
        }
        if (!eol)
            break;
        p = eol + 1;
    }
    return NULL;
}

const char *mmx_block_type_name(const MMXBlockEntry *b)
{
    if (b->n_sources == 0)
        return "AUDIO";
    if (b->n_sources == 1)
        return "REF";
    return "REF2";
}
