#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include "minimix/audio_buffer.h"

int mmx_audio_buffer_init(MMXAudioBuffer *buf, unsigned long sample_rate,
                          unsigned short channels, unsigned long long frame_count)
{
    size_t total;

    memset(buf, 0, sizeof(*buf));
    if (channels == 0)
        return -1;

    total = (size_t)frame_count * channels;
    buf->samples = (float *)calloc(total ? total : 1, sizeof(float));
    if (!buf->samples)
        return -1;

    buf->sample_rate = sample_rate;
    buf->channels = channels;
    buf->frame_count = frame_count;
    buf->exact = 1;          /* trusted until a reader finds bits the float could not keep */
    return 0;
}

int mmx_audio_buffer_alloc_int(MMXAudioBuffer *buf)
{
    size_t total = (size_t)buf->frame_count * buf->channels;
    free(buf->isamples);
    buf->isamples = (int *)calloc(total ? total : 1, sizeof(int));
    return buf->isamples ? 0 : -1;
}

void mmx_audio_buffer_free(MMXAudioBuffer *buf)
{
    if (!buf)
        return;
    free(buf->samples);
    free(buf->isamples);
    memset(buf, 0, sizeof(*buf));
}

float *mmx_audio_buffer_frame(const MMXAudioBuffer *buf, unsigned long long frame)
{
    return buf->samples + (size_t)frame * buf->channels;
}

double mmx_audio_buffer_duration_seconds(const MMXAudioBuffer *buf)
{
    if (buf->sample_rate == 0)
        return 0.0;
    return (double)buf->frame_count / (double)buf->sample_rate;
}

void mmx_format_duration(double seconds, char *out, unsigned long out_size)
{
    unsigned long total_ms = (unsigned long)(seconds * 1000.0 + 0.5);
    unsigned long minutes = total_ms / 60000UL;
    unsigned long secs = (total_ms / 1000UL) % 60UL;
    unsigned long ms = total_ms % 1000UL;
    snprintf(out, out_size, "%02lu:%02lu.%03lu", minutes, secs, ms);
}
