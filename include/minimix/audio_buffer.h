#ifndef MMX_AUDIO_BUFFER_H
#define MMX_AUDIO_BUFFER_H

/* Internal normalized audio representation: interleaved float in [-1, 1]. */

typedef struct
{
    unsigned long sample_rate;
    unsigned short channels;
    unsigned short source_bits;      /* bit depth of original input (16/24/32), 0 = unknown */
    unsigned short exact;            /* 1: every source sample survived the conversion to float exactly,
                                        so a lossless encode can reproduce the input bit for bit.
                                        0: 32-bit integer input with bits below the 24-bit float
                                        significand, or float input off the 24-bit grid - the buffer
                                        already lost them. Readers set it; the lossless encoder refuses
                                        input that is not exact instead of calling the result lossless. */
    unsigned long long frame_count;
    float *samples;                  /* interleaved, frame_count * channels */
    int *isamples;                   /* 32-bit integer sources only (NULL otherwise): the exact samples, interleaved, in the
                                        source's own integer scale - the float cannot hold 32 bits. The WAV reader fills
                                        it, the LL2 lossless encoder codes it, the LL2 decoder writes it and the WAV
                                        writer prefers it for 32-bit output. */
} MMXAudioBuffer;

/* Adds the exact integer samples to an initialised buffer (zeroed). Returns 0 on success. */
int mmx_audio_buffer_alloc_int(MMXAudioBuffer *buf);

/* Allocates zeroed sample storage. Returns 0 on success. */
int mmx_audio_buffer_init(MMXAudioBuffer *buf, unsigned long sample_rate,
                          unsigned short channels, unsigned long long frame_count);

void mmx_audio_buffer_free(MMXAudioBuffer *buf);

/* Pointer to the first sample of a frame. */
float *mmx_audio_buffer_frame(const MMXAudioBuffer *buf, unsigned long long frame);

double mmx_audio_buffer_duration_seconds(const MMXAudioBuffer *buf);

/* Formats seconds as MM:SS.mmm into out (at least 16 bytes). */
void mmx_format_duration(double seconds, char *out, unsigned long out_size);

#endif
