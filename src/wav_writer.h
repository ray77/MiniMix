#ifndef MMX_WAV_WRITER_H
#define MMX_WAV_WRITER_H

#include "minimix/audio_buffer.h"

/* Writes a PCM WAV file with the given bit depth (16, 24 or 32). Returns 0 on success. */
int mmx_wav_write(const char *path, const MMXAudioBuffer *buf, unsigned int bits);

#endif
