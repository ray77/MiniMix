#ifndef MMX_WAV_READER_H
#define MMX_WAV_READER_H

#include "minimix/audio_buffer.h"

/* Loads a RIFF/WAVE file (PCM 8/16/24/32-bit or IEEE float 32-bit) into a
   normalized float buffer. Returns 0 on success. */
int mmx_wav_read(const char *path, MMXAudioBuffer *out);

#endif
