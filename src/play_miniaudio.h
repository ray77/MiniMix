#ifndef MMX_PLAY_MINIAUDIO_H
#define MMX_PLAY_MINIAUDIO_H
/* miniaudio (third_party/miniaudio, public domain or MIT-0) with only what mmx play uses: the playback device. The
   same settings for the implementation (play_miniaudio.c) and for its user (play.c). */
#define MA_NO_DECODING
#define MA_NO_ENCODING
#define MA_NO_GENERATION
#define MA_NO_RESOURCE_MANAGER
#define MA_NO_NODE_GRAPH
#define MA_NO_ENGINE
#include "miniaudio.h"
#endif
