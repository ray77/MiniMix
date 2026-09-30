#ifndef MMX_PLAY_H
#define MMX_PLAY_H
#include "cli.h"

/* mmx play FILE...: plays .mmx files in the terminal while they decode. Returns the exit status. */
int mmx_cmd_play(const MMXOptions *opts);
#endif
