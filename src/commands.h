#ifndef MMX_COMMANDS_H
#define MMX_COMMANDS_H

#include "cli.h"

int mmx_cmd_encode(const MMXOptions *opts);
int mmx_cmd_decode(const MMXOptions *opts);
int mmx_cmd_analyze(const MMXOptions *opts);
int mmx_cmd_info(const MMXOptions *opts);
int mmx_cmd_compare(const MMXOptions *opts);
int mmx_cmd_tag(const MMXOptions *opts);

#endif
