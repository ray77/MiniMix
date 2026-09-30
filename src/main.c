#include <stdio.h>
#include "cli.h"
#include "commands.h"
#include "play.h"

int main(int argc, char **argv)
{
    MMXOptions opts;

    if (mmx_cli_parse(argc, argv, &opts) != 0)
        return 2;

    switch (opts.command)
    {
    case MMX_CMD_ENCODE:  return mmx_cmd_encode(&opts);
    case MMX_CMD_DECODE:  return mmx_cmd_decode(&opts);
    case MMX_CMD_ANALYZE: return mmx_cmd_analyze(&opts);
    case MMX_CMD_INFO:    return mmx_cmd_info(&opts);
    case MMX_CMD_COMPARE: return mmx_cmd_compare(&opts);
    case MMX_CMD_TAG:     return mmx_cmd_tag(&opts);
    case MMX_CMD_PLAY:    return mmx_cmd_play(&opts);
    case MMX_CMD_VERSION: printf("mmx %s\n", MMX_VERSION_STRING); return 0;
    default:              mmx_cli_usage(); return 0;
    }
}
