#include <stdio.h>
#include <stdarg.h>
#include "log.h"

static MMXLogLevel g_level = MMX_LOG_INFO;
static int g_quiet = 0;

static const char *level_name(MMXLogLevel level)
{
    switch (level)
    {
    case MMX_LOG_ERROR:   return "ERROR";
    case MMX_LOG_WARNING: return "WARN";
    case MMX_LOG_INFO:    return "INFO";
    case MMX_LOG_DEBUG:   return "DEBUG";
    case MMX_LOG_TRACE:   return "TRACE";
    }
    return "?";
}

void mmx_log_set_level(MMXLogLevel level)
{
    g_level = level;
}

MMXLogLevel mmx_log_get_level(void)
{
    return g_level;
}

void mmx_log_set_quiet(int quiet)
{
    g_quiet = quiet;
}

void mmx_log(MMXLogLevel level, const char *fmt, ...)
{
    va_list args;
    FILE *out;

    if (level > g_level)
        return;
    if (g_quiet && level != MMX_LOG_ERROR)
        return;

    out = (level <= MMX_LOG_WARNING) ? stderr : stdout;
    fprintf(out, "[%s] ", level_name(level));
    va_start(args, fmt);
    vfprintf(out, fmt, args);
    va_end(args);
    fputc('\n', out);
    fflush(out);
}
