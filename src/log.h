#ifndef MMX_LOG_H
#define MMX_LOG_H

/* Central console output. No scattered printf() in analysis code. */

typedef enum
{
    MMX_LOG_ERROR = 0,
    MMX_LOG_WARNING = 1,
    MMX_LOG_INFO = 2,
    MMX_LOG_DEBUG = 3,
    MMX_LOG_TRACE = 4
} MMXLogLevel;

void mmx_log_set_level(MMXLogLevel level);
MMXLogLevel mmx_log_get_level(void);

/* Suppress all non-error output (used by --json so stdout stays clean). */
void mmx_log_set_quiet(int quiet);

void mmx_log(MMXLogLevel level, const char *fmt, ...);

#define mmx_error(...)   mmx_log(MMX_LOG_ERROR, __VA_ARGS__)
#define mmx_warning(...) mmx_log(MMX_LOG_WARNING, __VA_ARGS__)
#define mmx_info(...)    mmx_log(MMX_LOG_INFO, __VA_ARGS__)
#define mmx_debug(...)   mmx_log(MMX_LOG_DEBUG, __VA_ARGS__)
#define mmx_trace(...)   mmx_log(MMX_LOG_TRACE, __VA_ARGS__)

#endif
