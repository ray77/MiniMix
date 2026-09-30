#ifndef MMX_WORKERS_H
#define MMX_WORKERS_H
/* Per-object helper threads for the decoder (the encoder's pool in threads.c is a process-wide singleton with a single
 * publisher; a player may run several decoders at once). mmx_workers_run(w, fn, ctx, count) runs fn(ctx, i) for
 * i = 0 .. count-1, item 0 on the calling thread and item i >= 1 on helper i - 1, and returns when all are done. Every
 * item writes only its own data, so results do not depend on threads; with no helpers (or MMX_DECODE_THREADS=1) the
 * caller runs the items in order. POSIX threads or Win32. */

typedef struct MMXWorkers MMXWorkers;
typedef void (*MMXWorkFn)(void *ctx, unsigned int i);

/* helpers: number of extra threads (0 = none). Returns NULL only on out of memory; failing thread creation leaves
   fewer helpers, which is still correct. */
MMXWorkers *mmx_workers_new(unsigned int helpers);
void mmx_workers_run(MMXWorkers *w, MMXWorkFn fn, void *ctx, unsigned int count);
void mmx_workers_free(MMXWorkers *w);
/* the machine's logical cores (at least 1) */
unsigned int mmx_workers_cores(void);

#endif
