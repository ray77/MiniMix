#ifndef MMX_THREADS_H
#define MMX_THREADS_H

/* A tiny fixed-size worker pool for the parts of the encoder whose per-item
   work is independent (the analyzer's psychoacoustic pass, its alignments and
   its candidate scoring). It exists to remove waste, never to change a result:
   every task writes only to storage indexed by its own item or by its own
   worker slot, so the outcome does not depend on the number of threads or on
   the order the scheduler happens to pick. `MMX_THREADS=1` takes the old
   single-threaded path (no pool is created, the caller simply runs the loop),
   and every other count produces the same bytes. */

/* Worker count: MMX_THREADS if set (1 = single-threaded), else the number of
   online cores, clamped to MMX_THREADS_MAX. Cached after the first call. */
unsigned int mmx_threads(void);

#define MMX_THREADS_MAX 64

/* fn(ctx, i, worker) for every i in [0, n). The calling thread is worker 0 and
   runs tasks as well; the call returns when every i has been done. Tasks of one
   call run concurrently in an unspecified order; nested calls are not allowed. */
typedef void (*MMXTaskFn)(void *ctx, unsigned long i, unsigned int worker);
void mmx_parallel_for(unsigned long n, MMXTaskFn fn, void *ctx);

#endif
