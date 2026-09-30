#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include <sched.h>
#ifndef _WIN32
#include <unistd.h>
#endif
#if defined(__APPLE__)
#include <sys/sysctl.h>
#endif
#include "threads.h"
#include "workers.h"

/* ---------------------------------------------------------------- pool ---
   One persistent pool of `mmx_threads() - 1` helpers plus the calling thread.
   The analyzer opens a parallel region several times per frame - tens of
   thousands of times on a full track - so a region must cost microseconds, not
   the round trip through a condition variable: the helpers poll the job
   counter for a few tens of microseconds before they block, and a helper that
   did fall asleep never holds the caller up, because the caller counts
   finished items and simply does the rest itself.

   That last property is what makes the job descriptor delicate. The caller
   returns as soon as every item is *done*, which it can be long before a
   helper that slept through the whole job has even looked at it. Such a helper
   then reads `fn`/`ctx`/`n` while the caller is already writing the next job
   into them. Reading a half-written descriptor is bad enough on its own, but
   the real damage is that the stale helper could pass the *new* item count to
   the *old* generation's cursor, claim an item of a job that has not started,
   run it a second time and leave `done` one too high - on which the caller's
   `done != n` wait never ends. So the descriptor is published under a seqlock:

     `seq` odd  - the caller is writing the descriptor, nobody may use it
     `seq` even - the descriptor belongs to generation `seq / 2`

   A helper reads the descriptor only at an even `seq` and keeps it only if
   `seq` is unchanged afterwards, so what it holds is exactly one generation's
   (fn, ctx, n) - never a mix of two. The item cursor carries that generation in
   its upper half, and the helper compares the cursor's count against *its own*
   `n`, so a helper that arrives late finds the cursor of its generation
   exhausted (the caller only moves on once every item is done) and claims
   nothing. Every item is therefore run exactly once, by exactly one worker
   slot, and `done` counts exactly the current job.

   Everything shared is read and written with the atomic builtins, so the
   outcome does not depend on the number of threads or on the order the
   scheduler happens to pick: each task writes only to storage indexed by its
   own item or by its own worker slot. `MMX_THREADS=1` takes the old
   single-threaded path (no pool is created, the caller simply runs the loop),
   and every other count produces the same bytes.

   Only the thread that created the pool publishes jobs; nested or concurrent
   parallel regions are not allowed (the analyzer opens them one at a time from
   its own thread). */

#define SPIN_POLLS 30000u      /* polls of the job counter before a helper blocks (tens of microseconds) */
#define YIELD_AFTER 2000u      /* polls the caller does before it yields while waiting for the last item */

typedef struct
{
    pthread_mutex_t m;
    pthread_cond_t work;
    pthread_t tid[MMX_THREADS_MAX];
    unsigned int helpers;

    MMXTaskFn fn;                   /* the descriptor: written only between an odd */
    void *ctx;                      /* and the following even `seq`, and read only */
    unsigned long n;                /* at an even `seq` that did not change meanwhile */
    unsigned long seq;              /* 2 * generation, +1 while the descriptor is written */
    unsigned long long cursor;      /* (generation tag << 32) | next item */
    unsigned long done;             /* items finished in this job */
} Pool;

static Pool pool;
static pthread_once_t pool_once = PTHREAD_ONCE_INIT;
static int pool_ready = 0;

unsigned int mmx_threads(void)
{
    static unsigned int n = 0;
    if (n == 0)
    {
        const char *e = getenv("MMX_THREADS");
        long long v = e ? atol(e) : 0;
        if (v <= 0)
        {
            long long cores = 0;
#if defined(__APPLE__)
            /* Asymmetric cores: _SC_NPROCESSORS_ONLN counts the efficiency cores too, and taking
               them costs more than they bring. Measured on an M1 (4 performance + 4 efficiency),
               one track at quality 7: with 4 threads 43.6 s analysis + 135.6 s coding = 179.2 s,
               with 8 threads 35.8 s + 167.0 s = 202.8 s. The extra four cores buy the parallel
               analysis 20 % and cost the serial coding 25 %. perflevel0 is the fast cluster. */
            {
                size_t len = sizeof(int);
                int perf = 0;
                if (sysctlbyname("hw.perflevel0.logicalcpu", &perf, &len, NULL, 0) == 0 && perf > 0)
                    cores = perf;
            }
#endif
            if (cores <= 0)
                cores = (long long)mmx_workers_cores();
            v = cores > 0 ? cores : 1;
        }
        if (v > MMX_THREADS_MAX) v = MMX_THREADS_MAX;
        n = (unsigned int)v;
    }
    return n;
}

/* Claims and runs items of generation `gen` until they are gone or the job
   changed. Every claim is a compare-and-swap on the packed cursor, so an item
   is handed out exactly once and only to a worker that holds that generation's
   own descriptor - `n` is the caller's parameter, never the pool's field. */
#define GEN_TAG(g) ((unsigned long long)((g) & 0xffffffffUL))
#define ITEM_MAX 0x7fffffffUL       /* items of one job: the cursor's lower half */

static void run_items(Pool *p, unsigned long gen, MMXTaskFn fn, void *ctx, unsigned long n, unsigned int worker)
{
    for (;;)
    {
        unsigned long long cur = __atomic_load_n(&p->cursor, __ATOMIC_ACQUIRE);
        unsigned long i;
        if ((cur >> 32) != GEN_TAG(gen))
            return;
        i = (unsigned long)(cur & 0xffffffffULL);
        if (i >= n)
            return;
        if (!__atomic_compare_exchange_n(&p->cursor, &cur, cur + 1, 1, __ATOMIC_ACQ_REL, __ATOMIC_RELAXED))
            continue;
        fn(ctx, i, worker);
        __atomic_fetch_add(&p->done, 1, __ATOMIC_RELEASE);
    }
}

/* Reads the descriptor of the job `s` names. Returns 0 if the caller started
   publishing another one meanwhile, in which case this helper simply sits the
   job out - the caller does every item nobody took. */
static int take_job(Pool *p, unsigned long s, MMXTaskFn *fn, void **ctx, unsigned long *n)
{
    *fn = __atomic_load_n(&p->fn, __ATOMIC_RELAXED);
    *ctx = __atomic_load_n(&p->ctx, __ATOMIC_RELAXED);
    *n = __atomic_load_n(&p->n, __ATOMIC_RELAXED);
    /* the three relaxed loads above must not be reordered after the validating load, or a helper
       could validate an old generation while already holding the next one's descriptor */
    __atomic_thread_fence(__ATOMIC_ACQUIRE);
    return __atomic_load_n(&p->seq, __ATOMIC_RELAXED) == s;
}

static void *worker_main(void *arg)
{
    Pool *p = &pool;
    unsigned int worker = (unsigned int)(size_t)arg;
    unsigned long seen = 0;

    for (;;)
    {
        unsigned long s;
        unsigned int spins = 0;
        MMXTaskFn fn;
        void *ctx;
        unsigned long n;

        /* wait for a published job (even `seq`) that is not the one just done */
        for (;;)
        {
            s = __atomic_load_n(&p->seq, __ATOMIC_ACQUIRE);
            if (s != seen && (s & 1UL) == 0)
                break;
            if (++spins < SPIN_POLLS)
                continue;
            pthread_mutex_lock(&p->m);
            while (((s = __atomic_load_n(&p->seq, __ATOMIC_ACQUIRE)) == seen) || (s & 1UL) != 0)
                pthread_cond_wait(&p->work, &p->m);
            pthread_mutex_unlock(&p->m);
            break;
        }
        if (!take_job(p, s, &fn, &ctx, &n))
            continue;                       /* the next job was already being published */
        seen = s;
        run_items(p, s >> 1, fn, ctx, n, worker);
    }
}

static void pool_start(void)
{
    Pool *p = &pool;
    unsigned int want = mmx_threads(), i;
    if (want < 2)
        return;
    if (pthread_mutex_init(&p->m, NULL) != 0 || pthread_cond_init(&p->work, NULL) != 0)
        return;
    for (i = 1; i < want; i++)
    {
        if (pthread_create(&p->tid[i], NULL, worker_main, (void *)(size_t)i) != 0)
            break;                      /* fewer helpers than asked: still correct, just slower */
        p->helpers++;
    }
    pool_ready = p->helpers > 0;
}

void mmx_parallel_for(unsigned long n, MMXTaskFn fn, void *ctx)
{
    Pool *p = &pool;
    unsigned long i, s, gen;
    unsigned int spins = 0;

    if (n == 0)
        return;
    if (n > 1 && n <= ITEM_MAX && mmx_threads() > 1)
        pthread_once(&pool_once, pool_start);
    if (n == 1 || n > ITEM_MAX || !pool_ready)
    {
        for (i = 0; i < n; i++)
            fn(ctx, i, 0);
        return;
    }

    s = __atomic_load_n(&p->seq, __ATOMIC_RELAXED);   /* even; only this thread publishes jobs */
    gen = (s >> 1) + 1;
    __atomic_store_n(&p->seq, s + 1, __ATOMIC_RELAXED);     /* odd: the descriptor is in flux */
    /* a release store does not stop the FOLLOWING stores from becoming visible first, which is what
       the odd marker has to prevent; the fence does */
    __atomic_thread_fence(__ATOMIC_RELEASE);
    __atomic_store_n(&p->fn, fn, __ATOMIC_RELAXED);
    __atomic_store_n(&p->ctx, ctx, __ATOMIC_RELAXED);
    __atomic_store_n(&p->n, n, __ATOMIC_RELAXED);
    __atomic_store_n(&p->done, 0, __ATOMIC_RELAXED);
    __atomic_store_n(&p->cursor, GEN_TAG(gen) << 32, __ATOMIC_RELEASE);
    pthread_mutex_lock(&p->m);                  /* under the lock, so a helper that is
                                                   about to block cannot miss the job */
    __atomic_store_n(&p->seq, s + 2, __ATOMIC_RELEASE);     /* even again: generation `gen` is open */
    pthread_cond_broadcast(&p->work);
    pthread_mutex_unlock(&p->m);

    run_items(p, gen, fn, ctx, n, 0);           /* the caller is worker 0 and works too */
    while (__atomic_load_n(&p->done, __ATOMIC_ACQUIRE) != n)
        if (++spins > YIELD_AFTER) { sched_yield(); spins = 0; }
}
