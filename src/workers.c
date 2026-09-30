#include <stdlib.h>
#include <string.h>
#include "workers.h"
#ifndef _WIN32
#include <unistd.h>
#endif

#define MAX_HELPERS 4

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
typedef CRITICAL_SECTION Mtx;
typedef CONDITION_VARIABLE Cnd;
typedef HANDLE Thr;
static void mtx_init(Mtx *m) { InitializeCriticalSection(m); }
static void mtx_free(Mtx *m) { DeleteCriticalSection(m); }
static void mtx_lock(Mtx *m) { EnterCriticalSection(m); }
static void mtx_unlock(Mtx *m) { LeaveCriticalSection(m); }
static void cnd_init_(Cnd *c) { InitializeConditionVariable(c); }
static void cnd_free(Cnd *c) { (void)c; }
static void cnd_wait_(Cnd *c, Mtx *m) { SleepConditionVariableCS(c, m, INFINITE); }
static void cnd_bcast(Cnd *c) { WakeAllConditionVariable(c); }
#else
#include <pthread.h>
typedef pthread_mutex_t Mtx;
typedef pthread_cond_t Cnd;
typedef pthread_t Thr;
static void mtx_init(Mtx *m) { pthread_mutex_init(m, NULL); }
static void mtx_free(Mtx *m) { pthread_mutex_destroy(m); }
static void mtx_lock(Mtx *m) { pthread_mutex_lock(m); }
static void mtx_unlock(Mtx *m) { pthread_mutex_unlock(m); }
static void cnd_init_(Cnd *c) { pthread_cond_init(c, NULL); }
static void cnd_free(Cnd *c) { pthread_cond_destroy(c); }
static void cnd_wait_(Cnd *c, Mtx *m) { pthread_cond_wait(c, m); }
static void cnd_bcast(Cnd *c) { pthread_cond_broadcast(c); }
#endif

typedef struct
{
    MMXWorkers *w;
    unsigned int index;                 /* item index + 1 of this helper */
} Helper;

struct MMXWorkers
{
    Mtx m;
    Cnd go, done;
    unsigned int helpers;
    Thr tid[MAX_HELPERS];
    Helper h[MAX_HELPERS];
    unsigned long gen;                  /* job generation */
    MMXWorkFn fn;
    void *ctx;
    unsigned int count;
    unsigned int finished;              /* helper items finished in this generation */
    int quit;
};

static void helper_loop(Helper *h)
{
    MMXWorkers *w = h->w;
    unsigned long seen = 0;
    for (;;)
    {
        MMXWorkFn fn;
        void *ctx;
        unsigned int count;
        mtx_lock(&w->m);
        while (!w->quit && w->gen == seen)
            cnd_wait_(&w->go, &w->m);
        if (w->quit) { mtx_unlock(&w->m); return; }
        seen = w->gen;
        fn = w->fn; ctx = w->ctx; count = w->count;
        mtx_unlock(&w->m);
        if (h->index < count)
            fn(ctx, h->index);
        mtx_lock(&w->m);
        w->finished++;
        cnd_bcast(&w->done);
        mtx_unlock(&w->m);
    }
}

#ifdef _WIN32
static DWORD WINAPI helper_main(LPVOID arg) { helper_loop((Helper *)arg); return 0; }
#else
static void *helper_main(void *arg) { helper_loop((Helper *)arg); return NULL; }
#endif

MMXWorkers *mmx_workers_new(unsigned int helpers)
{
    MMXWorkers *w = (MMXWorkers *)calloc(1, sizeof(MMXWorkers));
    const char *e = getenv("MMX_DECODE_THREADS");
    unsigned int i;
    if (!w) return NULL;
    if (e && atoi(e) >= 1 && (unsigned int)atoi(e) - 1 < helpers) helpers = (unsigned int)atoi(e) - 1;
    if (helpers > MAX_HELPERS) helpers = MAX_HELPERS;
    mtx_init(&w->m);
    cnd_init_(&w->go);
    cnd_init_(&w->done);
    for (i = 0; i < helpers; i++)
    {
        w->h[i].w = w;
        w->h[i].index = i + 1;
#ifdef _WIN32
        w->tid[i] = CreateThread(NULL, 0, helper_main, &w->h[i], 0, NULL);
        if (!w->tid[i]) break;
#else
        if (pthread_create(&w->tid[i], NULL, helper_main, &w->h[i]) != 0) break;
#endif
        w->helpers++;
    }
    return w;
}

void mmx_workers_run(MMXWorkers *w, MMXWorkFn fn, void *ctx, unsigned int count)
{
    unsigned int i;
    if (!w || w->helpers == 0 || count <= 1)
    {
        for (i = 0; i < count; i++) fn(ctx, i);
        return;
    }
    mtx_lock(&w->m);
    w->fn = fn; w->ctx = ctx; w->count = count; w->finished = 0; w->gen++;
    cnd_bcast(&w->go);
    mtx_unlock(&w->m);
    fn(ctx, 0);
    for (i = w->helpers + 1; i < count; i++) fn(ctx, i);   /* items beyond the helpers run here */
    mtx_lock(&w->m);
    while (w->finished < w->helpers)
        cnd_wait_(&w->done, &w->m);
    mtx_unlock(&w->m);
}

void mmx_workers_free(MMXWorkers *w)
{
    unsigned int i;
    if (!w) return;
    mtx_lock(&w->m);
    w->quit = 1;
    cnd_bcast(&w->go);
    mtx_unlock(&w->m);
    for (i = 0; i < w->helpers; i++)
    {
#ifdef _WIN32
        WaitForSingleObject(w->tid[i], INFINITE);
        CloseHandle(w->tid[i]);
#else
        pthread_join(w->tid[i], NULL);
#endif
    }
    cnd_free(&w->go);
    cnd_free(&w->done);
    mtx_free(&w->m);
    free(w);
}

unsigned int mmx_workers_cores(void)
{
#ifdef _WIN32
    SYSTEM_INFO si;
    GetSystemInfo(&si);
    return si.dwNumberOfProcessors ? (unsigned int)si.dwNumberOfProcessors : 1u;
#else
    long n = sysconf(_SC_NPROCESSORS_ONLN);
    return n > 0 ? (unsigned int)n : 1u;
#endif
}
