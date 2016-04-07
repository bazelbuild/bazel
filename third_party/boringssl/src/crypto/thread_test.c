/* Copyright (c) 2015, Google Inc.
 *
 * Permission to use, copy, modify, and/or distribute this software for any
 * purpose with or without fee is hereby granted, provided that the above
 * copyright notice and this permission notice appear in all copies.
 *
 * THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES
 * WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF
 * MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY
 * SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
 * WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN ACTION
 * OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF OR IN
 * CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE. */

#include "internal.h"

#include <stdio.h>


#if !defined(OPENSSL_NO_THREADS)

#if defined(OPENSSL_WINDOWS)

#pragma warning(push, 3)
#include <windows.h>
#pragma warning(pop)

typedef HANDLE thread_t;

static DWORD WINAPI thread_run(LPVOID arg) {
  void (*thread_func)(void);
  /* VC really doesn't like casting between data and function pointers. */
  memcpy(&thread_func, &arg, sizeof(thread_func));
  thread_func();
  return 0;
}

static int run_thread(thread_t *out_thread, void (*thread_func)(void)) {
  void *arg;
  /* VC really doesn't like casting between data and function pointers. */
  memcpy(&arg, &thread_func, sizeof(arg));

  *out_thread = CreateThread(NULL /* security attributes */,
                             0 /* default stack size */, thread_run, arg,
                             0 /* run immediately */, NULL /* ignore id */);
  return *out_thread != NULL;
}

static int wait_for_thread(thread_t thread) {
  return WaitForSingleObject(thread, INFINITE) == 0;
}

#else

#include <pthread.h>

typedef pthread_t thread_t;

static void *thread_run(void *arg) {
  void (*thread_func)(void) = arg;
  thread_func();
  return NULL;
}

static int run_thread(thread_t *out_thread, void (*thread_func)(void)) {
  return pthread_create(out_thread, NULL /* default attributes */, thread_run,
                        thread_func) == 0;
}

static int wait_for_thread(thread_t thread) {
  return pthread_join(thread, NULL) == 0;
}

#endif  /* OPENSSL_WINDOWS */

static unsigned g_once_init_called = 0;

static void once_init(void) {
  g_once_init_called++;
}

static CRYPTO_once_t g_test_once = CRYPTO_ONCE_INIT;

static void call_once_thread(void) {
  CRYPTO_once(&g_test_once, once_init);
}

static int test_once(void) {
  if (g_once_init_called != 0) {
    fprintf(stderr, "g_once_init_called was non-zero at start.\n");
    return 0;
  }

  thread_t thread;
  if (!run_thread(&thread, call_once_thread) ||
      !wait_for_thread(thread)) {
    fprintf(stderr, "thread failed.\n");
    return 0;
  }

  CRYPTO_once(&g_test_once, once_init);

  if (g_once_init_called != 1) {
    fprintf(stderr, "Expected init function to be called once, but found %u.\n",
            g_once_init_called);
    return 0;
  }

  return 1;
}


static int g_test_thread_ok = 0;
static unsigned g_destructor_called_count = 0;

static void thread_local_destructor(void *arg) {
  if (arg == NULL) {
    return;
  }

  unsigned *count = arg;
  (*count)++;
}

static void thread_local_test_thread(void) {
  void *ptr = CRYPTO_get_thread_local(OPENSSL_THREAD_LOCAL_TEST);
  if (ptr != NULL) {
    return;
  }

  if (!CRYPTO_set_thread_local(OPENSSL_THREAD_LOCAL_TEST,
                               &g_destructor_called_count,
                               thread_local_destructor)) {
    return;
  }

  if (CRYPTO_get_thread_local(OPENSSL_THREAD_LOCAL_TEST) !=
      &g_destructor_called_count) {
    return;
  }

  g_test_thread_ok = 1;
}

static void thread_local_test2_thread(void) {}

static int test_thread_local(void) {
  void *ptr = CRYPTO_get_thread_local(OPENSSL_THREAD_LOCAL_TEST);
  if (ptr != NULL) {
    fprintf(stderr, "Thread-local data was non-NULL at start.\n");
  }

  thread_t thread;
  if (!run_thread(&thread, thread_local_test_thread) ||
      !wait_for_thread(thread)) {
    fprintf(stderr, "thread failed.\n");
    return 0;
  }

  if (!g_test_thread_ok) {
    fprintf(stderr, "Thread-local data didn't work in thread.\n");
    return 0;
  }

  if (g_destructor_called_count != 1) {
    fprintf(stderr,
            "Destructor should have been called once, but actually called %u "
            "times.\n",
            g_destructor_called_count);
    return 0;
  }

  /* thread_local_test2_thread doesn't do anything, but it tests that the
   * thread destructor function works even if thread-local storage wasn't used
   * for a thread. */
  if (!run_thread(&thread, thread_local_test2_thread) ||
      !wait_for_thread(thread)) {
    fprintf(stderr, "thread failed.\n");
    return 0;
  }

  return 1;
}

int main(int argc, char **argv) {
  if (!test_once() ||
      !test_thread_local()) {
    return 1;
  }

  printf("PASS\n");
  return 0;
}

#else  /* OPENSSL_NO_THREADS */

int main(int argc, char **argv) {
  printf("PASS\n");
  return 0;
}

#endif
