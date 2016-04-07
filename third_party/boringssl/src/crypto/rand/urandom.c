/* Copyright (c) 2014, Google Inc.
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

#include <openssl/rand.h>

#if !defined(OPENSSL_WINDOWS)

#include <assert.h>
#include <errno.h>
#include <fcntl.h>
#include <string.h>
#include <unistd.h>

#include <openssl/thread.h>
#include <openssl/mem.h>

#include "internal.h"
#include "../internal.h"


/* This file implements a PRNG by reading from /dev/urandom, optionally with a
 * buffer, which is unsafe across |fork|. */

#define BUF_SIZE 4096

/* rand_buffer contains unused, random bytes, some of which may have been
 * consumed already. */
struct rand_buffer {
  size_t used;
  uint8_t rand[BUF_SIZE];
};

/* requested_lock is used to protect the |*_requested| variables. */
static struct CRYPTO_STATIC_MUTEX requested_lock = CRYPTO_STATIC_MUTEX_INIT;

/* urandom_fd_requested is set by |RAND_set_urandom_fd|.  It's protected by
 * |requested_lock|. */
static int urandom_fd_requested = -2;

/* urandom_fd is a file descriptor to /dev/urandom. It's protected by |once|. */
static int urandom_fd = -2;

/* urandom_buffering_requested is set by |RAND_enable_fork_unsafe_buffering|.
 * It's protected by |requested_lock|. */
static int urandom_buffering_requested = 0;

/* urandom_buffering controls whether buffering is enabled (1) or not (0). This
 * is protected by |once|. */
static int urandom_buffering = 0;

static CRYPTO_once_t once = CRYPTO_ONCE_INIT;

/* init_once initializes the state of this module to values previously
 * requested. This is the only function that modifies |urandom_fd| and
 * |urandom_buffering|, whose values may be read safely after calling the
 * once. */
static void init_once(void) {
  CRYPTO_STATIC_MUTEX_lock_read(&requested_lock);
  urandom_buffering = urandom_buffering_requested;
  int fd = urandom_fd_requested;
  CRYPTO_STATIC_MUTEX_unlock(&requested_lock);

  if (fd == -2) {
    do {
      fd = open("/dev/urandom", O_RDONLY);
    } while (fd == -1 && errno == EINTR);
  }

  if (fd < 0) {
    abort();
  }

  int flags = fcntl(fd, F_GETFD);
  if (flags == -1) {
    abort();
  }
  flags |= FD_CLOEXEC;
  if (fcntl(fd, F_SETFD, flags) == -1) {
    abort();
  }
  urandom_fd = fd;
}

void RAND_cleanup(void) {}

void RAND_set_urandom_fd(int fd) {
  fd = dup(fd);
  if (fd < 0) {
    abort();
  }

  CRYPTO_STATIC_MUTEX_lock_write(&requested_lock);
  urandom_fd_requested = fd;
  CRYPTO_STATIC_MUTEX_unlock(&requested_lock);

  CRYPTO_once(&once, init_once);
  if (urandom_fd != fd) {
    abort();  // Already initialized.
  }
}

void RAND_enable_fork_unsafe_buffering(int fd) {
  if (fd >= 0) {
    fd = dup(fd);
    if (fd < 0) {
      abort();
    }
  } else {
    fd = -2;
  }

  CRYPTO_STATIC_MUTEX_lock_write(&requested_lock);
  urandom_buffering_requested = 1;
  urandom_fd_requested = fd;
  CRYPTO_STATIC_MUTEX_unlock(&requested_lock);

  CRYPTO_once(&once, init_once);
  if (urandom_buffering != 1 || (fd >= 0 && urandom_fd != fd)) {
    abort();  // Already initialized.
  }
}

static struct rand_buffer *get_thread_local_buffer(void) {
  struct rand_buffer *buf =
      CRYPTO_get_thread_local(OPENSSL_THREAD_LOCAL_URANDOM_BUF);
  if (buf != NULL) {
    return buf;
  }

  buf = OPENSSL_malloc(sizeof(struct rand_buffer));
  if (buf == NULL) {
    return NULL;
  }
  buf->used = BUF_SIZE;  /* To trigger a |read_full| on first use. */
  if (!CRYPTO_set_thread_local(OPENSSL_THREAD_LOCAL_URANDOM_BUF, buf,
                               OPENSSL_free)) {
    OPENSSL_free(buf);
    return NULL;
  }

  return buf;
}

/* read_full reads exactly |len| bytes from |fd| into |out| and returns 1. In
 * the case of an error it returns 0. */
static char read_full(int fd, uint8_t *out, size_t len) {
  ssize_t r;

  while (len > 0) {
    do {
      r = read(fd, out, len);
    } while (r == -1 && errno == EINTR);

    if (r <= 0) {
      return 0;
    }
    out += r;
    len -= r;
  }

  return 1;
}

/* read_from_buffer reads |requested| random bytes from the buffer into |out|,
 * refilling it if necessary to satisfy the request. */
static void read_from_buffer(struct rand_buffer *buf,
                             uint8_t *out, size_t requested) {
  size_t remaining = BUF_SIZE - buf->used;

  while (requested > remaining) {
    memcpy(out, &buf->rand[buf->used], remaining);
    buf->used += remaining;
    out += remaining;
    requested -= remaining;

    if (!read_full(urandom_fd, buf->rand, BUF_SIZE)) {
      abort();
      return;
    }
    buf->used = 0;
    remaining = BUF_SIZE;
  }

  memcpy(out, &buf->rand[buf->used], requested);
  buf->used += requested;
}

/* CRYPTO_sysrand puts |requested| random bytes into |out|. */
void CRYPTO_sysrand(uint8_t *out, size_t requested) {
  if (requested == 0) {
    return;
  }

  CRYPTO_once(&once, init_once);
  if (urandom_buffering && requested < BUF_SIZE) {
    struct rand_buffer *buf = get_thread_local_buffer();
    if (buf != NULL) {
      read_from_buffer(buf, out, requested);
      return;
    }
  }

  if (!read_full(urandom_fd, out, requested)) {
    abort();
  }
}

#endif  /* !OPENSSL_WINDOWS */
