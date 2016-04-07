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

#include <openssl/type_check.h>


int main(int argc, char **argv) {
  CRYPTO_refcount_t count = 0;

  CRYPTO_refcount_inc(&count);
  if (count != 1) {
    fprintf(stderr, "Incrementing reference count did not work.\n");
    return 1;
  }
  if (!CRYPTO_refcount_dec_and_test_zero(&count) || count != 0) {
    fprintf(stderr, "Decrementing reference count to zero did not work.\n");
    return 1;
  }

  count = CRYPTO_REFCOUNT_MAX;
  CRYPTO_refcount_inc(&count);
  if (count != CRYPTO_REFCOUNT_MAX) {
    fprintf(stderr, "Count did not saturate correctly when incrementing.\n");
    return 1;
  }
  if (CRYPTO_refcount_dec_and_test_zero(&count) ||
      count != CRYPTO_REFCOUNT_MAX) {
    fprintf(stderr, "Count did not saturate correctly when decrementing.\n");
    return 1;
  }

  count = 2;
  if (CRYPTO_refcount_dec_and_test_zero(&count)) {
    fprintf(stderr, "Decrementing two resulted in zero!\n");
    return 1;
  }
  if (count != 1) {
    fprintf(stderr, "Decrementing two did not produce one!");
    return 1;
  }

  printf("PASS\n");
  return 0;
}
