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

#include <stdio.h>
#include <string.h>

#include <openssl/crypto.h>
#include <openssl/err.h>
#include <openssl/mem.h>


static bool TestOverflow() {
  for (unsigned i = 0; i < ERR_NUM_ERRORS*2; i++) {
    ERR_put_error(1, i+1, "function", "test", 1);
  }

  for (unsigned i = 0; i < ERR_NUM_ERRORS - 1; i++) {
    uint32_t err = ERR_get_error();
    /* Errors are returned in order they were pushed, with the least recent ones
     * removed, up to |ERR_NUM_ERRORS - 1| errors. So the errors returned are
     * |ERR_NUM_ERRORS + 2| through |ERR_NUM_ERRORS * 2|, inclusive. */
    if (err == 0 || ERR_GET_REASON(err) != i + ERR_NUM_ERRORS + 2) {
      fprintf(stderr, "ERR_get_error failed at %u\n", i);
      return false;
    }
  }

  if (ERR_get_error() != 0) {
    fprintf(stderr, "ERR_get_error more than the expected number of values.\n");
    return false;
  }

  return true;
}

static bool TestPutError() {
  if (ERR_get_error() != 0) {
    fprintf(stderr, "ERR_get_error returned value before an error was added.\n");
    return false;
  }

  ERR_put_error(1, 2, "function", "test", 4);
  ERR_add_error_data(1, "testing");

  int peeked_line, line, peeked_flags, flags;
  const char *peeked_file, *file, *peeked_data, *data;
  uint32_t peeked_packed_error =
      ERR_peek_error_line_data(&peeked_file, &peeked_line, &peeked_data,
                               &peeked_flags);
  const char *function = ERR_peek_function();
  uint32_t packed_error = ERR_get_error_line_data(&file, &line, &data, &flags);

  if (peeked_packed_error != packed_error ||
      peeked_file != file ||
      peeked_data != data ||
      peeked_flags != flags) {
    fprintf(stderr, "Bad peeked error data returned.\n");
    return false;
  }

  if (strcmp(function, "function") != 0 ||
      strcmp(file, "test") != 0 ||
      line != 4 ||
      (flags & ERR_FLAG_STRING) == 0 ||
      ERR_GET_LIB(packed_error) != 1 ||
      ERR_GET_REASON(packed_error) != 2 ||
      strcmp(data, "testing") != 0) {
    fprintf(stderr, "Bad error data returned.\n");
    return false;
  }

  return true;
}

static bool TestClearError() {
  if (ERR_get_error() != 0) {
    fprintf(stderr, "ERR_get_error returned value before an error was added.\n");
    return false;
  }

  ERR_put_error(1, 2, "function", "test", 4);
  ERR_clear_error();

  if (ERR_get_error() != 0) {
    fprintf(stderr, "Error remained after clearing.\n");
    return false;
  }

  return true;
}

static bool TestPrint() {
  ERR_put_error(1, 2, "function", "test", 4);
  ERR_add_error_data(1, "testing");
  uint32_t packed_error = ERR_get_error();

  char buf[256];
  for (size_t i = 0; i <= sizeof(buf); i++) {
    ERR_error_string_n(packed_error, buf, i);
  }

  return true;
}

static bool TestRelease() {
  ERR_put_error(1, 2, "function", "test", 4);
  ERR_remove_thread_state(NULL);
  return true;
}

static bool HasSuffix(const char *str, const char *suffix) {
  size_t suffix_len = strlen(suffix);
  size_t str_len = strlen(str);
  if (str_len < suffix_len) {
    return false;
  }
  return strcmp(str + str_len - suffix_len, suffix) == 0;
}

static bool TestPutMacro() {
  int expected_line = __LINE__ + 1;
  OPENSSL_PUT_ERROR(USER, ERR_R_INTERNAL_ERROR);

  int line;
  const char *file;
  const char *function = ERR_peek_function();
  uint32_t error = ERR_get_error_line(&file, &line);

  if (strcmp(function, "TestPutMacro") != 0 ||
      !HasSuffix(file, "err_test.cc") ||
      line != expected_line ||
      ERR_GET_LIB(error) != ERR_LIB_USER ||
      ERR_GET_REASON(error) != ERR_R_INTERNAL_ERROR) {
    fprintf(stderr, "Bad error data returned.\n");
    return false;
  }

  return true;
}

int main() {
  CRYPTO_library_init();

  if (!TestOverflow() ||
      !TestPutError() ||
      !TestClearError() ||
      !TestPrint() ||
      !TestRelease() ||
      !TestPutMacro()) {
    return 1;
  }

  printf("PASS\n");
  return 0;
}
