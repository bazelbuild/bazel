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

#include <stdio.h>
#include <string.h>

#include <vector>

#include <openssl/crypto.h>
#include <openssl/poly1305.h>

#include "../test/file_test.h"
#include "../test/stl_compat.h"


// |CRYPTO_poly1305_finish| requires a 16-byte-aligned output.
#if defined(OPENSSL_WINDOWS)
// MSVC doesn't support C++11 |alignas|.
#define ALIGNED __declspec(align(16))
#else
#define ALIGNED alignas(16)
#endif

static bool TestPoly1305(FileTest *t, void *arg) {
  std::vector<uint8_t> key, in, mac;
  if (!t->GetBytes(&key, "Key") ||
      !t->GetBytes(&in, "Input") ||
      !t->GetBytes(&mac, "MAC")) {
    return false;
  }
  if (key.size() != 32 || mac.size() != 16) {
    t->PrintLine("Invalid test");
    return false;
  }

  // Test single-shot operation.
  poly1305_state state;
  CRYPTO_poly1305_init(&state, bssl::vector_data(&key));
  CRYPTO_poly1305_update(&state, bssl::vector_data(&in), in.size());
  ALIGNED uint8_t out[16];
  CRYPTO_poly1305_finish(&state, out);
  if (!t->ExpectBytesEqual(out, 16, bssl::vector_data(&mac), mac.size())) {
    t->PrintLine("Single-shot Poly1305 failed.");
    return false;
  }

  // Test streaming byte-by-byte.
  CRYPTO_poly1305_init(&state, bssl::vector_data(&key));
  for (size_t i = 0; i < in.size(); i++) {
    CRYPTO_poly1305_update(&state, &in[i], 1);
  }
  CRYPTO_poly1305_finish(&state, out);
  if (!t->ExpectBytesEqual(out, 16, bssl::vector_data(&mac), mac.size())) {
    t->PrintLine("Streaming Poly1305 failed.");
    return false;
  }

  return true;
}

int main(int argc, char **argv) {
  CRYPTO_library_init();

  if (argc != 2) {
    fprintf(stderr, "%s <test file>\n", argv[0]);
    return 1;
  }

  return FileTestMain(TestPoly1305, nullptr, argv[1]);
}
