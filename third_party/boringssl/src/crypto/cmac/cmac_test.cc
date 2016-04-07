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

#include <algorithm>

#include <openssl/cmac.h>

#include "../test/scoped_types.h"
#include "../test/test_util.h"


static void dump(const uint8_t *got, const uint8_t *want, size_t len) {
  hexdump(stderr, "got :", got, len);
  hexdump(stderr, "want:", want, len);
  fflush(stderr);
}

static int test(const char *name, const uint8_t *key, size_t key_len,
                const uint8_t *msg, size_t msg_len, const uint8_t *expected) {
  uint8_t out[16];

  if (!AES_CMAC(out, key, key_len, msg, msg_len)) {
    fprintf(stderr, "%s: AES_CMAC failed\n", name);
    return 0;
  }

  if (CRYPTO_memcmp(out, expected, sizeof(out)) != 0) {
    fprintf(stderr, "%s: CMAC result differs:\n", name);
    dump(out, expected, sizeof(out));
    return 0;
  }

  ScopedCMAC_CTX ctx(CMAC_CTX_new());
  if (!CMAC_Init(ctx.get(), key, key_len, EVP_aes_128_cbc(), NULL)) {
    fprintf(stderr, "%s: CMAC_Init failed.\n", name);
    return 0;
  }

  for (unsigned chunk_size = 1; chunk_size <= msg_len; chunk_size++) {
    if (!CMAC_Reset(ctx.get())) {
      fprintf(stderr, "%s/%u: CMAC_Reset failed.\n", name, chunk_size);
      return 0;
    }

    size_t done = 0;
    while (done < msg_len) {
      size_t todo = std::min(msg_len - done, static_cast<size_t>(chunk_size));
      if (!CMAC_Update(ctx.get(), msg + done, todo)) {
        fprintf(stderr, "%s/%u: CMAC_Update failed.\n", name, chunk_size);
        return 0;
      }

      done += todo;
    }

    size_t out_len;
    if (!CMAC_Final(ctx.get(), out, &out_len)) {
      fprintf(stderr, "%s/%u: CMAC_Final failed.\n", name, chunk_size);
      return 0;
    }

    if (out_len != sizeof(out)) {
      fprintf(stderr, "%s/%u: incorrect out_len: %u.\n", name, chunk_size,
              static_cast<unsigned>(out_len));
      return 0;
    }

    if (CRYPTO_memcmp(out, expected, sizeof(out)) != 0) {
      fprintf(stderr, "%s/%u: CMAC result differs:\n", name, chunk_size);
      dump(out, expected, sizeof(out));
      return 0;
    }
  }

  return 1;
}

static int rfc_4493_test_vectors(void) {
  static const uint8_t kKey[16] = {
      0x2b, 0x7e, 0x15, 0x16, 0x28, 0xae, 0xd2, 0xa6,
      0xab, 0xf7, 0x15, 0x88, 0x09, 0xcf, 0x4f, 0x3c,
  };
  static const uint8_t kOut1[16] = {
      0xbb, 0x1d, 0x69, 0x29, 0xe9, 0x59, 0x37, 0x28,
      0x7f, 0xa3, 0x7d, 0x12, 0x9b, 0x75, 0x67, 0x46,
  };
  static const uint8_t kMsg2[] = {
      0x6b, 0xc1, 0xbe, 0xe2, 0x2e, 0x40, 0x9f, 0x96,
      0xe9, 0x3d, 0x7e, 0x11, 0x73, 0x93, 0x17, 0x2a,
  };
  static const uint8_t kOut2[16] = {
      0x07, 0x0a, 0x16, 0xb4, 0x6b, 0x4d, 0x41, 0x44,
      0xf7, 0x9b, 0xdd, 0x9d, 0xd0, 0x4a, 0x28, 0x7c,
  };
  static const uint8_t kMsg3[] = {
      0x6b, 0xc1, 0xbe, 0xe2, 0x2e, 0x40, 0x9f, 0x96,
      0xe9, 0x3d, 0x7e, 0x11, 0x73, 0x93, 0x17, 0x2a,
      0xae, 0x2d, 0x8a, 0x57, 0x1e, 0x03, 0xac, 0x9c,
      0x9e, 0xb7, 0x6f, 0xac, 0x45, 0xaf, 0x8e, 0x51,
      0x30, 0xc8, 0x1c, 0x46, 0xa3, 0x5c, 0xe4, 0x11,
  };
  static const uint8_t kOut3[16] = {
      0xdf, 0xa6, 0x67, 0x47, 0xde, 0x9a, 0xe6, 0x30,
      0x30, 0xca, 0x32, 0x61, 0x14, 0x97, 0xc8, 0x27,
  };
  static const uint8_t kMsg4[] = {
      0x6b, 0xc1, 0xbe, 0xe2, 0x2e, 0x40, 0x9f, 0x96,
      0xe9, 0x3d, 0x7e, 0x11, 0x73, 0x93, 0x17, 0x2a,
      0xae, 0x2d, 0x8a, 0x57, 0x1e, 0x03, 0xac, 0x9c,
      0x9e, 0xb7, 0x6f, 0xac, 0x45, 0xaf, 0x8e, 0x51,
      0x30, 0xc8, 0x1c, 0x46, 0xa3, 0x5c, 0xe4, 0x11,
      0xe5, 0xfb, 0xc1, 0x19, 0x1a, 0x0a, 0x52, 0xef,
      0xf6, 0x9f, 0x24, 0x45, 0xdf, 0x4f, 0x9b, 0x17,
      0xad, 0x2b, 0x41, 0x7b, 0xe6, 0x6c, 0x37, 0x10,
  };
  static const uint8_t kOut4[16] = {
      0x51, 0xf0, 0xbe, 0xbf, 0x7e, 0x3b, 0x9d, 0x92,
      0xfc, 0x49, 0x74, 0x17, 0x79, 0x36, 0x3c, 0xfe,
  };

  if (!test("RFC 4493 #1", kKey, sizeof(kKey), NULL, 0, kOut1) ||
      !test("RFC 4493 #2", kKey, sizeof(kKey), kMsg2, sizeof(kMsg2), kOut2) ||
      !test("RFC 4493 #3", kKey, sizeof(kKey), kMsg3, sizeof(kMsg3), kOut3) ||
      !test("RFC 4493 #4", kKey, sizeof(kKey), kMsg4, sizeof(kMsg4), kOut4)) {
    return 0;
  }

  return 1;
}

int main(int argc, char **argv) {
  if (!rfc_4493_test_vectors()) {
    return 1;
  }

  printf("PASS\n");
  return 0;
}
