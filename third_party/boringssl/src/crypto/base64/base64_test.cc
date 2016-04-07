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

#include <openssl/base64.h>
#include <openssl/crypto.h>
#include <openssl/err.h>


struct TestVector {
  const char *decoded;
  const char *encoded;
};

// Test vectors from RFC 4648.
static const TestVector kTestVectors[] = {
  { "", "" },
  { "f" , "Zg==" },
  { "fo", "Zm8=" },
  { "foo", "Zm9v" },
  { "foob", "Zm9vYg==" },
  { "fooba", "Zm9vYmE=" },
  { "foobar", "Zm9vYmFy" },
};

static const size_t kNumTests = sizeof(kTestVectors) / sizeof(kTestVectors[0]);

static bool TestEncode() {
  for (size_t i = 0; i < kNumTests; i++) {
    const TestVector *t = &kTestVectors[i];
    uint8_t out[9];
    size_t len = EVP_EncodeBlock(out, (const uint8_t*)t->decoded,
                                 strlen(t->decoded));
    if (len != strlen(t->encoded) ||
        memcmp(out, t->encoded, len) != 0) {
      fprintf(stderr, "encode(\"%s\") = \"%.*s\", want \"%s\"\n",
              t->decoded, (int)len, (const char*)out, t->encoded);
      return false;
    }
  }
  return true;
}

static bool TestDecode() {
  uint8_t out[6];
  size_t len;

  for (size_t i = 0; i < kNumTests; i++) {
    // Test the normal API.
    const TestVector *t = &kTestVectors[i];
    size_t expected_len = strlen(t->decoded);
    if (!EVP_DecodeBase64(out, &len, sizeof(out),
                          (const uint8_t*)t->encoded, strlen(t->encoded))) {
      fprintf(stderr, "decode(\"%s\") failed\n", t->encoded);
      return false;
    }
    if (len != strlen(t->decoded) ||
        memcmp(out, t->decoded, len) != 0) {
      fprintf(stderr, "decode(\"%s\") = \"%.*s\", want \"%s\"\n",
              t->encoded, (int)len, (const char*)out, t->decoded);
      return false;
    }

    // Test that the padding behavior of the deprecated API is preserved.
    int ret = EVP_DecodeBlock(out, (const uint8_t*)t->encoded,
                              strlen(t->encoded));
    if (ret < 0) {
      fprintf(stderr, "decode(\"%s\") failed\n", t->encoded);
      return false;
    }
    if (ret % 3 != 0) {
      fprintf(stderr, "EVP_DecodeBlock did not ignore padding\n");
      return false;
    }
    if (expected_len % 3 != 0) {
      ret -= 3 - (expected_len % 3);
    }
    if (static_cast<size_t>(ret) != strlen(t->decoded) ||
        memcmp(out, t->decoded, ret) != 0) {
      fprintf(stderr, "decode(\"%s\") = \"%.*s\", want \"%s\"\n",
              t->encoded, ret, (const char*)out, t->decoded);
      return false;
    }
  }

  if (EVP_DecodeBase64(out, &len, sizeof(out), (const uint8_t*)"a!bc", 4)) {
    fprintf(stderr, "Failed to reject invalid characters in the middle.\n");
    return false;
  }

  if (EVP_DecodeBase64(out, &len, sizeof(out), (const uint8_t*)"a=bc", 4)) {
    fprintf(stderr, "Failed to reject invalid characters in the middle.\n");
    return false;
  }

  if (EVP_DecodeBase64(out, &len, sizeof(out), (const uint8_t*)"abc", 4)) {
    fprintf(stderr, "Failed to reject invalid input length.\n");
    return false;
  }

  return true;
}

int main(void) {
  CRYPTO_library_init();
  ERR_load_crypto_strings();

  if (!TestEncode() ||
      !TestDecode()) {
    return 1;
  }

  printf("PASS\n");
  return 0;
}
