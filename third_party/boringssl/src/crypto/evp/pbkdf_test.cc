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

#include <openssl/bio.h>
#include <openssl/crypto.h>
#include <openssl/digest.h>
#include <openssl/err.h>
#include <openssl/evp.h>


// Prints out the data buffer as a sequence of hex bytes.
static void PrintDataHex(const void *data, size_t len) {
  for (size_t i = 0; i < len; ++i) {
    fprintf(stderr, "%02x", (int)((const uint8_t *)data)[i]);
  }
}

// Helper for testing that PBKDF2 derives the expected key from the given
// inputs. Returns 1 on success, 0 otherwise.
static bool TestPBKDF2(const void *password, size_t password_len,
                       const void *salt, size_t salt_len, unsigned iterations,
                       const EVP_MD *digest, size_t key_len,
                       const uint8_t *expected_key) {
  uint8_t key[64];

  if (key_len > sizeof(key)) {
    fprintf(stderr, "Output buffer is not large enough.\n");
    return false;
  }

  if (!PKCS5_PBKDF2_HMAC((const char *)password, password_len,
                         (const uint8_t *)salt, salt_len, iterations, digest,
                         key_len, key)) {
    fprintf(stderr, "Call to PKCS5_PBKDF2_HMAC failed\n");
    ERR_print_errors_fp(stderr);
    return false;
  }

  if (memcmp(key, expected_key, key_len) != 0) {
    fprintf(stderr, "Resulting key material does not match expectation\n");
    fprintf(stderr, "Expected:\n    ");
    PrintDataHex(expected_key, key_len);
    fprintf(stderr, "\nActual:\n    ");
    PrintDataHex(key, key_len);
    fprintf(stderr, "\n");
    return false;
  }

  return true;
}

// Tests deriving a key using an empty password (specified both as NULL and as
// non-NULL). Note that NULL has special meaning to HMAC initialization.
static bool TestEmptyPassword() {
  const uint8_t kKey[] = {0xa3, 0x3d, 0xdd, 0xc3, 0x04, 0x78, 0x18,
                          0x55, 0x15, 0x31, 0x1f, 0x87, 0x52, 0x89,
                          0x5d, 0x36, 0xea, 0x43, 0x63, 0xa2};

  if (!TestPBKDF2(NULL, 0, "salt", 4, 1, EVP_sha1(), sizeof(kKey), kKey) ||
      !TestPBKDF2("", 0, "salt", 4, 1, EVP_sha1(), sizeof(kKey), kKey)) {
    return false;
  }

  return true;
}

// Tests deriving a key using an empty salt. Note that the expectation was
// generated using OpenSSL itself, and hence is not verified.
static bool TestEmptySalt() {
  const uint8_t kKey[] = {0x8b, 0xc2, 0xf9, 0x16, 0x7a, 0x81, 0xcd, 0xcf,
                          0xad, 0x12, 0x35, 0xcd, 0x90, 0x47, 0xf1, 0x13,
                          0x62, 0x71, 0xc1, 0xf9, 0x78, 0xfc, 0xfc, 0xb3,
                          0x5e, 0x22, 0xdb, 0xea, 0xfa, 0x46, 0x34, 0xf6};

  if (!TestPBKDF2("password", 8, NULL, 0, 2, EVP_sha256(), sizeof(kKey),
                  kKey) ||
      !TestPBKDF2("password", 8, "", 0, 2, EVP_sha256(), sizeof(kKey), kKey)) {
    return false;
  }

  return true;
}

// Exercises test vectors taken from https://tools.ietf.org/html/rfc6070.
// Note that each of these test vectors uses SHA-1 as the digest.
static bool TestRFC6070Vectors() {
  const uint8_t kKey1[] = {0x0c, 0x60, 0xc8, 0x0f, 0x96, 0x1f, 0x0e,
                           0x71, 0xf3, 0xa9, 0xb5, 0x24, 0xaf, 0x60,
                           0x12, 0x06, 0x2f, 0xe0, 0x37, 0xa6};
  const uint8_t kKey2[] = {0xea, 0x6c, 0x01, 0x4d, 0xc7, 0x2d, 0x6f,
                           0x8c, 0xcd, 0x1e, 0xd9, 0x2a, 0xce, 0x1d,
                           0x41, 0xf0, 0xd8, 0xde, 0x89, 0x57};
  const uint8_t kKey3[] = {0x56, 0xfa, 0x6a, 0xa7, 0x55, 0x48, 0x09, 0x9d,
                           0xcc, 0x37, 0xd7, 0xf0, 0x34, 0x25, 0xe0, 0xc3};

  if (!TestPBKDF2("password", 8, "salt", 4, 1, EVP_sha1(), sizeof(kKey1),
                  kKey1) ||
      !TestPBKDF2("password", 8, "salt", 4, 2, EVP_sha1(), sizeof(kKey2),
                  kKey2) ||
      !TestPBKDF2("pass\0word", 9, "sa\0lt", 5, 4096, EVP_sha1(),
                  sizeof(kKey3), kKey3)) {
    return false;
  }

  return true;
}

// Tests key derivation using SHA-2 digests.
static bool TestSHA2() {
  // This test was taken from:
  // http://stackoverflow.com/questions/5130513/pbkdf2-hmac-sha2-test-vectors.
  const uint8_t kKey1[] = {0xae, 0x4d, 0x0c, 0x95, 0xaf, 0x6b, 0x46, 0xd3,
                           0x2d, 0x0a, 0xdf, 0xf9, 0x28, 0xf0, 0x6d, 0xd0,
                           0x2a, 0x30, 0x3f, 0x8e, 0xf3, 0xc2, 0x51, 0xdf,
                           0xd6, 0xe2, 0xd8, 0x5a, 0x95, 0x47, 0x4c, 0x43};

  // This test was taken from:
  // http://stackoverflow.com/questions/15593184/pbkdf2-hmac-sha-512-test-vectors.
  const uint8_t kKey2[] = {
      0x8c, 0x05, 0x11, 0xf4, 0xc6, 0xe5, 0x97, 0xc6, 0xac, 0x63, 0x15,
      0xd8, 0xf0, 0x36, 0x2e, 0x22, 0x5f, 0x3c, 0x50, 0x14, 0x95, 0xba,
      0x23, 0xb8, 0x68, 0xc0, 0x05, 0x17, 0x4d, 0xc4, 0xee, 0x71, 0x11,
      0x5b, 0x59, 0xf9, 0xe6, 0x0c, 0xd9, 0x53, 0x2f, 0xa3, 0x3e, 0x0f,
      0x75, 0xae, 0xfe, 0x30, 0x22, 0x5c, 0x58, 0x3a, 0x18, 0x6c, 0xd8,
      0x2b, 0xd4, 0xda, 0xea, 0x97, 0x24, 0xa3, 0xd3, 0xb8};

  if (!TestPBKDF2("password", 8, "salt", 4, 2, EVP_sha256(), sizeof(kKey1),
                  kKey1) ||
      !TestPBKDF2("passwordPASSWORDpassword", 24,
                  "saltSALTsaltSALTsaltSALTsaltSALTsalt", 36, 4096,
                  EVP_sha512(), sizeof(kKey2), kKey2)) {
    return false;
  }

  return true;
}

int main(void) {
  CRYPTO_library_init();
  ERR_load_crypto_strings();

  if (!TestEmptyPassword()) {
    fprintf(stderr, "TestEmptyPassword failed\n");
    return 1;
  }

  if (!TestEmptySalt()) {
    fprintf(stderr, "TestEmptySalt failed\n");
    return 1;
  }

  if (!TestRFC6070Vectors()) {
    fprintf(stderr, "TestRFC6070Vectors failed\n");
    return 1;
  }

  if (!TestSHA2()) {
    fprintf(stderr, "TestSHA2 failed\n");
    return 1;
  }

  printf("PASS\n");
  ERR_free_strings();
  return 0;
}
