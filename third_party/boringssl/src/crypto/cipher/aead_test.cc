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

#include <stdint.h>
#include <string.h>

#include <vector>

#include <openssl/aead.h>
#include <openssl/crypto.h>
#include <openssl/err.h>

#include "../test/file_test.h"
#include "../test/scoped_types.h"
#include "../test/stl_compat.h"


// This program tests an AEAD against a series of test vectors from a file,
// using the FileTest format. As an example, here's a valid test case:
//
//   KEY: 5a19f3173586b4c42f8412f4d5a786531b3231753e9e00998aec12fda8df10e4
//   NONCE: 978105dfce667bf4
//   IN: 6a4583908d
//   AD: b654574932
//   CT: 5294265a60
//   TAG: 1d45758621762e061368e68868e2f929

static bool TestAEAD(FileTest *t, void *arg) {
  const EVP_AEAD *aead = reinterpret_cast<const EVP_AEAD*>(arg);

  std::vector<uint8_t> key, nonce, in, ad, ct, tag;
  if (!t->GetBytes(&key, "KEY") ||
      !t->GetBytes(&nonce, "NONCE") ||
      !t->GetBytes(&in, "IN") ||
      !t->GetBytes(&ad, "AD") ||
      !t->GetBytes(&ct, "CT") ||
      !t->GetBytes(&tag, "TAG")) {
    return false;
  }

  ScopedEVP_AEAD_CTX ctx;
  if (!EVP_AEAD_CTX_init_with_direction(ctx.get(), aead,
                                        bssl::vector_data(&key), key.size(),
                                        tag.size(), evp_aead_seal)) {
    t->PrintLine("Failed to init AEAD.");
    return false;
  }

  std::vector<uint8_t> out(in.size() + EVP_AEAD_max_overhead(aead));
  if (!t->HasAttribute("NO_SEAL")) {
    size_t out_len;
    if (!EVP_AEAD_CTX_seal(ctx.get(), bssl::vector_data(&out), &out_len,
                           out.size(), bssl::vector_data(&nonce), nonce.size(),
                           bssl::vector_data(&in), in.size(),
                           bssl::vector_data(&ad), ad.size())) {
      t->PrintLine("Failed to run AEAD.");
      return false;
    }
    out.resize(out_len);

    if (out.size() != ct.size() + tag.size()) {
      t->PrintLine("Bad output length: %u vs %u.", (unsigned)out_len,
                   (unsigned)(ct.size() + tag.size()));
      return false;
    }
    if (!t->ExpectBytesEqual(bssl::vector_data(&ct), ct.size(),
                             bssl::vector_data(&out), ct.size()) ||
        !t->ExpectBytesEqual(bssl::vector_data(&tag), tag.size(),
                             bssl::vector_data(&out) + ct.size(), tag.size())) {
      return false;
    }
  } else {
    out.resize(ct.size() + tag.size());
    memcpy(bssl::vector_data(&out), bssl::vector_data(&ct), ct.size());
    memcpy(bssl::vector_data(&out) + ct.size(), bssl::vector_data(&tag),
           tag.size());
  }

  // The "stateful" AEADs for implementing pre-AEAD cipher suites need to be
  // reset after each operation.
  ctx.Reset();
  if (!EVP_AEAD_CTX_init_with_direction(ctx.get(), aead,
                                        bssl::vector_data(&key), key.size(),
                                        tag.size(), evp_aead_open)) {
    t->PrintLine("Failed to init AEAD.");
    return false;
  }

  std::vector<uint8_t> out2(out.size());
  size_t out2_len;
  int ret = EVP_AEAD_CTX_open(ctx.get(),
                              bssl::vector_data(&out2), &out2_len, out2.size(),
                              bssl::vector_data(&nonce), nonce.size(),
                              bssl::vector_data(&out), out.size(),
                              bssl::vector_data(&ad), ad.size());
  if (t->HasAttribute("FAILS")) {
    if (ret) {
      t->PrintLine("Decrypted bad data.");
      return false;
    }
    ERR_clear_error();
    return true;
  }

  if (!ret) {
    t->PrintLine("Failed to decrypt.");
    return false;
  }
  out2.resize(out2_len);
  if (!t->ExpectBytesEqual(bssl::vector_data(&in), in.size(),
                           bssl::vector_data(&out2), out2.size())) {
    return false;
  }

  // The "stateful" AEADs for implementing pre-AEAD cipher suites need to be
  // reset after each operation.
  ctx.Reset();
  if (!EVP_AEAD_CTX_init_with_direction(ctx.get(), aead,
                                        bssl::vector_data(&key), key.size(),
                                        tag.size(), evp_aead_open)) {
    t->PrintLine("Failed to init AEAD.");
    return false;
  }

  // Garbage at the end isn't ignored.
  out.push_back(0);
  out2.resize(out.size());
  if (EVP_AEAD_CTX_open(ctx.get(), bssl::vector_data(&out2), &out2_len,
                        out2.size(), bssl::vector_data(&nonce), nonce.size(),
                        bssl::vector_data(&out), out.size(),
                        bssl::vector_data(&ad), ad.size())) {
    t->PrintLine("Decrypted bad data with trailing garbage.");
    return false;
  }
  ERR_clear_error();

  // The "stateful" AEADs for implementing pre-AEAD cipher suites need to be
  // reset after each operation.
  ctx.Reset();
  if (!EVP_AEAD_CTX_init_with_direction(ctx.get(), aead,
                                        bssl::vector_data(&key), key.size(),
                                        tag.size(), evp_aead_open)) {
    t->PrintLine("Failed to init AEAD.");
    return false;
  }

  // Verify integrity is checked.
  out[0] ^= 0x80;
  out.resize(out.size() - 1);
  out2.resize(out.size());
  if (EVP_AEAD_CTX_open(ctx.get(), bssl::vector_data(&out2), &out2_len,
                        out2.size(), bssl::vector_data(&nonce), nonce.size(),
                        bssl::vector_data(&out), out.size(),
                        bssl::vector_data(&ad), ad.size())) {
    t->PrintLine("Decrypted bad data with corrupted byte.");
    return false;
  }
  ERR_clear_error();

  return true;
}

static int TestCleanupAfterInitFailure(const EVP_AEAD *aead) {
  EVP_AEAD_CTX ctx;
  uint8_t key[128];

  memset(key, 0, sizeof(key));
  const size_t key_len = EVP_AEAD_key_length(aead);
  if (key_len > sizeof(key)) {
    fprintf(stderr, "Key length of AEAD too long.\n");
    return 0;
  }

  if (EVP_AEAD_CTX_init(&ctx, aead, key, key_len,
                        9999 /* a silly tag length to trigger an error */,
                        NULL /* ENGINE */) != 0) {
    fprintf(stderr, "A silly tag length didn't trigger an error!\n");
    return 0;
  }
  ERR_clear_error();

  /* Running a second, failed _init should not cause a memory leak. */
  if (EVP_AEAD_CTX_init(&ctx, aead, key, key_len,
                        9999 /* a silly tag length to trigger an error */,
                        NULL /* ENGINE */) != 0) {
    fprintf(stderr, "A silly tag length didn't trigger an error!\n");
    return 0;
  }
  ERR_clear_error();

  /* Calling _cleanup on an |EVP_AEAD_CTX| after a failed _init should be a
   * no-op. */
  EVP_AEAD_CTX_cleanup(&ctx);
  return 1;
}

struct AEADName {
  const char name[40];
  const EVP_AEAD *(*func)(void);
};

static const struct AEADName kAEADs[] = {
  { "aes-128-gcm", EVP_aead_aes_128_gcm },
  { "aes-256-gcm", EVP_aead_aes_256_gcm },
  { "chacha20-poly1305", EVP_aead_chacha20_poly1305 },
  { "rc4-md5-tls", EVP_aead_rc4_md5_tls },
  { "rc4-sha1-tls", EVP_aead_rc4_sha1_tls },
  { "aes-128-cbc-sha1-tls", EVP_aead_aes_128_cbc_sha1_tls },
  { "aes-128-cbc-sha1-tls-implicit-iv", EVP_aead_aes_128_cbc_sha1_tls_implicit_iv },
  { "aes-128-cbc-sha256-tls", EVP_aead_aes_128_cbc_sha256_tls },
  { "aes-256-cbc-sha1-tls", EVP_aead_aes_256_cbc_sha1_tls },
  { "aes-256-cbc-sha1-tls-implicit-iv", EVP_aead_aes_256_cbc_sha1_tls_implicit_iv },
  { "aes-256-cbc-sha256-tls", EVP_aead_aes_256_cbc_sha256_tls },
  { "aes-256-cbc-sha384-tls", EVP_aead_aes_256_cbc_sha384_tls },
  { "des-ede3-cbc-sha1-tls", EVP_aead_des_ede3_cbc_sha1_tls },
  { "des-ede3-cbc-sha1-tls-implicit-iv", EVP_aead_des_ede3_cbc_sha1_tls_implicit_iv },
  { "rc4-md5-ssl3", EVP_aead_rc4_md5_ssl3 },
  { "rc4-sha1-ssl3", EVP_aead_rc4_sha1_ssl3 },
  { "aes-128-cbc-sha1-ssl3", EVP_aead_aes_128_cbc_sha1_ssl3 },
  { "aes-256-cbc-sha1-ssl3", EVP_aead_aes_256_cbc_sha1_ssl3 },
  { "des-ede3-cbc-sha1-ssl3", EVP_aead_des_ede3_cbc_sha1_ssl3 },
  { "aes-128-key-wrap", EVP_aead_aes_128_key_wrap },
  { "aes-256-key-wrap", EVP_aead_aes_256_key_wrap },
  { "aes-128-ctr-hmac-sha256", EVP_aead_aes_128_ctr_hmac_sha256 },
  { "aes-256-ctr-hmac-sha256", EVP_aead_aes_256_ctr_hmac_sha256 },
  { "", NULL },
};

int main(int argc, char **argv) {
  CRYPTO_library_init();

  if (argc != 3) {
    fprintf(stderr, "%s <aead> <test file.txt>\n", argv[0]);
    return 1;
  }

  const EVP_AEAD *aead;
  for (unsigned i = 0;; i++) {
    const struct AEADName &aead_name = kAEADs[i];
    if (aead_name.func == NULL) {
      fprintf(stderr, "Unknown AEAD: %s\n", argv[1]);
      return 2;
    }
    if (strcmp(aead_name.name, argv[1]) == 0) {
      aead = aead_name.func();
      break;
    }
  }

  if (!TestCleanupAfterInitFailure(aead)) {
    return 1;
  }

  return FileTestMain(TestAEAD, const_cast<EVP_AEAD*>(aead), argv[2]);
}
