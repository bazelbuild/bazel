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

#include <string>
#include <functional>
#include <memory>
#include <vector>

#include <stdint.h>
#include <string.h>

#include <openssl/aead.h>
#include <openssl/digest.h>
#include <openssl/err.h>
#include <openssl/obj.h>
#include <openssl/rand.h>
#include <openssl/rsa.h>

#if defined(OPENSSL_WINDOWS)
#pragma warning(push, 3)
#include <windows.h>
#pragma warning(pop)
#elif defined(OPENSSL_APPLE)
#include <sys/time.h>
#endif

#include "../crypto/test/scoped_types.h"
#include "internal.h"


// TimeResults represents the results of benchmarking a function.
struct TimeResults {
  // num_calls is the number of function calls done in the time period.
  unsigned num_calls;
  // us is the number of microseconds that elapsed in the time period.
  unsigned us;

  void Print(const std::string &description) {
    printf("Did %u %s operations in %uus (%.1f ops/sec)\n", num_calls,
           description.c_str(), us,
           (static_cast<double>(num_calls) / us) * 1000000);
  }

  void PrintWithBytes(const std::string &description, size_t bytes_per_call) {
    printf("Did %u %s operations in %uus (%.1f ops/sec): %.1f MB/s\n",
           num_calls, description.c_str(), us,
           (static_cast<double>(num_calls) / us) * 1000000,
           static_cast<double>(bytes_per_call * num_calls) / us);
  }
};

#if defined(OPENSSL_WINDOWS)
static uint64_t time_now() { return GetTickCount64() * 1000; }
#elif defined(OPENSSL_APPLE)
static uint64_t time_now() {
  struct timeval tv;
  uint64_t ret;

  gettimeofday(&tv, NULL);
  ret = tv.tv_sec;
  ret *= 1000000;
  ret += tv.tv_usec;
  return ret;
}
#else
static uint64_t time_now() {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);

  uint64_t ret = ts.tv_sec;
  ret *= 1000000;
  ret += ts.tv_nsec / 1000;
  return ret;
}
#endif

static bool TimeFunction(TimeResults *results, std::function<bool()> func) {
  // kTotalMS is the total amount of time that we'll aim to measure a function
  // for.
  static const uint64_t kTotalUS = 1000000;
  uint64_t start = time_now(), now, delta;
  unsigned done = 0, iterations_between_time_checks;

  if (!func()) {
    return false;
  }
  now = time_now();
  delta = now - start;
  if (delta == 0) {
    iterations_between_time_checks = 250;
  } else {
    // Aim for about 100ms between time checks.
    iterations_between_time_checks =
        static_cast<double>(100000) / static_cast<double>(delta);
    if (iterations_between_time_checks > 1000) {
      iterations_between_time_checks = 1000;
    } else if (iterations_between_time_checks < 1) {
      iterations_between_time_checks = 1;
    }
  }

  for (;;) {
    for (unsigned i = 0; i < iterations_between_time_checks; i++) {
      if (!func()) {
        return false;
      }
      done++;
    }

    now = time_now();
    if (now - start > kTotalUS) {
      break;
    }
  }

  results->us = now - start;
  results->num_calls = done;
  return true;
}

static bool SpeedRSA(const std::string &key_name, RSA *key,
                     const std::string &selected) {
  if (!selected.empty() && key_name.find(selected) == std::string::npos) {
    return true;
  }

  std::unique_ptr<uint8_t[]> sig(new uint8_t[RSA_size(key)]);
  const uint8_t fake_sha256_hash[32] = {0};
  unsigned sig_len;

  TimeResults results;
  if (!TimeFunction(&results,
                    [key, &sig, &fake_sha256_hash, &sig_len]() -> bool {
        return RSA_sign(NID_sha256, fake_sha256_hash, sizeof(fake_sha256_hash),
                        sig.get(), &sig_len, key);
      })) {
    fprintf(stderr, "RSA_sign failed.\n");
    ERR_print_errors_fp(stderr);
    return false;
  }
  results.Print(key_name + " signing");

  if (!TimeFunction(&results,
                    [key, &fake_sha256_hash, &sig, sig_len]() -> bool {
        return RSA_verify(NID_sha256, fake_sha256_hash,
                          sizeof(fake_sha256_hash), sig.get(), sig_len, key);
      })) {
    fprintf(stderr, "RSA_verify failed.\n");
    ERR_print_errors_fp(stderr);
    return false;
  }
  results.Print(key_name + " verify");

  return true;
}

static uint8_t *align(uint8_t *in, unsigned alignment) {
  return reinterpret_cast<uint8_t *>(
      (reinterpret_cast<uintptr_t>(in) + alignment) &
      ~static_cast<size_t>(alignment - 1));
}

static bool SpeedAEADChunk(const EVP_AEAD *aead, const std::string &name,
                           size_t chunk_len, size_t ad_len) {
  static const unsigned kAlignment = 16;

  EVP_AEAD_CTX ctx;
  const size_t key_len = EVP_AEAD_key_length(aead);
  const size_t nonce_len = EVP_AEAD_nonce_length(aead);
  const size_t overhead_len = EVP_AEAD_max_overhead(aead);

  std::unique_ptr<uint8_t[]> key(new uint8_t[key_len]);
  memset(key.get(), 0, key_len);
  std::unique_ptr<uint8_t[]> nonce(new uint8_t[nonce_len]);
  memset(nonce.get(), 0, nonce_len);
  std::unique_ptr<uint8_t[]> in_storage(new uint8_t[chunk_len + kAlignment]);
  std::unique_ptr<uint8_t[]> out_storage(new uint8_t[chunk_len + overhead_len + kAlignment]);
  std::unique_ptr<uint8_t[]> ad(new uint8_t[ad_len]);
  memset(ad.get(), 0, ad_len);

  uint8_t *const in = align(in_storage.get(), kAlignment);
  memset(in, 0, chunk_len);
  uint8_t *const out = align(out_storage.get(), kAlignment);
  memset(out, 0, chunk_len + overhead_len);

  if (!EVP_AEAD_CTX_init_with_direction(&ctx, aead, key.get(), key_len,
                                        EVP_AEAD_DEFAULT_TAG_LENGTH,
                                        evp_aead_seal)) {
    fprintf(stderr, "Failed to create EVP_AEAD_CTX.\n");
    ERR_print_errors_fp(stderr);
    return false;
  }

  TimeResults results;
  if (!TimeFunction(&results, [chunk_len, overhead_len, nonce_len, ad_len, in,
                               out, &ctx, &nonce, &ad]() -> bool {
        size_t out_len;

        return EVP_AEAD_CTX_seal(
            &ctx, out, &out_len, chunk_len + overhead_len, nonce.get(),
            nonce_len, in, chunk_len, ad.get(), ad_len);
      })) {
    fprintf(stderr, "EVP_AEAD_CTX_seal failed.\n");
    ERR_print_errors_fp(stderr);
    return false;
  }

  results.PrintWithBytes(name + " seal", chunk_len);

  EVP_AEAD_CTX_cleanup(&ctx);

  return true;
}

static bool SpeedAEAD(const EVP_AEAD *aead, const std::string &name,
                      size_t ad_len, const std::string &selected) {
  if (!selected.empty() && name.find(selected) == std::string::npos) {
    return true;
  }

  return SpeedAEADChunk(aead, name + " (16 bytes)", 16, ad_len) &&
         SpeedAEADChunk(aead, name + " (1350 bytes)", 1350, ad_len) &&
         SpeedAEADChunk(aead, name + " (8192 bytes)", 8192, ad_len);
}

static bool SpeedHashChunk(const EVP_MD *md, const std::string &name,
                           size_t chunk_len) {
  EVP_MD_CTX *ctx = EVP_MD_CTX_create();
  uint8_t scratch[8192];

  if (chunk_len > sizeof(scratch)) {
    return false;
  }

  TimeResults results;
  if (!TimeFunction(&results, [ctx, md, chunk_len, &scratch]() -> bool {
        uint8_t digest[EVP_MAX_MD_SIZE];
        unsigned int md_len;

        return EVP_DigestInit_ex(ctx, md, NULL /* ENGINE */) &&
               EVP_DigestUpdate(ctx, scratch, chunk_len) &&
               EVP_DigestFinal_ex(ctx, digest, &md_len);
      })) {
    fprintf(stderr, "EVP_DigestInit_ex failed.\n");
    ERR_print_errors_fp(stderr);
    return false;
  }

  results.PrintWithBytes(name, chunk_len);

  EVP_MD_CTX_destroy(ctx);

  return true;
}
static bool SpeedHash(const EVP_MD *md, const std::string &name,
                      const std::string &selected) {
  if (!selected.empty() && name.find(selected) == std::string::npos) {
    return true;
  }

  return SpeedHashChunk(md, name + " (16 bytes)", 16) &&
         SpeedHashChunk(md, name + " (256 bytes)", 256) &&
         SpeedHashChunk(md, name + " (8192 bytes)", 8192);
}

static bool SpeedRandomChunk(const std::string name, size_t chunk_len) {
  uint8_t scratch[8192];

  if (chunk_len > sizeof(scratch)) {
    return false;
  }

  TimeResults results;
  if (!TimeFunction(&results, [chunk_len, &scratch]() -> bool {
        RAND_bytes(scratch, chunk_len);
        return true;
      })) {
    return false;
  }

  results.PrintWithBytes(name, chunk_len);
  return true;
}

static bool SpeedRandom(const std::string &selected) {
  if (!selected.empty() && selected != "RNG") {
    return true;
  }

  return SpeedRandomChunk("RNG (16 bytes)", 16) &&
         SpeedRandomChunk("RNG (256 bytes)", 256) &&
         SpeedRandomChunk("RNG (8192 bytes)", 8192);
}

static bool SpeedECDHCurve(const std::string &name, int nid,
                           const std::string &selected) {
  if (!selected.empty() && name.find(selected) == std::string::npos) {
    return true;
  }

  TimeResults results;
  if (!TimeFunction(&results, [nid]() -> bool {
        ScopedEC_KEY key(EC_KEY_new_by_curve_name(nid));
        if (!key ||
            !EC_KEY_generate_key(key.get())) {
          return false;
        }
        const EC_GROUP *const group = EC_KEY_get0_group(key.get());
        ScopedEC_POINT point(EC_POINT_new(group));
        ScopedBN_CTX ctx(BN_CTX_new());

        ScopedBIGNUM x(BN_new());
        ScopedBIGNUM y(BN_new());

        if (!point || !ctx || !x || !y ||
            !EC_POINT_mul(group, point.get(), NULL,
                          EC_KEY_get0_public_key(key.get()),
                          EC_KEY_get0_private_key(key.get()), ctx.get()) ||
            !EC_POINT_get_affine_coordinates_GFp(group, point.get(), x.get(),
                                                 y.get(), ctx.get())) {
          return false;
        }

        return true;
      })) {
    return false;
  }

  results.Print(name);
  return true;
}

static bool SpeedECDSACurve(const std::string &name, int nid,
                            const std::string &selected) {
  if (!selected.empty() && name.find(selected) == std::string::npos) {
    return true;
  }

  ScopedEC_KEY key(EC_KEY_new_by_curve_name(nid));
  if (!key ||
      !EC_KEY_generate_key(key.get())) {
    return false;
  }

  uint8_t signature[256];
  if (ECDSA_size(key.get()) > sizeof(signature)) {
    return false;
  }
  uint8_t digest[20];
  memset(digest, 42, sizeof(digest));
  unsigned sig_len;

  TimeResults results;
  if (!TimeFunction(&results, [&key, &signature, &digest, &sig_len]() -> bool {
        return ECDSA_sign(0, digest, sizeof(digest), signature, &sig_len,
                          key.get()) == 1;
      })) {
    return false;
  }

  results.Print(name + " signing");

  if (!TimeFunction(&results, [&key, &signature, &digest, sig_len]() -> bool {
        return ECDSA_verify(0, digest, sizeof(digest), signature, sig_len,
                            key.get()) == 1;
      })) {
    return false;
  }

  results.Print(name + " verify");

  return true;
}

static bool SpeedECDH(const std::string &selected) {
  return SpeedECDHCurve("ECDH P-224", NID_secp224r1, selected) &&
         SpeedECDHCurve("ECDH P-256", NID_X9_62_prime256v1, selected) &&
         SpeedECDHCurve("ECDH P-384", NID_secp384r1, selected) &&
         SpeedECDHCurve("ECDH P-521", NID_secp521r1, selected);
}

static bool SpeedECDSA(const std::string &selected) {
  return SpeedECDSACurve("ECDSA P-224", NID_secp224r1, selected) &&
         SpeedECDSACurve("ECDSA P-256", NID_X9_62_prime256v1, selected) &&
         SpeedECDSACurve("ECDSA P-384", NID_secp384r1, selected) &&
         SpeedECDSACurve("ECDSA P-521", NID_secp521r1, selected);
}

bool Speed(const std::vector<std::string> &args) {
  std::string selected;
  if (args.size() > 1) {
    fprintf(stderr, "Usage: bssl speed [speed test selector, i.e. 'RNG']\n");
    return false;
  }
  if (args.size() > 0) {
    selected = args[0];
  }

  RSA *key = RSA_private_key_from_bytes(kDERRSAPrivate2048,
                                        kDERRSAPrivate2048Len);
  if (key == NULL) {
    fprintf(stderr, "Failed to parse RSA key.\n");
    ERR_print_errors_fp(stderr);
    return false;
  }

  if (!SpeedRSA("RSA 2048", key, selected)) {
    return false;
  }

  RSA_free(key);
  key = RSA_private_key_from_bytes(kDERRSAPrivate3Prime2048,
                                   kDERRSAPrivate3Prime2048Len);
  if (key == NULL) {
    fprintf(stderr, "Failed to parse RSA key.\n");
    ERR_print_errors_fp(stderr);
    return false;
  }

  if (!SpeedRSA("RSA 2048 (3 prime, e=3)", key, selected)) {
    return false;
  }

  RSA_free(key);
  key = RSA_private_key_from_bytes(kDERRSAPrivate4096,
                                   kDERRSAPrivate4096Len);
  if (key == NULL) {
    fprintf(stderr, "Failed to parse 4096-bit RSA key.\n");
    ERR_print_errors_fp(stderr);
    return 1;
  }

  if (!SpeedRSA("RSA 4096", key, selected)) {
    return false;
  }

  RSA_free(key);

  // kTLSADLen is the number of bytes of additional data that TLS passes to
  // AEADs.
  static const size_t kTLSADLen = 13;
  // kLegacyADLen is the number of bytes that TLS passes to the "legacy" AEADs.
  // These are AEADs that weren't originally defined as AEADs, but which we use
  // via the AEAD interface. In order for that to work, they have some TLS
  // knowledge in them and construct a couple of the AD bytes internally.
  static const size_t kLegacyADLen = kTLSADLen - 2;

  if (!SpeedAEAD(EVP_aead_aes_128_gcm(), "AES-128-GCM", kTLSADLen, selected) ||
      !SpeedAEAD(EVP_aead_aes_256_gcm(), "AES-256-GCM", kTLSADLen, selected) ||
      !SpeedAEAD(EVP_aead_chacha20_poly1305(), "ChaCha20-Poly1305", kTLSADLen,
                 selected) ||
      !SpeedAEAD(EVP_aead_rc4_md5_tls(), "RC4-MD5", kLegacyADLen, selected) ||
      !SpeedAEAD(EVP_aead_aes_128_cbc_sha1_tls(), "AES-128-CBC-SHA1",
                 kLegacyADLen, selected) ||
      !SpeedAEAD(EVP_aead_aes_256_cbc_sha1_tls(), "AES-256-CBC-SHA1",
                 kLegacyADLen, selected) ||
      !SpeedHash(EVP_sha1(), "SHA-1", selected) ||
      !SpeedHash(EVP_sha256(), "SHA-256", selected) ||
      !SpeedHash(EVP_sha512(), "SHA-512", selected) ||
      !SpeedRandom(selected) ||
      !SpeedECDH(selected) ||
      !SpeedECDSA(selected)) {
    return false;
  }

  return true;
}
