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

#include <vector>

#include <openssl/crypto.h>
#include <openssl/ec_key.h>
#include <openssl/err.h>
#include <openssl/mem.h>

#include "../test/scoped_types.h"
#include "../test/stl_compat.h"


// kECKeyWithoutPublic is an ECPrivateKey with the optional publicKey field
// omitted.
static const uint8_t kECKeyWithoutPublic[] = {
  0x30, 0x31, 0x02, 0x01, 0x01, 0x04, 0x20, 0xc6, 0xc1, 0xaa, 0xda, 0x15, 0xb0,
  0x76, 0x61, 0xf8, 0x14, 0x2c, 0x6c, 0xaf, 0x0f, 0xdb, 0x24, 0x1a, 0xff, 0x2e,
  0xfe, 0x46, 0xc0, 0x93, 0x8b, 0x74, 0xf2, 0xbc, 0xc5, 0x30, 0x52, 0xb0, 0x77,
  0xa0, 0x0a, 0x06, 0x08, 0x2a, 0x86, 0x48, 0xce, 0x3d, 0x03, 0x01, 0x07,
};

// kECKeyMissingZeros is an ECPrivateKey containing a degenerate P-256 key where
// the private key is one. The private key is incorrectly encoded without zero
// padding.
static const uint8_t kECKeyMissingZeros[] = {
  0x30, 0x58, 0x02, 0x01, 0x01, 0x04, 0x01, 0x01, 0xa0, 0x0a, 0x06, 0x08, 0x2a,
  0x86, 0x48, 0xce, 0x3d, 0x03, 0x01, 0x07, 0xa1, 0x44, 0x03, 0x42, 0x00, 0x04,
  0x6b, 0x17, 0xd1, 0xf2, 0xe1, 0x2c, 0x42, 0x47, 0xf8, 0xbc, 0xe6, 0xe5, 0x63,
  0xa4, 0x40, 0xf2, 0x77, 0x03, 0x7d, 0x81, 0x2d, 0xeb, 0x33, 0xa0, 0xf4, 0xa1,
  0x39, 0x45, 0xd8, 0x98, 0xc2, 0x96, 0x4f, 0xe3, 0x42, 0xe2, 0xfe, 0x1a, 0x7f,
  0x9b, 0x8e, 0xe7, 0xeb, 0x4a, 0x7c, 0x0f, 0x9e, 0x16, 0x2b, 0xce, 0x33, 0x57,
  0x6b, 0x31, 0x5e, 0xce, 0xcb, 0xb6, 0x40, 0x68, 0x37, 0xbf, 0x51, 0xf5,
};

// kECKeyMissingZeros is an ECPrivateKey containing a degenerate P-256 key where
// the private key is one. The private key is encoded with the required zero
// padding.
static const uint8_t kECKeyWithZeros[] = {
  0x30, 0x77, 0x02, 0x01, 0x01, 0x04, 0x20, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01,
  0xa0, 0x0a, 0x06, 0x08, 0x2a, 0x86, 0x48, 0xce, 0x3d, 0x03, 0x01, 0x07, 0xa1,
  0x44, 0x03, 0x42, 0x00, 0x04, 0x6b, 0x17, 0xd1, 0xf2, 0xe1, 0x2c, 0x42, 0x47,
  0xf8, 0xbc, 0xe6, 0xe5, 0x63, 0xa4, 0x40, 0xf2, 0x77, 0x03, 0x7d, 0x81, 0x2d,
  0xeb, 0x33, 0xa0, 0xf4, 0xa1, 0x39, 0x45, 0xd8, 0x98, 0xc2, 0x96, 0x4f, 0xe3,
  0x42, 0xe2, 0xfe, 0x1a, 0x7f, 0x9b, 0x8e, 0xe7, 0xeb, 0x4a, 0x7c, 0x0f, 0x9e,
  0x16, 0x2b, 0xce, 0x33, 0x57, 0x6b, 0x31, 0x5e, 0xce, 0xcb, 0xb6, 0x40, 0x68,
  0x37, 0xbf, 0x51, 0xf5,
};

// DecodeECPrivateKey decodes |in| as an ECPrivateKey structure and returns the
// result or nullptr on error.
static ScopedEC_KEY DecodeECPrivateKey(const uint8_t *in, size_t in_len) {
  const uint8_t *inp = in;
  ScopedEC_KEY ret(d2i_ECPrivateKey(NULL, &inp, in_len));
  if (!ret || inp != in + in_len) {
    return nullptr;
  }
  return ret;
}

// EncodeECPrivateKey encodes |key| as an ECPrivateKey structure into |*out|. It
// returns true on success or false on error.
static bool EncodeECPrivateKey(std::vector<uint8_t> *out, EC_KEY *key) {
  int len = i2d_ECPrivateKey(key, NULL);
  out->resize(len);
  uint8_t *outp = bssl::vector_data(out);
  return i2d_ECPrivateKey(key, &outp) == len;
}

bool Testd2i_ECPrivateKey() {
  ScopedEC_KEY key = DecodeECPrivateKey(kECKeyWithoutPublic,
                                        sizeof(kECKeyWithoutPublic));
  if (!key) {
    fprintf(stderr, "Failed to parse private key.\n");
    ERR_print_errors_fp(stderr);
    return false;
  }

  std::vector<uint8_t> out;
  if (!EncodeECPrivateKey(&out, key.get())) {
    fprintf(stderr, "Failed to serialize private key.\n");
    ERR_print_errors_fp(stderr);
    return false;
  }

  if (std::vector<uint8_t>(kECKeyWithoutPublic,
                           kECKeyWithoutPublic + sizeof(kECKeyWithoutPublic)) !=
      out) {
    fprintf(stderr, "Serialisation of key doesn't match original.\n");
    return false;
  }

  const EC_POINT *pub_key = EC_KEY_get0_public_key(key.get());
  if (pub_key == NULL) {
    fprintf(stderr, "Public key missing.\n");
    return false;
  }

  ScopedBIGNUM x(BN_new());
  ScopedBIGNUM y(BN_new());
  if (!x || !y) {
    return false;
  }
  if (!EC_POINT_get_affine_coordinates_GFp(EC_KEY_get0_group(key.get()),
                                           pub_key, x.get(), y.get(), NULL)) {
    fprintf(stderr, "Failed to get public key in affine coordinates.\n");
    return false;
  }
  ScopedOpenSSLString x_hex(BN_bn2hex(x.get()));
  ScopedOpenSSLString y_hex(BN_bn2hex(y.get()));
  if (!x_hex || !y_hex) {
    return false;
  }
  if (0 != strcmp(
          x_hex.get(),
          "c81561ecf2e54edefe6617db1c7a34a70744ddb261f269b83dacfcd2ade5a681") ||
      0 != strcmp(
          y_hex.get(),
          "e0e2afa3f9b6abe4c698ef6495f1be49a3196c5056acb3763fe4507eec596e88")) {
    fprintf(stderr, "Incorrect public key: %s %s\n", x_hex.get(), y_hex.get());
    return false;
  }

  return true;
}

static bool TestZeroPadding() {
  // Check that the correct encoding round-trips.
  ScopedEC_KEY key = DecodeECPrivateKey(kECKeyWithZeros,
                                        sizeof(kECKeyWithZeros));
  std::vector<uint8_t> out;
  if (!key || !EncodeECPrivateKey(&out, key.get())) {
    ERR_print_errors_fp(stderr);
    return false;
  }

  if (std::vector<uint8_t>(kECKeyWithZeros,
                           kECKeyWithZeros + sizeof(kECKeyWithZeros)) != out) {
    fprintf(stderr, "Serialisation of key was incorrect.\n");
    return false;
  }

  // Keys without leading zeros also parse, but they encode correctly.
  key = DecodeECPrivateKey(kECKeyMissingZeros, sizeof(kECKeyMissingZeros));
  if (!key || !EncodeECPrivateKey(&out, key.get())) {
    ERR_print_errors_fp(stderr);
    return false;
  }

  if (std::vector<uint8_t>(kECKeyWithZeros,
                           kECKeyWithZeros + sizeof(kECKeyWithZeros)) != out) {
    fprintf(stderr, "Serialisation of key was incorrect.\n");
    return false;
  }

  return true;
}

int main(void) {
  CRYPTO_library_init();
  ERR_load_crypto_strings();

  if (!Testd2i_ECPrivateKey() ||
      !TestZeroPadding()) {
    fprintf(stderr, "failed\n");
    return 1;
  }

  printf("PASS\n");
  return 0;
}
