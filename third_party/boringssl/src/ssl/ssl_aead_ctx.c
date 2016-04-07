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

#include <openssl/ssl.h>

#include <assert.h>
#include <string.h>

#include <openssl/aead.h>
#include <openssl/err.h>
#include <openssl/rand.h>
#include <openssl/type_check.h>

#include "internal.h"


OPENSSL_COMPILE_ASSERT(EVP_AEAD_MAX_NONCE_LENGTH < 256,
                       variable_nonce_len_doesnt_fit_in_uint8_t);

SSL_AEAD_CTX *SSL_AEAD_CTX_new(enum evp_aead_direction_t direction,
                               uint16_t version, const SSL_CIPHER *cipher,
                               const uint8_t *enc_key, size_t enc_key_len,
                               const uint8_t *mac_key, size_t mac_key_len,
                               const uint8_t *fixed_iv, size_t fixed_iv_len) {
  const EVP_AEAD *aead;
  size_t discard;
  if (!ssl_cipher_get_evp_aead(&aead, &discard, &discard, cipher, version)) {
    OPENSSL_PUT_ERROR(SSL, ERR_R_INTERNAL_ERROR);
    return 0;
  }

  uint8_t merged_key[EVP_AEAD_MAX_KEY_LENGTH];
  if (mac_key_len > 0) {
    /* This is a "stateful" AEAD (for compatibility with pre-AEAD cipher
     * suites). */
    if (mac_key_len + enc_key_len + fixed_iv_len > sizeof(merged_key)) {
      OPENSSL_PUT_ERROR(SSL, ERR_R_INTERNAL_ERROR);
      return 0;
    }
    memcpy(merged_key, mac_key, mac_key_len);
    memcpy(merged_key + mac_key_len, enc_key, enc_key_len);
    memcpy(merged_key + mac_key_len + enc_key_len, fixed_iv, fixed_iv_len);
    enc_key = merged_key;
    enc_key_len += mac_key_len;
    enc_key_len += fixed_iv_len;
  }

  SSL_AEAD_CTX *aead_ctx = (SSL_AEAD_CTX *)OPENSSL_malloc(sizeof(SSL_AEAD_CTX));
  if (aead_ctx == NULL) {
    OPENSSL_PUT_ERROR(SSL, ERR_R_MALLOC_FAILURE);
    return NULL;
  }
  memset(aead_ctx, 0, sizeof(SSL_AEAD_CTX));
  aead_ctx->cipher = cipher;

  if (!EVP_AEAD_CTX_init_with_direction(
          &aead_ctx->ctx, aead, enc_key, enc_key_len,
          EVP_AEAD_DEFAULT_TAG_LENGTH, direction)) {
    OPENSSL_free(aead_ctx);
    return NULL;
  }

  assert(EVP_AEAD_nonce_length(aead) <= EVP_AEAD_MAX_NONCE_LENGTH);
  aead_ctx->variable_nonce_len = (uint8_t)EVP_AEAD_nonce_length(aead);
  if (mac_key_len == 0) {
    /* For a real AEAD, the IV is the fixed part of the nonce. */
    if (fixed_iv_len > sizeof(aead_ctx->fixed_nonce) ||
        fixed_iv_len > aead_ctx->variable_nonce_len) {
      SSL_AEAD_CTX_free(aead_ctx);
      OPENSSL_PUT_ERROR(SSL, ERR_R_INTERNAL_ERROR);
      return 0;
    }
    aead_ctx->variable_nonce_len -= fixed_iv_len;

    memcpy(aead_ctx->fixed_nonce, fixed_iv, fixed_iv_len);
    aead_ctx->fixed_nonce_len = fixed_iv_len;
    /* AES-GCM uses an explicit nonce. */
    if (cipher->algorithm_enc & (SSL_AES128GCM | SSL_AES256GCM)) {
      aead_ctx->variable_nonce_included_in_record = 1;
    }
  } else {
    aead_ctx->variable_nonce_included_in_record = 1;
    aead_ctx->random_variable_nonce = 1;
    aead_ctx->omit_length_in_ad = 1;
    aead_ctx->omit_version_in_ad = (version == SSL3_VERSION);
  }

  return aead_ctx;
}

void SSL_AEAD_CTX_free(SSL_AEAD_CTX *aead) {
  if (aead == NULL) {
    return;
  }
  EVP_AEAD_CTX_cleanup(&aead->ctx);
  OPENSSL_free(aead);
}

size_t SSL_AEAD_CTX_explicit_nonce_len(SSL_AEAD_CTX *aead) {
  if (aead != NULL && aead->variable_nonce_included_in_record) {
    return aead->variable_nonce_len;
  }
  return 0;
}

size_t SSL_AEAD_CTX_max_overhead(SSL_AEAD_CTX *aead) {
  if (aead == NULL) {
    return 0;
  }
  return EVP_AEAD_max_overhead(aead->ctx.aead) +
      SSL_AEAD_CTX_explicit_nonce_len(aead);
}

/* ssl_aead_ctx_get_ad writes the additional data for |aead| into |out| and
 * returns the number of bytes written. */
static size_t ssl_aead_ctx_get_ad(SSL_AEAD_CTX *aead, uint8_t out[13],
                                  uint8_t type, uint16_t wire_version,
                                  const uint8_t seqnum[8],
                                  size_t plaintext_len) {
  memcpy(out, seqnum, 8);
  size_t len = 8;
  out[len++] = type;
  if (!aead->omit_version_in_ad) {
    out[len++] = (uint8_t)(wire_version >> 8);
    out[len++] = (uint8_t)wire_version;
  }
  if (!aead->omit_length_in_ad) {
    out[len++] = (uint8_t)(plaintext_len >> 8);
    out[len++] = (uint8_t)plaintext_len;
  }
  return len;
}

int SSL_AEAD_CTX_open(SSL_AEAD_CTX *aead, uint8_t *out, size_t *out_len,
                      size_t max_out, uint8_t type, uint16_t wire_version,
                      const uint8_t seqnum[8], const uint8_t *in,
                      size_t in_len) {
  if (aead == NULL) {
    /* Handle the initial NULL cipher. */
    if (in_len > max_out) {
      OPENSSL_PUT_ERROR(SSL, SSL_R_BUFFER_TOO_SMALL);
      return 0;
    }
    memmove(out, in, in_len);
    *out_len = in_len;
    return 1;
  }

  /* TLS 1.2 AEADs include the length in the AD and are assumed to have fixed
   * overhead. Otherwise the parameter is unused. */
  size_t plaintext_len = 0;
  if (!aead->omit_length_in_ad) {
    size_t overhead = SSL_AEAD_CTX_max_overhead(aead);
    if (in_len < overhead) {
      /* Publicly invalid. */
      OPENSSL_PUT_ERROR(SSL, SSL_R_BAD_PACKET_LENGTH);
      return 0;
    }
    plaintext_len = in_len - overhead;
  }
  uint8_t ad[13];
  size_t ad_len = ssl_aead_ctx_get_ad(aead, ad, type, wire_version, seqnum,
                                      plaintext_len);

  /* Assemble the nonce. */
  uint8_t nonce[EVP_AEAD_MAX_NONCE_LENGTH];
  size_t nonce_len = 0;
  memcpy(nonce, aead->fixed_nonce, aead->fixed_nonce_len);
  nonce_len += aead->fixed_nonce_len;
  if (aead->variable_nonce_included_in_record) {
    if (in_len < aead->variable_nonce_len) {
      /* Publicly invalid. */
      OPENSSL_PUT_ERROR(SSL, SSL_R_BAD_PACKET_LENGTH);
      return 0;
    }
    memcpy(nonce + nonce_len, in, aead->variable_nonce_len);
    in += aead->variable_nonce_len;
    in_len -= aead->variable_nonce_len;
  } else {
    assert(aead->variable_nonce_len == 8);
    memcpy(nonce + nonce_len, seqnum, aead->variable_nonce_len);
  }
  nonce_len += aead->variable_nonce_len;

  return EVP_AEAD_CTX_open(&aead->ctx, out, out_len, max_out, nonce, nonce_len,
                           in, in_len, ad, ad_len);
}

int SSL_AEAD_CTX_seal(SSL_AEAD_CTX *aead, uint8_t *out, size_t *out_len,
                      size_t max_out, uint8_t type, uint16_t wire_version,
                      const uint8_t seqnum[8], const uint8_t *in,
                      size_t in_len) {
  if (aead == NULL) {
    /* Handle the initial NULL cipher. */
    if (in_len > max_out) {
      OPENSSL_PUT_ERROR(SSL, SSL_R_BUFFER_TOO_SMALL);
      return 0;
    }
    memmove(out, in, in_len);
    *out_len = in_len;
    return 1;
  }

  uint8_t ad[13];
  size_t ad_len = ssl_aead_ctx_get_ad(aead, ad, type, wire_version, seqnum,
                                      in_len);

  /* Assemble the nonce. */
  uint8_t nonce[EVP_AEAD_MAX_NONCE_LENGTH];
  size_t nonce_len = 0;
  memcpy(nonce, aead->fixed_nonce, aead->fixed_nonce_len);
  nonce_len += aead->fixed_nonce_len;
  if (aead->random_variable_nonce) {
    assert(aead->variable_nonce_included_in_record);
    if (!RAND_bytes(nonce + nonce_len, aead->variable_nonce_len)) {
      return 0;
    }
  } else {
    /* When sending we use the sequence number as the variable part of the
     * nonce. */
    assert(aead->variable_nonce_len == 8);
    memcpy(nonce + nonce_len, ad, aead->variable_nonce_len);
  }
  nonce_len += aead->variable_nonce_len;

  /* Emit the variable nonce if included in the record. */
  size_t extra_len = 0;
  if (aead->variable_nonce_included_in_record) {
    if (max_out < aead->variable_nonce_len) {
      OPENSSL_PUT_ERROR(SSL, SSL_R_BUFFER_TOO_SMALL);
      return 0;
    }
    if (out < in + in_len && in < out + aead->variable_nonce_len) {
      OPENSSL_PUT_ERROR(SSL, SSL_R_OUTPUT_ALIASES_INPUT);
      return 0;
    }
    memcpy(out, nonce + aead->fixed_nonce_len, aead->variable_nonce_len);
    extra_len = aead->variable_nonce_len;
    out += aead->variable_nonce_len;
    max_out -= aead->variable_nonce_len;
  }

  if (!EVP_AEAD_CTX_seal(&aead->ctx, out, out_len, max_out, nonce, nonce_len,
                         in, in_len, ad, ad_len)) {
    return 0;
  }
  *out_len += extra_len;
  return 1;
}
