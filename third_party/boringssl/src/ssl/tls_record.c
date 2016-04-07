/* Copyright (C) 1995-1998 Eric Young (eay@cryptsoft.com)
 * All rights reserved.
 *
 * This package is an SSL implementation written
 * by Eric Young (eay@cryptsoft.com).
 * The implementation was written so as to conform with Netscapes SSL.
 *
 * This library is free for commercial and non-commercial use as long as
 * the following conditions are aheared to.  The following conditions
 * apply to all code found in this distribution, be it the RC4, RSA,
 * lhash, DES, etc., code; not just the SSL code.  The SSL documentation
 * included with this distribution is covered by the same copyright terms
 * except that the holder is Tim Hudson (tjh@cryptsoft.com).
 *
 * Copyright remains Eric Young's, and as such any Copyright notices in
 * the code are not to be removed.
 * If this package is used in a product, Eric Young should be given attribution
 * as the author of the parts of the library used.
 * This can be in the form of a textual message at program startup or
 * in documentation (online or textual) provided with the package.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 * 1. Redistributions of source code must retain the copyright
 *    notice, this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 * 3. All advertising materials mentioning features or use of this software
 *    must display the following acknowledgement:
 *    "This product includes cryptographic software written by
 *     Eric Young (eay@cryptsoft.com)"
 *    The word 'cryptographic' can be left out if the rouines from the library
 *    being used are not cryptographic related :-).
 * 4. If you include any Windows specific code (or a derivative thereof) from
 *    the apps directory (application code) you must include an acknowledgement:
 *    "This product includes software written by Tim Hudson (tjh@cryptsoft.com)"
 *
 * THIS SOFTWARE IS PROVIDED BY ERIC YOUNG ``AS IS'' AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED.  IN NO EVENT SHALL THE AUTHOR OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
 * OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
 * HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 * LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
 * OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
 * SUCH DAMAGE.
 *
 * The licence and distribution terms for any publically available version or
 * derivative of this code cannot be changed.  i.e. this code cannot simply be
 * copied and put under another distribution licence
 * [including the GNU Public Licence.]
 */
/* ====================================================================
 * Copyright (c) 1998-2002 The OpenSSL Project.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in
 *    the documentation and/or other materials provided with the
 *    distribution.
 *
 * 3. All advertising materials mentioning features or use of this
 *    software must display the following acknowledgment:
 *    "This product includes software developed by the OpenSSL Project
 *    for use in the OpenSSL Toolkit. (http://www.openssl.org/)"
 *
 * 4. The names "OpenSSL Toolkit" and "OpenSSL Project" must not be used to
 *    endorse or promote products derived from this software without
 *    prior written permission. For written permission, please contact
 *    openssl-core@openssl.org.
 *
 * 5. Products derived from this software may not be called "OpenSSL"
 *    nor may "OpenSSL" appear in their names without prior written
 *    permission of the OpenSSL Project.
 *
 * 6. Redistributions of any form whatsoever must retain the following
 *    acknowledgment:
 *    "This product includes software developed by the OpenSSL Project
 *    for use in the OpenSSL Toolkit (http://www.openssl.org/)"
 *
 * THIS SOFTWARE IS PROVIDED BY THE OpenSSL PROJECT ``AS IS'' AND ANY
 * EXPRESSED OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE OpenSSL PROJECT OR
 * ITS CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 * SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT
 * NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
 * HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
 * STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED
 * OF THE POSSIBILITY OF SUCH DAMAGE.
 * ====================================================================
 *
 * This product includes cryptographic software written by Eric Young
 * (eay@cryptsoft.com).  This product includes software written by Tim
 * Hudson (tjh@cryptsoft.com). */

#include <openssl/ssl.h>

#include <assert.h>

#include <openssl/bytestring.h>
#include <openssl/err.h>

#include "internal.h"


/* kMaxEmptyRecords is the number of consecutive, empty records that will be
 * processed. Without this limit an attacker could send empty records at a
 * faster rate than we can process and cause record processing to loop
 * forever. */
static const uint8_t kMaxEmptyRecords = 32;

size_t ssl_record_prefix_len(const SSL *ssl) {
  if (SSL_IS_DTLS(ssl)) {
    return DTLS1_RT_HEADER_LENGTH +
           SSL_AEAD_CTX_explicit_nonce_len(ssl->aead_read_ctx);
  } else {
    return SSL3_RT_HEADER_LENGTH +
           SSL_AEAD_CTX_explicit_nonce_len(ssl->aead_read_ctx);
  }
}

size_t ssl_seal_prefix_len(const SSL *ssl) {
  if (SSL_IS_DTLS(ssl)) {
    return DTLS1_RT_HEADER_LENGTH +
           SSL_AEAD_CTX_explicit_nonce_len(ssl->aead_write_ctx);
  } else {
    size_t ret = SSL3_RT_HEADER_LENGTH +
                 SSL_AEAD_CTX_explicit_nonce_len(ssl->aead_write_ctx);
    if (ssl->s3->need_record_splitting) {
      ret += SSL3_RT_HEADER_LENGTH;
      ret += ssl_cipher_get_record_split_len(ssl->aead_write_ctx->cipher);
    }
    return ret;
  }
}

size_t ssl_max_seal_overhead(const SSL *ssl) {
  if (SSL_IS_DTLS(ssl)) {
    return DTLS1_RT_HEADER_LENGTH +
           SSL_AEAD_CTX_max_overhead(ssl->aead_write_ctx);
  } else {
    size_t ret = SSL3_RT_HEADER_LENGTH +
                 SSL_AEAD_CTX_max_overhead(ssl->aead_write_ctx);
    if (ssl->s3->need_record_splitting) {
      ret *= 2;
    }
    return ret;
  }
}

enum ssl_open_record_t tls_open_record(
    SSL *ssl, uint8_t *out_type, uint8_t *out, size_t *out_len,
    size_t *out_consumed, uint8_t *out_alert, size_t max_out, const uint8_t *in,
    size_t in_len) {
  CBS cbs;
  CBS_init(&cbs, in, in_len);

  /* Decode the record header. */
  uint8_t type;
  uint16_t version, ciphertext_len;
  if (!CBS_get_u8(&cbs, &type) ||
      !CBS_get_u16(&cbs, &version) ||
      !CBS_get_u16(&cbs, &ciphertext_len)) {
    *out_consumed = SSL3_RT_HEADER_LENGTH;
    return ssl_open_record_partial;
  }

  /* Check the version. */
  if ((ssl->s3->have_version && version != ssl->version) ||
      (version >> 8) != SSL3_VERSION_MAJOR) {
    OPENSSL_PUT_ERROR(SSL, SSL_R_WRONG_VERSION_NUMBER);
    *out_alert = SSL_AD_PROTOCOL_VERSION;
    return ssl_open_record_error;
  }

  /* Check the ciphertext length. */
  size_t extra = 0;
  if (ssl->options & SSL_OP_MICROSOFT_BIG_SSLV3_BUFFER) {
    extra = SSL3_RT_MAX_EXTRA;
  }
  if (ciphertext_len > SSL3_RT_MAX_ENCRYPTED_LENGTH + extra) {
    OPENSSL_PUT_ERROR(SSL, SSL_R_ENCRYPTED_LENGTH_TOO_LONG);
    *out_alert = SSL_AD_RECORD_OVERFLOW;
    return ssl_open_record_error;
  }

  /* Extract the body. */
  CBS body;
  if (!CBS_get_bytes(&cbs, &body, ciphertext_len)) {
    *out_consumed = SSL3_RT_HEADER_LENGTH + (size_t)ciphertext_len;
    return ssl_open_record_partial;
  }

  if (ssl->msg_callback != NULL) {
    ssl->msg_callback(0 /* read */, 0, SSL3_RT_HEADER, in,
                      SSL3_RT_HEADER_LENGTH, ssl, ssl->msg_callback_arg);
  }

  /* Decrypt the body. */
  size_t plaintext_len;
  if (!SSL_AEAD_CTX_open(ssl->aead_read_ctx, out, &plaintext_len, max_out,
                         type, version, ssl->s3->read_sequence, CBS_data(&body),
                         CBS_len(&body))) {
    OPENSSL_PUT_ERROR(SSL, SSL_R_DECRYPTION_FAILED_OR_BAD_RECORD_MAC);
    *out_alert = SSL_AD_BAD_RECORD_MAC;
    return ssl_open_record_error;
  }
  if (!ssl3_record_sequence_update(ssl->s3->read_sequence, 8)) {
    *out_alert = SSL_AD_INTERNAL_ERROR;
    return ssl_open_record_error;
  }

  /* Check the plaintext length. */
  if (plaintext_len > SSL3_RT_MAX_PLAIN_LENGTH + extra) {
    OPENSSL_PUT_ERROR(SSL, SSL_R_DATA_LENGTH_TOO_LONG);
    *out_alert = SSL_AD_RECORD_OVERFLOW;
    return ssl_open_record_error;
  }

  /* Limit the number of consecutive empty records. */
  if (plaintext_len == 0) {
    ssl->s3->empty_record_count++;
    if (ssl->s3->empty_record_count > kMaxEmptyRecords) {
      OPENSSL_PUT_ERROR(SSL, SSL_R_TOO_MANY_EMPTY_FRAGMENTS);
      *out_alert = SSL_AD_UNEXPECTED_MESSAGE;
      return ssl_open_record_error;
    }
    /* Apart from the limit, empty records are returned up to the caller. This
     * allows the caller to reject records of the wrong type. */
  } else {
    ssl->s3->empty_record_count = 0;
  }

  *out_type = type;
  *out_len = plaintext_len;
  *out_consumed = in_len - CBS_len(&cbs);
  return ssl_open_record_success;
}

static int do_seal_record(SSL *ssl, uint8_t *out, size_t *out_len,
                          size_t max_out, uint8_t type, const uint8_t *in,
                          size_t in_len) {
  if (max_out < SSL3_RT_HEADER_LENGTH) {
    OPENSSL_PUT_ERROR(SSL, SSL_R_BUFFER_TOO_SMALL);
    return 0;
  }
  /* Check the record header does not alias any part of the input.
   * |SSL_AEAD_CTX_seal| will internally enforce other aliasing requirements. */
  if (in < out + SSL3_RT_HEADER_LENGTH && out < in + in_len) {
    OPENSSL_PUT_ERROR(SSL, SSL_R_OUTPUT_ALIASES_INPUT);
    return 0;
  }

  out[0] = type;

  /* Some servers hang if initial ClientHello is larger than 256 bytes and
   * record version number > TLS 1.0. */
  uint16_t wire_version = ssl->version;
  if (!ssl->s3->have_version && ssl->version > SSL3_VERSION) {
    wire_version = TLS1_VERSION;
  }
  out[1] = wire_version >> 8;
  out[2] = wire_version & 0xff;

  size_t ciphertext_len;
  if (!SSL_AEAD_CTX_seal(ssl->aead_write_ctx, out + SSL3_RT_HEADER_LENGTH,
                         &ciphertext_len, max_out - SSL3_RT_HEADER_LENGTH,
                         type, wire_version, ssl->s3->write_sequence, in,
                         in_len) ||
      !ssl3_record_sequence_update(ssl->s3->write_sequence, 8)) {
    return 0;
  }

  if (ciphertext_len >= 1 << 16) {
    OPENSSL_PUT_ERROR(SSL, ERR_R_OVERFLOW);
    return 0;
  }
  out[3] = ciphertext_len >> 8;
  out[4] = ciphertext_len & 0xff;

  *out_len = SSL3_RT_HEADER_LENGTH + ciphertext_len;

  if (ssl->msg_callback) {
    ssl->msg_callback(1 /* write */, 0, SSL3_RT_HEADER, out,
                      SSL3_RT_HEADER_LENGTH, ssl, ssl->msg_callback_arg);
  }

  return 1;
}

int tls_seal_record(SSL *ssl, uint8_t *out, size_t *out_len, size_t max_out,
                    uint8_t type, const uint8_t *in, size_t in_len) {
  size_t frag_len = 0;
  if (ssl->s3->need_record_splitting && type == SSL3_RT_APPLICATION_DATA &&
      in_len > 1) {
    /* |do_seal_record| will notice if it clobbers |in[0]|, but not if it
     * aliases the rest of |in|. */
    if (in + 1 <= out && out < in + in_len) {
      OPENSSL_PUT_ERROR(SSL, SSL_R_OUTPUT_ALIASES_INPUT);
      return 0;
    }
    /* Ensure |do_seal_record| does not write beyond |in[0]|. */
    size_t frag_max_out = max_out;
    if (out <= in + 1 && in + 1 < out + frag_max_out) {
      frag_max_out = (size_t)(in + 1 - out);
    }
    if (!do_seal_record(ssl, out, &frag_len, frag_max_out, type, in, 1)) {
      return 0;
    }
    in++;
    in_len--;
    out += frag_len;
    max_out -= frag_len;

    assert(SSL3_RT_HEADER_LENGTH +
               ssl_cipher_get_record_split_len(ssl->aead_write_ctx->cipher) ==
           frag_len);
  }

  if (!do_seal_record(ssl, out, out_len, max_out, type, in, in_len)) {
    return 0;
  }
  *out_len += frag_len;
  return 1;
}
