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
 * Copyright 2005 Nokia. All rights reserved.
 *
 * The portions of the attached software ("Contribution") is developed by
 * Nokia Corporation and is licensed pursuant to the OpenSSL open source
 * license.
 *
 * The Contribution, originally written by Mika Kousa and Pasi Eronen of
 * Nokia Corporation, consists of the "PSK" (Pre-Shared Key) ciphersuites
 * support (see RFC 4279) to OpenSSL.
 *
 * No patent licenses or other rights except those expressly stated in
 * the OpenSSL open source license shall be deemed granted or received
 * expressly, by implication, estoppel, or otherwise.
 *
 * No assurances are provided by Nokia that the Contribution does not
 * infringe the patent or other intellectual property rights of any third
 * party or that the license provides you with all the necessary rights
 * to make use of the Contribution.
 *
 * THE SOFTWARE IS PROVIDED "AS IS" WITHOUT WARRANTY OF ANY KIND. IN
 * ADDITION TO THE DISCLAIMERS INCLUDED IN THE LICENSE, NOKIA
 * SPECIFICALLY DISCLAIMS ANY LIABILITY FOR CLAIMS BROUGHT BY YOU OR ANY
 * OTHER ENTITY BASED ON INFRINGEMENT OF INTELLECTUAL PROPERTY RIGHTS OR
 * OTHERWISE. */

#include <openssl/ssl.h>

#include <limits.h>
#include <string.h>

#include <openssl/bytestring.h>
#include <openssl/err.h>
#include <openssl/mem.h>
#include <openssl/x509.h>

#include "internal.h"


/* An SSL_SESSION is serialized as the following ASN.1 structure:
 *
 * SSLSession ::= SEQUENCE {
 *     version                     INTEGER (1),  -- ignored
 *     sslVersion                  INTEGER,      -- protocol version number
 *     cipher                      OCTET STRING, -- two bytes long
 *     sessionID                   OCTET STRING,
 *     masterKey                   OCTET STRING,
 *     time                    [1] INTEGER OPTIONAL, -- seconds since UNIX epoch
 *     timeout                 [2] INTEGER OPTIONAL, -- in seconds
 *     peer                    [3] Certificate OPTIONAL,
 *     sessionIDContext        [4] OCTET STRING OPTIONAL,
 *     verifyResult            [5] INTEGER OPTIONAL,  -- one of X509_V_* codes
 *     hostName                [6] OCTET STRING OPTIONAL,
 *                                 -- from server_name extension
 *     pskIdentity             [8] OCTET STRING OPTIONAL,
 *     ticketLifetimeHint      [9] INTEGER OPTIONAL,       -- client-only
 *     ticket                  [10] OCTET STRING OPTIONAL, -- client-only
 *     peerSHA256              [13] OCTET STRING OPTIONAL,
 *     originalHandshakeHash   [14] OCTET STRING OPTIONAL,
 *     signedCertTimestampList [15] OCTET STRING OPTIONAL,
 *                                  -- contents of SCT extension
 *     ocspResponse            [16] OCTET STRING OPTIONAL,
 *                                  -- stapled OCSP response from the server
 *     extendedMasterSecret    [17] BOOLEAN OPTIONAL,
 *     keyExchangeInfo         [18] INTEGER OPTIONAL,
 *     certChain               [19] SEQUENCE OF Certificate OPTIONAL,
 * }
 *
 * Note: historically this serialization has included other optional
 * fields. Their presense is currently treated as a parse error:
 *
 *     keyArg                  [0] IMPLICIT OCTET STRING OPTIONAL,
 *     pskIdentityHint         [7] OCTET STRING OPTIONAL,
 *     compressionMethod       [11] OCTET STRING OPTIONAL,
 *     srpUsername             [12] OCTET STRING OPTIONAL, */

static const int kTimeTag =
    CBS_ASN1_CONSTRUCTED | CBS_ASN1_CONTEXT_SPECIFIC | 1;
static const int kTimeoutTag =
    CBS_ASN1_CONSTRUCTED | CBS_ASN1_CONTEXT_SPECIFIC | 2;
static const int kPeerTag =
    CBS_ASN1_CONSTRUCTED | CBS_ASN1_CONTEXT_SPECIFIC | 3;
 static const int kSessionIDContextTag =
    CBS_ASN1_CONSTRUCTED | CBS_ASN1_CONTEXT_SPECIFIC | 4;
static const int kVerifyResultTag =
    CBS_ASN1_CONSTRUCTED | CBS_ASN1_CONTEXT_SPECIFIC | 5;
static const int kHostNameTag =
    CBS_ASN1_CONSTRUCTED | CBS_ASN1_CONTEXT_SPECIFIC | 6;
static const int kPSKIdentityTag =
    CBS_ASN1_CONSTRUCTED | CBS_ASN1_CONTEXT_SPECIFIC | 8;
static const int kTicketLifetimeHintTag =
    CBS_ASN1_CONSTRUCTED | CBS_ASN1_CONTEXT_SPECIFIC | 9;
static const int kTicketTag =
    CBS_ASN1_CONSTRUCTED | CBS_ASN1_CONTEXT_SPECIFIC | 10;
static const int kPeerSHA256Tag =
    CBS_ASN1_CONSTRUCTED | CBS_ASN1_CONTEXT_SPECIFIC | 13;
static const int kOriginalHandshakeHashTag =
    CBS_ASN1_CONSTRUCTED | CBS_ASN1_CONTEXT_SPECIFIC | 14;
static const int kSignedCertTimestampListTag =
    CBS_ASN1_CONSTRUCTED | CBS_ASN1_CONTEXT_SPECIFIC | 15;
static const int kOCSPResponseTag =
    CBS_ASN1_CONSTRUCTED | CBS_ASN1_CONTEXT_SPECIFIC | 16;
static const int kExtendedMasterSecretTag =
    CBS_ASN1_CONSTRUCTED | CBS_ASN1_CONTEXT_SPECIFIC | 17;
static const int kKeyExchangeInfoTag =
    CBS_ASN1_CONSTRUCTED | CBS_ASN1_CONTEXT_SPECIFIC | 18;
static const int kCertChainTag =
    CBS_ASN1_CONSTRUCTED | CBS_ASN1_CONTEXT_SPECIFIC | 19;

static int add_X509(CBB *cbb, X509 *x509) {
  int len = i2d_X509(x509, NULL);
  if (len < 0) {
    return 0;
  }
  uint8_t *buf;
  if (!CBB_add_space(cbb, &buf, len)) {
    OPENSSL_PUT_ERROR(SSL, ERR_R_MALLOC_FAILURE);
    return 0;
  }
  if (buf != NULL && i2d_X509(x509, &buf) < 0) {
    return 0;
  }
  return 1;
}

static int SSL_SESSION_to_bytes_full(const SSL_SESSION *in, uint8_t **out_data,
                                     size_t *out_len, int for_ticket) {
  CBB cbb, session, child, child2;

  if (in == NULL || in->cipher == NULL) {
    return 0;
  }

  CBB_zero(&cbb);
  if (!CBB_init(&cbb, 0) ||
      !CBB_add_asn1(&cbb, &session, CBS_ASN1_SEQUENCE) ||
      !CBB_add_asn1_uint64(&session, SSL_SESSION_ASN1_VERSION) ||
      !CBB_add_asn1_uint64(&session, in->ssl_version) ||
      !CBB_add_asn1(&session, &child, CBS_ASN1_OCTETSTRING) ||
      !CBB_add_u16(&child, (uint16_t)(in->cipher->id & 0xffff)) ||
      !CBB_add_asn1(&session, &child, CBS_ASN1_OCTETSTRING) ||
      /* The session ID is irrelevant for a session ticket. */
      !CBB_add_bytes(&child, in->session_id,
                     for_ticket ? 0 : in->session_id_length) ||
      !CBB_add_asn1(&session, &child, CBS_ASN1_OCTETSTRING) ||
      !CBB_add_bytes(&child, in->master_key, in->master_key_length)) {
    OPENSSL_PUT_ERROR(SSL, ERR_R_MALLOC_FAILURE);
    goto err;
  }

  if (in->time != 0) {
    if (!CBB_add_asn1(&session, &child, kTimeTag) ||
        !CBB_add_asn1_uint64(&child, in->time)) {
      OPENSSL_PUT_ERROR(SSL, ERR_R_MALLOC_FAILURE);
      goto err;
    }
  }

  if (in->timeout != 0) {
    if (!CBB_add_asn1(&session, &child, kTimeoutTag) ||
        !CBB_add_asn1_uint64(&child, in->timeout)) {
      OPENSSL_PUT_ERROR(SSL, ERR_R_MALLOC_FAILURE);
      goto err;
    }
  }

  /* The peer certificate is only serialized if the SHA-256 isn't
   * serialized instead. */
  if (in->peer && !in->peer_sha256_valid) {
    if (!CBB_add_asn1(&session, &child, kPeerTag)) {
      OPENSSL_PUT_ERROR(SSL, ERR_R_MALLOC_FAILURE);
      goto err;
    }
    if (!add_X509(&child, in->peer)) {
      goto err;
    }
  }

  /* Although it is OPTIONAL and usually empty, OpenSSL has
   * historically always encoded the sid_ctx. */
  if (!CBB_add_asn1(&session, &child, kSessionIDContextTag) ||
      !CBB_add_asn1(&child, &child2, CBS_ASN1_OCTETSTRING) ||
      !CBB_add_bytes(&child2, in->sid_ctx, in->sid_ctx_length)) {
    OPENSSL_PUT_ERROR(SSL, ERR_R_MALLOC_FAILURE);
    goto err;
  }

  if (in->verify_result != X509_V_OK) {
    if (!CBB_add_asn1(&session, &child, kVerifyResultTag) ||
        !CBB_add_asn1_uint64(&child, in->verify_result)) {
      OPENSSL_PUT_ERROR(SSL, ERR_R_MALLOC_FAILURE);
      goto err;
    }
  }

  if (in->tlsext_hostname) {
    if (!CBB_add_asn1(&session, &child, kHostNameTag) ||
        !CBB_add_asn1(&child, &child2, CBS_ASN1_OCTETSTRING) ||
        !CBB_add_bytes(&child2, (const uint8_t *)in->tlsext_hostname,
                       strlen(in->tlsext_hostname))) {
      OPENSSL_PUT_ERROR(SSL, ERR_R_MALLOC_FAILURE);
      goto err;
    }
  }

  if (in->psk_identity) {
    if (!CBB_add_asn1(&session, &child, kPSKIdentityTag) ||
        !CBB_add_asn1(&child, &child2, CBS_ASN1_OCTETSTRING) ||
        !CBB_add_bytes(&child2, (const uint8_t *)in->psk_identity,
                       strlen(in->psk_identity))) {
      OPENSSL_PUT_ERROR(SSL, ERR_R_MALLOC_FAILURE);
      goto err;
    }
  }

  if (in->tlsext_tick_lifetime_hint > 0) {
    if (!CBB_add_asn1(&session, &child, kTicketLifetimeHintTag) ||
        !CBB_add_asn1_uint64(&child, in->tlsext_tick_lifetime_hint)) {
      OPENSSL_PUT_ERROR(SSL, ERR_R_MALLOC_FAILURE);
      goto err;
    }
  }

  if (in->tlsext_tick && !for_ticket) {
    if (!CBB_add_asn1(&session, &child, kTicketTag) ||
        !CBB_add_asn1(&child, &child2, CBS_ASN1_OCTETSTRING) ||
        !CBB_add_bytes(&child2, in->tlsext_tick, in->tlsext_ticklen)) {
      OPENSSL_PUT_ERROR(SSL, ERR_R_MALLOC_FAILURE);
      goto err;
    }
  }

  if (in->peer_sha256_valid) {
    if (!CBB_add_asn1(&session, &child, kPeerSHA256Tag) ||
        !CBB_add_asn1(&child, &child2, CBS_ASN1_OCTETSTRING) ||
        !CBB_add_bytes(&child2, in->peer_sha256, sizeof(in->peer_sha256))) {
      OPENSSL_PUT_ERROR(SSL, ERR_R_MALLOC_FAILURE);
      goto err;
    }
  }

  if (in->original_handshake_hash_len > 0) {
    if (!CBB_add_asn1(&session, &child, kOriginalHandshakeHashTag) ||
        !CBB_add_asn1(&child, &child2, CBS_ASN1_OCTETSTRING) ||
        !CBB_add_bytes(&child2, in->original_handshake_hash,
                       in->original_handshake_hash_len)) {
      OPENSSL_PUT_ERROR(SSL, ERR_R_MALLOC_FAILURE);
      goto err;
    }
  }

  if (in->tlsext_signed_cert_timestamp_list_length > 0) {
    if (!CBB_add_asn1(&session, &child, kSignedCertTimestampListTag) ||
        !CBB_add_asn1(&child, &child2, CBS_ASN1_OCTETSTRING) ||
        !CBB_add_bytes(&child2, in->tlsext_signed_cert_timestamp_list,
                       in->tlsext_signed_cert_timestamp_list_length)) {
      OPENSSL_PUT_ERROR(SSL, ERR_R_MALLOC_FAILURE);
      goto err;
    }
  }

  if (in->ocsp_response_length > 0) {
    if (!CBB_add_asn1(&session, &child, kOCSPResponseTag) ||
        !CBB_add_asn1(&child, &child2, CBS_ASN1_OCTETSTRING) ||
        !CBB_add_bytes(&child2, in->ocsp_response, in->ocsp_response_length)) {
      OPENSSL_PUT_ERROR(SSL, ERR_R_MALLOC_FAILURE);
      goto err;
    }
  }

  if (in->extended_master_secret) {
    if (!CBB_add_asn1(&session, &child, kExtendedMasterSecretTag) ||
        !CBB_add_asn1(&child, &child2, CBS_ASN1_BOOLEAN) ||
        !CBB_add_u8(&child2, 0xff)) {
      OPENSSL_PUT_ERROR(SSL, ERR_R_MALLOC_FAILURE);
      goto err;
    }
  }

  if (in->key_exchange_info > 0 &&
      (!CBB_add_asn1(&session, &child, kKeyExchangeInfoTag) ||
       !CBB_add_asn1_uint64(&child, in->key_exchange_info))) {
    OPENSSL_PUT_ERROR(SSL, ERR_R_MALLOC_FAILURE);
    goto err;
  }

  /* The certificate chain is only serialized if the leaf's SHA-256 isn't
   * serialized instead. */
  if (in->cert_chain != NULL && !in->peer_sha256_valid) {
    if (!CBB_add_asn1(&session, &child, kCertChainTag)) {
      OPENSSL_PUT_ERROR(SSL, ERR_R_MALLOC_FAILURE);
      goto err;
    }
    size_t i;
    for (i = 0; i < sk_X509_num(in->cert_chain); i++) {
      if (!add_X509(&child, sk_X509_value(in->cert_chain, i))) {
        goto err;
      }
    }
  }

  if (!CBB_finish(&cbb, out_data, out_len)) {
    OPENSSL_PUT_ERROR(SSL, ERR_R_MALLOC_FAILURE);
    goto err;
  }
  return 1;

 err:
  CBB_cleanup(&cbb);
  return 0;
}

int SSL_SESSION_to_bytes(const SSL_SESSION *in, uint8_t **out_data,
                         size_t *out_len) {
  return SSL_SESSION_to_bytes_full(in, out_data, out_len, 0);
}

int SSL_SESSION_to_bytes_for_ticket(const SSL_SESSION *in, uint8_t **out_data,
                                    size_t *out_len) {
  return SSL_SESSION_to_bytes_full(in, out_data, out_len, 1);
}

int i2d_SSL_SESSION(SSL_SESSION *in, uint8_t **pp) {
  uint8_t *out;
  size_t len;

  if (!SSL_SESSION_to_bytes(in, &out, &len)) {
    return -1;
  }

  if (len > INT_MAX) {
    OPENSSL_free(out);
    OPENSSL_PUT_ERROR(SSL, ERR_R_OVERFLOW);
    return -1;
  }

  if (pp) {
    memcpy(*pp, out, len);
    *pp += len;
  }
  OPENSSL_free(out);

  return len;
}

/* SSL_SESSION_parse_string gets an optional ASN.1 OCTET STRING
 * explicitly tagged with |tag| from |cbs| and saves it in |*out|. On
 * entry, if |*out| is not NULL, it frees the existing contents. If
 * the element was not found, it sets |*out| to NULL. It returns one
 * on success, whether or not the element was found, and zero on
 * decode error. */
static int SSL_SESSION_parse_string(CBS *cbs, char **out, unsigned tag) {
  CBS value;
  int present;
  if (!CBS_get_optional_asn1_octet_string(cbs, &value, &present, tag)) {
    OPENSSL_PUT_ERROR(SSL, SSL_R_INVALID_SSL_SESSION);
    return 0;
  }
  if (present) {
    if (CBS_contains_zero_byte(&value)) {
      OPENSSL_PUT_ERROR(SSL, SSL_R_INVALID_SSL_SESSION);
      return 0;
    }
    if (!CBS_strdup(&value, out)) {
      OPENSSL_PUT_ERROR(SSL, ERR_R_MALLOC_FAILURE);
      return 0;
    }
  } else {
    OPENSSL_free(*out);
    *out = NULL;
  }
  return 1;
}

/* SSL_SESSION_parse_string gets an optional ASN.1 OCTET STRING
 * explicitly tagged with |tag| from |cbs| and stows it in |*out_ptr|
 * and |*out_len|. If |*out_ptr| is not NULL, it frees the existing
 * contents. On entry, if the element was not found, it sets
 * |*out_ptr| to NULL. It returns one on success, whether or not the
 * element was found, and zero on decode error. */
static int SSL_SESSION_parse_octet_string(CBS *cbs, uint8_t **out_ptr,
                                          size_t *out_len, unsigned tag) {
  CBS value;
  if (!CBS_get_optional_asn1_octet_string(cbs, &value, NULL, tag)) {
    OPENSSL_PUT_ERROR(SSL, SSL_R_INVALID_SSL_SESSION);
    return 0;
  }
  if (!CBS_stow(&value, out_ptr, out_len)) {
    OPENSSL_PUT_ERROR(SSL, ERR_R_MALLOC_FAILURE);
    return 0;
  }
  return 1;
}

/* SSL_SESSION_parse_bounded_octet_string parses an optional ASN.1 OCTET STRING
 * explicitly tagged with |tag| of size at most |max_out|. */
static int SSL_SESSION_parse_bounded_octet_string(
    CBS *cbs, uint8_t *out, unsigned *out_len, unsigned max_out, unsigned tag) {
  CBS value;
  if (!CBS_get_optional_asn1_octet_string(cbs, &value, NULL, tag) ||
      CBS_len(&value) > max_out) {
    OPENSSL_PUT_ERROR(SSL, SSL_R_INVALID_SSL_SESSION);
    return 0;
  }
  memcpy(out, CBS_data(&value), CBS_len(&value));
  *out_len = (unsigned)CBS_len(&value);
  return 1;
}

static int SSL_SESSION_parse_long(CBS *cbs, long *out, unsigned tag,
                                  long default_value) {
  uint64_t value;
  if (!CBS_get_optional_asn1_uint64(cbs, &value, tag,
                                    (uint64_t)default_value) ||
      value > LONG_MAX) {
    OPENSSL_PUT_ERROR(SSL, SSL_R_INVALID_SSL_SESSION);
    return 0;
  }
  *out = (long)value;
  return 1;
}

static int SSL_SESSION_parse_u32(CBS *cbs, uint32_t *out, unsigned tag,
                                 uint32_t default_value) {
  uint64_t value;
  if (!CBS_get_optional_asn1_uint64(cbs, &value, tag,
                                    (uint64_t)default_value) ||
      value > 0xffffffff) {
    OPENSSL_PUT_ERROR(SSL, SSL_R_INVALID_SSL_SESSION);
    return 0;
  }
  *out = (uint32_t)value;
  return 1;
}

static X509 *parse_x509(CBS *cbs) {
  const uint8_t *ptr = CBS_data(cbs);
  X509 *ret = d2i_X509(NULL, &ptr, CBS_len(cbs));
  if (ret == NULL) {
    return NULL;
  }
  CBS_skip(cbs, ptr - CBS_data(cbs));
  return ret;
}

static SSL_SESSION *SSL_SESSION_parse(CBS *cbs) {
  SSL_SESSION *ret = SSL_SESSION_new();
  if (ret == NULL) {
    goto err;
  }

  CBS session;
  uint64_t version, ssl_version;
  if (!CBS_get_asn1(cbs, &session, CBS_ASN1_SEQUENCE) ||
      !CBS_get_asn1_uint64(&session, &version) ||
      version != SSL_SESSION_ASN1_VERSION ||
      !CBS_get_asn1_uint64(&session, &ssl_version)) {
    OPENSSL_PUT_ERROR(SSL, SSL_R_INVALID_SSL_SESSION);
    goto err;
  }
  /* Only support SSLv3/TLS and DTLS. */
  if ((ssl_version >> 8) != SSL3_VERSION_MAJOR &&
      (ssl_version >> 8) != (DTLS1_VERSION >> 8)) {
    OPENSSL_PUT_ERROR(SSL, SSL_R_UNKNOWN_SSL_VERSION);
    goto err;
  }
  ret->ssl_version = ssl_version;

  CBS cipher;
  uint16_t cipher_value;
  if (!CBS_get_asn1(&session, &cipher, CBS_ASN1_OCTETSTRING) ||
      !CBS_get_u16(&cipher, &cipher_value) ||
      CBS_len(&cipher) != 0) {
    OPENSSL_PUT_ERROR(SSL, SSL_R_INVALID_SSL_SESSION);
    goto err;
  }
  ret->cipher = SSL_get_cipher_by_value(cipher_value);
  if (ret->cipher == NULL) {
    OPENSSL_PUT_ERROR(SSL, SSL_R_UNSUPPORTED_CIPHER);
    goto err;
  }

  CBS session_id, master_key;
  if (!CBS_get_asn1(&session, &session_id, CBS_ASN1_OCTETSTRING) ||
      CBS_len(&session_id) > SSL3_MAX_SSL_SESSION_ID_LENGTH ||
      !CBS_get_asn1(&session, &master_key, CBS_ASN1_OCTETSTRING) ||
      CBS_len(&master_key) > SSL_MAX_MASTER_KEY_LENGTH) {
    OPENSSL_PUT_ERROR(SSL, SSL_R_INVALID_SSL_SESSION);
    goto err;
  }
  memcpy(ret->session_id, CBS_data(&session_id), CBS_len(&session_id));
  ret->session_id_length = CBS_len(&session_id);
  memcpy(ret->master_key, CBS_data(&master_key), CBS_len(&master_key));
  ret->master_key_length = CBS_len(&master_key);

  if (!SSL_SESSION_parse_long(&session, &ret->time, kTimeTag, time(NULL)) ||
      !SSL_SESSION_parse_long(&session, &ret->timeout, kTimeoutTag, 3)) {
    OPENSSL_PUT_ERROR(SSL, SSL_R_INVALID_SSL_SESSION);
    goto err;
  }

  CBS peer;
  int has_peer;
  if (!CBS_get_optional_asn1(&session, &peer, &has_peer, kPeerTag)) {
    OPENSSL_PUT_ERROR(SSL, SSL_R_INVALID_SSL_SESSION);
    goto err;
  }
  X509_free(ret->peer);
  ret->peer = NULL;
  if (has_peer) {
    ret->peer = parse_x509(&peer);
    if (ret->peer == NULL) {
      goto err;
    }
    if (CBS_len(&peer) != 0) {
      OPENSSL_PUT_ERROR(SSL, SSL_R_INVALID_SSL_SESSION);
      goto err;
    }
  }

  if (!SSL_SESSION_parse_bounded_octet_string(
          &session, ret->sid_ctx, &ret->sid_ctx_length, sizeof(ret->sid_ctx),
          kSessionIDContextTag) ||
      !SSL_SESSION_parse_long(&session, &ret->verify_result, kVerifyResultTag,
                              X509_V_OK) ||
      !SSL_SESSION_parse_string(&session, &ret->tlsext_hostname,
                                kHostNameTag) ||
      !SSL_SESSION_parse_string(&session, &ret->psk_identity,
                                kPSKIdentityTag) ||
      !SSL_SESSION_parse_u32(&session, &ret->tlsext_tick_lifetime_hint,
                             kTicketLifetimeHintTag, 0) ||
      !SSL_SESSION_parse_octet_string(&session, &ret->tlsext_tick,
                                      &ret->tlsext_ticklen, kTicketTag)) {
    goto err;
  }

  if (CBS_peek_asn1_tag(&session, kPeerSHA256Tag)) {
    CBS child, peer_sha256;
    if (!CBS_get_asn1(&session, &child, kPeerSHA256Tag) ||
        !CBS_get_asn1(&child, &peer_sha256, CBS_ASN1_OCTETSTRING) ||
        CBS_len(&peer_sha256) != sizeof(ret->peer_sha256) ||
        CBS_len(&child) != 0) {
      OPENSSL_PUT_ERROR(SSL, SSL_R_INVALID_SSL_SESSION);
      goto err;
    }
    memcpy(ret->peer_sha256, CBS_data(&peer_sha256), sizeof(ret->peer_sha256));
    ret->peer_sha256_valid = 1;
  } else {
    ret->peer_sha256_valid = 0;
  }

  if (!SSL_SESSION_parse_bounded_octet_string(
          &session, ret->original_handshake_hash,
          &ret->original_handshake_hash_len,
          sizeof(ret->original_handshake_hash), kOriginalHandshakeHashTag) ||
      !SSL_SESSION_parse_octet_string(
          &session, &ret->tlsext_signed_cert_timestamp_list,
          &ret->tlsext_signed_cert_timestamp_list_length,
          kSignedCertTimestampListTag) ||
      !SSL_SESSION_parse_octet_string(
          &session, &ret->ocsp_response, &ret->ocsp_response_length,
          kOCSPResponseTag)) {
    goto err;
  }

  int extended_master_secret;
  if (!CBS_get_optional_asn1_bool(&session, &extended_master_secret,
                                  kExtendedMasterSecretTag,
                                  0 /* default to false */)) {
    OPENSSL_PUT_ERROR(SSL, SSL_R_INVALID_SSL_SESSION);
    goto err;
  }
  ret->extended_master_secret = !!extended_master_secret;

  if (!SSL_SESSION_parse_u32(&session, &ret->key_exchange_info,
                             kKeyExchangeInfoTag, 0)) {
    OPENSSL_PUT_ERROR(SSL, SSL_R_INVALID_SSL_SESSION);
    goto err;
  }

  CBS cert_chain;
  int has_cert_chain;
  if (!CBS_get_optional_asn1(&session, &cert_chain, &has_cert_chain,
                             kCertChainTag)) {
    OPENSSL_PUT_ERROR(SSL, SSL_R_INVALID_SSL_SESSION);
    goto err;
  }
  sk_X509_pop_free(ret->cert_chain, X509_free);
  ret->cert_chain = NULL;
  if (has_cert_chain) {
    ret->cert_chain = sk_X509_new_null();
    if (ret->cert_chain == NULL) {
      OPENSSL_PUT_ERROR(SSL, ERR_R_MALLOC_FAILURE);
      goto err;
    }
    while (CBS_len(&cert_chain) > 0) {
      X509 *x509 = parse_x509(&cert_chain);
      if (x509 == NULL) {
        goto err;
      }
      if (!sk_X509_push(ret->cert_chain, x509)) {
        OPENSSL_PUT_ERROR(SSL, ERR_R_MALLOC_FAILURE);
        X509_free(x509);
        goto err;
      }
    }
  }

  if (CBS_len(&session) != 0) {
    OPENSSL_PUT_ERROR(SSL, SSL_R_INVALID_SSL_SESSION);
    goto err;
  }

  return ret;

err:
  SSL_SESSION_free(ret);
  return NULL;
}

SSL_SESSION *SSL_SESSION_from_bytes(const uint8_t *in, size_t in_len) {
  CBS cbs;
  CBS_init(&cbs, in, in_len);
  SSL_SESSION *ret = SSL_SESSION_parse(&cbs);
  if (ret == NULL) {
    return NULL;
  }
  if (CBS_len(&cbs) != 0) {
    OPENSSL_PUT_ERROR(SSL, SSL_R_INVALID_SSL_SESSION);
    SSL_SESSION_free(ret);
    return NULL;
  }
  return ret;
}

SSL_SESSION *d2i_SSL_SESSION(SSL_SESSION **a, const uint8_t **pp, long length) {
  if (length < 0) {
    OPENSSL_PUT_ERROR(SSL, ERR_R_INTERNAL_ERROR);
    return NULL;
  }

  CBS cbs;
  CBS_init(&cbs, *pp, length);

  SSL_SESSION *ret = SSL_SESSION_parse(&cbs);
  if (ret == NULL) {
    return NULL;
  }

  if (a) {
    SSL_SESSION_free(*a);
    *a = ret;
  }
  *pp = CBS_data(&cbs);
  return ret;
}
