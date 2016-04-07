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
 * Copyright (c) 1998-2007 The OpenSSL Project.  All rights reserved.
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
 * Hudson (tjh@cryptsoft.com).
 *
 */
/* ====================================================================
 * Copyright 2002 Sun Microsystems, Inc. ALL RIGHTS RESERVED.
 *
 * Portions of the attached software ("Contribution") are developed by 
 * SUN MICROSYSTEMS, INC., and are contributed to the OpenSSL project.
 *
 * The Contribution is licensed pursuant to the OpenSSL open source
 * license provided above.
 *
 * ECC cipher suite support in OpenSSL originally written by
 * Vipul Gupta and Sumit Gupta of Sun Microsystems Laboratories.
 *
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

#include <assert.h>
#include <stdio.h>
#include <string.h>

#include <openssl/bn.h>
#include <openssl/buf.h>
#include <openssl/bytestring.h>
#include <openssl/cipher.h>
#include <openssl/dh.h>
#include <openssl/ec.h>
#include <openssl/ecdsa.h>
#include <openssl/err.h>
#include <openssl/evp.h>
#include <openssl/hmac.h>
#include <openssl/md5.h>
#include <openssl/mem.h>
#include <openssl/obj.h>
#include <openssl/rand.h>
#include <openssl/sha.h>
#include <openssl/x509.h>

#include "internal.h"
#include "../crypto/internal.h"
#include "../crypto/dh/internal.h"


int ssl3_accept(SSL *s) {
  BUF_MEM *buf = NULL;
  uint32_t alg_a;
  void (*cb)(const SSL *ssl, int type, int val) = NULL;
  int ret = -1;
  int new_state, state, skip = 0;

  assert(s->handshake_func == ssl3_accept);
  assert(s->server);
  assert(!SSL_IS_DTLS(s));

  ERR_clear_error();
  ERR_clear_system_error();

  if (s->info_callback != NULL) {
    cb = s->info_callback;
  } else if (s->ctx->info_callback != NULL) {
    cb = s->ctx->info_callback;
  }

  s->in_handshake++;

  if (s->cert == NULL) {
    OPENSSL_PUT_ERROR(SSL, SSL_R_NO_CERTIFICATE_SET);
    return -1;
  }

  for (;;) {
    state = s->state;

    switch (s->state) {
      case SSL_ST_ACCEPT:
        if (cb != NULL) {
          cb(s, SSL_CB_HANDSHAKE_START, 1);
        }

        if (s->init_buf == NULL) {
          buf = BUF_MEM_new();
          if (!buf || !BUF_MEM_grow(buf, SSL3_RT_MAX_PLAIN_LENGTH)) {
            ret = -1;
            goto end;
          }
          s->init_buf = buf;
          buf = NULL;
        }
        s->init_num = 0;

        /* Enable a write buffer. This groups handshake messages within a flight
         * into a single write. */
        if (!ssl_init_wbio_buffer(s, 1)) {
          ret = -1;
          goto end;
        }

        if (!ssl3_init_handshake_buffer(s)) {
          OPENSSL_PUT_ERROR(SSL, ERR_R_INTERNAL_ERROR);
          ret = -1;
          goto end;
        }

        if (!s->s3->have_version) {
          s->state = SSL3_ST_SR_INITIAL_BYTES;
        } else {
          s->state = SSL3_ST_SR_CLNT_HELLO_A;
        }
        break;

      case SSL3_ST_SR_INITIAL_BYTES:
        ret = ssl3_get_initial_bytes(s);
        if (ret <= 0) {
          goto end;
        }
        /* ssl3_get_initial_bytes sets s->state to one of
         * SSL3_ST_SR_V2_CLIENT_HELLO or SSL3_ST_SR_CLNT_HELLO_A on success. */
        break;

      case SSL3_ST_SR_V2_CLIENT_HELLO:
        ret = ssl3_get_v2_client_hello(s);
        if (ret <= 0) {
          goto end;
        }
        s->state = SSL3_ST_SR_CLNT_HELLO_A;
        break;

      case SSL3_ST_SR_CLNT_HELLO_A:
      case SSL3_ST_SR_CLNT_HELLO_B:
      case SSL3_ST_SR_CLNT_HELLO_C:
      case SSL3_ST_SR_CLNT_HELLO_D:
        s->shutdown = 0;
        ret = ssl3_get_client_hello(s);
        if (ret <= 0) {
          goto end;
        }
        s->state = SSL3_ST_SW_SRVR_HELLO_A;
        s->init_num = 0;
        break;

      case SSL3_ST_SW_SRVR_HELLO_A:
      case SSL3_ST_SW_SRVR_HELLO_B:
        ret = ssl3_send_server_hello(s);
        if (ret <= 0) {
          goto end;
        }
        if (s->hit) {
          if (s->tlsext_ticket_expected) {
            s->state = SSL3_ST_SW_SESSION_TICKET_A;
          } else {
            s->state = SSL3_ST_SW_CHANGE_A;
          }
        } else {
          s->state = SSL3_ST_SW_CERT_A;
        }
        s->init_num = 0;
        break;

      case SSL3_ST_SW_CERT_A:
      case SSL3_ST_SW_CERT_B:
        if (ssl_cipher_has_server_public_key(s->s3->tmp.new_cipher)) {
          ret = ssl3_send_server_certificate(s);
          if (ret <= 0) {
            goto end;
          }
          if (s->s3->tmp.certificate_status_expected) {
            s->state = SSL3_ST_SW_CERT_STATUS_A;
          } else {
            s->state = SSL3_ST_SW_KEY_EXCH_A;
          }
        } else {
          skip = 1;
          s->state = SSL3_ST_SW_KEY_EXCH_A;
        }
        s->init_num = 0;
        break;

      case SSL3_ST_SW_CERT_STATUS_A:
      case SSL3_ST_SW_CERT_STATUS_B:
        ret = ssl3_send_certificate_status(s);
        if (ret <= 0) {
          goto end;
        }
        s->state = SSL3_ST_SW_KEY_EXCH_A;
        s->init_num = 0;
        break;

      case SSL3_ST_SW_KEY_EXCH_A:
      case SSL3_ST_SW_KEY_EXCH_B:
      case SSL3_ST_SW_KEY_EXCH_C:
        alg_a = s->s3->tmp.new_cipher->algorithm_auth;

        /* Send a ServerKeyExchange message if:
         * - The key exchange is ephemeral or anonymous
         *   Diffie-Hellman.
         * - There is a PSK identity hint.
         *
         * TODO(davidben): This logic is currently duplicated in d1_srvr.c. Fix
         * this. In the meantime, keep them in sync. */
        if (ssl_cipher_requires_server_key_exchange(s->s3->tmp.new_cipher) ||
            ((alg_a & SSL_aPSK) && s->psk_identity_hint)) {
          ret = ssl3_send_server_key_exchange(s);
          if (ret <= 0) {
            goto end;
          }
        } else {
          skip = 1;
        }

        s->state = SSL3_ST_SW_CERT_REQ_A;
        s->init_num = 0;
        break;

      case SSL3_ST_SW_CERT_REQ_A:
      case SSL3_ST_SW_CERT_REQ_B:
        if (s->s3->tmp.cert_request) {
          ret = ssl3_send_certificate_request(s);
          if (ret <= 0) {
            goto end;
          }
        } else {
          skip = 1;
        }
        s->state = SSL3_ST_SW_SRVR_DONE_A;
        s->init_num = 0;
        break;

      case SSL3_ST_SW_SRVR_DONE_A:
      case SSL3_ST_SW_SRVR_DONE_B:
        ret = ssl3_send_server_done(s);
        if (ret <= 0) {
          goto end;
        }
        s->s3->tmp.next_state = SSL3_ST_SR_CERT_A;
        s->state = SSL3_ST_SW_FLUSH;
        s->init_num = 0;
        break;

      case SSL3_ST_SW_FLUSH:
        /* This code originally checked to see if any data was pending using
         * BIO_CTRL_INFO and then flushed. This caused problems as documented
         * in PR#1939. The proposed fix doesn't completely resolve this issue
         * as buggy implementations of BIO_CTRL_PENDING still exist. So instead
         * we just flush unconditionally. */
        s->rwstate = SSL_WRITING;
        if (BIO_flush(s->wbio) <= 0) {
          ret = -1;
          goto end;
        }
        s->rwstate = SSL_NOTHING;

        s->state = s->s3->tmp.next_state;
        break;

      case SSL3_ST_SR_CERT_A:
      case SSL3_ST_SR_CERT_B:
        if (s->s3->tmp.cert_request) {
          ret = ssl3_get_client_certificate(s);
          if (ret <= 0) {
            goto end;
          }
        }
        s->init_num = 0;
        s->state = SSL3_ST_SR_KEY_EXCH_A;
        break;

      case SSL3_ST_SR_KEY_EXCH_A:
      case SSL3_ST_SR_KEY_EXCH_B:
        ret = ssl3_get_client_key_exchange(s);
        if (ret <= 0) {
          goto end;
        }
        s->state = SSL3_ST_SR_CERT_VRFY_A;
        s->init_num = 0;
        break;

      case SSL3_ST_SR_CERT_VRFY_A:
      case SSL3_ST_SR_CERT_VRFY_B:
        ret = ssl3_get_cert_verify(s);
        if (ret <= 0) {
          goto end;
        }

        s->state = SSL3_ST_SR_CHANGE;
        s->init_num = 0;
        break;

      case SSL3_ST_SR_CHANGE: {
        char next_proto_neg = 0;
        char channel_id = 0;
        next_proto_neg = s->s3->next_proto_neg_seen;
        channel_id = s->s3->tlsext_channel_id_valid;

        /* At this point, the next message must be entirely behind a
         * ChangeCipherSpec. */
        if (!ssl3_expect_change_cipher_spec(s)) {
          ret = -1;
          goto end;
        }
        if (next_proto_neg) {
          s->state = SSL3_ST_SR_NEXT_PROTO_A;
        } else if (channel_id) {
          s->state = SSL3_ST_SR_CHANNEL_ID_A;
        } else {
          s->state = SSL3_ST_SR_FINISHED_A;
        }
        break;
      }

      case SSL3_ST_SR_NEXT_PROTO_A:
      case SSL3_ST_SR_NEXT_PROTO_B:
        ret = ssl3_get_next_proto(s);
        if (ret <= 0) {
          goto end;
        }
        s->init_num = 0;
        if (s->s3->tlsext_channel_id_valid) {
          s->state = SSL3_ST_SR_CHANNEL_ID_A;
        } else {
          s->state = SSL3_ST_SR_FINISHED_A;
        }
        break;

      case SSL3_ST_SR_CHANNEL_ID_A:
      case SSL3_ST_SR_CHANNEL_ID_B:
        ret = ssl3_get_channel_id(s);
        if (ret <= 0) {
          goto end;
        }
        s->init_num = 0;
        s->state = SSL3_ST_SR_FINISHED_A;
        break;

      case SSL3_ST_SR_FINISHED_A:
      case SSL3_ST_SR_FINISHED_B:
        ret =
            ssl3_get_finished(s, SSL3_ST_SR_FINISHED_A, SSL3_ST_SR_FINISHED_B);
        if (ret <= 0) {
          goto end;
        }

        if (s->hit) {
          s->state = SSL_ST_OK;
        } else if (s->tlsext_ticket_expected) {
          s->state = SSL3_ST_SW_SESSION_TICKET_A;
        } else {
          s->state = SSL3_ST_SW_CHANGE_A;
        }
        /* If this is a full handshake with ChannelID then record the hashshake
         * hashes in |s->session| in case we need them to verify a ChannelID
         * signature on a resumption of this session in the future. */
        if (!s->hit && s->s3->tlsext_channel_id_valid) {
          ret = tls1_record_handshake_hashes_for_channel_id(s);
          if (ret <= 0) {
            goto end;
          }
        }
        s->init_num = 0;
        break;

      case SSL3_ST_SW_SESSION_TICKET_A:
      case SSL3_ST_SW_SESSION_TICKET_B:
        ret = ssl3_send_new_session_ticket(s);
        if (ret <= 0) {
          goto end;
        }
        s->state = SSL3_ST_SW_CHANGE_A;
        s->init_num = 0;
        break;

      case SSL3_ST_SW_CHANGE_A:
      case SSL3_ST_SW_CHANGE_B:
        s->session->cipher = s->s3->tmp.new_cipher;
        if (!s->enc_method->setup_key_block(s)) {
          ret = -1;
          goto end;
        }

        ret = ssl3_send_change_cipher_spec(s, SSL3_ST_SW_CHANGE_A,
                                           SSL3_ST_SW_CHANGE_B);
        if (ret <= 0) {
          goto end;
        }
        s->state = SSL3_ST_SW_FINISHED_A;
        s->init_num = 0;

        if (!s->enc_method->change_cipher_state(
                s, SSL3_CHANGE_CIPHER_SERVER_WRITE)) {
          ret = -1;
          goto end;
        }
        break;

      case SSL3_ST_SW_FINISHED_A:
      case SSL3_ST_SW_FINISHED_B:
        ret =
            ssl3_send_finished(s, SSL3_ST_SW_FINISHED_A, SSL3_ST_SW_FINISHED_B,
                               s->enc_method->server_finished_label,
                               s->enc_method->server_finished_label_len);
        if (ret <= 0) {
          goto end;
        }
        s->state = SSL3_ST_SW_FLUSH;
        if (s->hit) {
          s->s3->tmp.next_state = SSL3_ST_SR_CHANGE;
        } else {
          s->s3->tmp.next_state = SSL_ST_OK;
        }
        s->init_num = 0;
        break;

      case SSL_ST_OK:
        /* clean a few things up */
        ssl3_cleanup_key_block(s);

        BUF_MEM_free(s->init_buf);
        s->init_buf = NULL;

        /* remove buffering on output */
        ssl_free_wbio_buffer(s);

        s->init_num = 0;

        /* If we aren't retaining peer certificates then we can discard it
         * now. */
        if (s->ctx->retain_only_sha256_of_client_certs) {
          X509_free(s->session->peer);
          s->session->peer = NULL;
          sk_X509_pop_free(s->session->cert_chain, X509_free);
          s->session->cert_chain = NULL;
        }

        s->s3->initial_handshake_complete = 1;

        ssl_update_cache(s, SSL_SESS_CACHE_SERVER);

        if (cb != NULL) {
          cb(s, SSL_CB_HANDSHAKE_DONE, 1);
        }

        ret = 1;
        goto end;

      default:
        OPENSSL_PUT_ERROR(SSL, SSL_R_UNKNOWN_STATE);
        ret = -1;
        goto end;
    }

    if (!s->s3->tmp.reuse_message && !skip && cb != NULL && s->state != state) {
      new_state = s->state;
      s->state = state;
      cb(s, SSL_CB_ACCEPT_LOOP, 1);
      s->state = new_state;
    }
    skip = 0;
  }

end:
  s->in_handshake--;
  BUF_MEM_free(buf);
  if (cb != NULL) {
    cb(s, SSL_CB_ACCEPT_EXIT, ret);
  }
  return ret;
}

int ssl3_get_initial_bytes(SSL *s) {
  /* Read the first 5 bytes, the size of the TLS record header. This is
   * sufficient to detect a V2ClientHello and ensures that we never read beyond
   * the first record. */
  int ret = ssl_read_buffer_extend_to(s, SSL3_RT_HEADER_LENGTH);
  if (ret <= 0) {
    return ret;
  }
  assert(ssl_read_buffer_len(s) == SSL3_RT_HEADER_LENGTH);
  const uint8_t *p = ssl_read_buffer(s);

  /* Some dedicated error codes for protocol mixups should the application wish
   * to interpret them differently. (These do not overlap with ClientHello or
   * V2ClientHello.) */
  if (strncmp("GET ", (const char *)p, 4) == 0 ||
      strncmp("POST ", (const char *)p, 5) == 0 ||
      strncmp("HEAD ", (const char *)p, 5) == 0 ||
      strncmp("PUT ", (const char *)p, 4) == 0) {
    OPENSSL_PUT_ERROR(SSL, SSL_R_HTTP_REQUEST);
    return -1;
  }
  if (strncmp("CONNE", (const char *)p, 5) == 0) {
    OPENSSL_PUT_ERROR(SSL, SSL_R_HTTPS_PROXY_REQUEST);
    return -1;
  }

  /* Determine if this is a V2ClientHello. */
  if ((p[0] & 0x80) && p[2] == SSL2_MT_CLIENT_HELLO &&
      p[3] >= SSL3_VERSION_MAJOR) {
    /* This is a V2ClientHello. */
    s->state = SSL3_ST_SR_V2_CLIENT_HELLO;
    return 1;
  }

  /* Fall through to the standard logic. */
  s->state = SSL3_ST_SR_CLNT_HELLO_A;
  return 1;
}

int ssl3_get_v2_client_hello(SSL *s) {
  const uint8_t *p;
  int ret;
  CBS v2_client_hello, cipher_specs, session_id, challenge;
  size_t msg_length, rand_len, len;
  uint8_t msg_type;
  uint16_t version, cipher_spec_length, session_id_length, challenge_length;
  CBB client_hello, hello_body, cipher_suites;
  uint8_t random[SSL3_RANDOM_SIZE];

  /* Determine the length of the V2ClientHello. */
  assert(ssl_read_buffer_len(s) >= SSL3_RT_HEADER_LENGTH);
  p = ssl_read_buffer(s);
  msg_length = ((p[0] & 0x7f) << 8) | p[1];
  if (msg_length > (1024 * 4)) {
    OPENSSL_PUT_ERROR(SSL, SSL_R_RECORD_TOO_LARGE);
    return -1;
  }
  if (msg_length < SSL3_RT_HEADER_LENGTH - 2) {
    /* Reject lengths that are too short early. We have already read
     * |SSL3_RT_HEADER_LENGTH| bytes, so we should not attempt to process an
     * (invalid) V2ClientHello which would be shorter than that. */
    OPENSSL_PUT_ERROR(SSL, SSL_R_RECORD_LENGTH_MISMATCH);
    return -1;
  }

  /* Read the remainder of the V2ClientHello. */
  ret = ssl_read_buffer_extend_to(s, 2 + msg_length);
  if (ret <= 0) {
    return ret;
  }
  assert(ssl_read_buffer_len(s) == msg_length + 2);
  CBS_init(&v2_client_hello, ssl_read_buffer(s) + 2, msg_length);

  /* The V2ClientHello without the length is incorporated into the handshake
   * hash. */
  if (!ssl3_update_handshake_hash(s, CBS_data(&v2_client_hello),
                                  CBS_len(&v2_client_hello))) {
    return -1;
  }
  if (s->msg_callback) {
    s->msg_callback(0, SSL2_VERSION, 0, CBS_data(&v2_client_hello),
                    CBS_len(&v2_client_hello), s, s->msg_callback_arg);
  }

  if (!CBS_get_u8(&v2_client_hello, &msg_type) ||
      !CBS_get_u16(&v2_client_hello, &version) ||
      !CBS_get_u16(&v2_client_hello, &cipher_spec_length) ||
      !CBS_get_u16(&v2_client_hello, &session_id_length) ||
      !CBS_get_u16(&v2_client_hello, &challenge_length) ||
      !CBS_get_bytes(&v2_client_hello, &cipher_specs, cipher_spec_length) ||
      !CBS_get_bytes(&v2_client_hello, &session_id, session_id_length) ||
      !CBS_get_bytes(&v2_client_hello, &challenge, challenge_length) ||
      CBS_len(&v2_client_hello) != 0) {
    OPENSSL_PUT_ERROR(SSL, SSL_R_DECODE_ERROR);
    return -1;
  }

  /* msg_type has already been checked. */
  assert(msg_type == SSL2_MT_CLIENT_HELLO);

  /* The client_random is the V2ClientHello challenge. Truncate or
   * left-pad with zeros as needed. */
  memset(random, 0, SSL3_RANDOM_SIZE);
  rand_len = CBS_len(&challenge);
  if (rand_len > SSL3_RANDOM_SIZE) {
    rand_len = SSL3_RANDOM_SIZE;
  }
  memcpy(random + (SSL3_RANDOM_SIZE - rand_len), CBS_data(&challenge),
         rand_len);

  /* Write out an equivalent SSLv3 ClientHello. */
  CBB_zero(&client_hello);
  if (!CBB_init_fixed(&client_hello, (uint8_t *)s->init_buf->data,
                      s->init_buf->max) ||
      !CBB_add_u8(&client_hello, SSL3_MT_CLIENT_HELLO) ||
      !CBB_add_u24_length_prefixed(&client_hello, &hello_body) ||
      !CBB_add_u16(&hello_body, version) ||
      !CBB_add_bytes(&hello_body, random, SSL3_RANDOM_SIZE) ||
      /* No session id. */
      !CBB_add_u8(&hello_body, 0) ||
      !CBB_add_u16_length_prefixed(&hello_body, &cipher_suites)) {
    CBB_cleanup(&client_hello);
    OPENSSL_PUT_ERROR(SSL, ERR_R_MALLOC_FAILURE);
    return -1;
  }

  /* Copy the cipher suites. */
  while (CBS_len(&cipher_specs) > 0) {
    uint32_t cipher_spec;
    if (!CBS_get_u24(&cipher_specs, &cipher_spec)) {
      CBB_cleanup(&client_hello);
      OPENSSL_PUT_ERROR(SSL, SSL_R_DECODE_ERROR);
      return -1;
    }

    /* Skip SSLv2 ciphers. */
    if ((cipher_spec & 0xff0000) != 0) {
      continue;
    }
    if (!CBB_add_u16(&cipher_suites, cipher_spec)) {
      CBB_cleanup(&client_hello);
      OPENSSL_PUT_ERROR(SSL, ERR_R_INTERNAL_ERROR);
      return -1;
    }
  }

  /* Add the null compression scheme and finish. */
  if (!CBB_add_u8(&hello_body, 1) || !CBB_add_u8(&hello_body, 0) ||
      !CBB_finish(&client_hello, NULL, &len)) {
    CBB_cleanup(&client_hello);
    OPENSSL_PUT_ERROR(SSL, ERR_R_INTERNAL_ERROR);
    return -1;
  }

  /* Mark the message for "re"-use by the version-specific method. */
  s->s3->tmp.reuse_message = 1;
  s->s3->tmp.message_type = SSL3_MT_CLIENT_HELLO;
  /* The handshake message header is 4 bytes. */
  s->s3->tmp.message_size = len - 4;

  /* Consume and discard the V2ClientHello. */
  ssl_read_buffer_consume(s, 2 + msg_length);
  ssl_read_buffer_discard(s);

  return 1;
}

int ssl3_get_client_hello(SSL *s) {
  int ok, al = SSL_AD_INTERNAL_ERROR, ret = -1;
  long n;
  const SSL_CIPHER *c;
  STACK_OF(SSL_CIPHER) *ciphers = NULL;
  struct ssl_early_callback_ctx early_ctx;
  CBS client_hello;
  uint16_t client_version;
  CBS client_random, session_id, cipher_suites, compression_methods;
  SSL_SESSION *session = NULL;

  /* We do this so that we will respond with our native type. If we are TLSv1
   * and we get SSLv3, we will respond with TLSv1, This down switching should
   * be handled by a different method. If we are SSLv3, we will respond with
   * SSLv3, even if prompted with TLSv1. */
  switch (s->state) {
    case SSL3_ST_SR_CLNT_HELLO_A:
    case SSL3_ST_SR_CLNT_HELLO_B:
      n = s->method->ssl_get_message(
          s, SSL3_ST_SR_CLNT_HELLO_A, SSL3_ST_SR_CLNT_HELLO_B,
          SSL3_MT_CLIENT_HELLO, SSL3_RT_MAX_PLAIN_LENGTH,
          ssl_hash_message, &ok);

      if (!ok) {
        return n;
      }

      s->state = SSL3_ST_SR_CLNT_HELLO_C;
      /* fallthrough */
    case SSL3_ST_SR_CLNT_HELLO_C:
    case SSL3_ST_SR_CLNT_HELLO_D:
      /* We have previously parsed the ClientHello message, and can't call
       * ssl_get_message again without hashing the message into the Finished
       * digest again. */
      n = s->init_num;

      memset(&early_ctx, 0, sizeof(early_ctx));
      early_ctx.ssl = s;
      early_ctx.client_hello = s->init_msg;
      early_ctx.client_hello_len = n;
      if (!ssl_early_callback_init(&early_ctx)) {
        al = SSL_AD_DECODE_ERROR;
        OPENSSL_PUT_ERROR(SSL, SSL_R_CLIENTHELLO_PARSE_FAILED);
        goto f_err;
      }

      if (s->state == SSL3_ST_SR_CLNT_HELLO_C &&
          s->ctx->select_certificate_cb != NULL) {
        s->state = SSL3_ST_SR_CLNT_HELLO_D;
        switch (s->ctx->select_certificate_cb(&early_ctx)) {
          case 0:
            s->rwstate = SSL_CERTIFICATE_SELECTION_PENDING;
            goto err;

          case -1:
            /* Connection rejected. */
            al = SSL_AD_ACCESS_DENIED;
            OPENSSL_PUT_ERROR(SSL, SSL_R_CONNECTION_REJECTED);
            goto f_err;

          default:
            /* fallthrough */;
        }
      }
      s->state = SSL3_ST_SR_CLNT_HELLO_D;
      break;

    default:
      OPENSSL_PUT_ERROR(SSL, SSL_R_UNKNOWN_STATE);
      return -1;
  }

  CBS_init(&client_hello, s->init_msg, n);
  if (!CBS_get_u16(&client_hello, &client_version) ||
      !CBS_get_bytes(&client_hello, &client_random, SSL3_RANDOM_SIZE) ||
      !CBS_get_u8_length_prefixed(&client_hello, &session_id) ||
      CBS_len(&session_id) > SSL_MAX_SSL_SESSION_ID_LENGTH) {
    al = SSL_AD_DECODE_ERROR;
    OPENSSL_PUT_ERROR(SSL, SSL_R_DECODE_ERROR);
    goto f_err;
  }

  /* use version from inside client hello, not from record header (may differ:
   * see RFC 2246, Appendix E, second paragraph) */
  s->client_version = client_version;

  /* Load the client random. */
  memcpy(s->s3->client_random, CBS_data(&client_random), SSL3_RANDOM_SIZE);

  if (SSL_IS_DTLS(s)) {
    CBS cookie;

    if (!CBS_get_u8_length_prefixed(&client_hello, &cookie) ||
        CBS_len(&cookie) > DTLS1_COOKIE_LENGTH) {
      al = SSL_AD_DECODE_ERROR;
      OPENSSL_PUT_ERROR(SSL, SSL_R_DECODE_ERROR);
      goto f_err;
    }
  }

  /* Note: This codepath may run twice if |ssl_get_prev_session| completes
   * asynchronously.
   *
   * TODO(davidben): Clean up the order of events around ClientHello
   * processing. */
  if (!s->s3->have_version) {
    /* Select version to use */
    uint16_t version = ssl3_get_mutual_version(s, client_version);
    if (version == 0) {
      OPENSSL_PUT_ERROR(SSL, SSL_R_UNSUPPORTED_PROTOCOL);
      s->version = s->client_version;
      al = SSL_AD_PROTOCOL_VERSION;
      goto f_err;
    }
    s->version = version;
    s->enc_method = ssl3_get_enc_method(version);
    assert(s->enc_method != NULL);
    /* At this point, the connection's version is known and |s->version| is
     * fixed. Begin enforcing the record-layer version. */
    s->s3->have_version = 1;
  } else if (SSL_IS_DTLS(s) ? (s->client_version > s->version)
                            : (s->client_version < s->version)) {
    OPENSSL_PUT_ERROR(SSL, SSL_R_WRONG_VERSION_NUMBER);
    al = SSL_AD_PROTOCOL_VERSION;
    goto f_err;
  }

  s->hit = 0;
  int send_new_ticket = 0;
  switch (ssl_get_prev_session(s, &session, &send_new_ticket, &early_ctx)) {
    case ssl_session_success:
      break;
    case ssl_session_error:
      goto err;
    case ssl_session_retry:
      s->rwstate = SSL_PENDING_SESSION;
      goto err;
  }
  s->tlsext_ticket_expected = send_new_ticket;

  /* The EMS state is needed when making the resumption decision, but
   * extensions are not normally parsed until later. This detects the EMS
   * extension for the resumption decision and it's checked against the result
   * of the normal parse later in this function. */
  const uint8_t *ems_data;
  size_t ems_len;
  int have_extended_master_secret =
      s->version != SSL3_VERSION &&
      SSL_early_callback_ctx_extension_get(&early_ctx,
                                           TLSEXT_TYPE_extended_master_secret,
                                           &ems_data, &ems_len) &&
      ems_len == 0;

  if (session != NULL) {
    if (session->extended_master_secret &&
        !have_extended_master_secret) {
      /* A ClientHello without EMS that attempts to resume a session with EMS
       * is fatal to the connection. */
      al = SSL_AD_HANDSHAKE_FAILURE;
      OPENSSL_PUT_ERROR(SSL, SSL_R_RESUMED_EMS_SESSION_WITHOUT_EMS_EXTENSION);
      goto f_err;
    }

    s->hit =
        /* Only resume if the session's version matches the negotiated version:
         * most clients do not accept a mismatch. */
        s->version == session->ssl_version &&
        /* If the client offers the EMS extension, but the previous session
         * didn't use it, then negotiate a new session. */
        have_extended_master_secret == session->extended_master_secret;
  }

  if (s->hit) {
    /* Use the new session. */
    SSL_SESSION_free(s->session);
    s->session = session;
    session = NULL;

    s->verify_result = s->session->verify_result;
  } else if (!ssl_get_new_session(s, 1)) {
    goto err;
  }

  if (s->ctx->dos_protection_cb != NULL && s->ctx->dos_protection_cb(&early_ctx) == 0) {
    /* Connection rejected for DOS reasons. */
    al = SSL_AD_ACCESS_DENIED;
    OPENSSL_PUT_ERROR(SSL, SSL_R_CONNECTION_REJECTED);
    goto f_err;
  }

  if (!CBS_get_u16_length_prefixed(&client_hello, &cipher_suites) ||
      CBS_len(&cipher_suites) == 0 ||
      CBS_len(&cipher_suites) % 2 != 0 ||
      !CBS_get_u8_length_prefixed(&client_hello, &compression_methods) ||
      CBS_len(&compression_methods) == 0) {
    al = SSL_AD_DECODE_ERROR;
    OPENSSL_PUT_ERROR(SSL, SSL_R_DECODE_ERROR);
    goto f_err;
  }

  ciphers = ssl_bytes_to_cipher_list(s, &cipher_suites);
  if (ciphers == NULL) {
    goto err;
  }

  /* If it is a hit, check that the cipher is in the list. */
  if (s->hit) {
    size_t j;
    int found_cipher = 0;
    uint32_t id = s->session->cipher->id;

    for (j = 0; j < sk_SSL_CIPHER_num(ciphers); j++) {
      c = sk_SSL_CIPHER_value(ciphers, j);
      if (c->id == id) {
        found_cipher = 1;
        break;
      }
    }

    if (!found_cipher) {
      /* we need to have the cipher in the cipher list if we are asked to reuse
       * it */
      al = SSL_AD_ILLEGAL_PARAMETER;
      OPENSSL_PUT_ERROR(SSL, SSL_R_REQUIRED_CIPHER_MISSING);
      goto f_err;
    }
  }

  /* Only null compression is supported. */
  if (memchr(CBS_data(&compression_methods), 0,
             CBS_len(&compression_methods)) == NULL) {
    al = SSL_AD_ILLEGAL_PARAMETER;
    OPENSSL_PUT_ERROR(SSL, SSL_R_NO_COMPRESSION_SPECIFIED);
    goto f_err;
  }

  /* TLS extensions. */
  if (s->version >= SSL3_VERSION &&
      !ssl_parse_clienthello_tlsext(s, &client_hello)) {
    OPENSSL_PUT_ERROR(SSL, SSL_R_PARSE_TLSEXT);
    goto err;
  }

  /* There should be nothing left over in the record. */
  if (CBS_len(&client_hello) != 0) {
    /* wrong packet length */
    al = SSL_AD_DECODE_ERROR;
    OPENSSL_PUT_ERROR(SSL, SSL_R_BAD_PACKET_LENGTH);
    goto f_err;
  }

  if (have_extended_master_secret != s->s3->tmp.extended_master_secret) {
    al = SSL_AD_INTERNAL_ERROR;
    OPENSSL_PUT_ERROR(SSL, SSL_R_EMS_STATE_INCONSISTENT);
    goto f_err;
  }

  /* Given ciphers and SSL_get_ciphers, we must pick a cipher */
  if (!s->hit) {
    if (ciphers == NULL) {
      al = SSL_AD_ILLEGAL_PARAMETER;
      OPENSSL_PUT_ERROR(SSL, SSL_R_NO_CIPHERS_PASSED);
      goto f_err;
    }

    /* Let cert callback update server certificates if required */
    if (s->cert->cert_cb) {
      int rv = s->cert->cert_cb(s, s->cert->cert_cb_arg);
      if (rv == 0) {
        al = SSL_AD_INTERNAL_ERROR;
        OPENSSL_PUT_ERROR(SSL, SSL_R_CERT_CB_ERROR);
        goto f_err;
      }
      if (rv < 0) {
        s->rwstate = SSL_X509_LOOKUP;
        goto err;
      }
      s->rwstate = SSL_NOTHING;
    }
    c = ssl3_choose_cipher(s, ciphers, ssl_get_cipher_preferences(s));

    if (c == NULL) {
      al = SSL_AD_HANDSHAKE_FAILURE;
      OPENSSL_PUT_ERROR(SSL, SSL_R_NO_SHARED_CIPHER);
      goto f_err;
    }
    s->s3->tmp.new_cipher = c;

    /* Determine whether to request a client certificate. */
    s->s3->tmp.cert_request = !!(s->verify_mode & SSL_VERIFY_PEER);
    /* Only request a certificate if Channel ID isn't negotiated. */
    if ((s->verify_mode & SSL_VERIFY_PEER_IF_NO_OBC) &&
        s->s3->tlsext_channel_id_valid) {
      s->s3->tmp.cert_request = 0;
    }
    /* Plain PSK forbids Certificate and CertificateRequest. */
    if (s->s3->tmp.new_cipher->algorithm_mkey & SSL_kPSK) {
      s->s3->tmp.cert_request = 0;
    }
  } else {
    /* Session-id reuse */
    s->s3->tmp.new_cipher = s->session->cipher;
    s->s3->tmp.cert_request = 0;
  }

  /* Now that the cipher is known, initialize the handshake hash. */
  if (!ssl3_init_handshake_hash(s)) {
    goto f_err;
  }

  /* In TLS 1.2, client authentication requires hashing the handshake transcript
   * under a different hash. Otherwise, release the handshake buffer. */
  if (!SSL_USE_SIGALGS(s) || !s->s3->tmp.cert_request) {
    ssl3_free_handshake_buffer(s);
  }

  /* we now have the following setup;
   * client_random
   * cipher_list        - our prefered list of ciphers
   * ciphers            - the clients prefered list of ciphers
   * compression        - basically ignored right now
   * ssl version is set - sslv3
   * s->session         - The ssl session has been setup.
   * s->hit             - session reuse flag
   * s->tmp.new_cipher  - the new cipher to use. */

  if (ret < 0) {
    ret = -ret;
  }

  if (0) {
  f_err:
    ssl3_send_alert(s, SSL3_AL_FATAL, al);
  }

err:
  sk_SSL_CIPHER_free(ciphers);
  SSL_SESSION_free(session);
  return ret;
}

int ssl3_send_server_hello(SSL *s) {
  uint8_t *buf;
  uint8_t *p, *d;
  int sl;
  unsigned long l;

  if (s->state == SSL3_ST_SW_SRVR_HELLO_A) {
    /* We only accept ChannelIDs on connections with ECDHE in order to avoid a
     * known attack while we fix ChannelID itself. */
    if (s->s3->tlsext_channel_id_valid &&
        (s->s3->tmp.new_cipher->algorithm_mkey & SSL_kECDHE) == 0) {
      s->s3->tlsext_channel_id_valid = 0;
    }

    /* If this is a resumption and the original handshake didn't support
     * ChannelID then we didn't record the original handshake hashes in the
     * session and so cannot resume with ChannelIDs. */
    if (s->hit && s->session->original_handshake_hash_len == 0) {
      s->s3->tlsext_channel_id_valid = 0;
    }

    buf = (uint8_t *)s->init_buf->data;
    /* Do the message type and length last */
    d = p = ssl_handshake_start(s);

    *(p++) = s->version >> 8;
    *(p++) = s->version & 0xff;

    /* Random stuff */
    if (!ssl_fill_hello_random(s->s3->server_random, SSL3_RANDOM_SIZE,
                               1 /* server */)) {
      OPENSSL_PUT_ERROR(SSL, ERR_R_INTERNAL_ERROR);
      return -1;
    }
    memcpy(p, s->s3->server_random, SSL3_RANDOM_SIZE);
    p += SSL3_RANDOM_SIZE;

    /* There are several cases for the session ID to send
     * back in the server hello:
     * - For session reuse from the session cache, we send back the old session
     *   ID.
     * - If stateless session reuse (using a session ticket) is successful, we
     *   send back the client's "session ID" (which doesn't actually identify
     *   the session).
     * - If it is a new session, we send back the new session ID.
     * - However, if we want the new session to be single-use, we send back a
     *   0-length session ID.
     * s->hit is non-zero in either case of session reuse, so the following
     * won't overwrite an ID that we're supposed to send back. */
    if (!(s->ctx->session_cache_mode & SSL_SESS_CACHE_SERVER) && !s->hit) {
      s->session->session_id_length = 0;
    }

    sl = s->session->session_id_length;
    if (sl > (int)sizeof(s->session->session_id)) {
      OPENSSL_PUT_ERROR(SSL, ERR_R_INTERNAL_ERROR);
      return -1;
    }
    *(p++) = sl;
    memcpy(p, s->session->session_id, sl);
    p += sl;

    /* put the cipher */
    s2n(ssl_cipher_get_value(s->s3->tmp.new_cipher), p);

    /* put the compression method */
    *(p++) = 0;

    p = ssl_add_serverhello_tlsext(s, p, buf + SSL3_RT_MAX_PLAIN_LENGTH);
    if (p == NULL) {
      OPENSSL_PUT_ERROR(SSL, ERR_R_INTERNAL_ERROR);
      return -1;
    }

    /* do the header */
    l = (p - d);
    if (!ssl_set_handshake_header(s, SSL3_MT_SERVER_HELLO, l)) {
      return -1;
    }
    s->state = SSL3_ST_SW_SRVR_HELLO_B;
  }

  /* SSL3_ST_SW_SRVR_HELLO_B */
  return ssl_do_write(s);
}

int ssl3_send_certificate_status(SSL *ssl) {
  if (ssl->state == SSL3_ST_SW_CERT_STATUS_A) {
    CBB out, ocsp_response;
    size_t length;

    CBB_zero(&out);
    if (!CBB_init_fixed(&out, ssl_handshake_start(ssl),
                        ssl->init_buf->max - SSL_HM_HEADER_LENGTH(ssl)) ||
        !CBB_add_u8(&out, TLSEXT_STATUSTYPE_ocsp) ||
        !CBB_add_u24_length_prefixed(&out, &ocsp_response) ||
        !CBB_add_bytes(&ocsp_response, ssl->ctx->ocsp_response,
                       ssl->ctx->ocsp_response_length) ||
        !CBB_finish(&out, NULL, &length) ||
        !ssl_set_handshake_header(ssl, SSL3_MT_CERTIFICATE_STATUS, length)) {
      OPENSSL_PUT_ERROR(SSL, ERR_R_INTERNAL_ERROR);
      CBB_cleanup(&out);
      return -1;
    }

    ssl->state = SSL3_ST_SW_CERT_STATUS_B;
  }

  /* SSL3_ST_SW_CERT_STATUS_B */
  return ssl_do_write(ssl);
}

int ssl3_send_server_done(SSL *s) {
  if (s->state == SSL3_ST_SW_SRVR_DONE_A) {
    if (!ssl_set_handshake_header(s, SSL3_MT_SERVER_DONE, 0)) {
      return -1;
    }
    s->state = SSL3_ST_SW_SRVR_DONE_B;
  }

  /* SSL3_ST_SW_SRVR_DONE_B */
  return ssl_do_write(s);
}

int ssl3_send_server_key_exchange(SSL *s) {
  DH *dh = NULL, *dhp;
  EC_KEY *ecdh = NULL;
  uint8_t *encodedPoint = NULL;
  int encodedlen = 0;
  uint16_t curve_id = 0;
  BN_CTX *bn_ctx = NULL;
  const char *psk_identity_hint = NULL;
  size_t psk_identity_hint_len = 0;
  size_t sig_len;
  size_t max_sig_len;
  uint8_t *p, *d;
  int al, i;
  uint32_t alg_k;
  uint32_t alg_a;
  int n;
  CERT *cert;
  BIGNUM *r[4];
  int nr[4];
  BUF_MEM *buf;
  EVP_MD_CTX md_ctx;

  if (s->state == SSL3_ST_SW_KEY_EXCH_C) {
    return ssl_do_write(s);
  }

  if (ssl_cipher_has_server_public_key(s->s3->tmp.new_cipher)) {
    if (!ssl_has_private_key(s)) {
      al = SSL_AD_INTERNAL_ERROR;
      goto f_err;
    }
    max_sig_len = ssl_private_key_max_signature_len(s);
  } else {
    max_sig_len = 0;
  }

  EVP_MD_CTX_init(&md_ctx);
  enum ssl_private_key_result_t sign_result;
  if (s->state == SSL3_ST_SW_KEY_EXCH_A) {
    alg_k = s->s3->tmp.new_cipher->algorithm_mkey;
    alg_a = s->s3->tmp.new_cipher->algorithm_auth;
    cert = s->cert;

    buf = s->init_buf;

    r[0] = r[1] = r[2] = r[3] = NULL;
    n = 0;
    if (alg_a & SSL_aPSK) {
      /* size for PSK identity hint */
      psk_identity_hint = s->psk_identity_hint;
      if (psk_identity_hint) {
        psk_identity_hint_len = strlen(psk_identity_hint);
      } else {
        psk_identity_hint_len = 0;
      }
      n += 2 + psk_identity_hint_len;
    }

    if (alg_k & SSL_kDHE) {
      dhp = cert->dh_tmp;
      if (dhp == NULL && s->cert->dh_tmp_cb != NULL) {
        dhp = s->cert->dh_tmp_cb(s, 0, 1024);
      }
      if (dhp == NULL) {
        al = SSL_AD_HANDSHAKE_FAILURE;
        OPENSSL_PUT_ERROR(SSL, SSL_R_MISSING_TMP_DH_KEY);
        goto f_err;
      }

      if (s->s3->tmp.dh != NULL) {
        OPENSSL_PUT_ERROR(SSL, ERR_R_INTERNAL_ERROR);
        goto err;
      }
      dh = DHparams_dup(dhp);
      if (dh == NULL) {
        OPENSSL_PUT_ERROR(SSL, ERR_R_DH_LIB);
        goto err;
      }
      s->s3->tmp.dh = dh;

      if (!DH_generate_key(dh)) {
        OPENSSL_PUT_ERROR(SSL, ERR_R_DH_LIB);
        goto err;
      }

      r[0] = dh->p;
      r[1] = dh->g;
      r[2] = dh->pub_key;
    } else if (alg_k & SSL_kECDHE) {
      /* Determine the curve to use. */
      int nid = NID_undef;
      if (cert->ecdh_nid != NID_undef) {
        nid = cert->ecdh_nid;
      } else if (cert->ecdh_tmp_cb != NULL) {
        /* Note: |ecdh_tmp_cb| does NOT pass ownership of the result
         * to the caller. */
        EC_KEY *template = s->cert->ecdh_tmp_cb(s, 0, 1024);
        if (template != NULL && EC_KEY_get0_group(template) != NULL) {
          nid = EC_GROUP_get_curve_name(EC_KEY_get0_group(template));
        }
      } else {
        nid = tls1_get_shared_curve(s);
      }
      if (nid == NID_undef) {
        al = SSL_AD_HANDSHAKE_FAILURE;
        OPENSSL_PUT_ERROR(SSL, SSL_R_MISSING_TMP_ECDH_KEY);
        goto f_err;
      }

      if (s->s3->tmp.ecdh != NULL) {
        OPENSSL_PUT_ERROR(SSL, ERR_R_INTERNAL_ERROR);
        goto err;
      }
      ecdh = EC_KEY_new_by_curve_name(nid);
      if (ecdh == NULL) {
        goto err;
      }
      s->s3->tmp.ecdh = ecdh;

      if (!EC_KEY_generate_key(ecdh)) {
        OPENSSL_PUT_ERROR(SSL, ERR_R_ECDH_LIB);
        goto err;
      }

      /* We only support ephemeral ECDH keys over named (not generic) curves. */
      const EC_GROUP *group = EC_KEY_get0_group(ecdh);
      if (!tls1_ec_nid2curve_id(&curve_id, EC_GROUP_get_curve_name(group))) {
        OPENSSL_PUT_ERROR(SSL, SSL_R_UNSUPPORTED_ELLIPTIC_CURVE);
        goto err;
      }

      /* Encode the public key. First check the size of encoding and allocate
       * memory accordingly. */
      encodedlen =
          EC_POINT_point2oct(group, EC_KEY_get0_public_key(ecdh),
                             POINT_CONVERSION_UNCOMPRESSED, NULL, 0, NULL);

      encodedPoint = (uint8_t *)OPENSSL_malloc(encodedlen * sizeof(uint8_t));
      bn_ctx = BN_CTX_new();
      if (encodedPoint == NULL || bn_ctx == NULL) {
        OPENSSL_PUT_ERROR(SSL, ERR_R_MALLOC_FAILURE);
        goto err;
      }

      encodedlen = EC_POINT_point2oct(group, EC_KEY_get0_public_key(ecdh),
                                      POINT_CONVERSION_UNCOMPRESSED,
                                      encodedPoint, encodedlen, bn_ctx);

      if (encodedlen == 0) {
        OPENSSL_PUT_ERROR(SSL, ERR_R_ECDH_LIB);
        goto err;
      }

      BN_CTX_free(bn_ctx);
      bn_ctx = NULL;

      /* We only support named (not generic) curves in ECDH ephemeral key
       * exchanges. In this situation, we need four additional bytes to encode
       * the entire ServerECDHParams structure. */
      n += 4 + encodedlen;

      /* We'll generate the serverKeyExchange message explicitly so we can set
       * these to NULLs */
      r[0] = NULL;
      r[1] = NULL;
      r[2] = NULL;
      r[3] = NULL;
    } else if (!(alg_k & SSL_kPSK)) {
      al = SSL_AD_HANDSHAKE_FAILURE;
      OPENSSL_PUT_ERROR(SSL, SSL_R_UNKNOWN_KEY_EXCHANGE_TYPE);
      goto f_err;
    }

    for (i = 0; i < 4 && r[i] != NULL; i++) {
      nr[i] = BN_num_bytes(r[i]);
      n += 2 + nr[i];
    }

    if (!BUF_MEM_grow_clean(buf, n + SSL_HM_HEADER_LENGTH(s) + max_sig_len)) {
      OPENSSL_PUT_ERROR(SSL, ERR_LIB_BUF);
      goto err;
    }
    d = p = ssl_handshake_start(s);

    for (i = 0; i < 4 && r[i] != NULL; i++) {
      s2n(nr[i], p);
      BN_bn2bin(r[i], p);
      p += nr[i];
    }

    /* Note: ECDHE PSK ciphersuites use SSL_kECDHE and SSL_aPSK. When one of
     * them is used, the server key exchange record needs to have both the
     * psk_identity_hint and the ServerECDHParams. */
    if (alg_a & SSL_aPSK) {
      /* copy PSK identity hint (if provided) */
      s2n(psk_identity_hint_len, p);
      if (psk_identity_hint_len > 0) {
        memcpy(p, psk_identity_hint, psk_identity_hint_len);
        p += psk_identity_hint_len;
      }
    }

    if (alg_k & SSL_kECDHE) {
      /* We only support named (not generic) curves. In this situation, the
       * serverKeyExchange message has:
       * [1 byte CurveType], [2 byte CurveName]
       * [1 byte length of encoded point], followed by
       * the actual encoded point itself. */
      *(p++) = NAMED_CURVE_TYPE;
      *(p++) = (uint8_t)(curve_id >> 8);
      *(p++) = (uint8_t)(curve_id & 0xff);
      *(p++) = encodedlen;
      memcpy(p, encodedPoint, encodedlen);
      p += encodedlen;
      OPENSSL_free(encodedPoint);
      encodedPoint = NULL;
    }

    if (ssl_cipher_has_server_public_key(s->s3->tmp.new_cipher)) {
      /* n is the length of the params, they start at d and p points to
       * the space at the end. */
      const EVP_MD *md;
      uint8_t digest[EVP_MAX_MD_SIZE];
      unsigned int digest_length;

      const int pkey_type = ssl_private_key_type(s);

      /* Determine signature algorithm. */
      if (SSL_USE_SIGALGS(s)) {
        md = tls1_choose_signing_digest(s);
        if (!tls12_get_sigandhash(s, p, md)) {
          /* Should never happen */
          al = SSL_AD_INTERNAL_ERROR;
          OPENSSL_PUT_ERROR(SSL, ERR_R_INTERNAL_ERROR);
          goto f_err;
        }
        p += 2;
      } else if (pkey_type == EVP_PKEY_RSA) {
        md = EVP_md5_sha1();
      } else {
        md = EVP_sha1();
      }

      if (!EVP_DigestInit_ex(&md_ctx, md, NULL) ||
          !EVP_DigestUpdate(&md_ctx, s->s3->client_random, SSL3_RANDOM_SIZE) ||
          !EVP_DigestUpdate(&md_ctx, s->s3->server_random, SSL3_RANDOM_SIZE) ||
          !EVP_DigestUpdate(&md_ctx, d, n) ||
          !EVP_DigestFinal_ex(&md_ctx, digest, &digest_length)) {
        OPENSSL_PUT_ERROR(SSL, ERR_LIB_EVP);
        goto err;
      }

      sign_result = ssl_private_key_sign(s, &p[2], &sig_len, max_sig_len,
                                         EVP_MD_CTX_md(&md_ctx), digest,
                                         digest_length);
    } else {
      /* This key exchange doesn't involve a signature. */
      sign_result = ssl_private_key_success;
      sig_len = 0;
    }
  } else {
    assert(s->state == SSL3_ST_SW_KEY_EXCH_B);
    /* Restore |p|. */
    p = ssl_handshake_start(s) + s->init_num - SSL_HM_HEADER_LENGTH(s);
    sign_result = ssl_private_key_sign_complete(s, &p[2], &sig_len,
                                                max_sig_len);
  }

  switch (sign_result) {
    case ssl_private_key_success:
      s->rwstate = SSL_NOTHING;
      break;
    case ssl_private_key_failure:
      s->rwstate = SSL_NOTHING;
      goto err;
    case ssl_private_key_retry:
      s->rwstate = SSL_PRIVATE_KEY_OPERATION;
      /* Stash away |p|. */
      s->init_num = p - ssl_handshake_start(s) + SSL_HM_HEADER_LENGTH(s);
      s->state = SSL3_ST_SW_KEY_EXCH_B;
      goto err;
  }

  if (ssl_cipher_has_server_public_key(s->s3->tmp.new_cipher)) {
    s2n(sig_len, p);
    p += sig_len;
  }
  if (!ssl_set_handshake_header(s, SSL3_MT_SERVER_KEY_EXCHANGE,
                                p - ssl_handshake_start(s))) {
    goto err;
  }
  s->state = SSL3_ST_SW_KEY_EXCH_C;

  EVP_MD_CTX_cleanup(&md_ctx);
  return ssl_do_write(s);

f_err:
  ssl3_send_alert(s, SSL3_AL_FATAL, al);
err:
  OPENSSL_free(encodedPoint);
  BN_CTX_free(bn_ctx);
  EVP_MD_CTX_cleanup(&md_ctx);
  return -1;
}

int ssl3_send_certificate_request(SSL *s) {
  uint8_t *p, *d;
  size_t i;
  int j, nl, off, n;
  STACK_OF(X509_NAME) *sk = NULL;
  X509_NAME *name;
  BUF_MEM *buf;

  if (s->state == SSL3_ST_SW_CERT_REQ_A) {
    buf = s->init_buf;

    d = p = ssl_handshake_start(s);

    /* get the list of acceptable cert types */
    p++;
    n = ssl3_get_req_cert_type(s, p);
    d[0] = n;
    p += n;
    n++;

    if (SSL_USE_SIGALGS(s)) {
      const uint8_t *psigs;
      nl = tls12_get_psigalgs(s, &psigs);
      s2n(nl, p);
      memcpy(p, psigs, nl);
      p += nl;
      n += nl + 2;
    }

    off = n;
    p += 2;
    n += 2;

    sk = SSL_get_client_CA_list(s);
    nl = 0;
    if (sk != NULL) {
      for (i = 0; i < sk_X509_NAME_num(sk); i++) {
        name = sk_X509_NAME_value(sk, i);
        j = i2d_X509_NAME(name, NULL);
        if (!BUF_MEM_grow_clean(buf, SSL_HM_HEADER_LENGTH(s) + n + j + 2)) {
          OPENSSL_PUT_ERROR(SSL, ERR_R_BUF_LIB);
          goto err;
        }
        p = ssl_handshake_start(s) + n;
        s2n(j, p);
        i2d_X509_NAME(name, &p);
        n += 2 + j;
        nl += 2 + j;
      }
    }

    /* else no CA names */
    p = ssl_handshake_start(s) + off;
    s2n(nl, p);

    if (!ssl_set_handshake_header(s, SSL3_MT_CERTIFICATE_REQUEST, n)) {
      goto err;
    }
    s->state = SSL3_ST_SW_CERT_REQ_B;
  }

  /* SSL3_ST_SW_CERT_REQ_B */
  return ssl_do_write(s);

err:
  return -1;
}

int ssl3_get_client_key_exchange(SSL *s) {
  int al, ok;
  long n;
  CBS client_key_exchange;
  uint32_t alg_k;
  uint32_t alg_a;
  uint8_t *premaster_secret = NULL;
  size_t premaster_secret_len = 0;
  RSA *rsa = NULL;
  uint8_t *decrypt_buf = NULL;
  EVP_PKEY *pkey = NULL;
  BIGNUM *pub = NULL;
  DH *dh_srvr;

  EC_KEY *srvr_ecdh = NULL;
  EVP_PKEY *clnt_pub_pkey = NULL;
  EC_POINT *clnt_ecpoint = NULL;
  BN_CTX *bn_ctx = NULL;
  unsigned int psk_len = 0;
  uint8_t psk[PSK_MAX_PSK_LEN];

  n = s->method->ssl_get_message(s, SSL3_ST_SR_KEY_EXCH_A,
                                 SSL3_ST_SR_KEY_EXCH_B,
                                 SSL3_MT_CLIENT_KEY_EXCHANGE, 2048, /* ??? */
                                 ssl_hash_message, &ok);

  if (!ok) {
    return n;
  }

  CBS_init(&client_key_exchange, s->init_msg, n);

  alg_k = s->s3->tmp.new_cipher->algorithm_mkey;
  alg_a = s->s3->tmp.new_cipher->algorithm_auth;

  /* If using a PSK key exchange, prepare the pre-shared key. */
  if (alg_a & SSL_aPSK) {
    CBS psk_identity;

    /* If using PSK, the ClientKeyExchange contains a psk_identity. If PSK,
     * then this is the only field in the message. */
    if (!CBS_get_u16_length_prefixed(&client_key_exchange, &psk_identity) ||
        ((alg_k & SSL_kPSK) && CBS_len(&client_key_exchange) != 0)) {
      OPENSSL_PUT_ERROR(SSL, SSL_R_DECODE_ERROR);
      al = SSL_AD_DECODE_ERROR;
      goto f_err;
    }

    if (s->psk_server_callback == NULL) {
      OPENSSL_PUT_ERROR(SSL, SSL_R_PSK_NO_SERVER_CB);
      al = SSL_AD_INTERNAL_ERROR;
      goto f_err;
    }

    if (CBS_len(&psk_identity) > PSK_MAX_IDENTITY_LEN ||
        CBS_contains_zero_byte(&psk_identity)) {
      OPENSSL_PUT_ERROR(SSL, SSL_R_DATA_LENGTH_TOO_LONG);
      al = SSL_AD_ILLEGAL_PARAMETER;
      goto f_err;
    }

    if (!CBS_strdup(&psk_identity, &s->session->psk_identity)) {
      al = SSL_AD_INTERNAL_ERROR;
      OPENSSL_PUT_ERROR(SSL, ERR_R_MALLOC_FAILURE);
      goto f_err;
    }

    /* Look up the key for the identity. */
    psk_len =
        s->psk_server_callback(s, s->session->psk_identity, psk, sizeof(psk));
    if (psk_len > PSK_MAX_PSK_LEN) {
      OPENSSL_PUT_ERROR(SSL, ERR_R_INTERNAL_ERROR);
      al = SSL_AD_INTERNAL_ERROR;
      goto f_err;
    } else if (psk_len == 0) {
      /* PSK related to the given identity not found */
      OPENSSL_PUT_ERROR(SSL, SSL_R_PSK_IDENTITY_NOT_FOUND);
      al = SSL_AD_UNKNOWN_PSK_IDENTITY;
      goto f_err;
    }
  }

  /* Depending on the key exchange method, compute |premaster_secret| and
   * |premaster_secret_len|. */
  if (alg_k & SSL_kRSA) {
    CBS encrypted_premaster_secret;
    uint8_t rand_premaster_secret[SSL_MAX_MASTER_KEY_LENGTH];
    uint8_t good;
    size_t rsa_size, decrypt_len, premaster_index, j;

    pkey = s->cert->privatekey;
    if (pkey == NULL || pkey->type != EVP_PKEY_RSA || pkey->pkey.rsa == NULL) {
      al = SSL_AD_HANDSHAKE_FAILURE;
      OPENSSL_PUT_ERROR(SSL, SSL_R_MISSING_RSA_CERTIFICATE);
      goto f_err;
    }
    rsa = pkey->pkey.rsa;

    /* TLS and [incidentally] DTLS{0xFEFF} */
    if (s->version > SSL3_VERSION) {
      CBS copy = client_key_exchange;
      if (!CBS_get_u16_length_prefixed(&client_key_exchange,
                                       &encrypted_premaster_secret) ||
          CBS_len(&client_key_exchange) != 0) {
        if (!(s->options & SSL_OP_TLS_D5_BUG)) {
          al = SSL_AD_DECODE_ERROR;
          OPENSSL_PUT_ERROR(SSL, SSL_R_TLS_RSA_ENCRYPTED_VALUE_LENGTH_IS_WRONG);
          goto f_err;
        } else {
          encrypted_premaster_secret = copy;
        }
      }
    } else {
      encrypted_premaster_secret = client_key_exchange;
    }

    /* Reject overly short RSA keys because we want to be sure that the buffer
     * size makes it safe to iterate over the entire size of a premaster secret
     * (SSL_MAX_MASTER_KEY_LENGTH). The actual expected size is larger due to
     * RSA padding, but the bound is sufficient to be safe. */
    rsa_size = RSA_size(rsa);
    if (rsa_size < SSL_MAX_MASTER_KEY_LENGTH) {
      al = SSL_AD_DECRYPT_ERROR;
      OPENSSL_PUT_ERROR(SSL, SSL_R_DECRYPTION_FAILED);
      goto f_err;
    }

    /* We must not leak whether a decryption failure occurs because of
     * Bleichenbacher's attack on PKCS #1 v1.5 RSA padding (see RFC 2246,
     * section 7.4.7.1). The code follows that advice of the TLS RFC and
     * generates a random premaster secret for the case that the decrypt fails.
     * See https://tools.ietf.org/html/rfc5246#section-7.4.7.1 */
    if (!RAND_bytes(rand_premaster_secret, sizeof(rand_premaster_secret))) {
      goto err;
    }

    /* Allocate a buffer large enough for an RSA decryption. */
    decrypt_buf = OPENSSL_malloc(rsa_size);
    if (decrypt_buf == NULL) {
      OPENSSL_PUT_ERROR(SSL, ERR_R_MALLOC_FAILURE);
      goto err;
    }

    /* Decrypt with no padding. PKCS#1 padding will be removed as part of the
     * timing-sensitive code below. */
    if (!RSA_decrypt(rsa, &decrypt_len, decrypt_buf, rsa_size,
                     CBS_data(&encrypted_premaster_secret),
                     CBS_len(&encrypted_premaster_secret), RSA_NO_PADDING)) {
      goto err;
    }
    if (decrypt_len != rsa_size) {
      /* This should never happen, but do a check so we do not read
       * uninitialized memory. */
      OPENSSL_PUT_ERROR(SSL, ERR_R_INTERNAL_ERROR);
      goto err;
    }

    /* Remove the PKCS#1 padding and adjust |decrypt_len| as appropriate.
     * |good| will be 0xff if the premaster is acceptable and zero otherwise.
     * */
    good =
        constant_time_eq_int_8(RSA_message_index_PKCS1_type_2(
                                   decrypt_buf, decrypt_len, &premaster_index),
                               1);
    decrypt_len = decrypt_len - premaster_index;

    /* decrypt_len should be SSL_MAX_MASTER_KEY_LENGTH. */
    good &= constant_time_eq_8(decrypt_len, SSL_MAX_MASTER_KEY_LENGTH);

    /* Copy over the unpadded premaster. Whatever the value of
     * |decrypt_good_mask|, copy as if the premaster were the right length. It
     * is important the memory access pattern be constant. */
    premaster_secret =
        BUF_memdup(decrypt_buf + (rsa_size - SSL_MAX_MASTER_KEY_LENGTH),
                   SSL_MAX_MASTER_KEY_LENGTH);
    if (premaster_secret == NULL) {
      OPENSSL_PUT_ERROR(SSL, ERR_R_MALLOC_FAILURE);
      goto err;
    }
    OPENSSL_free(decrypt_buf);
    decrypt_buf = NULL;

    /* If the version in the decrypted pre-master secret is correct then
     * version_good will be 0xff, otherwise it'll be zero. The
     * Klima-Pokorny-Rosa extension of Bleichenbacher's attack
     * (http://eprint.iacr.org/2003/052/) exploits the version number check as
     * a "bad version oracle". Thus version checks are done in constant time
     * and are treated like any other decryption error. */
    good &= constant_time_eq_8(premaster_secret[0],
                               (unsigned)(s->client_version >> 8));
    good &= constant_time_eq_8(premaster_secret[1],
                               (unsigned)(s->client_version & 0xff));

    /* Now copy rand_premaster_secret over premaster_secret using
     * decrypt_good_mask. */
    for (j = 0; j < sizeof(rand_premaster_secret); j++) {
      premaster_secret[j] = constant_time_select_8(good, premaster_secret[j],
                                                   rand_premaster_secret[j]);
    }

    premaster_secret_len = sizeof(rand_premaster_secret);
  } else if (alg_k & SSL_kDHE) {
    CBS dh_Yc;
    int dh_len;

    if (!CBS_get_u16_length_prefixed(&client_key_exchange, &dh_Yc) ||
        CBS_len(&dh_Yc) == 0 || CBS_len(&client_key_exchange) != 0) {
      OPENSSL_PUT_ERROR(SSL, SSL_R_DH_PUBLIC_VALUE_LENGTH_IS_WRONG);
      al = SSL_R_DECODE_ERROR;
      goto f_err;
    }

    if (s->s3->tmp.dh == NULL) {
      al = SSL_AD_HANDSHAKE_FAILURE;
      OPENSSL_PUT_ERROR(SSL, SSL_R_MISSING_TMP_DH_KEY);
      goto f_err;
    }
    dh_srvr = s->s3->tmp.dh;

    pub = BN_bin2bn(CBS_data(&dh_Yc), CBS_len(&dh_Yc), NULL);
    if (pub == NULL) {
      OPENSSL_PUT_ERROR(SSL, SSL_R_BN_LIB);
      goto err;
    }

    /* Allocate a buffer for the premaster secret. */
    premaster_secret = OPENSSL_malloc(DH_size(dh_srvr));
    if (premaster_secret == NULL) {
      OPENSSL_PUT_ERROR(SSL, ERR_R_MALLOC_FAILURE);
      BN_clear_free(pub);
      goto err;
    }

    dh_len = DH_compute_key(premaster_secret, pub, dh_srvr);
    if (dh_len <= 0) {
      OPENSSL_PUT_ERROR(SSL, ERR_R_DH_LIB);
      BN_clear_free(pub);
      goto err;
    }

    DH_free(s->s3->tmp.dh);
    s->s3->tmp.dh = NULL;
    BN_clear_free(pub);
    pub = NULL;

    premaster_secret_len = dh_len;
  } else if (alg_k & SSL_kECDHE) {
    int field_size = 0, ecdh_len;
    const EC_KEY *tkey;
    const EC_GROUP *group;
    const BIGNUM *priv_key;
    CBS ecdh_Yc;

    /* initialize structures for server's ECDH key pair */
    srvr_ecdh = EC_KEY_new();
    if (srvr_ecdh == NULL) {
      OPENSSL_PUT_ERROR(SSL, ERR_R_MALLOC_FAILURE);
      goto err;
    }

    /* Use the ephermeral values we saved when generating the ServerKeyExchange
     * msg. */
    tkey = s->s3->tmp.ecdh;

    group = EC_KEY_get0_group(tkey);
    priv_key = EC_KEY_get0_private_key(tkey);

    if (!EC_KEY_set_group(srvr_ecdh, group) ||
        !EC_KEY_set_private_key(srvr_ecdh, priv_key)) {
      OPENSSL_PUT_ERROR(SSL, ERR_R_EC_LIB);
      goto err;
    }

    /* Let's get client's public key */
    clnt_ecpoint = EC_POINT_new(group);
    if (clnt_ecpoint == NULL) {
      OPENSSL_PUT_ERROR(SSL, ERR_R_MALLOC_FAILURE);
      goto err;
    }

    /* Get client's public key from encoded point in the ClientKeyExchange
     * message. */
    if (!CBS_get_u8_length_prefixed(&client_key_exchange, &ecdh_Yc) ||
        CBS_len(&client_key_exchange) != 0) {
      al = SSL_AD_DECODE_ERROR;
      OPENSSL_PUT_ERROR(SSL, SSL_R_DECODE_ERROR);
      goto f_err;
    }

    bn_ctx = BN_CTX_new();
    if (bn_ctx == NULL) {
      OPENSSL_PUT_ERROR(SSL, ERR_R_MALLOC_FAILURE);
      goto err;
    }

    if (!EC_POINT_oct2point(group, clnt_ecpoint, CBS_data(&ecdh_Yc),
                            CBS_len(&ecdh_Yc), bn_ctx)) {
      OPENSSL_PUT_ERROR(SSL, ERR_R_EC_LIB);
      goto err;
    }

    /* Allocate a buffer for both the secret and the PSK. */
    field_size = EC_GROUP_get_degree(group);
    if (field_size <= 0) {
      OPENSSL_PUT_ERROR(SSL, ERR_R_ECDH_LIB);
      goto err;
    }

    ecdh_len = (field_size + 7) / 8;
    premaster_secret = OPENSSL_malloc(ecdh_len);
    if (premaster_secret == NULL) {
      OPENSSL_PUT_ERROR(SSL, ERR_R_MALLOC_FAILURE);
      goto err;
    }

    /* Compute the shared pre-master secret */
    ecdh_len = ECDH_compute_key(premaster_secret, ecdh_len, clnt_ecpoint,
                                srvr_ecdh, NULL);
    if (ecdh_len <= 0) {
      OPENSSL_PUT_ERROR(SSL, ERR_R_ECDH_LIB);
      goto err;
    }

    EVP_PKEY_free(clnt_pub_pkey);
    clnt_pub_pkey = NULL;
    EC_POINT_free(clnt_ecpoint);
    clnt_ecpoint = NULL;
    EC_KEY_free(srvr_ecdh);
    srvr_ecdh = NULL;
    BN_CTX_free(bn_ctx);
    bn_ctx = NULL;
    EC_KEY_free(s->s3->tmp.ecdh);
    s->s3->tmp.ecdh = NULL;

    premaster_secret_len = ecdh_len;
  } else if (alg_k & SSL_kPSK) {
    /* For plain PSK, other_secret is a block of 0s with the same length as the
     * pre-shared key. */
    premaster_secret_len = psk_len;
    premaster_secret = OPENSSL_malloc(premaster_secret_len);
    if (premaster_secret == NULL) {
      OPENSSL_PUT_ERROR(SSL, ERR_R_MALLOC_FAILURE);
      goto err;
    }
    memset(premaster_secret, 0, premaster_secret_len);
  } else {
    al = SSL_AD_HANDSHAKE_FAILURE;
    OPENSSL_PUT_ERROR(SSL, SSL_R_UNKNOWN_CIPHER_TYPE);
    goto f_err;
  }

  /* For a PSK cipher suite, the actual pre-master secret is combined with the
   * pre-shared key. */
  if (alg_a & SSL_aPSK) {
    CBB new_premaster, child;
    uint8_t *new_data;
    size_t new_len;

    CBB_zero(&new_premaster);
    if (!CBB_init(&new_premaster, 2 + psk_len + 2 + premaster_secret_len) ||
        !CBB_add_u16_length_prefixed(&new_premaster, &child) ||
        !CBB_add_bytes(&child, premaster_secret, premaster_secret_len) ||
        !CBB_add_u16_length_prefixed(&new_premaster, &child) ||
        !CBB_add_bytes(&child, psk, psk_len) ||
        !CBB_finish(&new_premaster, &new_data, &new_len)) {
      OPENSSL_PUT_ERROR(SSL, ERR_R_MALLOC_FAILURE);
      CBB_cleanup(&new_premaster);
      goto err;
    }

    OPENSSL_cleanse(premaster_secret, premaster_secret_len);
    OPENSSL_free(premaster_secret);
    premaster_secret = new_data;
    premaster_secret_len = new_len;
  }

  /* Compute the master secret */
  s->session->master_key_length = s->enc_method->generate_master_secret(
      s, s->session->master_key, premaster_secret, premaster_secret_len);
  if (s->session->master_key_length == 0) {
    goto err;
  }
  s->session->extended_master_secret = s->s3->tmp.extended_master_secret;

  OPENSSL_cleanse(premaster_secret, premaster_secret_len);
  OPENSSL_free(premaster_secret);
  return 1;

f_err:
  ssl3_send_alert(s, SSL3_AL_FATAL, al);
err:
  if (premaster_secret) {
    if (premaster_secret_len) {
      OPENSSL_cleanse(premaster_secret, premaster_secret_len);
    }
    OPENSSL_free(premaster_secret);
  }
  OPENSSL_free(decrypt_buf);
  EVP_PKEY_free(clnt_pub_pkey);
  EC_POINT_free(clnt_ecpoint);
  EC_KEY_free(srvr_ecdh);
  BN_CTX_free(bn_ctx);

  return -1;
}

int ssl3_get_cert_verify(SSL *s) {
  int al, ok, ret = 0;
  long n;
  CBS certificate_verify, signature;
  X509 *peer = s->session->peer;
  EVP_PKEY *pkey = NULL;
  const EVP_MD *md = NULL;
  uint8_t digest[EVP_MAX_MD_SIZE];
  size_t digest_length;
  EVP_PKEY_CTX *pctx = NULL;

  /* Only RSA and ECDSA client certificates are supported, so a
   * CertificateVerify is required if and only if there's a client certificate.
   * */
  if (peer == NULL) {
    ssl3_free_handshake_buffer(s);
    return 1;
  }

  n = s->method->ssl_get_message(
      s, SSL3_ST_SR_CERT_VRFY_A, SSL3_ST_SR_CERT_VRFY_B,
      SSL3_MT_CERTIFICATE_VERIFY, SSL3_RT_MAX_PLAIN_LENGTH,
      ssl_dont_hash_message, &ok);

  if (!ok) {
    return n;
  }

  /* Filter out unsupported certificate types. */
  pkey = X509_get_pubkey(peer);
  if (pkey == NULL) {
    goto err;
  }
  if (!(X509_certificate_type(peer, pkey) & EVP_PKT_SIGN) ||
      (pkey->type != EVP_PKEY_RSA && pkey->type != EVP_PKEY_EC)) {
    al = SSL_AD_UNSUPPORTED_CERTIFICATE;
    OPENSSL_PUT_ERROR(SSL, SSL_R_PEER_ERROR_UNSUPPORTED_CERTIFICATE_TYPE);
    goto f_err;
  }

  CBS_init(&certificate_verify, s->init_msg, n);

  /* Determine the digest type if needbe. */
  if (SSL_USE_SIGALGS(s) &&
      !tls12_check_peer_sigalg(&md, &al, s, &certificate_verify, pkey)) {
    goto f_err;
  }

  /* Compute the digest. */
  if (!ssl3_cert_verify_hash(s, digest, &digest_length, &md, pkey->type)) {
    goto err;
  }

  /* The handshake buffer is no longer necessary, and we may hash the current
   * message.*/
  ssl3_free_handshake_buffer(s);
  if (!ssl3_hash_current_message(s)) {
    goto err;
  }

  /* Parse and verify the signature. */
  if (!CBS_get_u16_length_prefixed(&certificate_verify, &signature) ||
      CBS_len(&certificate_verify) != 0) {
    al = SSL_AD_DECODE_ERROR;
    OPENSSL_PUT_ERROR(SSL, SSL_R_DECODE_ERROR);
    goto f_err;
  }

  pctx = EVP_PKEY_CTX_new(pkey, NULL);
  if (pctx == NULL) {
    goto err;
  }
  if (!EVP_PKEY_verify_init(pctx) ||
      !EVP_PKEY_CTX_set_signature_md(pctx, md) ||
      !EVP_PKEY_verify(pctx, CBS_data(&signature), CBS_len(&signature), digest,
                       digest_length)) {
    al = SSL_AD_DECRYPT_ERROR;
    OPENSSL_PUT_ERROR(SSL, SSL_R_BAD_SIGNATURE);
    goto f_err;
  }

  ret = 1;

  if (0) {
  f_err:
    ssl3_send_alert(s, SSL3_AL_FATAL, al);
  }

err:
  EVP_PKEY_CTX_free(pctx);
  EVP_PKEY_free(pkey);

  return ret;
}

int ssl3_get_client_certificate(SSL *s) {
  int i, ok, al, ret = -1;
  X509 *x = NULL;
  unsigned long n;
  STACK_OF(X509) *sk = NULL;
  SHA256_CTX sha256;
  CBS certificate_msg, certificate_list;
  int is_first_certificate = 1;

  n = s->method->ssl_get_message(s, SSL3_ST_SR_CERT_A, SSL3_ST_SR_CERT_B, -1,
                                 (long)s->max_cert_list, ssl_hash_message, &ok);

  if (!ok) {
    return n;
  }

  if (s->s3->tmp.message_type == SSL3_MT_CLIENT_KEY_EXCHANGE) {
    if ((s->verify_mode & SSL_VERIFY_PEER) &&
        (s->verify_mode & SSL_VERIFY_FAIL_IF_NO_PEER_CERT)) {
      OPENSSL_PUT_ERROR(SSL, SSL_R_PEER_DID_NOT_RETURN_A_CERTIFICATE);
      al = SSL_AD_HANDSHAKE_FAILURE;
      goto f_err;
    }

    /* If tls asked for a client cert, the client must return a 0 list */
    if (s->version > SSL3_VERSION && s->s3->tmp.cert_request) {
      OPENSSL_PUT_ERROR(SSL,
                        SSL_R_TLS_PEER_DID_NOT_RESPOND_WITH_CERTIFICATE_LIST);
      al = SSL_AD_UNEXPECTED_MESSAGE;
      goto f_err;
    }
    s->s3->tmp.reuse_message = 1;

    return 1;
  }

  if (s->s3->tmp.message_type != SSL3_MT_CERTIFICATE) {
    al = SSL_AD_UNEXPECTED_MESSAGE;
    OPENSSL_PUT_ERROR(SSL, SSL_R_WRONG_MESSAGE_TYPE);
    goto f_err;
  }

  CBS_init(&certificate_msg, s->init_msg, n);

  sk = sk_X509_new_null();
  if (sk == NULL) {
    OPENSSL_PUT_ERROR(SSL, ERR_R_MALLOC_FAILURE);
    goto err;
  }

  if (!CBS_get_u24_length_prefixed(&certificate_msg, &certificate_list) ||
      CBS_len(&certificate_msg) != 0) {
    al = SSL_AD_DECODE_ERROR;
    OPENSSL_PUT_ERROR(SSL, SSL_R_DECODE_ERROR);
    goto f_err;
  }

  while (CBS_len(&certificate_list) > 0) {
    CBS certificate;
    const uint8_t *data;

    if (!CBS_get_u24_length_prefixed(&certificate_list, &certificate)) {
      al = SSL_AD_DECODE_ERROR;
      OPENSSL_PUT_ERROR(SSL, SSL_R_DECODE_ERROR);
      goto f_err;
    }

    if (is_first_certificate && s->ctx->retain_only_sha256_of_client_certs) {
      /* If this is the first certificate, and we don't want to keep peer
       * certificates in memory, then we hash it right away. */
      SHA256_Init(&sha256);
      SHA256_Update(&sha256, CBS_data(&certificate), CBS_len(&certificate));
      SHA256_Final(s->session->peer_sha256, &sha256);
      s->session->peer_sha256_valid = 1;
    }
    is_first_certificate = 0;

    data = CBS_data(&certificate);
    x = d2i_X509(NULL, &data, CBS_len(&certificate));
    if (x == NULL) {
      al = SSL_AD_BAD_CERTIFICATE;
      OPENSSL_PUT_ERROR(SSL, ERR_R_ASN1_LIB);
      goto f_err;
    }
    if (data != CBS_data(&certificate) + CBS_len(&certificate)) {
      al = SSL_AD_DECODE_ERROR;
      OPENSSL_PUT_ERROR(SSL, SSL_R_CERT_LENGTH_MISMATCH);
      goto f_err;
    }
    if (!sk_X509_push(sk, x)) {
      OPENSSL_PUT_ERROR(SSL, ERR_R_MALLOC_FAILURE);
      goto err;
    }
    x = NULL;
  }

  if (sk_X509_num(sk) <= 0) {
    /* No client certificate so the handshake buffer may be discarded. */
    ssl3_free_handshake_buffer(s);

    /* TLS does not mind 0 certs returned */
    if (s->version == SSL3_VERSION) {
      al = SSL_AD_HANDSHAKE_FAILURE;
      OPENSSL_PUT_ERROR(SSL, SSL_R_NO_CERTIFICATES_RETURNED);
      goto f_err;
    } else if ((s->verify_mode & SSL_VERIFY_PEER) &&
             (s->verify_mode & SSL_VERIFY_FAIL_IF_NO_PEER_CERT)) {
      /* Fail for TLS only if we required a certificate */
      OPENSSL_PUT_ERROR(SSL, SSL_R_PEER_DID_NOT_RETURN_A_CERTIFICATE);
      al = SSL_AD_HANDSHAKE_FAILURE;
      goto f_err;
    }
  } else {
    i = ssl_verify_cert_chain(s, sk);
    if (i <= 0) {
      al = ssl_verify_alarm_type(s->verify_result);
      OPENSSL_PUT_ERROR(SSL, SSL_R_CERTIFICATE_VERIFY_FAILED);
      goto f_err;
    }
  }

  X509_free(s->session->peer);
  s->session->peer = sk_X509_shift(sk);
  s->session->verify_result = s->verify_result;

  sk_X509_pop_free(s->session->cert_chain, X509_free);
  s->session->cert_chain = sk;
  /* Inconsistency alert: cert_chain does *not* include the peer's own
   * certificate, while we do include it in s3_clnt.c */

  sk = NULL;

  ret = 1;

  if (0) {
  f_err:
    ssl3_send_alert(s, SSL3_AL_FATAL, al);
  }

err:
  X509_free(x);
  sk_X509_pop_free(sk, X509_free);
  return ret;
}

int ssl3_send_server_certificate(SSL *s) {
  if (s->state == SSL3_ST_SW_CERT_A) {
    if (!ssl3_output_cert_chain(s)) {
      return 0;
    }
    s->state = SSL3_ST_SW_CERT_B;
  }

  /* SSL3_ST_SW_CERT_B */
  return ssl_do_write(s);
}

/* send a new session ticket (not necessarily for a new session) */
int ssl3_send_new_session_ticket(SSL *s) {
  int ret = -1;
  uint8_t *session = NULL;
  size_t session_len;
  EVP_CIPHER_CTX ctx;
  HMAC_CTX hctx;

  EVP_CIPHER_CTX_init(&ctx);
  HMAC_CTX_init(&hctx);

  if (s->state == SSL3_ST_SW_SESSION_TICKET_A) {
    uint8_t *p, *macstart;
    int len;
    unsigned int hlen;
    SSL_CTX *tctx = s->initial_ctx;
    uint8_t iv[EVP_MAX_IV_LENGTH];
    uint8_t key_name[16];
    /* The maximum overhead of encrypting the session is 16 (key name) + IV +
     * one block of encryption overhead + HMAC.  */
    const size_t max_ticket_overhead =
        16 + EVP_MAX_IV_LENGTH + EVP_MAX_BLOCK_LENGTH + EVP_MAX_MD_SIZE;

    /* Serialize the SSL_SESSION to be encoded into the ticket. */
    if (!SSL_SESSION_to_bytes_for_ticket(s->session, &session, &session_len)) {
      goto err;
    }

    /* If the session is too long, emit a dummy value rather than abort the
     * connection. */
    if (session_len > 0xFFFF - max_ticket_overhead) {
      static const char kTicketPlaceholder[] = "TICKET TOO LARGE";
      const size_t placeholder_len = strlen(kTicketPlaceholder);

      OPENSSL_free(session);
      session = NULL;

      p = ssl_handshake_start(s);
      /* Emit ticket_lifetime_hint. */
      l2n(0, p);
      /* Emit ticket. */
      s2n(placeholder_len, p);
      memcpy(p, kTicketPlaceholder, placeholder_len);
      p += placeholder_len;

      len = p - ssl_handshake_start(s);
      if (!ssl_set_handshake_header(s, SSL3_MT_NEWSESSION_TICKET, len)) {
        goto err;
      }
      s->state = SSL3_ST_SW_SESSION_TICKET_B;
      return ssl_do_write(s);
    }

    /* Grow buffer if need be: the length calculation is as follows:
     * handshake_header_length + 4 (ticket lifetime hint) + 2 (ticket length) +
     * max_ticket_overhead + * session_length */
    if (!BUF_MEM_grow(s->init_buf, SSL_HM_HEADER_LENGTH(s) + 6 +
                                       max_ticket_overhead + session_len)) {
      goto err;
    }
    p = ssl_handshake_start(s);
    /* Initialize HMAC and cipher contexts. If callback present it does all the
     * work otherwise use generated values from parent ctx. */
    if (tctx->tlsext_ticket_key_cb) {
      if (tctx->tlsext_ticket_key_cb(s, key_name, iv, &ctx, &hctx,
                                     1 /* encrypt */) < 0) {
        goto err;
      }
    } else {
      if (!RAND_bytes(iv, 16) ||
          !EVP_EncryptInit_ex(&ctx, EVP_aes_128_cbc(), NULL,
                              tctx->tlsext_tick_aes_key, iv) ||
          !HMAC_Init_ex(&hctx, tctx->tlsext_tick_hmac_key, 16, tlsext_tick_md(),
                        NULL)) {
        goto err;
      }
      memcpy(key_name, tctx->tlsext_tick_key_name, 16);
    }

    /* Ticket lifetime hint (advisory only): We leave this unspecified for
     * resumed session (for simplicity), and guess that tickets for new
     * sessions will live as long as their sessions. */
    l2n(s->hit ? 0 : s->session->timeout, p);

    /* Skip ticket length for now */
    p += 2;
    /* Output key name */
    macstart = p;
    memcpy(p, key_name, 16);
    p += 16;
    /* output IV */
    memcpy(p, iv, EVP_CIPHER_CTX_iv_length(&ctx));
    p += EVP_CIPHER_CTX_iv_length(&ctx);
    /* Encrypt session data */
    if (!EVP_EncryptUpdate(&ctx, p, &len, session, session_len)) {
      goto err;
    }
    p += len;
    if (!EVP_EncryptFinal_ex(&ctx, p, &len)) {
      goto err;
    }
    p += len;

    if (!HMAC_Update(&hctx, macstart, p - macstart) ||
        !HMAC_Final(&hctx, p, &hlen)) {
      goto err;
    }

    p += hlen;
    /* Now write out lengths: p points to end of data written */
    /* Total length */
    len = p - ssl_handshake_start(s);
    /* Skip ticket lifetime hint */
    p = ssl_handshake_start(s) + 4;
    s2n(len - 6, p);
    if (!ssl_set_handshake_header(s, SSL3_MT_NEWSESSION_TICKET, len)) {
      goto err;
    }
    s->state = SSL3_ST_SW_SESSION_TICKET_B;
  }

  /* SSL3_ST_SW_SESSION_TICKET_B */
  ret = ssl_do_write(s);

err:
  OPENSSL_free(session);
  EVP_CIPHER_CTX_cleanup(&ctx);
  HMAC_CTX_cleanup(&hctx);
  return ret;
}

/* ssl3_get_next_proto reads a Next Protocol Negotiation handshake message. It
 * sets the next_proto member in s if found */
int ssl3_get_next_proto(SSL *s) {
  int ok;
  long n;
  CBS next_protocol, selected_protocol, padding;

  /* Clients cannot send a NextProtocol message if we didn't see the extension
   * in their ClientHello */
  if (!s->s3->next_proto_neg_seen) {
    OPENSSL_PUT_ERROR(SSL, SSL_R_GOT_NEXT_PROTO_WITHOUT_EXTENSION);
    return -1;
  }

  n = s->method->ssl_get_message(s, SSL3_ST_SR_NEXT_PROTO_A,
                                 SSL3_ST_SR_NEXT_PROTO_B, SSL3_MT_NEXT_PROTO,
                                 514, /* See the payload format below */
                                 ssl_hash_message, &ok);

  if (!ok) {
    return n;
  }

  /* s->state doesn't reflect whether ChangeCipherSpec has been received in
   * this handshake, but s->s3->change_cipher_spec does (will be reset by
   * ssl3_get_finished).
   *
   * TODO(davidben): Is this check now redundant with
   * SSL3_FLAGS_EXPECT_CCS? */
  if (!s->s3->change_cipher_spec) {
    OPENSSL_PUT_ERROR(SSL, SSL_R_GOT_NEXT_PROTO_BEFORE_A_CCS);
    return -1;
  }

  CBS_init(&next_protocol, s->init_msg, n);

  /* The payload looks like:
   *   uint8 proto_len;
   *   uint8 proto[proto_len];
   *   uint8 padding_len;
   *   uint8 padding[padding_len]; */
  if (!CBS_get_u8_length_prefixed(&next_protocol, &selected_protocol) ||
      !CBS_get_u8_length_prefixed(&next_protocol, &padding) ||
      CBS_len(&next_protocol) != 0 ||
      !CBS_stow(&selected_protocol, &s->next_proto_negotiated,
                &s->next_proto_negotiated_len)) {
    return 0;
  }

  return 1;
}

/* ssl3_get_channel_id reads and verifies a ClientID handshake message. */
int ssl3_get_channel_id(SSL *s) {
  int ret = -1, ok;
  long n;
  uint8_t channel_id_hash[EVP_MAX_MD_SIZE];
  size_t channel_id_hash_len;
  const uint8_t *p;
  uint16_t extension_type;
  EC_GROUP *p256 = NULL;
  EC_KEY *key = NULL;
  EC_POINT *point = NULL;
  ECDSA_SIG sig;
  BIGNUM x, y;
  CBS encrypted_extensions, extension;

  n = s->method->ssl_get_message(
      s, SSL3_ST_SR_CHANNEL_ID_A, SSL3_ST_SR_CHANNEL_ID_B,
      SSL3_MT_ENCRYPTED_EXTENSIONS, 2 + 2 + TLSEXT_CHANNEL_ID_SIZE,
      ssl_dont_hash_message, &ok);

  if (!ok) {
    return n;
  }

  /* Before incorporating the EncryptedExtensions message to the handshake
   * hash, compute the hash that should have been signed. */
  if (!tls1_channel_id_hash(s, channel_id_hash, &channel_id_hash_len)) {
    return -1;
  }
  assert(channel_id_hash_len == SHA256_DIGEST_LENGTH);

  if (!ssl3_hash_current_message(s)) {
    return -1;
  }

  /* s->state doesn't reflect whether ChangeCipherSpec has been received in
   * this handshake, but s->s3->change_cipher_spec does (will be reset by
   * ssl3_get_finished).
   *
   * TODO(davidben): Is this check now redundant with SSL3_FLAGS_EXPECT_CCS? */
  if (!s->s3->change_cipher_spec) {
    OPENSSL_PUT_ERROR(SSL, SSL_R_GOT_CHANNEL_ID_BEFORE_A_CCS);
    return -1;
  }

  CBS_init(&encrypted_extensions, s->init_msg, n);

  /* EncryptedExtensions could include multiple extensions, but the only
   * extension that could be negotiated is ChannelID, so there can only be one
   * entry.
   *
   * The payload looks like:
   *   uint16 extension_type
   *   uint16 extension_len;
   *   uint8 x[32];
   *   uint8 y[32];
   *   uint8 r[32];
   *   uint8 s[32]; */

  if (!CBS_get_u16(&encrypted_extensions, &extension_type) ||
      !CBS_get_u16_length_prefixed(&encrypted_extensions, &extension) ||
      CBS_len(&encrypted_extensions) != 0 ||
      extension_type != TLSEXT_TYPE_channel_id ||
      CBS_len(&extension) != TLSEXT_CHANNEL_ID_SIZE) {
    OPENSSL_PUT_ERROR(SSL, SSL_R_INVALID_MESSAGE);
    return -1;
  }

  p256 = EC_GROUP_new_by_curve_name(NID_X9_62_prime256v1);
  if (!p256) {
    OPENSSL_PUT_ERROR(SSL, SSL_R_NO_P256_SUPPORT);
    return -1;
  }

  BN_init(&x);
  BN_init(&y);
  sig.r = BN_new();
  sig.s = BN_new();
  if (sig.r == NULL || sig.s == NULL) {
    goto err;
  }

  p = CBS_data(&extension);
  if (BN_bin2bn(p + 0, 32, &x) == NULL ||
      BN_bin2bn(p + 32, 32, &y) == NULL ||
      BN_bin2bn(p + 64, 32, sig.r) == NULL ||
      BN_bin2bn(p + 96, 32, sig.s) == NULL) {
    goto err;
  }

  point = EC_POINT_new(p256);
  if (!point || !EC_POINT_set_affine_coordinates_GFp(p256, point, &x, &y, NULL)) {
    goto err;
  }

  key = EC_KEY_new();
  if (!key || !EC_KEY_set_group(key, p256) ||
      !EC_KEY_set_public_key(key, point)) {
    goto err;
  }

  /* We stored the handshake hash in |tlsext_channel_id| the first time that we
   * were called. */
  if (!ECDSA_do_verify(channel_id_hash, channel_id_hash_len, &sig, key)) {
    OPENSSL_PUT_ERROR(SSL, SSL_R_CHANNEL_ID_SIGNATURE_INVALID);
    s->s3->tlsext_channel_id_valid = 0;
    goto err;
  }

  memcpy(s->s3->tlsext_channel_id, p, 64);
  ret = 1;

err:
  BN_free(&x);
  BN_free(&y);
  BN_free(sig.r);
  BN_free(sig.s);
  EC_KEY_free(key);
  EC_POINT_free(point);
  EC_GROUP_free(p256);
  return ret;
}
