/* ssl/s3_clnt.c */
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
 * OTHERWISE.
 */

#include <openssl/ssl.h>

#include <assert.h>
#include <stdio.h>
#include <string.h>

#include <openssl/bn.h>
#include <openssl/buf.h>
#include <openssl/bytestring.h>
#include <openssl/dh.h>
#include <openssl/ec_key.h>
#include <openssl/ecdsa.h>
#include <openssl/err.h>
#include <openssl/evp.h>
#include <openssl/md5.h>
#include <openssl/mem.h>
#include <openssl/obj.h>
#include <openssl/rand.h>
#include <openssl/x509.h>
#include <openssl/x509v3.h>

#include "internal.h"
#include "../crypto/dh/internal.h"


int ssl3_connect(SSL *s) {
  BUF_MEM *buf = NULL;
  void (*cb)(const SSL *ssl, int type, int val) = NULL;
  int ret = -1;
  int new_state, state, skip = 0;

  assert(s->handshake_func == ssl3_connect);
  assert(!s->server);
  assert(!SSL_IS_DTLS(s));

  ERR_clear_error();
  ERR_clear_system_error();

  if (s->info_callback != NULL) {
    cb = s->info_callback;
  } else if (s->ctx->info_callback != NULL) {
    cb = s->ctx->info_callback;
  }

  s->in_handshake++;

  for (;;) {
    state = s->state;

    switch (s->state) {
      case SSL_ST_CONNECT:
        if (cb != NULL) {
          cb(s, SSL_CB_HANDSHAKE_START, 1);
        }

        if (s->init_buf == NULL) {
          buf = BUF_MEM_new();
          if (buf == NULL ||
              !BUF_MEM_grow(buf, SSL3_RT_MAX_PLAIN_LENGTH)) {
            ret = -1;
            goto end;
          }

          s->init_buf = buf;
          buf = NULL;
        }

        if (!ssl_init_wbio_buffer(s, 0)) {
          ret = -1;
          goto end;
        }

        /* don't push the buffering BIO quite yet */

        if (!ssl3_init_handshake_buffer(s)) {
          OPENSSL_PUT_ERROR(SSL, ERR_R_INTERNAL_ERROR);
          ret = -1;
          goto end;
        }

        s->state = SSL3_ST_CW_CLNT_HELLO_A;
        s->init_num = 0;
        break;

      case SSL3_ST_CW_CLNT_HELLO_A:
      case SSL3_ST_CW_CLNT_HELLO_B:
        s->shutdown = 0;
        ret = ssl3_send_client_hello(s);
        if (ret <= 0) {
          goto end;
        }
        s->state = SSL3_ST_CR_SRVR_HELLO_A;
        s->init_num = 0;

        /* turn on buffering for the next lot of output */
        if (s->bbio != s->wbio) {
          s->wbio = BIO_push(s->bbio, s->wbio);
        }

        break;

      case SSL3_ST_CR_SRVR_HELLO_A:
      case SSL3_ST_CR_SRVR_HELLO_B:
        ret = ssl3_get_server_hello(s);
        if (ret <= 0) {
          goto end;
        }

        if (s->hit) {
          s->state = SSL3_ST_CR_CHANGE;
          if (s->tlsext_ticket_expected) {
            /* receive renewed session ticket */
            s->state = SSL3_ST_CR_SESSION_TICKET_A;
          }
        } else {
          s->state = SSL3_ST_CR_CERT_A;
        }
        s->init_num = 0;
        break;

      case SSL3_ST_CR_CERT_A:
      case SSL3_ST_CR_CERT_B:
        if (ssl_cipher_has_server_public_key(s->s3->tmp.new_cipher)) {
          ret = ssl3_get_server_certificate(s);
          if (ret <= 0) {
            goto end;
          }
          if (s->s3->tmp.certificate_status_expected) {
            s->state = SSL3_ST_CR_CERT_STATUS_A;
          } else {
            s->state = SSL3_ST_VERIFY_SERVER_CERT;
          }
        } else {
          skip = 1;
          s->state = SSL3_ST_CR_KEY_EXCH_A;
        }
        s->init_num = 0;
        break;

      case SSL3_ST_VERIFY_SERVER_CERT:
        ret = ssl3_verify_server_cert(s);
        if (ret <= 0) {
          goto end;
        }

        s->state = SSL3_ST_CR_KEY_EXCH_A;
        s->init_num = 0;
        break;

      case SSL3_ST_CR_KEY_EXCH_A:
      case SSL3_ST_CR_KEY_EXCH_B:
        ret = ssl3_get_server_key_exchange(s);
        if (ret <= 0) {
          goto end;
        }
        s->state = SSL3_ST_CR_CERT_REQ_A;
        s->init_num = 0;
        break;

      case SSL3_ST_CR_CERT_REQ_A:
      case SSL3_ST_CR_CERT_REQ_B:
        ret = ssl3_get_certificate_request(s);
        if (ret <= 0) {
          goto end;
        }
        s->state = SSL3_ST_CR_SRVR_DONE_A;
        s->init_num = 0;
        break;

      case SSL3_ST_CR_SRVR_DONE_A:
      case SSL3_ST_CR_SRVR_DONE_B:
        ret = ssl3_get_server_done(s);
        if (ret <= 0) {
          goto end;
        }
        if (s->s3->tmp.cert_req) {
          s->state = SSL3_ST_CW_CERT_A;
        } else {
          s->state = SSL3_ST_CW_KEY_EXCH_A;
        }
        s->init_num = 0;

        break;

      case SSL3_ST_CW_CERT_A:
      case SSL3_ST_CW_CERT_B:
      case SSL3_ST_CW_CERT_C:
      case SSL3_ST_CW_CERT_D:
        ret = ssl3_send_client_certificate(s);
        if (ret <= 0) {
          goto end;
        }
        s->state = SSL3_ST_CW_KEY_EXCH_A;
        s->init_num = 0;
        break;

      case SSL3_ST_CW_KEY_EXCH_A:
      case SSL3_ST_CW_KEY_EXCH_B:
        ret = ssl3_send_client_key_exchange(s);
        if (ret <= 0) {
          goto end;
        }
        /* For TLS, cert_req is set to 2, so a cert chain
         * of nothing is sent, but no verify packet is sent */
        if (s->s3->tmp.cert_req == 1) {
          s->state = SSL3_ST_CW_CERT_VRFY_A;
        } else {
          s->state = SSL3_ST_CW_CHANGE_A;
          s->s3->change_cipher_spec = 0;
        }

        s->init_num = 0;
        break;

      case SSL3_ST_CW_CERT_VRFY_A:
      case SSL3_ST_CW_CERT_VRFY_B:
      case SSL3_ST_CW_CERT_VRFY_C:
        ret = ssl3_send_cert_verify(s);
        if (ret <= 0) {
          goto end;
        }
        s->state = SSL3_ST_CW_CHANGE_A;
        s->init_num = 0;
        s->s3->change_cipher_spec = 0;
        break;

      case SSL3_ST_CW_CHANGE_A:
      case SSL3_ST_CW_CHANGE_B:
        ret = ssl3_send_change_cipher_spec(s, SSL3_ST_CW_CHANGE_A,
                                           SSL3_ST_CW_CHANGE_B);
        if (ret <= 0) {
          goto end;
        }

        s->state = SSL3_ST_CW_FINISHED_A;
        if (s->s3->tlsext_channel_id_valid) {
          s->state = SSL3_ST_CW_CHANNEL_ID_A;
        }
        if (s->s3->next_proto_neg_seen) {
          s->state = SSL3_ST_CW_NEXT_PROTO_A;
        }
        s->init_num = 0;

        s->session->cipher = s->s3->tmp.new_cipher;
        if (!s->enc_method->setup_key_block(s) ||
            !s->enc_method->change_cipher_state(
                s, SSL3_CHANGE_CIPHER_CLIENT_WRITE)) {
          ret = -1;
          goto end;
        }

        break;

      case SSL3_ST_CW_NEXT_PROTO_A:
      case SSL3_ST_CW_NEXT_PROTO_B:
        ret = ssl3_send_next_proto(s);
        if (ret <= 0) {
          goto end;
        }

        if (s->s3->tlsext_channel_id_valid) {
          s->state = SSL3_ST_CW_CHANNEL_ID_A;
        } else {
          s->state = SSL3_ST_CW_FINISHED_A;
        }
        break;

      case SSL3_ST_CW_CHANNEL_ID_A:
      case SSL3_ST_CW_CHANNEL_ID_B:
        ret = ssl3_send_channel_id(s);
        if (ret <= 0) {
          goto end;
        }
        s->state = SSL3_ST_CW_FINISHED_A;
        break;

      case SSL3_ST_CW_FINISHED_A:
      case SSL3_ST_CW_FINISHED_B:
        ret =
            ssl3_send_finished(s, SSL3_ST_CW_FINISHED_A, SSL3_ST_CW_FINISHED_B,
                               s->enc_method->client_finished_label,
                               s->enc_method->client_finished_label_len);
        if (ret <= 0) {
          goto end;
        }
        s->state = SSL3_ST_CW_FLUSH;

        if (s->hit) {
          s->s3->tmp.next_state = SSL_ST_OK;
        } else {
          /* This is a non-resumption handshake. If it involves ChannelID, then
           * record the handshake hashes at this point in the session so that
           * any resumption of this session with ChannelID can sign those
           * hashes. */
          ret = tls1_record_handshake_hashes_for_channel_id(s);
          if (ret <= 0) {
            goto end;
          }
          if ((SSL_get_mode(s) & SSL_MODE_ENABLE_FALSE_START) &&
              ssl3_can_false_start(s) &&
              /* No False Start on renegotiation (would complicate the state
               * machine). */
              !s->s3->initial_handshake_complete) {
            s->s3->tmp.next_state = SSL3_ST_FALSE_START;
          } else {
            /* Allow NewSessionTicket if ticket expected */
            if (s->tlsext_ticket_expected) {
              s->s3->tmp.next_state = SSL3_ST_CR_SESSION_TICKET_A;
            } else {
              s->s3->tmp.next_state = SSL3_ST_CR_CHANGE;
            }
          }
        }
        s->init_num = 0;
        break;

      case SSL3_ST_CR_SESSION_TICKET_A:
      case SSL3_ST_CR_SESSION_TICKET_B:
        ret = ssl3_get_new_session_ticket(s);
        if (ret <= 0) {
          goto end;
        }
        s->state = SSL3_ST_CR_CHANGE;
        s->init_num = 0;
        break;

      case SSL3_ST_CR_CERT_STATUS_A:
      case SSL3_ST_CR_CERT_STATUS_B:
        ret = ssl3_get_cert_status(s);
        if (ret <= 0) {
          goto end;
        }
        s->state = SSL3_ST_VERIFY_SERVER_CERT;
        s->init_num = 0;
        break;

      case SSL3_ST_CR_CHANGE:
        /* At this point, the next message must be entirely behind a
         * ChangeCipherSpec. */
        if (!ssl3_expect_change_cipher_spec(s)) {
          ret = -1;
          goto end;
        }
        s->state = SSL3_ST_CR_FINISHED_A;
        break;

      case SSL3_ST_CR_FINISHED_A:
      case SSL3_ST_CR_FINISHED_B:
        ret =
            ssl3_get_finished(s, SSL3_ST_CR_FINISHED_A, SSL3_ST_CR_FINISHED_B);
        if (ret <= 0) {
          goto end;
        }

        if (s->hit) {
          s->state = SSL3_ST_CW_CHANGE_A;
        } else {
          s->state = SSL_ST_OK;
        }
        s->init_num = 0;
        break;

      case SSL3_ST_CW_FLUSH:
        s->rwstate = SSL_WRITING;
        if (BIO_flush(s->wbio) <= 0) {
          ret = -1;
          goto end;
        }
        s->rwstate = SSL_NOTHING;
        s->state = s->s3->tmp.next_state;
        break;

      case SSL3_ST_FALSE_START:
        /* Allow NewSessionTicket if ticket expected */
        if (s->tlsext_ticket_expected) {
          s->state = SSL3_ST_CR_SESSION_TICKET_A;
        } else {
          s->state = SSL3_ST_CR_CHANGE;
        }
        s->s3->tmp.in_false_start = 1;

        ssl_free_wbio_buffer(s);
        ret = 1;
        goto end;

      case SSL_ST_OK:
        /* clean a few things up */
        ssl3_cleanup_key_block(s);

        BUF_MEM_free(s->init_buf);
        s->init_buf = NULL;

        /* Remove write buffering now. */
        ssl_free_wbio_buffer(s);

        const int is_initial_handshake = !s->s3->initial_handshake_complete;

        s->init_num = 0;
        s->s3->tmp.in_false_start = 0;
        s->s3->initial_handshake_complete = 1;

        if (is_initial_handshake) {
          /* Renegotiations do not participate in session resumption. */
          ssl_update_cache(s, SSL_SESS_CACHE_CLIENT);
        }

        ret = 1;
        /* s->server=0; */

        if (cb != NULL) {
          cb(s, SSL_CB_HANDSHAKE_DONE, 1);
        }

        goto end;

      default:
        OPENSSL_PUT_ERROR(SSL, SSL_R_UNKNOWN_STATE);
        ret = -1;
        goto end;
    }

    if (!s->s3->tmp.reuse_message && !skip) {
      if (cb != NULL && s->state != state) {
        new_state = s->state;
        s->state = state;
        cb(s, SSL_CB_CONNECT_LOOP, 1);
        s->state = new_state;
      }
    }
    skip = 0;
  }

end:
  s->in_handshake--;
  BUF_MEM_free(buf);
  if (cb != NULL) {
    cb(s, SSL_CB_CONNECT_EXIT, ret);
  }
  return ret;
}

int ssl3_send_client_hello(SSL *s) {
  uint8_t *buf, *p, *d;
  int i;
  unsigned long l;

  buf = (uint8_t *)s->init_buf->data;
  if (s->state == SSL3_ST_CW_CLNT_HELLO_A) {
    if (!s->s3->have_version) {
      uint16_t max_version = ssl3_get_max_client_version(s);
      /* Disabling all versions is silly: return an error. */
      if (max_version == 0) {
        OPENSSL_PUT_ERROR(SSL, SSL_R_WRONG_SSL_VERSION);
        goto err;
      }
      s->version = max_version;
      s->client_version = max_version;
    }

    /* If the configured session was created at a version higher than our
     * maximum version, drop it. */
    if (s->session &&
        (s->session->session_id_length == 0 || s->session->not_resumable ||
         (!SSL_IS_DTLS(s) && s->session->ssl_version > s->version) ||
         (SSL_IS_DTLS(s) && s->session->ssl_version < s->version))) {
      SSL_set_session(s, NULL);
    }

    /* else use the pre-loaded session */
    p = s->s3->client_random;

    /* If resending the ClientHello in DTLS after a HelloVerifyRequest, don't
     * renegerate the client_random. The random must be reused. */
    if ((!SSL_IS_DTLS(s) || !s->d1->send_cookie) &&
        !ssl_fill_hello_random(p, sizeof(s->s3->client_random),
                               0 /* client */)) {
      goto err;
    }

    /* Do the message type and length last. Note: the final argument to
     * ssl_add_clienthello_tlsext below depends on the size of this prefix. */
    d = p = ssl_handshake_start(s);

    /* version indicates the negotiated version: for example from an SSLv2/v3
     * compatible client hello). The client_version field is the maximum
     * version we permit and it is also used in RSA encrypted premaster
     * secrets. Some servers can choke if we initially report a higher version
     * then renegotiate to a lower one in the premaster secret. This didn't
     * happen with TLS 1.0 as most servers supported it but it can with TLS 1.1
     * or later if the server only supports 1.0.
     *
     * Possible scenario with previous logic:
     *   1. Client hello indicates TLS 1.2
     *   2. Server hello says TLS 1.0
     *   3. RSA encrypted premaster secret uses 1.2.
     *   4. Handhaked proceeds using TLS 1.0.
     *   5. Server sends hello request to renegotiate.
     *   6. Client hello indicates TLS v1.0 as we now
     *      know that is maximum server supports.
     *   7. Server chokes on RSA encrypted premaster secret
     *      containing version 1.0.
     *
     * For interoperability it should be OK to always use the maximum version
     * we support in client hello and then rely on the checking of version to
     * ensure the servers isn't being inconsistent: for example initially
     * negotiating with TLS 1.0 and renegotiating with TLS 1.2. We do this by
     * using client_version in client hello and not resetting it to the
     * negotiated version. */
    *(p++) = s->client_version >> 8;
    *(p++) = s->client_version & 0xff;

    /* Random stuff */
    memcpy(p, s->s3->client_random, SSL3_RANDOM_SIZE);
    p += SSL3_RANDOM_SIZE;

    /* Session ID */
    if (s->s3->initial_handshake_complete || s->session == NULL) {
      /* Renegotiations do not participate in session resumption. */
      i = 0;
    } else {
      i = s->session->session_id_length;
    }
    *(p++) = i;
    if (i != 0) {
      if (i > (int)sizeof(s->session->session_id)) {
        OPENSSL_PUT_ERROR(SSL, ERR_R_INTERNAL_ERROR);
        goto err;
      }
      memcpy(p, s->session->session_id, i);
      p += i;
    }

    /* cookie stuff for DTLS */
    if (SSL_IS_DTLS(s)) {
      if (s->d1->cookie_len > sizeof(s->d1->cookie)) {
        OPENSSL_PUT_ERROR(SSL, ERR_R_INTERNAL_ERROR);
        goto err;
      }
      *(p++) = s->d1->cookie_len;
      memcpy(p, s->d1->cookie, s->d1->cookie_len);
      p += s->d1->cookie_len;
    }

    /* Ciphers supported */
    i = ssl_cipher_list_to_bytes(s, SSL_get_ciphers(s), &p[2]);
    if (i == 0) {
      OPENSSL_PUT_ERROR(SSL, SSL_R_NO_CIPHERS_AVAILABLE);
      goto err;
    }
    s2n(i, p);
    p += i;

    /* COMPRESSION */
    *(p++) = 1;
    *(p++) = 0; /* Add the NULL method */

    /* TLS extensions*/
    p = ssl_add_clienthello_tlsext(s, p, buf + SSL3_RT_MAX_PLAIN_LENGTH,
                                   p - buf);
    if (p == NULL) {
      OPENSSL_PUT_ERROR(SSL, ERR_R_INTERNAL_ERROR);
      goto err;
    }

    l = p - d;
    if (!ssl_set_handshake_header(s, SSL3_MT_CLIENT_HELLO, l)) {
      goto err;
    }
    s->state = SSL3_ST_CW_CLNT_HELLO_B;
  }

  /* SSL3_ST_CW_CLNT_HELLO_B */
  return ssl_do_write(s);

err:
  return -1;
}

int ssl3_get_server_hello(SSL *s) {
  STACK_OF(SSL_CIPHER) *sk;
  const SSL_CIPHER *c;
  CERT *ct = s->cert;
  int al = SSL_AD_INTERNAL_ERROR, ok;
  long n;
  CBS server_hello, server_random, session_id;
  uint16_t server_version, cipher_suite;
  uint8_t compression_method;
  uint32_t mask_ssl;

  n = s->method->ssl_get_message(s, SSL3_ST_CR_SRVR_HELLO_A,
                                 SSL3_ST_CR_SRVR_HELLO_B, SSL3_MT_SERVER_HELLO,
                                 20000, /* ?? */
                                 ssl_hash_message, &ok);

  if (!ok) {
    uint32_t err = ERR_peek_error();
    if (ERR_GET_LIB(err) == ERR_LIB_SSL &&
        ERR_GET_REASON(err) == SSL_R_SSLV3_ALERT_HANDSHAKE_FAILURE) {
      /* Add a dedicated error code to the queue for a handshake_failure alert
       * in response to ClientHello. This matches NSS's client behavior and
       * gives a better error on a (probable) failure to negotiate initial
       * parameters. Note: this error code comes after the original one.
       *
       * See https://crbug.com/446505. */
      OPENSSL_PUT_ERROR(SSL, SSL_R_HANDSHAKE_FAILURE_ON_CLIENT_HELLO);
    }
    return n;
  }

  CBS_init(&server_hello, s->init_msg, n);

  if (!CBS_get_u16(&server_hello, &server_version) ||
      !CBS_get_bytes(&server_hello, &server_random, SSL3_RANDOM_SIZE) ||
      !CBS_get_u8_length_prefixed(&server_hello, &session_id) ||
      CBS_len(&session_id) > SSL3_SESSION_ID_SIZE ||
      !CBS_get_u16(&server_hello, &cipher_suite) ||
      !CBS_get_u8(&server_hello, &compression_method)) {
    al = SSL_AD_DECODE_ERROR;
    OPENSSL_PUT_ERROR(SSL, SSL_R_DECODE_ERROR);
    goto f_err;
  }

  assert(s->s3->have_version == s->s3->initial_handshake_complete);
  if (!s->s3->have_version) {
    if (!ssl3_is_version_enabled(s, server_version)) {
      OPENSSL_PUT_ERROR(SSL, SSL_R_UNSUPPORTED_PROTOCOL);
      s->version = server_version;
      /* Mark the version as fixed so the record-layer version is not clamped
       * to TLS 1.0. */
      s->s3->have_version = 1;
      al = SSL_AD_PROTOCOL_VERSION;
      goto f_err;
    }
    s->version = server_version;
    s->enc_method = ssl3_get_enc_method(server_version);
    assert(s->enc_method != NULL);
    /* At this point, the connection's version is known and s->version is
     * fixed. Begin enforcing the record-layer version. */
    s->s3->have_version = 1;
  } else if (server_version != s->version) {
    OPENSSL_PUT_ERROR(SSL, SSL_R_WRONG_SSL_VERSION);
    al = SSL_AD_PROTOCOL_VERSION;
    goto f_err;
  }

  /* Copy over the server random. */
  memcpy(s->s3->server_random, CBS_data(&server_random), SSL3_RANDOM_SIZE);

  assert(s->session == NULL || s->session->session_id_length > 0);
  if (!s->s3->initial_handshake_complete && s->session != NULL &&
      CBS_mem_equal(&session_id, s->session->session_id,
                    s->session->session_id_length)) {
    if (s->sid_ctx_length != s->session->sid_ctx_length ||
        memcmp(s->session->sid_ctx, s->sid_ctx, s->sid_ctx_length)) {
      /* actually a client application bug */
      al = SSL_AD_ILLEGAL_PARAMETER;
      OPENSSL_PUT_ERROR(SSL,
                        SSL_R_ATTEMPT_TO_REUSE_SESSION_IN_DIFFERENT_CONTEXT);
      goto f_err;
    }
    s->hit = 1;
  } else {
    /* The session wasn't resumed. Create a fresh SSL_SESSION to
     * fill out. */
    s->hit = 0;
    if (!ssl_get_new_session(s, 0)) {
      goto f_err;
    }
    /* Note: session_id could be empty. */
    s->session->session_id_length = CBS_len(&session_id);
    memcpy(s->session->session_id, CBS_data(&session_id), CBS_len(&session_id));
  }

  c = SSL_get_cipher_by_value(cipher_suite);
  if (c == NULL) {
    /* unknown cipher */
    al = SSL_AD_ILLEGAL_PARAMETER;
    OPENSSL_PUT_ERROR(SSL, SSL_R_UNKNOWN_CIPHER_RETURNED);
    goto f_err;
  }
  /* ct->mask_ssl was computed from client capabilities. Now
   * that the final version is known, compute a new mask_ssl. */
  if (!SSL_USE_TLS1_2_CIPHERS(s)) {
    mask_ssl = SSL_TLSV1_2;
  } else {
    mask_ssl = 0;
  }
  /* If the cipher is disabled then we didn't sent it in the ClientHello, so if
   * the server selected it, it's an error. */
  if ((c->algorithm_ssl & mask_ssl) ||
      (c->algorithm_mkey & ct->mask_k) ||
      (c->algorithm_auth & ct->mask_a)) {
    al = SSL_AD_ILLEGAL_PARAMETER;
    OPENSSL_PUT_ERROR(SSL, SSL_R_WRONG_CIPHER_RETURNED);
    goto f_err;
  }

  sk = ssl_get_ciphers_by_id(s);
  if (!sk_SSL_CIPHER_find(sk, NULL, c)) {
    /* we did not say we would use this cipher */
    al = SSL_AD_ILLEGAL_PARAMETER;
    OPENSSL_PUT_ERROR(SSL, SSL_R_WRONG_CIPHER_RETURNED);
    goto f_err;
  }

  if (s->hit) {
    if (s->session->cipher != c) {
      al = SSL_AD_ILLEGAL_PARAMETER;
      OPENSSL_PUT_ERROR(SSL, SSL_R_OLD_SESSION_CIPHER_NOT_RETURNED);
      goto f_err;
    }
    if (s->session->ssl_version != s->version) {
      al = SSL_AD_ILLEGAL_PARAMETER;
      OPENSSL_PUT_ERROR(SSL, SSL_R_OLD_SESSION_VERSION_NOT_RETURNED);
      goto f_err;
    }
  }
  s->s3->tmp.new_cipher = c;

  /* Now that the cipher is known, initialize the handshake hash. */
  if (!ssl3_init_handshake_hash(s)) {
    goto f_err;
  }

  /* If doing a full handshake with TLS 1.2, the server may request a client
   * certificate which requires hashing the handshake transcript under a
   * different hash. Otherwise, the handshake buffer may be released. */
  if (!SSL_USE_SIGALGS(s) || s->hit) {
    ssl3_free_handshake_buffer(s);
  }

  /* Only the NULL compression algorithm is supported. */
  if (compression_method != 0) {
    al = SSL_AD_ILLEGAL_PARAMETER;
    OPENSSL_PUT_ERROR(SSL, SSL_R_UNSUPPORTED_COMPRESSION_ALGORITHM);
    goto f_err;
  }

  /* TLS extensions */
  if (!ssl_parse_serverhello_tlsext(s, &server_hello)) {
    OPENSSL_PUT_ERROR(SSL, SSL_R_PARSE_TLSEXT);
    goto err;
  }

  /* There should be nothing left over in the record. */
  if (CBS_len(&server_hello) != 0) {
    /* wrong packet length */
    al = SSL_AD_DECODE_ERROR;
    OPENSSL_PUT_ERROR(SSL, SSL_R_BAD_PACKET_LENGTH);
    goto f_err;
  }

  if (s->hit &&
      s->s3->tmp.extended_master_secret != s->session->extended_master_secret) {
    al = SSL_AD_HANDSHAKE_FAILURE;
    if (s->session->extended_master_secret) {
      OPENSSL_PUT_ERROR(SSL, SSL_R_RESUMED_EMS_SESSION_WITHOUT_EMS_EXTENSION);
    } else {
      OPENSSL_PUT_ERROR(SSL, SSL_R_RESUMED_NON_EMS_SESSION_WITH_EMS_EXTENSION);
    }
    goto f_err;
  }

  return 1;

f_err:
  ssl3_send_alert(s, SSL3_AL_FATAL, al);
err:
  return -1;
}

/* ssl3_check_certificate_for_cipher returns one if |leaf| is a suitable server
 * certificate type for |cipher|. Otherwise, it returns zero and pushes an error
 * on the error queue. */
static int ssl3_check_certificate_for_cipher(X509 *leaf,
                                             const SSL_CIPHER *cipher) {
  int ret = 0;
  EVP_PKEY *pkey = X509_get_pubkey(leaf);
  if (pkey == NULL) {
    goto err;
  }

  /* Check the certificate's type matches the cipher. */
  int expected_type = ssl_cipher_get_key_type(cipher);
  assert(expected_type != EVP_PKEY_NONE);
  if (pkey->type != expected_type) {
    OPENSSL_PUT_ERROR(SSL, SSL_R_WRONG_CERTIFICATE_TYPE);
    goto err;
  }

  /* TODO(davidben): This behavior is preserved from upstream. Should key usages
   * be checked in other cases as well? */
  if (cipher->algorithm_auth & SSL_aECDSA) {
    /* This call populates the ex_flags field correctly */
    X509_check_purpose(leaf, -1, 0);
    if ((leaf->ex_flags & EXFLAG_KUSAGE) &&
        !(leaf->ex_kusage & X509v3_KU_DIGITAL_SIGNATURE)) {
      OPENSSL_PUT_ERROR(SSL, SSL_R_ECC_CERT_NOT_FOR_SIGNING);
      goto err;
    }
  }

  ret = 1;

err:
  EVP_PKEY_free(pkey);
  return ret;
}

int ssl3_get_server_certificate(SSL *s) {
  int al, ok, ret = -1;
  unsigned long n;
  X509 *x = NULL;
  STACK_OF(X509) *sk = NULL;
  EVP_PKEY *pkey = NULL;
  CBS cbs, certificate_list;
  const uint8_t *data;

  n = s->method->ssl_get_message(s, SSL3_ST_CR_CERT_A, SSL3_ST_CR_CERT_B,
                                 SSL3_MT_CERTIFICATE, (long)s->max_cert_list,
                                 ssl_hash_message, &ok);

  if (!ok) {
    return n;
  }

  CBS_init(&cbs, s->init_msg, n);

  sk = sk_X509_new_null();
  if (sk == NULL) {
    OPENSSL_PUT_ERROR(SSL, ERR_R_MALLOC_FAILURE);
    goto err;
  }

  if (!CBS_get_u24_length_prefixed(&cbs, &certificate_list) ||
      CBS_len(&certificate_list) == 0 ||
      CBS_len(&cbs) != 0) {
    al = SSL_AD_DECODE_ERROR;
    OPENSSL_PUT_ERROR(SSL, SSL_R_DECODE_ERROR);
    goto f_err;
  }

  while (CBS_len(&certificate_list) > 0) {
    CBS certificate;
    if (!CBS_get_u24_length_prefixed(&certificate_list, &certificate)) {
      al = SSL_AD_DECODE_ERROR;
      OPENSSL_PUT_ERROR(SSL, SSL_R_CERT_LENGTH_MISMATCH);
      goto f_err;
    }
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

  X509 *leaf = sk_X509_value(sk, 0);
  if (!ssl3_check_certificate_for_cipher(leaf, s->s3->tmp.new_cipher)) {
    al = SSL_AD_ILLEGAL_PARAMETER;
    goto f_err;
  }

  /* NOTE: Unlike the server half, the client's copy of |cert_chain| includes
   * the leaf. */
  sk_X509_pop_free(s->session->cert_chain, X509_free);
  s->session->cert_chain = sk;
  sk = NULL;

  X509_free(s->session->peer);
  s->session->peer = X509_up_ref(leaf);

  s->session->verify_result = s->verify_result;

  ret = 1;

  if (0) {
  f_err:
    ssl3_send_alert(s, SSL3_AL_FATAL, al);
  }

err:
  EVP_PKEY_free(pkey);
  X509_free(x);
  sk_X509_pop_free(sk, X509_free);
  return ret;
}

int ssl3_get_server_key_exchange(SSL *s) {
  EVP_MD_CTX md_ctx;
  int al, ok;
  long n, alg_k, alg_a;
  EVP_PKEY *pkey = NULL;
  const EVP_MD *md = NULL;
  RSA *rsa = NULL;
  DH *dh = NULL;
  EC_KEY *ecdh = NULL;
  BN_CTX *bn_ctx = NULL;
  EC_POINT *srvr_ecpoint = NULL;
  CBS server_key_exchange, server_key_exchange_orig, parameter;

  /* use same message size as in ssl3_get_certificate_request() as
   * ServerKeyExchange message may be skipped */
  n = s->method->ssl_get_message(s, SSL3_ST_CR_KEY_EXCH_A,
                                 SSL3_ST_CR_KEY_EXCH_B, -1, s->max_cert_list,
                                 ssl_hash_message, &ok);
  if (!ok) {
    return n;
  }

  if (s->s3->tmp.message_type != SSL3_MT_SERVER_KEY_EXCHANGE) {
    if (ssl_cipher_requires_server_key_exchange(s->s3->tmp.new_cipher)) {
      OPENSSL_PUT_ERROR(SSL, SSL_R_UNEXPECTED_MESSAGE);
      ssl3_send_alert(s, SSL3_AL_FATAL, SSL_AD_UNEXPECTED_MESSAGE);
      return -1;
    }

    /* In plain PSK ciphersuite, ServerKeyExchange may be omitted to send no
     * identity hint. */
    if (s->s3->tmp.new_cipher->algorithm_auth & SSL_aPSK) {
      /* TODO(davidben): This should be reset in one place with the rest of the
       * handshake state. */
      OPENSSL_free(s->s3->tmp.peer_psk_identity_hint);
      s->s3->tmp.peer_psk_identity_hint = NULL;
    }
    s->s3->tmp.reuse_message = 1;
    return 1;
  }

  /* Retain a copy of the original CBS to compute the signature over. */
  CBS_init(&server_key_exchange, s->init_msg, n);
  server_key_exchange_orig = server_key_exchange;

  alg_k = s->s3->tmp.new_cipher->algorithm_mkey;
  alg_a = s->s3->tmp.new_cipher->algorithm_auth;
  EVP_MD_CTX_init(&md_ctx);

  if (alg_a & SSL_aPSK) {
    CBS psk_identity_hint;

    /* Each of the PSK key exchanges begins with a psk_identity_hint. */
    if (!CBS_get_u16_length_prefixed(&server_key_exchange,
                                     &psk_identity_hint)) {
      al = SSL_AD_DECODE_ERROR;
      OPENSSL_PUT_ERROR(SSL, SSL_R_DECODE_ERROR);
      goto f_err;
    }

    /* Store PSK identity hint for later use, hint is used in
     * ssl3_send_client_key_exchange.  Assume that the maximum length of a PSK
     * identity hint can be as long as the maximum length of a PSK identity.
     * Also do not allow NULL characters; identities are saved as C strings.
     *
     * TODO(davidben): Should invalid hints be ignored? It's a hint rather than
     * a specific identity. */
    if (CBS_len(&psk_identity_hint) > PSK_MAX_IDENTITY_LEN ||
        CBS_contains_zero_byte(&psk_identity_hint)) {
      al = SSL_AD_HANDSHAKE_FAILURE;
      OPENSSL_PUT_ERROR(SSL, SSL_R_DATA_LENGTH_TOO_LONG);
      goto f_err;
    }

    /* Save the identity hint as a C string. */
    if (!CBS_strdup(&psk_identity_hint, &s->s3->tmp.peer_psk_identity_hint)) {
      al = SSL_AD_INTERNAL_ERROR;
      OPENSSL_PUT_ERROR(SSL, ERR_R_MALLOC_FAILURE);
      goto f_err;
    }
  }

  if (alg_k & SSL_kDHE) {
    CBS dh_p, dh_g, dh_Ys;

    if (!CBS_get_u16_length_prefixed(&server_key_exchange, &dh_p) ||
        CBS_len(&dh_p) == 0 ||
        !CBS_get_u16_length_prefixed(&server_key_exchange, &dh_g) ||
        CBS_len(&dh_g) == 0 ||
        !CBS_get_u16_length_prefixed(&server_key_exchange, &dh_Ys) ||
        CBS_len(&dh_Ys) == 0) {
      al = SSL_AD_DECODE_ERROR;
      OPENSSL_PUT_ERROR(SSL, SSL_R_DECODE_ERROR);
      goto f_err;
    }

    dh = DH_new();
    if (dh == NULL) {
      OPENSSL_PUT_ERROR(SSL, ERR_R_DH_LIB);
      goto err;
    }

    if ((dh->p = BN_bin2bn(CBS_data(&dh_p), CBS_len(&dh_p), NULL)) == NULL ||
        (dh->g = BN_bin2bn(CBS_data(&dh_g), CBS_len(&dh_g), NULL)) == NULL ||
        (dh->pub_key = BN_bin2bn(CBS_data(&dh_Ys), CBS_len(&dh_Ys), NULL)) ==
            NULL) {
      OPENSSL_PUT_ERROR(SSL, ERR_R_BN_LIB);
      goto err;
    }

    s->session->key_exchange_info = DH_num_bits(dh);
    if (s->session->key_exchange_info < 1024) {
      OPENSSL_PUT_ERROR(SSL, SSL_R_BAD_DH_P_LENGTH);
      goto err;
    }
    DH_free(s->s3->tmp.peer_dh_tmp);
    s->s3->tmp.peer_dh_tmp = dh;
    dh = NULL;
  } else if (alg_k & SSL_kECDHE) {
    uint16_t curve_id;
    int curve_nid = 0;
    const EC_GROUP *group;
    CBS point;

    /* Extract elliptic curve parameters and the server's ephemeral ECDH public
     * key.  Check curve is one of our preferences, if not server has sent an
     * invalid curve. */
    if (!tls1_check_curve(s, &server_key_exchange, &curve_id)) {
      al = SSL_AD_DECODE_ERROR;
      OPENSSL_PUT_ERROR(SSL, SSL_R_WRONG_CURVE);
      goto f_err;
    }

    curve_nid = tls1_ec_curve_id2nid(curve_id);
    if (curve_nid == 0) {
      al = SSL_AD_INTERNAL_ERROR;
      OPENSSL_PUT_ERROR(SSL, SSL_R_UNABLE_TO_FIND_ECDH_PARAMETERS);
      goto f_err;
    }

    ecdh = EC_KEY_new_by_curve_name(curve_nid);
    s->session->key_exchange_info = curve_id;
    if (ecdh == NULL) {
      OPENSSL_PUT_ERROR(SSL, ERR_R_EC_LIB);
      goto err;
    }

    group = EC_KEY_get0_group(ecdh);

    /* Next, get the encoded ECPoint */
    if (!CBS_get_u8_length_prefixed(&server_key_exchange, &point)) {
      al = SSL_AD_DECODE_ERROR;
      OPENSSL_PUT_ERROR(SSL, SSL_R_DECODE_ERROR);
      goto f_err;
    }

    if (((srvr_ecpoint = EC_POINT_new(group)) == NULL) ||
        ((bn_ctx = BN_CTX_new()) == NULL)) {
      OPENSSL_PUT_ERROR(SSL, ERR_R_MALLOC_FAILURE);
      goto err;
    }

    if (!EC_POINT_oct2point(group, srvr_ecpoint, CBS_data(&point),
                            CBS_len(&point), bn_ctx)) {
      al = SSL_AD_DECODE_ERROR;
      OPENSSL_PUT_ERROR(SSL, SSL_R_BAD_ECPOINT);
      goto f_err;
    }
    EC_KEY_set_public_key(ecdh, srvr_ecpoint);
    EC_KEY_free(s->s3->tmp.peer_ecdh_tmp);
    s->s3->tmp.peer_ecdh_tmp = ecdh;
    ecdh = NULL;
    BN_CTX_free(bn_ctx);
    bn_ctx = NULL;
    EC_POINT_free(srvr_ecpoint);
    srvr_ecpoint = NULL;
  } else if (!(alg_k & SSL_kPSK)) {
    al = SSL_AD_UNEXPECTED_MESSAGE;
    OPENSSL_PUT_ERROR(SSL, SSL_R_UNEXPECTED_MESSAGE);
    goto f_err;
  }

  /* At this point, |server_key_exchange| contains the signature, if any, while
   * |server_key_exchange_orig| contains the entire message. From that, derive
   * a CBS containing just the parameter. */
  CBS_init(&parameter, CBS_data(&server_key_exchange_orig),
           CBS_len(&server_key_exchange_orig) - CBS_len(&server_key_exchange));

  /* ServerKeyExchange should be signed by the server's public key. */
  if (ssl_cipher_has_server_public_key(s->s3->tmp.new_cipher)) {
    pkey = X509_get_pubkey(s->session->peer);
    if (pkey == NULL) {
      goto err;
    }

    if (SSL_USE_SIGALGS(s)) {
      if (!tls12_check_peer_sigalg(&md, &al, s, &server_key_exchange, pkey)) {
        goto f_err;
      }
    } else if (pkey->type == EVP_PKEY_RSA) {
      md = EVP_md5_sha1();
    } else {
      md = EVP_sha1();
    }

    /* The last field in |server_key_exchange| is the signature. */
    CBS signature;
    if (!CBS_get_u16_length_prefixed(&server_key_exchange, &signature) ||
        CBS_len(&server_key_exchange) != 0) {
      al = SSL_AD_DECODE_ERROR;
      OPENSSL_PUT_ERROR(SSL, SSL_R_DECODE_ERROR);
      goto f_err;
    }

    if (!EVP_DigestVerifyInit(&md_ctx, NULL, md, NULL, pkey) ||
        !EVP_DigestVerifyUpdate(&md_ctx, s->s3->client_random,
                                SSL3_RANDOM_SIZE) ||
        !EVP_DigestVerifyUpdate(&md_ctx, s->s3->server_random,
                                SSL3_RANDOM_SIZE) ||
        !EVP_DigestVerifyUpdate(&md_ctx, CBS_data(&parameter),
                                CBS_len(&parameter)) ||
        !EVP_DigestVerifyFinal(&md_ctx, CBS_data(&signature),
                               CBS_len(&signature))) {
      /* bad signature */
      al = SSL_AD_DECRYPT_ERROR;
      OPENSSL_PUT_ERROR(SSL, SSL_R_BAD_SIGNATURE);
      goto f_err;
    }
  } else {
    /* PSK ciphers are the only supported certificate-less ciphers. */
    assert(alg_a == SSL_aPSK);

    if (CBS_len(&server_key_exchange) > 0) {
      al = SSL_AD_DECODE_ERROR;
      OPENSSL_PUT_ERROR(SSL, SSL_R_EXTRA_DATA_IN_MESSAGE);
      goto f_err;
    }
  }
  EVP_PKEY_free(pkey);
  EVP_MD_CTX_cleanup(&md_ctx);
  return 1;

f_err:
  ssl3_send_alert(s, SSL3_AL_FATAL, al);
err:
  EVP_PKEY_free(pkey);
  RSA_free(rsa);
  DH_free(dh);
  BN_CTX_free(bn_ctx);
  EC_POINT_free(srvr_ecpoint);
  EC_KEY_free(ecdh);
  EVP_MD_CTX_cleanup(&md_ctx);
  return -1;
}

static int ca_dn_cmp(const X509_NAME **a, const X509_NAME **b) {
  return X509_NAME_cmp(*a, *b);
}

int ssl3_get_certificate_request(SSL *s) {
  int ok, ret = 0;
  unsigned long n;
  X509_NAME *xn = NULL;
  STACK_OF(X509_NAME) *ca_sk = NULL;
  CBS cbs;
  CBS certificate_types;
  CBS certificate_authorities;
  const uint8_t *data;

  n = s->method->ssl_get_message(s, SSL3_ST_CR_CERT_REQ_A,
                                 SSL3_ST_CR_CERT_REQ_B, -1, s->max_cert_list,
                                 ssl_hash_message, &ok);

  if (!ok) {
    return n;
  }

  s->s3->tmp.cert_req = 0;

  if (s->s3->tmp.message_type == SSL3_MT_SERVER_DONE) {
    s->s3->tmp.reuse_message = 1;
    /* If we get here we don't need the handshake buffer as we won't be doing
     * client auth. */
    ssl3_free_handshake_buffer(s);
    return 1;
  }

  if (s->s3->tmp.message_type != SSL3_MT_CERTIFICATE_REQUEST) {
    ssl3_send_alert(s, SSL3_AL_FATAL, SSL_AD_UNEXPECTED_MESSAGE);
    OPENSSL_PUT_ERROR(SSL, SSL_R_WRONG_MESSAGE_TYPE);
    goto err;
  }

  CBS_init(&cbs, s->init_msg, n);

  ca_sk = sk_X509_NAME_new(ca_dn_cmp);
  if (ca_sk == NULL) {
    OPENSSL_PUT_ERROR(SSL, ERR_R_MALLOC_FAILURE);
    goto err;
  }

  /* get the certificate types */
  if (!CBS_get_u8_length_prefixed(&cbs, &certificate_types)) {
    ssl3_send_alert(s, SSL3_AL_FATAL, SSL_AD_DECODE_ERROR);
    OPENSSL_PUT_ERROR(SSL, SSL_R_DECODE_ERROR);
    goto err;
  }

  if (!CBS_stow(&certificate_types, &s->s3->tmp.certificate_types,
                &s->s3->tmp.num_certificate_types)) {
    ssl3_send_alert(s, SSL3_AL_FATAL, SSL_AD_INTERNAL_ERROR);
    goto err;
  }

  if (SSL_USE_SIGALGS(s)) {
    CBS supported_signature_algorithms;
    if (!CBS_get_u16_length_prefixed(&cbs, &supported_signature_algorithms) ||
        !tls1_parse_peer_sigalgs(s, &supported_signature_algorithms)) {
      ssl3_send_alert(s, SSL3_AL_FATAL, SSL_AD_DECODE_ERROR);
      OPENSSL_PUT_ERROR(SSL, SSL_R_DECODE_ERROR);
      goto err;
    }
  }

  /* get the CA RDNs */
  if (!CBS_get_u16_length_prefixed(&cbs, &certificate_authorities)) {
    ssl3_send_alert(s, SSL3_AL_FATAL, SSL_AD_DECODE_ERROR);
    OPENSSL_PUT_ERROR(SSL, SSL_R_LENGTH_MISMATCH);
    goto err;
  }

  while (CBS_len(&certificate_authorities) > 0) {
    CBS distinguished_name;
    if (!CBS_get_u16_length_prefixed(&certificate_authorities,
                                     &distinguished_name)) {
      ssl3_send_alert(s, SSL3_AL_FATAL, SSL_AD_DECODE_ERROR);
      OPENSSL_PUT_ERROR(SSL, SSL_R_CA_DN_TOO_LONG);
      goto err;
    }

    data = CBS_data(&distinguished_name);

    xn = d2i_X509_NAME(NULL, &data, CBS_len(&distinguished_name));
    if (xn == NULL) {
      ssl3_send_alert(s, SSL3_AL_FATAL, SSL_AD_DECODE_ERROR);
      OPENSSL_PUT_ERROR(SSL, ERR_R_ASN1_LIB);
      goto err;
    }

    if (!CBS_skip(&distinguished_name, data - CBS_data(&distinguished_name))) {
      ssl3_send_alert(s, SSL3_AL_FATAL, SSL_AD_DECODE_ERROR);
      OPENSSL_PUT_ERROR(SSL, ERR_R_INTERNAL_ERROR);
      goto err;
    }

    if (CBS_len(&distinguished_name) != 0) {
      ssl3_send_alert(s, SSL3_AL_FATAL, SSL_AD_DECODE_ERROR);
      OPENSSL_PUT_ERROR(SSL, SSL_R_CA_DN_LENGTH_MISMATCH);
      goto err;
    }

    if (!sk_X509_NAME_push(ca_sk, xn)) {
      OPENSSL_PUT_ERROR(SSL, ERR_R_MALLOC_FAILURE);
      goto err;
    }
  }

  /* we should setup a certificate to return.... */
  s->s3->tmp.cert_req = 1;
  sk_X509_NAME_pop_free(s->s3->tmp.ca_names, X509_NAME_free);
  s->s3->tmp.ca_names = ca_sk;
  ca_sk = NULL;

  ret = 1;

err:
  sk_X509_NAME_pop_free(ca_sk, X509_NAME_free);
  return ret;
}

int ssl3_get_new_session_ticket(SSL *s) {
  int ok, al;
  long n;
  CBS new_session_ticket, ticket;

  n = s->method->ssl_get_message(
      s, SSL3_ST_CR_SESSION_TICKET_A, SSL3_ST_CR_SESSION_TICKET_B,
      SSL3_MT_NEWSESSION_TICKET, 16384, ssl_hash_message, &ok);

  if (!ok) {
    return n;
  }

  if (s->hit) {
    /* The server is sending a new ticket for an existing session. Sessions are
     * immutable once established, so duplicate all but the ticket of the
     * existing session. */
    uint8_t *bytes;
    size_t bytes_len;
    if (!SSL_SESSION_to_bytes_for_ticket(s->session, &bytes, &bytes_len)) {
      goto err;
    }
    SSL_SESSION *new_session = SSL_SESSION_from_bytes(bytes, bytes_len);
    OPENSSL_free(bytes);
    if (new_session == NULL) {
      /* This should never happen. */
      OPENSSL_PUT_ERROR(SSL, ERR_R_INTERNAL_ERROR);
      goto err;
    }

    SSL_SESSION_free(s->session);
    s->session = new_session;
  }

  CBS_init(&new_session_ticket, s->init_msg, n);

  if (!CBS_get_u32(&new_session_ticket,
                   &s->session->tlsext_tick_lifetime_hint) ||
      !CBS_get_u16_length_prefixed(&new_session_ticket, &ticket) ||
      CBS_len(&new_session_ticket) != 0) {
    al = SSL_AD_DECODE_ERROR;
    OPENSSL_PUT_ERROR(SSL, SSL_R_DECODE_ERROR);
    goto f_err;
  }

  if (!CBS_stow(&ticket, &s->session->tlsext_tick,
                &s->session->tlsext_ticklen)) {
    OPENSSL_PUT_ERROR(SSL, ERR_R_MALLOC_FAILURE);
    goto err;
  }

  /* Generate a session ID for this session based on the session ticket. We use
   * the session ID mechanism for detecting ticket resumption. This also fits in
   * with assumptions elsewhere in OpenSSL.*/
  if (!EVP_Digest(CBS_data(&ticket), CBS_len(&ticket), s->session->session_id,
                  &s->session->session_id_length, EVP_sha256(), NULL)) {
    goto err;
  }

  return 1;

f_err:
  ssl3_send_alert(s, SSL3_AL_FATAL, al);
err:
  return -1;
}

int ssl3_get_cert_status(SSL *s) {
  int ok, al;
  long n;
  CBS certificate_status, ocsp_response;
  uint8_t status_type;

  n = s->method->ssl_get_message(
      s, SSL3_ST_CR_CERT_STATUS_A, SSL3_ST_CR_CERT_STATUS_B,
      -1, 16384, ssl_hash_message, &ok);

  if (!ok) {
    return n;
  }

  if (s->s3->tmp.message_type != SSL3_MT_CERTIFICATE_STATUS) {
    /* A server may send status_request in ServerHello and then change
     * its mind about sending CertificateStatus. */
    s->s3->tmp.reuse_message = 1;
    return 1;
  }

  CBS_init(&certificate_status, s->init_msg, n);
  if (!CBS_get_u8(&certificate_status, &status_type) ||
      status_type != TLSEXT_STATUSTYPE_ocsp ||
      !CBS_get_u24_length_prefixed(&certificate_status, &ocsp_response) ||
      CBS_len(&ocsp_response) == 0 ||
      CBS_len(&certificate_status) != 0) {
    al = SSL_AD_DECODE_ERROR;
    OPENSSL_PUT_ERROR(SSL, SSL_R_DECODE_ERROR);
    goto f_err;
  }

  if (!CBS_stow(&ocsp_response, &s->session->ocsp_response,
                &s->session->ocsp_response_length)) {
    al = SSL_AD_INTERNAL_ERROR;
    OPENSSL_PUT_ERROR(SSL, ERR_R_MALLOC_FAILURE);
    goto f_err;
  }
  return 1;

f_err:
  ssl3_send_alert(s, SSL3_AL_FATAL, al);
  return -1;
}

int ssl3_get_server_done(SSL *s) {
  int ok;
  long n;

  n = s->method->ssl_get_message(s, SSL3_ST_CR_SRVR_DONE_A,
                                 SSL3_ST_CR_SRVR_DONE_B, SSL3_MT_SERVER_DONE,
                                 30, /* should be very small, like 0 :-) */
                                 ssl_hash_message, &ok);

  if (!ok) {
    return n;
  }

  if (n > 0) {
    /* should contain no data */
    ssl3_send_alert(s, SSL3_AL_FATAL, SSL_AD_DECODE_ERROR);
    OPENSSL_PUT_ERROR(SSL, SSL_R_LENGTH_MISMATCH);
    return -1;
  }

  return 1;
}


int ssl3_send_client_key_exchange(SSL *s) {
  uint8_t *p;
  int n = 0;
  uint32_t alg_k;
  uint32_t alg_a;
  uint8_t *q;
  EVP_PKEY *pkey = NULL;
  EC_KEY *clnt_ecdh = NULL;
  const EC_POINT *srvr_ecpoint = NULL;
  EVP_PKEY *srvr_pub_pkey = NULL;
  uint8_t *encodedPoint = NULL;
  int encoded_pt_len = 0;
  BN_CTX *bn_ctx = NULL;
  unsigned int psk_len = 0;
  uint8_t psk[PSK_MAX_PSK_LEN];
  uint8_t *pms = NULL;
  size_t pms_len = 0;

  if (s->state == SSL3_ST_CW_KEY_EXCH_A) {
    p = ssl_handshake_start(s);

    alg_k = s->s3->tmp.new_cipher->algorithm_mkey;
    alg_a = s->s3->tmp.new_cipher->algorithm_auth;

    /* If using a PSK key exchange, prepare the pre-shared key. */
    if (alg_a & SSL_aPSK) {
      char identity[PSK_MAX_IDENTITY_LEN + 1];
      size_t identity_len;

      if (s->psk_client_callback == NULL) {
        OPENSSL_PUT_ERROR(SSL, SSL_R_PSK_NO_CLIENT_CB);
        goto err;
      }

      memset(identity, 0, sizeof(identity));
      psk_len =
          s->psk_client_callback(s, s->s3->tmp.peer_psk_identity_hint, identity,
                                 sizeof(identity), psk, sizeof(psk));
      if (psk_len > PSK_MAX_PSK_LEN) {
        OPENSSL_PUT_ERROR(SSL, ERR_R_INTERNAL_ERROR);
        goto err;
      } else if (psk_len == 0) {
        OPENSSL_PUT_ERROR(SSL, SSL_R_PSK_IDENTITY_NOT_FOUND);
        ssl3_send_alert(s, SSL3_AL_FATAL, SSL_AD_HANDSHAKE_FAILURE);
        goto err;
      }

      identity_len = OPENSSL_strnlen(identity, sizeof(identity));
      if (identity_len > PSK_MAX_IDENTITY_LEN) {
        OPENSSL_PUT_ERROR(SSL, ERR_R_INTERNAL_ERROR);
        goto err;
      }

      OPENSSL_free(s->session->psk_identity);
      s->session->psk_identity = BUF_strdup(identity);
      if (s->session->psk_identity == NULL) {
        OPENSSL_PUT_ERROR(SSL, ERR_R_MALLOC_FAILURE);
        goto err;
      }

      /* Write out psk_identity. */
      s2n(identity_len, p);
      memcpy(p, identity, identity_len);
      p += identity_len;
      n = 2 + identity_len;
    }

    /* Depending on the key exchange method, compute |pms| and |pms_len|. */
    if (alg_k & SSL_kRSA) {
      RSA *rsa;
      size_t enc_pms_len;

      pms_len = SSL_MAX_MASTER_KEY_LENGTH;
      pms = OPENSSL_malloc(pms_len);
      if (pms == NULL) {
        OPENSSL_PUT_ERROR(SSL, ERR_R_MALLOC_FAILURE);
        goto err;
      }

      pkey = X509_get_pubkey(s->session->peer);
      if (pkey == NULL ||
          pkey->type != EVP_PKEY_RSA ||
          pkey->pkey.rsa == NULL) {
        OPENSSL_PUT_ERROR(SSL, ERR_R_INTERNAL_ERROR);
        EVP_PKEY_free(pkey);
        goto err;
      }

      s->session->key_exchange_info = EVP_PKEY_bits(pkey);
      rsa = pkey->pkey.rsa;
      EVP_PKEY_free(pkey);

      pms[0] = s->client_version >> 8;
      pms[1] = s->client_version & 0xff;
      if (!RAND_bytes(&pms[2], SSL_MAX_MASTER_KEY_LENGTH - 2)) {
        goto err;
      }

      s->session->master_key_length = SSL_MAX_MASTER_KEY_LENGTH;

      q = p;
      /* In TLS and beyond, reserve space for the length prefix. */
      if (s->version > SSL3_VERSION) {
        p += 2;
        n += 2;
      }
      if (!RSA_encrypt(rsa, &enc_pms_len, p, RSA_size(rsa), pms, pms_len,
                       RSA_PKCS1_PADDING)) {
        OPENSSL_PUT_ERROR(SSL, SSL_R_BAD_RSA_ENCRYPT);
        goto err;
      }
      n += enc_pms_len;

      /* Log the premaster secret, if logging is enabled. */
      if (!ssl_ctx_log_rsa_client_key_exchange(s->ctx, p, enc_pms_len, pms,
                                               pms_len)) {
        goto err;
      }

      /* Fill in the length prefix. */
      if (s->version > SSL3_VERSION) {
        s2n(enc_pms_len, q);
      }
    } else if (alg_k & SSL_kDHE) {
      DH *dh_srvr, *dh_clnt;
      int dh_len;
      size_t pub_len;

      if (s->s3->tmp.peer_dh_tmp == NULL) {
        OPENSSL_PUT_ERROR(SSL, ERR_R_INTERNAL_ERROR);
        goto err;
      }
      dh_srvr = s->s3->tmp.peer_dh_tmp;

      /* generate a new random key */
      dh_clnt = DHparams_dup(dh_srvr);
      if (dh_clnt == NULL) {
        OPENSSL_PUT_ERROR(SSL, ERR_R_DH_LIB);
        goto err;
      }
      if (!DH_generate_key(dh_clnt)) {
        OPENSSL_PUT_ERROR(SSL, ERR_R_DH_LIB);
        DH_free(dh_clnt);
        goto err;
      }

      pms_len = DH_size(dh_clnt);
      pms = OPENSSL_malloc(pms_len);
      if (pms == NULL) {
        OPENSSL_PUT_ERROR(SSL, ERR_R_MALLOC_FAILURE);
        DH_free(dh_clnt);
        goto err;
      }

      dh_len = DH_compute_key(pms, dh_srvr->pub_key, dh_clnt);
      if (dh_len <= 0) {
        OPENSSL_PUT_ERROR(SSL, ERR_R_DH_LIB);
        DH_free(dh_clnt);
        goto err;
      }
      pms_len = dh_len;

      /* send off the data */
      pub_len = BN_num_bytes(dh_clnt->pub_key);
      s2n(pub_len, p);
      BN_bn2bin(dh_clnt->pub_key, p);
      n += 2 + pub_len;

      DH_free(dh_clnt);
    } else if (alg_k & SSL_kECDHE) {
      const EC_GROUP *srvr_group = NULL;
      EC_KEY *tkey;
      int field_size = 0, ecdh_len;

      if (s->s3->tmp.peer_ecdh_tmp == NULL) {
        OPENSSL_PUT_ERROR(SSL, ERR_R_INTERNAL_ERROR);
        goto err;
      }

      tkey = s->s3->tmp.peer_ecdh_tmp;

      srvr_group = EC_KEY_get0_group(tkey);
      srvr_ecpoint = EC_KEY_get0_public_key(tkey);
      if (srvr_group == NULL || srvr_ecpoint == NULL) {
        OPENSSL_PUT_ERROR(SSL, ERR_R_INTERNAL_ERROR);
        goto err;
      }

      clnt_ecdh = EC_KEY_new();
      if (clnt_ecdh == NULL) {
        OPENSSL_PUT_ERROR(SSL, ERR_R_MALLOC_FAILURE);
        goto err;
      }

      if (!EC_KEY_set_group(clnt_ecdh, srvr_group)) {
        OPENSSL_PUT_ERROR(SSL, ERR_R_EC_LIB);
        goto err;
      }

      /* Generate a new ECDH key pair */
      if (!EC_KEY_generate_key(clnt_ecdh)) {
        OPENSSL_PUT_ERROR(SSL, ERR_R_ECDH_LIB);
        goto err;
      }

      field_size = EC_GROUP_get_degree(srvr_group);
      if (field_size <= 0) {
        OPENSSL_PUT_ERROR(SSL, ERR_R_ECDH_LIB);
        goto err;
      }

      pms_len = (field_size + 7) / 8;
      pms = OPENSSL_malloc(pms_len);
      if (pms == NULL) {
        OPENSSL_PUT_ERROR(SSL, ERR_R_MALLOC_FAILURE);
        goto err;
      }

      ecdh_len = ECDH_compute_key(pms, pms_len, srvr_ecpoint, clnt_ecdh, NULL);
      if (ecdh_len <= 0) {
        OPENSSL_PUT_ERROR(SSL, ERR_R_ECDH_LIB);
        goto err;
      }
      pms_len = ecdh_len;

      /* First check the size of encoding and allocate memory accordingly. */
      encoded_pt_len =
          EC_POINT_point2oct(srvr_group, EC_KEY_get0_public_key(clnt_ecdh),
                             POINT_CONVERSION_UNCOMPRESSED, NULL, 0, NULL);

      encodedPoint =
          (uint8_t *)OPENSSL_malloc(encoded_pt_len * sizeof(uint8_t));
      bn_ctx = BN_CTX_new();
      if (encodedPoint == NULL || bn_ctx == NULL) {
        OPENSSL_PUT_ERROR(SSL, ERR_R_MALLOC_FAILURE);
        goto err;
      }

      /* Encode the public key */
      encoded_pt_len = EC_POINT_point2oct(
          srvr_group, EC_KEY_get0_public_key(clnt_ecdh),
          POINT_CONVERSION_UNCOMPRESSED, encodedPoint, encoded_pt_len, bn_ctx);

      *p = encoded_pt_len; /* length of encoded point */
      /* Encoded point will be copied here */
      p += 1;
      n += 1;
      /* copy the point */
      memcpy(p, encodedPoint, encoded_pt_len);
      /* increment n to account for length field */
      n += encoded_pt_len;

      /* Free allocated memory */
      BN_CTX_free(bn_ctx);
      bn_ctx = NULL;
      OPENSSL_free(encodedPoint);
      encodedPoint = NULL;
      EC_KEY_free(clnt_ecdh);
      clnt_ecdh = NULL;
      EVP_PKEY_free(srvr_pub_pkey);
      srvr_pub_pkey = NULL;
    } else if (alg_k & SSL_kPSK) {
      /* For plain PSK, other_secret is a block of 0s with the same length as
       * the pre-shared key. */
      pms_len = psk_len;
      pms = OPENSSL_malloc(pms_len);
      if (pms == NULL) {
        OPENSSL_PUT_ERROR(SSL, ERR_R_MALLOC_FAILURE);
        goto err;
      }
      memset(pms, 0, pms_len);
    } else {
      ssl3_send_alert(s, SSL3_AL_FATAL, SSL_AD_HANDSHAKE_FAILURE);
      OPENSSL_PUT_ERROR(SSL, ERR_R_INTERNAL_ERROR);
      goto err;
    }

    /* For a PSK cipher suite, other_secret is combined with the pre-shared
     * key. */
    if (alg_a & SSL_aPSK) {
      CBB cbb, child;
      uint8_t *new_pms;
      size_t new_pms_len;

      CBB_zero(&cbb);
      if (!CBB_init(&cbb, 2 + psk_len + 2 + pms_len) ||
          !CBB_add_u16_length_prefixed(&cbb, &child) ||
          !CBB_add_bytes(&child, pms, pms_len) ||
          !CBB_add_u16_length_prefixed(&cbb, &child) ||
          !CBB_add_bytes(&child, psk, psk_len) ||
          !CBB_finish(&cbb, &new_pms, &new_pms_len)) {
        CBB_cleanup(&cbb);
        OPENSSL_PUT_ERROR(SSL, ERR_R_MALLOC_FAILURE);
        goto err;
      }
      OPENSSL_cleanse(pms, pms_len);
      OPENSSL_free(pms);
      pms = new_pms;
      pms_len = new_pms_len;
    }

    /* The message must be added to the finished hash before calculating the
     * master secret. */
    if (!ssl_set_handshake_header(s, SSL3_MT_CLIENT_KEY_EXCHANGE, n)) {
      goto err;
    }
    s->state = SSL3_ST_CW_KEY_EXCH_B;

    s->session->master_key_length = s->enc_method->generate_master_secret(
        s, s->session->master_key, pms, pms_len);
    if (s->session->master_key_length == 0) {
      goto err;
    }
    s->session->extended_master_secret = s->s3->tmp.extended_master_secret;
    OPENSSL_cleanse(pms, pms_len);
    OPENSSL_free(pms);
  }

  /* SSL3_ST_CW_KEY_EXCH_B */
  return s->method->do_write(s);

err:
  BN_CTX_free(bn_ctx);
  OPENSSL_free(encodedPoint);
  EC_KEY_free(clnt_ecdh);
  EVP_PKEY_free(srvr_pub_pkey);
  if (pms) {
    OPENSSL_cleanse(pms, pms_len);
    OPENSSL_free(pms);
  }
  return -1;
}

int ssl3_send_cert_verify(SSL *s) {
  if (s->state == SSL3_ST_CW_CERT_VRFY_A ||
      s->state == SSL3_ST_CW_CERT_VRFY_B) {
    enum ssl_private_key_result_t sign_result;
    uint8_t *p = ssl_handshake_start(s);
    size_t signature_length = 0;
    unsigned long n = 0;
    assert(ssl_has_private_key(s));

    if (s->state == SSL3_ST_CW_CERT_VRFY_A) {
      uint8_t *buf = (uint8_t *)s->init_buf->data;
      const EVP_MD *md = NULL;
      uint8_t digest[EVP_MAX_MD_SIZE];
      size_t digest_length;

      /* Write out the digest type if need be. */
      if (SSL_USE_SIGALGS(s)) {
        md = tls1_choose_signing_digest(s);
        if (!tls12_get_sigandhash(s, p, md)) {
          OPENSSL_PUT_ERROR(SSL, ERR_R_INTERNAL_ERROR);
          return -1;
        }
        p += 2;
        n += 2;
      }

      /* Compute the digest. */
      const int pkey_type = ssl_private_key_type(s);
      if (!ssl3_cert_verify_hash(s, digest, &digest_length, &md, pkey_type)) {
        return -1;
      }

      /* The handshake buffer is no longer necessary. */
      ssl3_free_handshake_buffer(s);

      /* Sign the digest. */
      signature_length = ssl_private_key_max_signature_len(s);
      if (p + 2 + signature_length > buf + SSL3_RT_MAX_PLAIN_LENGTH) {
        OPENSSL_PUT_ERROR(SSL, SSL_R_DATA_LENGTH_TOO_LONG);
        return -1;
      }

      s->rwstate = SSL_PRIVATE_KEY_OPERATION;
      sign_result = ssl_private_key_sign(s, &p[2], &signature_length,
                                         signature_length, md, digest,
                                         digest_length);
    } else {
      if (SSL_USE_SIGALGS(s)) {
        /* The digest has already been selected and written. */
        p += 2;
        n += 2;
      }
      signature_length = ssl_private_key_max_signature_len(s);
      s->rwstate = SSL_PRIVATE_KEY_OPERATION;
      sign_result = ssl_private_key_sign_complete(s, &p[2], &signature_length,
                                                  signature_length);
    }

    if (sign_result == ssl_private_key_retry) {
      s->state = SSL3_ST_CW_CERT_VRFY_B;
      return -1;
    }
    s->rwstate = SSL_NOTHING;
    if (sign_result != ssl_private_key_success) {
      return -1;
    }

    s2n(signature_length, p);
    n += signature_length + 2;
    if (!ssl_set_handshake_header(s, SSL3_MT_CERTIFICATE_VERIFY, n)) {
      return -1;
    }
    s->state = SSL3_ST_CW_CERT_VRFY_C;
  }

  return ssl_do_write(s);
}

/* ssl3_has_client_certificate returns true if a client certificate is
 * configured. */
static int ssl3_has_client_certificate(SSL *ssl) {
  return ssl->cert && ssl->cert->x509 && ssl_has_private_key(ssl);
}

int ssl3_send_client_certificate(SSL *s) {
  X509 *x509 = NULL;
  EVP_PKEY *pkey = NULL;
  int i;

  if (s->state == SSL3_ST_CW_CERT_A) {
    /* Let cert callback update client certificates if required */
    if (s->cert->cert_cb) {
      i = s->cert->cert_cb(s, s->cert->cert_cb_arg);
      if (i < 0) {
        s->rwstate = SSL_X509_LOOKUP;
        return -1;
      }
      if (i == 0) {
        ssl3_send_alert(s, SSL3_AL_FATAL, SSL_AD_INTERNAL_ERROR);
        return 0;
      }
      s->rwstate = SSL_NOTHING;
    }

    if (ssl3_has_client_certificate(s)) {
      s->state = SSL3_ST_CW_CERT_C;
    } else {
      s->state = SSL3_ST_CW_CERT_B;
    }
  }

  /* We need to get a client cert */
  if (s->state == SSL3_ST_CW_CERT_B) {
    /* If we get an error, we need to:
     *   ssl->rwstate=SSL_X509_LOOKUP; return(-1);
     * We then get retried later */
    i = ssl_do_client_cert_cb(s, &x509, &pkey);
    if (i < 0) {
      s->rwstate = SSL_X509_LOOKUP;
      return -1;
    }
    s->rwstate = SSL_NOTHING;
    if (i == 1 && pkey != NULL && x509 != NULL) {
      s->state = SSL3_ST_CW_CERT_B;
      if (!SSL_use_certificate(s, x509) || !SSL_use_PrivateKey(s, pkey)) {
        i = 0;
      }
    } else if (i == 1) {
      i = 0;
      OPENSSL_PUT_ERROR(SSL, SSL_R_BAD_DATA_RETURNED_BY_CALLBACK);
    }

    X509_free(x509);
    EVP_PKEY_free(pkey);
    if (i && !ssl3_has_client_certificate(s)) {
      i = 0;
    }
    if (i == 0) {
      if (s->version == SSL3_VERSION) {
        s->s3->tmp.cert_req = 0;
        ssl3_send_alert(s, SSL3_AL_WARNING, SSL_AD_NO_CERTIFICATE);
        return 1;
      } else {
        s->s3->tmp.cert_req = 2;
        /* There is no client certificate, so the handshake buffer may be
         * released. */
        ssl3_free_handshake_buffer(s);
      }
    }

    /* Ok, we have a cert */
    s->state = SSL3_ST_CW_CERT_C;
  }

  if (s->state == SSL3_ST_CW_CERT_C) {
    if (s->s3->tmp.cert_req == 2) {
      /* Send an empty Certificate message. */
      uint8_t *p = ssl_handshake_start(s);
      l2n3(0, p);
      if (!ssl_set_handshake_header(s, SSL3_MT_CERTIFICATE, 3)) {
        return -1;
      }
    } else if (!ssl3_output_cert_chain(s)) {
      return -1;
    }
    s->state = SSL3_ST_CW_CERT_D;
  }

  /* SSL3_ST_CW_CERT_D */
  return ssl_do_write(s);
}

int ssl3_send_next_proto(SSL *s) {
  unsigned int len, padding_len;
  uint8_t *d, *p;

  if (s->state == SSL3_ST_CW_NEXT_PROTO_A) {
    len = s->next_proto_negotiated_len;
    padding_len = 32 - ((len + 2) % 32);

    d = p = ssl_handshake_start(s);
    *(p++) = len;
    memcpy(p, s->next_proto_negotiated, len);
    p += len;
    *(p++) = padding_len;
    memset(p, 0, padding_len);
    p += padding_len;

    if (!ssl_set_handshake_header(s, SSL3_MT_NEXT_PROTO, p - d)) {
      return -1;
    }
    s->state = SSL3_ST_CW_NEXT_PROTO_B;
  }

  return ssl_do_write(s);
}

int ssl3_send_channel_id(SSL *s) {
  uint8_t *d;
  int ret = -1, public_key_len;
  EVP_MD_CTX md_ctx;
  ECDSA_SIG *sig = NULL;
  uint8_t *public_key = NULL, *derp, *der_sig = NULL;

  if (s->state != SSL3_ST_CW_CHANNEL_ID_A) {
    return ssl_do_write(s);
  }

  if (!s->tlsext_channel_id_private && s->ctx->channel_id_cb) {
    EVP_PKEY *key = NULL;
    s->ctx->channel_id_cb(s, &key);
    if (key != NULL) {
      s->tlsext_channel_id_private = key;
    }
  }

  if (!s->tlsext_channel_id_private) {
    s->rwstate = SSL_CHANNEL_ID_LOOKUP;
    return -1;
  }
  s->rwstate = SSL_NOTHING;

  if (EVP_PKEY_id(s->tlsext_channel_id_private) != EVP_PKEY_EC) {
    OPENSSL_PUT_ERROR(SSL, ERR_R_INTERNAL_ERROR);
    return -1;
  }
  EC_KEY *ec_key = s->tlsext_channel_id_private->pkey.ec;

  d = ssl_handshake_start(s);
  s2n(TLSEXT_TYPE_channel_id, d);
  s2n(TLSEXT_CHANNEL_ID_SIZE, d);

  EVP_MD_CTX_init(&md_ctx);

  public_key_len = i2o_ECPublicKey(ec_key, NULL);
  if (public_key_len <= 0) {
    OPENSSL_PUT_ERROR(SSL, SSL_R_CANNOT_SERIALIZE_PUBLIC_KEY);
    goto err;
  }

  /* i2o_ECPublicKey will produce an ANSI X9.62 public key which, for a
   * P-256 key, is 0x04 (meaning uncompressed) followed by the x and y
   * field elements as 32-byte, big-endian numbers. */
  if (public_key_len != 65) {
    OPENSSL_PUT_ERROR(SSL, SSL_R_CHANNEL_ID_NOT_P256);
    goto err;
  }
  public_key = OPENSSL_malloc(public_key_len);
  if (!public_key) {
    OPENSSL_PUT_ERROR(SSL, ERR_R_MALLOC_FAILURE);
    goto err;
  }

  derp = public_key;
  i2o_ECPublicKey(ec_key, &derp);

  uint8_t digest[EVP_MAX_MD_SIZE];
  size_t digest_len;
  if (!tls1_channel_id_hash(s, digest, &digest_len)) {
    goto err;
  }

  sig = ECDSA_do_sign(digest, digest_len, ec_key);
  if (sig == NULL) {
    goto err;
  }

  /* The first byte of public_key will be 0x4, denoting an uncompressed key. */
  memcpy(d, public_key + 1, 64);
  d += 64;
  if (!BN_bn2bin_padded(d, 32, sig->r) ||
      !BN_bn2bin_padded(d + 32, 32, sig->s)) {
    OPENSSL_PUT_ERROR(SSL, ERR_R_INTERNAL_ERROR);
    goto err;
  }

  if (!ssl_set_handshake_header(s, SSL3_MT_ENCRYPTED_EXTENSIONS,
                                2 + 2 + TLSEXT_CHANNEL_ID_SIZE)) {
    goto err;
  }
  s->state = SSL3_ST_CW_CHANNEL_ID_B;

  ret = ssl_do_write(s);

err:
  EVP_MD_CTX_cleanup(&md_ctx);
  OPENSSL_free(public_key);
  OPENSSL_free(der_sig);
  ECDSA_SIG_free(sig);

  return ret;
}

int ssl_do_client_cert_cb(SSL *s, X509 **px509, EVP_PKEY **ppkey) {
  int i = 0;
  if (s->ctx->client_cert_cb) {
    i = s->ctx->client_cert_cb(s, px509, ppkey);
  }
  return i;
}

int ssl3_verify_server_cert(SSL *s) {
  int ret = ssl_verify_cert_chain(s, s->session->cert_chain);
  if (s->verify_mode != SSL_VERIFY_NONE && ret <= 0) {
    int al = ssl_verify_alarm_type(s->verify_result);
    ssl3_send_alert(s, SSL3_AL_FATAL, al);
    OPENSSL_PUT_ERROR(SSL, SSL_R_CERTIFICATE_VERIFY_FAILED);
  } else {
    ret = 1;
    ERR_clear_error(); /* but we keep s->verify_result */
  }

  return ret;
}
