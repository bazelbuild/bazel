/*
 * DTLS implementation written by Nagendra Modadugu
 * (nagendra@cs.stanford.edu) for the OpenSSL project 2005. 
 */
/* ====================================================================
 * Copyright (c) 1999-2007 The OpenSSL Project.  All rights reserved.
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
 *    for use in the OpenSSL Toolkit. (http://www.OpenSSL.org/)"
 *
 * 4. The names "OpenSSL Toolkit" and "OpenSSL Project" must not be used to
 *    endorse or promote products derived from this software without
 *    prior written permission. For written permission, please contact
 *    openssl-core@OpenSSL.org.
 *
 * 5. Products derived from this software may not be called "OpenSSL"
 *    nor may "OpenSSL" appear in their names without prior written
 *    permission of the OpenSSL Project.
 *
 * 6. Redistributions of any form whatsoever must retain the following
 *    acknowledgment:
 *    "This product includes software developed by the OpenSSL Project
 *    for use in the OpenSSL Toolkit (http://www.OpenSSL.org/)"
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

#include <openssl/ssl.h>

#include <assert.h>
#include <stdio.h>

#include <openssl/bn.h>
#include <openssl/buf.h>
#include <openssl/dh.h>
#include <openssl/err.h>
#include <openssl/evp.h>
#include <openssl/md5.h>
#include <openssl/obj.h>
#include <openssl/rand.h>
#include <openssl/x509.h>

#include "internal.h"


int dtls1_accept(SSL *s) {
  BUF_MEM *buf = NULL;
  void (*cb)(const SSL *ssl, int type, int val) = NULL;
  uint32_t alg_a;
  int ret = -1;
  int new_state, state, skip = 0;

  assert(s->handshake_func == dtls1_accept);
  assert(s->server);
  assert(SSL_IS_DTLS(s));

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
      case SSL_ST_ACCEPT:
        if (cb != NULL) {
          cb(s, SSL_CB_HANDSHAKE_START, 1);
        }

        if (s->init_buf == NULL) {
          buf = BUF_MEM_new();
          if (buf == NULL || !BUF_MEM_grow(buf, SSL3_RT_MAX_PLAIN_LENGTH)) {
            ret = -1;
            goto end;
          }
          s->init_buf = buf;
          buf = NULL;
        }

        s->init_num = 0;

        if (!ssl_init_wbio_buffer(s, 1)) {
          ret = -1;
          goto end;
        }

        if (!ssl3_init_handshake_buffer(s)) {
          OPENSSL_PUT_ERROR(SSL, ERR_R_INTERNAL_ERROR);
          ret = -1;
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
        dtls1_stop_timer(s);
        s->state = SSL3_ST_SW_SRVR_HELLO_A;
        s->init_num = 0;
        break;

      case SSL3_ST_SW_SRVR_HELLO_A:
      case SSL3_ST_SW_SRVR_HELLO_B:
        dtls1_start_timer(s);
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
          dtls1_start_timer(s);
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
         * TODO(davidben): This logic is currently duplicated
         * in s3_srvr.c. Fix this. In the meantime, keep them
         * in sync. */
        if (ssl_cipher_requires_server_key_exchange(s->s3->tmp.new_cipher) ||
            ((alg_a & SSL_aPSK) && s->psk_identity_hint)) {
          dtls1_start_timer(s);
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
          dtls1_start_timer(s);
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
        dtls1_start_timer(s);
        ret = ssl3_send_server_done(s);
        if (ret <= 0) {
          goto end;
        }
        s->s3->tmp.next_state = SSL3_ST_SR_CERT_A;
        s->state = SSL3_ST_SW_FLUSH;
        s->init_num = 0;
        break;

      case SSL3_ST_SW_FLUSH:
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
        s->state = SSL3_ST_SR_FINISHED_A;
        s->init_num = 0;
        break;

      case SSL3_ST_SR_FINISHED_A:
      case SSL3_ST_SR_FINISHED_B:
        s->d1->change_cipher_spec_ok = 1;
        ret =
            ssl3_get_finished(s, SSL3_ST_SR_FINISHED_A, SSL3_ST_SR_FINISHED_B);
        if (ret <= 0) {
          goto end;
        }
        dtls1_stop_timer(s);
        if (s->hit) {
          s->state = SSL_ST_OK;
        } else if (s->tlsext_ticket_expected) {
          s->state = SSL3_ST_SW_SESSION_TICKET_A;
        } else {
          s->state = SSL3_ST_SW_CHANGE_A;
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

        ret = dtls1_send_change_cipher_spec(s, SSL3_ST_SW_CHANGE_A,
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

        dtls1_reset_seq_numbers(s, SSL3_CC_WRITE);
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
          s->s3->tmp.next_state = SSL3_ST_SR_FINISHED_A;
        } else {
          s->s3->tmp.next_state = SSL_ST_OK;
        }
        s->init_num = 0;
        break;

      case SSL_ST_OK:
        ssl3_cleanup_key_block(s);

        /* remove buffering on output */
        ssl_free_wbio_buffer(s);

        s->init_num = 0;
        s->s3->initial_handshake_complete = 1;

        ssl_update_cache(s, SSL_SESS_CACHE_SERVER);

        if (cb != NULL) {
          cb(s, SSL_CB_HANDSHAKE_DONE, 1);
        }

        ret = 1;

        /* done handshaking, next message is client hello */
        s->d1->handshake_read_seq = 0;
        /* next message is server hello */
        s->d1->handshake_write_seq = 0;
        s->d1->next_handshake_write_seq = 0;
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
        cb(s, SSL_CB_ACCEPT_LOOP, 1);
        s->state = new_state;
      }
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
