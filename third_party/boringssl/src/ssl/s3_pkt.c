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
#include <limits.h>
#include <stdio.h>
#include <string.h>

#include <openssl/buf.h>
#include <openssl/err.h>
#include <openssl/evp.h>
#include <openssl/mem.h>
#include <openssl/rand.h>

#include "internal.h"


static int do_ssl3_write(SSL *s, int type, const uint8_t *buf, unsigned len);

/* kMaxWarningAlerts is the number of consecutive warning alerts that will be
 * processed. */
static const uint8_t kMaxWarningAlerts = 4;

/* ssl3_get_record reads a new input record. On success, it places it in
 * |ssl->s3->rrec| and returns one. Otherwise it returns <= 0 on error or if
 * more data is needed. */
static int ssl3_get_record(SSL *ssl) {
  int ret;
again:
  /* Ensure the buffer is large enough to decrypt in-place. */
  ret = ssl_read_buffer_extend_to(ssl, ssl_record_prefix_len(ssl));
  if (ret <= 0) {
    return ret;
  }
  assert(ssl_read_buffer_len(ssl) >= ssl_record_prefix_len(ssl));

  uint8_t *out = ssl_read_buffer(ssl) + ssl_record_prefix_len(ssl);
  size_t max_out = ssl_read_buffer_len(ssl) - ssl_record_prefix_len(ssl);
  uint8_t type, alert;
  size_t len, consumed;
  switch (tls_open_record(ssl, &type, out, &len, &consumed, &alert, max_out,
                          ssl_read_buffer(ssl), ssl_read_buffer_len(ssl))) {
    case ssl_open_record_success:
      ssl_read_buffer_consume(ssl, consumed);

      if (len > 0xffff) {
        OPENSSL_PUT_ERROR(SSL, ERR_R_OVERFLOW);
        return -1;
      }

      SSL3_RECORD *rr = &ssl->s3->rrec;
      rr->type = type;
      rr->length = (uint16_t)len;
      rr->off = 0;
      rr->data = out;
      return 1;

    case ssl_open_record_partial:
      ret = ssl_read_buffer_extend_to(ssl, consumed);
      if (ret <= 0) {
        return ret;
      }
      goto again;

    case ssl_open_record_discard:
      ssl_read_buffer_consume(ssl, consumed);
      goto again;

    case ssl_open_record_error:
      ssl3_send_alert(ssl, SSL3_AL_FATAL, alert);
      return -1;
  }

  assert(0);
  OPENSSL_PUT_ERROR(SSL, ERR_R_INTERNAL_ERROR);
  return -1;
}

int ssl3_write_app_data(SSL *ssl, const void *buf, int len) {
  return ssl3_write_bytes(ssl, SSL3_RT_APPLICATION_DATA, buf, len);
}

/* Call this to write data in records of type |type|. It will return <= 0 if
 * not all data has been sent or non-blocking IO. */
int ssl3_write_bytes(SSL *s, int type, const void *buf_, int len) {
  const uint8_t *buf = buf_;
  unsigned int tot, n, nw;
  int i;

  s->rwstate = SSL_NOTHING;
  assert(s->s3->wnum <= INT_MAX);
  tot = s->s3->wnum;
  s->s3->wnum = 0;

  if (!s->in_handshake && SSL_in_init(s) && !SSL_in_false_start(s)) {
    i = s->handshake_func(s);
    if (i < 0) {
      return i;
    }
    if (i == 0) {
      OPENSSL_PUT_ERROR(SSL, SSL_R_SSL_HANDSHAKE_FAILURE);
      return -1;
    }
  }

  /* Ensure that if we end up with a smaller value of data to write out than
   * the the original len from a write which didn't complete for non-blocking
   * I/O and also somehow ended up avoiding the check for this in
   * ssl3_write_pending/SSL_R_BAD_WRITE_RETRY as it must never be possible to
   * end up with (len-tot) as a large number that will then promptly send
   * beyond the end of the users buffer ... so we trap and report the error in
   * a way the user will notice. */
  if (len < 0 || (size_t)len < tot) {
    OPENSSL_PUT_ERROR(SSL, SSL_R_BAD_LENGTH);
    return -1;
  }

  n = (len - tot);
  for (;;) {
    /* max contains the maximum number of bytes that we can put into a
     * record. */
    unsigned max = s->max_send_fragment;
    if (n > max) {
      nw = max;
    } else {
      nw = n;
    }

    i = do_ssl3_write(s, type, &buf[tot], nw);
    if (i <= 0) {
      s->s3->wnum = tot;
      return i;
    }

    if (i == (int)n || (type == SSL3_RT_APPLICATION_DATA &&
                        (s->mode & SSL_MODE_ENABLE_PARTIAL_WRITE))) {
      return tot + i;
    }

    n -= i;
    tot += i;
  }
}

/* do_ssl3_write writes an SSL record of the given type. */
static int do_ssl3_write(SSL *s, int type, const uint8_t *buf, unsigned len) {
  /* If there is still data from the previous record, flush it. */
  if (ssl_write_buffer_is_pending(s)) {
    return ssl3_write_pending(s, type, buf, len);
  }

  /* If we have an alert to send, lets send it */
  if (s->s3->alert_dispatch) {
    int ret = s->method->ssl_dispatch_alert(s);
    if (ret <= 0) {
      return ret;
    }
    /* if it went, fall through and send more stuff */
  }

  if (len > SSL3_RT_MAX_PLAIN_LENGTH) {
    OPENSSL_PUT_ERROR(SSL, ERR_R_INTERNAL_ERROR);
    return -1;
  }

  if (len == 0) {
    return 0;
  }

  size_t max_out = len + ssl_max_seal_overhead(s);
  if (max_out < len) {
    OPENSSL_PUT_ERROR(SSL, ERR_R_OVERFLOW);
    return -1;
  }
  uint8_t *out;
  size_t ciphertext_len;
  if (!ssl_write_buffer_init(s, &out, max_out) ||
      !tls_seal_record(s, out, &ciphertext_len, max_out, type, buf, len)) {
    return -1;
  }
  ssl_write_buffer_set_len(s, ciphertext_len);

  /* memorize arguments so that ssl3_write_pending can detect bad write retries
   * later */
  s->s3->wpend_tot = len;
  s->s3->wpend_buf = buf;
  s->s3->wpend_type = type;
  s->s3->wpend_ret = len;

  /* we now just need to write the buffer */
  return ssl3_write_pending(s, type, buf, len);
}

int ssl3_write_pending(SSL *s, int type, const uint8_t *buf, unsigned int len) {
  if (s->s3->wpend_tot > (int)len ||
      (s->s3->wpend_buf != buf &&
       !(s->mode & SSL_MODE_ACCEPT_MOVING_WRITE_BUFFER)) ||
      s->s3->wpend_type != type) {
    OPENSSL_PUT_ERROR(SSL, SSL_R_BAD_WRITE_RETRY);
    return -1;
  }

  int ret = ssl_write_buffer_flush(s);
  if (ret <= 0) {
    return ret;
  }
  return s->s3->wpend_ret;
}

/* ssl3_expect_change_cipher_spec informs the record layer that a
 * ChangeCipherSpec record is required at this point. If a Handshake record is
 * received before ChangeCipherSpec, the connection will fail. Moreover, if
 * there are unprocessed handshake bytes, the handshake will also fail and the
 * function returns zero. Otherwise, the function returns one. */
int ssl3_expect_change_cipher_spec(SSL *s) {
  if (s->s3->handshake_fragment_len > 0 || s->s3->tmp.reuse_message) {
    OPENSSL_PUT_ERROR(SSL, SSL_R_UNPROCESSED_HANDSHAKE_DATA);
    return 0;
  }

  s->s3->flags |= SSL3_FLAGS_EXPECT_CCS;
  return 1;
}

int ssl3_read_app_data(SSL *ssl, uint8_t *buf, int len, int peek) {
  return ssl3_read_bytes(ssl, SSL3_RT_APPLICATION_DATA, buf, len, peek);
}

void ssl3_read_close_notify(SSL *ssl) {
  ssl3_read_bytes(ssl, 0, NULL, 0, 0);
}

/* Return up to 'len' payload bytes received in 'type' records.
 * 'type' is one of the following:
 *
 *   -  SSL3_RT_HANDSHAKE (when ssl3_get_message calls us)
 *   -  SSL3_RT_APPLICATION_DATA (when ssl3_read calls us)
 *   -  0 (during a shutdown, no data has to be returned)
 *
 * If we don't have stored data to work from, read a SSL/TLS record first
 * (possibly multiple records if we still don't have anything to return).
 *
 * This function must handle any surprises the peer may have for us, such as
 * Alert records (e.g. close_notify), ChangeCipherSpec records (not really
 * a surprise, but handled as if it were), or renegotiation requests.
 * Also if record payloads contain fragments too small to process, we store
 * them until there is enough for the respective protocol (the record protocol
 * may use arbitrary fragmentation and even interleaving):
 *     Change cipher spec protocol
 *             just 1 byte needed, no need for keeping anything stored
 *     Alert protocol
 *             2 bytes needed (AlertLevel, AlertDescription)
 *     Handshake protocol
 *             4 bytes needed (HandshakeType, uint24 length) -- we just have
 *             to detect unexpected Client Hello and Hello Request messages
 *             here, anything else is handled by higher layers
 *     Application data protocol
 *             none of our business
 */
int ssl3_read_bytes(SSL *s, int type, uint8_t *buf, int len, int peek) {
  int al, i, ret;
  unsigned int n;
  SSL3_RECORD *rr;
  void (*cb)(const SSL *ssl, int type2, int val) = NULL;

  if ((type && type != SSL3_RT_APPLICATION_DATA && type != SSL3_RT_HANDSHAKE) ||
      (peek && type != SSL3_RT_APPLICATION_DATA)) {
    OPENSSL_PUT_ERROR(SSL, ERR_R_INTERNAL_ERROR);
    return -1;
  }

  if (type == SSL3_RT_HANDSHAKE && s->s3->handshake_fragment_len > 0) {
    /* (partially) satisfy request from storage */
    uint8_t *src = s->s3->handshake_fragment;
    uint8_t *dst = buf;
    unsigned int k;

    /* peek == 0 */
    n = 0;
    while (len > 0 && s->s3->handshake_fragment_len > 0) {
      *dst++ = *src++;
      len--;
      s->s3->handshake_fragment_len--;
      n++;
    }
    /* move any remaining fragment bytes: */
    for (k = 0; k < s->s3->handshake_fragment_len; k++) {
      s->s3->handshake_fragment[k] = *src++;
    }
    return n;
  }

  /* Now s->s3->handshake_fragment_len == 0 if type == SSL3_RT_HANDSHAKE. */

  /* This may require multiple iterations. False Start will cause
   * |s->handshake_func| to signal success one step early, but the handshake
   * must be completely finished before other modes are accepted.
   *
   * TODO(davidben): Move this check up to a higher level. */
  while (!s->in_handshake && SSL_in_init(s)) {
    assert(type == SSL3_RT_APPLICATION_DATA);
    i = s->handshake_func(s);
    if (i < 0) {
      return i;
    }
    if (i == 0) {
      OPENSSL_PUT_ERROR(SSL, SSL_R_SSL_HANDSHAKE_FAILURE);
      return -1;
    }
  }

start:
  s->rwstate = SSL_NOTHING;

  /* s->s3->rrec.type    - is the type of record
   * s->s3->rrec.data    - data
   * s->s3->rrec.off     - offset into 'data' for next read
   * s->s3->rrec.length  - number of bytes. */
  rr = &s->s3->rrec;

  /* get new packet if necessary */
  if (rr->length == 0) {
    ret = ssl3_get_record(s);
    if (ret <= 0) {
      return ret;
    }
  }

  /* we now have a packet which can be read and processed */

  /* |change_cipher_spec is set when we receive a ChangeCipherSpec and reset by
   * ssl3_get_finished. */
  if (s->s3->change_cipher_spec && rr->type != SSL3_RT_HANDSHAKE &&
      rr->type != SSL3_RT_ALERT) {
    al = SSL_AD_UNEXPECTED_MESSAGE;
    OPENSSL_PUT_ERROR(SSL, SSL_R_DATA_BETWEEN_CCS_AND_FINISHED);
    goto f_err;
  }

  /* If we are expecting a ChangeCipherSpec, it is illegal to receive a
   * Handshake record. */
  if (rr->type == SSL3_RT_HANDSHAKE && (s->s3->flags & SSL3_FLAGS_EXPECT_CCS)) {
    al = SSL_AD_UNEXPECTED_MESSAGE;
    OPENSSL_PUT_ERROR(SSL, SSL_R_HANDSHAKE_RECORD_BEFORE_CCS);
    goto f_err;
  }

  /* If the other end has shut down, throw anything we read away (even in
   * 'peek' mode) */
  if (s->shutdown & SSL_RECEIVED_SHUTDOWN) {
    rr->length = 0;
    s->rwstate = SSL_NOTHING;
    return 0;
  }

  if (type != 0 && type == rr->type) {
    s->s3->warning_alert_count = 0;

    /* SSL3_RT_APPLICATION_DATA or SSL3_RT_HANDSHAKE */
    /* make sure that we are not getting application data when we are doing a
     * handshake for the first time */
    if (SSL_in_init(s) && type == SSL3_RT_APPLICATION_DATA &&
        s->aead_read_ctx == NULL) {
      /* TODO(davidben): Is this check redundant with the handshake_func
       * check? */
      al = SSL_AD_UNEXPECTED_MESSAGE;
      OPENSSL_PUT_ERROR(SSL, SSL_R_APP_DATA_IN_HANDSHAKE);
      goto f_err;
    }

    /* Discard empty records. */
    if (rr->length == 0) {
      goto start;
    }

    if (len <= 0) {
      return len;
    }

    if ((unsigned int)len > rr->length) {
      n = rr->length;
    } else {
      n = (unsigned int)len;
    }

    memcpy(buf, &(rr->data[rr->off]), n);
    if (!peek) {
      rr->length -= n;
      rr->off += n;
      if (rr->length == 0) {
        rr->off = 0;
        /* The record has been consumed, so we may now clear the buffer. */
        ssl_read_buffer_discard(s);
      }
    }

    return n;
  }

  /* Process unexpected records. */

  if (rr->type == SSL3_RT_HANDSHAKE) {
    /* If peer renegotiations are disabled, all out-of-order handshake records
     * are fatal. Renegotiations as a server are never supported. */
    if (!s->accept_peer_renegotiations || s->server) {
      al = SSL_AD_NO_RENEGOTIATION;
      OPENSSL_PUT_ERROR(SSL, SSL_R_NO_RENEGOTIATION);
      goto f_err;
    }

    /* HelloRequests may be fragmented across multiple records. */
    const size_t size = sizeof(s->s3->handshake_fragment);
    const size_t avail = size - s->s3->handshake_fragment_len;
    const size_t todo = (rr->length < avail) ? rr->length : avail;
    memcpy(s->s3->handshake_fragment + s->s3->handshake_fragment_len,
           &rr->data[rr->off], todo);
    rr->off += todo;
    rr->length -= todo;
    s->s3->handshake_fragment_len += todo;
    if (s->s3->handshake_fragment_len < size) {
      goto start; /* fragment was too small */
    }

    /* Parse out and consume a HelloRequest. */
    if (s->s3->handshake_fragment[0] != SSL3_MT_HELLO_REQUEST ||
        s->s3->handshake_fragment[1] != 0 ||
        s->s3->handshake_fragment[2] != 0 ||
        s->s3->handshake_fragment[3] != 0) {
      al = SSL_AD_DECODE_ERROR;
      OPENSSL_PUT_ERROR(SSL, SSL_R_BAD_HELLO_REQUEST);
      goto f_err;
    }
    s->s3->handshake_fragment_len = 0;

    if (s->msg_callback) {
      s->msg_callback(0, s->version, SSL3_RT_HANDSHAKE,
                      s->s3->handshake_fragment, 4, s, s->msg_callback_arg);
    }

    if (!SSL_is_init_finished(s) || !s->s3->initial_handshake_complete) {
      /* This cannot happen. If a handshake is in progress, |type| must be
       * |SSL3_RT_HANDSHAKE|. */
      assert(0);
      OPENSSL_PUT_ERROR(SSL, ERR_R_INTERNAL_ERROR);
      goto err;
    }

    /* Renegotiation is only supported at quiescent points in the application
     * protocol, namely in HTTPS, just before reading the HTTP response. Require
     * the record-layer be idle and avoid complexities of sending a handshake
     * record while an application_data record is being written. */
    if (ssl_write_buffer_is_pending(s)) {
      al = SSL_AD_NO_RENEGOTIATION;
      OPENSSL_PUT_ERROR(SSL, SSL_R_NO_RENEGOTIATION);
      goto f_err;
    }

    /* Begin a new handshake. */
    s->state = SSL_ST_CONNECT;
    i = s->handshake_func(s);
    if (i < 0) {
      return i;
    }
    if (i == 0) {
      OPENSSL_PUT_ERROR(SSL, SSL_R_SSL_HANDSHAKE_FAILURE);
      return -1;
    }

    /* The handshake completed synchronously. Continue reading records. */
    goto start;
  }

  /* If an alert record, process one alert out of the record. Note that we allow
   * a single record to contain multiple alerts. */
  if (rr->type == SSL3_RT_ALERT) {
    /* Alerts may not be fragmented. */
    if (rr->length < 2) {
      al = SSL_AD_DECODE_ERROR;
      OPENSSL_PUT_ERROR(SSL, SSL_R_BAD_ALERT);
      goto f_err;
    }

    if (s->msg_callback) {
      s->msg_callback(0, s->version, SSL3_RT_ALERT, &rr->data[rr->off], 2, s,
                      s->msg_callback_arg);
    }
    const uint8_t alert_level = rr->data[rr->off++];
    const uint8_t alert_descr = rr->data[rr->off++];
    rr->length -= 2;

    if (s->info_callback != NULL) {
      cb = s->info_callback;
    } else if (s->ctx->info_callback != NULL) {
      cb = s->ctx->info_callback;
    }

    if (cb != NULL) {
      uint16_t alert = (alert_level << 8) | alert_descr;
      cb(s, SSL_CB_READ_ALERT, alert);
    }

    if (alert_level == SSL3_AL_WARNING) {
      s->s3->warn_alert = alert_descr;
      if (alert_descr == SSL_AD_CLOSE_NOTIFY) {
        s->shutdown |= SSL_RECEIVED_SHUTDOWN;
        return 0;
      }

      /* This is a warning but we receive it if we requested renegotiation and
       * the peer denied it. Terminate with a fatal alert because if
       * application tried to renegotiatie it presumably had a good reason and
       * expects it to succeed.
       *
       * In future we might have a renegotiation where we don't care if the
       * peer refused it where we carry on. */
      else if (alert_descr == SSL_AD_NO_RENEGOTIATION) {
        al = SSL_AD_HANDSHAKE_FAILURE;
        OPENSSL_PUT_ERROR(SSL, SSL_R_NO_RENEGOTIATION);
        goto f_err;
      }

      s->s3->warning_alert_count++;
      if (s->s3->warning_alert_count > kMaxWarningAlerts) {
        al = SSL_AD_UNEXPECTED_MESSAGE;
        OPENSSL_PUT_ERROR(SSL, SSL_R_TOO_MANY_WARNING_ALERTS);
        goto f_err;
      }
    } else if (alert_level == SSL3_AL_FATAL) {
      char tmp[16];

      s->rwstate = SSL_NOTHING;
      s->s3->fatal_alert = alert_descr;
      OPENSSL_PUT_ERROR(SSL, SSL_AD_REASON_OFFSET + alert_descr);
      BIO_snprintf(tmp, sizeof(tmp), "%d", alert_descr);
      ERR_add_error_data(2, "SSL alert number ", tmp);
      s->shutdown |= SSL_RECEIVED_SHUTDOWN;
      SSL_CTX_remove_session(s->ctx, s->session);
      return 0;
    } else {
      al = SSL_AD_ILLEGAL_PARAMETER;
      OPENSSL_PUT_ERROR(SSL, SSL_R_UNKNOWN_ALERT_TYPE);
      goto f_err;
    }

    goto start;
  }

  if (s->shutdown & SSL_SENT_SHUTDOWN) {
    /* close_notify has been sent, so discard all records other than alerts. */
    rr->length = 0;
    goto start;
  }

  if (rr->type == SSL3_RT_CHANGE_CIPHER_SPEC) {
    /* 'Change Cipher Spec' is just a single byte, so we know exactly what the
     * record payload has to look like */
    if (rr->length != 1 || rr->off != 0 || rr->data[0] != SSL3_MT_CCS) {
      al = SSL_AD_ILLEGAL_PARAMETER;
      OPENSSL_PUT_ERROR(SSL, SSL_R_BAD_CHANGE_CIPHER_SPEC);
      goto f_err;
    }

    /* Check we have a cipher to change to */
    if (s->s3->tmp.new_cipher == NULL) {
      al = SSL_AD_UNEXPECTED_MESSAGE;
      OPENSSL_PUT_ERROR(SSL, SSL_R_CCS_RECEIVED_EARLY);
      goto f_err;
    }

    if (!(s->s3->flags & SSL3_FLAGS_EXPECT_CCS)) {
      al = SSL_AD_UNEXPECTED_MESSAGE;
      OPENSSL_PUT_ERROR(SSL, SSL_R_CCS_RECEIVED_EARLY);
      goto f_err;
    }

    s->s3->flags &= ~SSL3_FLAGS_EXPECT_CCS;

    rr->length = 0;

    if (s->msg_callback) {
      s->msg_callback(0, s->version, SSL3_RT_CHANGE_CIPHER_SPEC, rr->data, 1, s,
                      s->msg_callback_arg);
    }

    s->s3->change_cipher_spec = 1;
    if (!ssl3_do_change_cipher_spec(s)) {
      goto err;
    } else {
      goto start;
    }
  }

  /* We already handled these. */
  assert(rr->type != SSL3_RT_CHANGE_CIPHER_SPEC && rr->type != SSL3_RT_ALERT &&
         rr->type != SSL3_RT_HANDSHAKE);

  al = SSL_AD_UNEXPECTED_MESSAGE;
  OPENSSL_PUT_ERROR(SSL, SSL_R_UNEXPECTED_RECORD);

f_err:
  ssl3_send_alert(s, SSL3_AL_FATAL, al);
err:
  return -1;
}

int ssl3_do_change_cipher_spec(SSL *s) {
  int i;

  if (s->state & SSL_ST_ACCEPT) {
    i = SSL3_CHANGE_CIPHER_SERVER_READ;
  } else {
    i = SSL3_CHANGE_CIPHER_CLIENT_READ;
  }

  if (s->s3->tmp.key_block == NULL) {
    if (s->session == NULL || s->session->master_key_length == 0) {
      /* might happen if dtls1_read_bytes() calls this */
      OPENSSL_PUT_ERROR(SSL, SSL_R_CCS_RECEIVED_EARLY);
      return 0;
    }

    s->session->cipher = s->s3->tmp.new_cipher;
    if (!s->enc_method->setup_key_block(s)) {
      return 0;
    }
  }

  if (!s->enc_method->change_cipher_state(s, i)) {
    return 0;
  }

  return 1;
}

int ssl3_send_alert(SSL *s, int level, int desc) {
  /* Map tls/ssl alert value to correct one */
  desc = s->enc_method->alert_value(desc);
  if (s->version == SSL3_VERSION && desc == SSL_AD_PROTOCOL_VERSION) {
    /* SSL 3.0 does not have protocol_version alerts */
    desc = SSL_AD_HANDSHAKE_FAILURE;
  }
  if (desc < 0) {
    return -1;
  }

  /* If a fatal one, remove from cache */
  if (level == 2 && s->session != NULL) {
    SSL_CTX_remove_session(s->ctx, s->session);
  }

  s->s3->alert_dispatch = 1;
  s->s3->send_alert[0] = level;
  s->s3->send_alert[1] = desc;
  if (!ssl_write_buffer_is_pending(s)) {
    /* Nothing is being written out, so the alert may be dispatched
     * immediately. */
    return s->method->ssl_dispatch_alert(s);
  }

  /* else data is still being written out, we will get written some time in the
   * future */
  return -1;
}

int ssl3_dispatch_alert(SSL *s) {
  int i, j;
  void (*cb)(const SSL *ssl, int type, int val) = NULL;

  s->s3->alert_dispatch = 0;
  i = do_ssl3_write(s, SSL3_RT_ALERT, &s->s3->send_alert[0], 2);
  if (i <= 0) {
    s->s3->alert_dispatch = 1;
  } else {
    /* Alert sent to BIO.  If it is important, flush it now. If the message
     * does not get sent due to non-blocking IO, we will not worry too much. */
    if (s->s3->send_alert[0] == SSL3_AL_FATAL) {
      BIO_flush(s->wbio);
    }

    if (s->msg_callback) {
      s->msg_callback(1, s->version, SSL3_RT_ALERT, s->s3->send_alert, 2, s,
                      s->msg_callback_arg);
    }

    if (s->info_callback != NULL) {
      cb = s->info_callback;
    } else if (s->ctx->info_callback != NULL) {
      cb = s->ctx->info_callback;
    }

    if (cb != NULL) {
      j = (s->s3->send_alert[0] << 8) | s->s3->send_alert[1];
      cb(s, SSL_CB_WRITE_ALERT, j);
    }
  }

  return i;
}
