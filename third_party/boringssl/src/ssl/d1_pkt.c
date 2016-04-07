/* DTLS implementation written by Nagendra Modadugu
 * (nagendra@cs.stanford.edu) for the OpenSSL project 2005. */
/* ====================================================================
 * Copyright (c) 1998-2005 The OpenSSL Project.  All rights reserved.
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
 * [including the GNU Public Licence.] */

#include <openssl/ssl.h>

#include <assert.h>
#include <stdio.h>
#include <string.h>

#include <openssl/buf.h>
#include <openssl/mem.h>
#include <openssl/evp.h>
#include <openssl/err.h>
#include <openssl/rand.h>

#include "internal.h"


static int do_dtls1_write(SSL *s, int type, const uint8_t *buf,
                          unsigned int len, enum dtls1_use_epoch_t use_epoch);

/* dtls1_get_record reads a new input record. On success, it places it in
 * |ssl->s3->rrec| and returns one. Otherwise it returns <= 0 on error or if
 * more data is needed. */
static int dtls1_get_record(SSL *ssl) {
again:
  /* Read a new packet if there is no unconsumed one. */
  if (ssl_read_buffer_len(ssl) == 0) {
    int ret = ssl_read_buffer_extend_to(ssl, 0 /* unused */);
    if (ret <= 0) {
      return ret;
    }
  }
  assert(ssl_read_buffer_len(ssl) > 0);

  /* Ensure the packet is large enough to decrypt in-place. */
  if (ssl_read_buffer_len(ssl) < ssl_record_prefix_len(ssl)) {
    ssl_read_buffer_clear(ssl);
    goto again;
  }

  uint8_t *out = ssl_read_buffer(ssl) + ssl_record_prefix_len(ssl);
  size_t max_out = ssl_read_buffer_len(ssl) - ssl_record_prefix_len(ssl);
  uint8_t type, alert;
  size_t len, consumed;
  switch (dtls_open_record(ssl, &type, out, &len, &consumed, &alert, max_out,
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

    case ssl_open_record_discard:
      ssl_read_buffer_consume(ssl, consumed);
      goto again;

    case ssl_open_record_error:
      ssl3_send_alert(ssl, SSL3_AL_FATAL, alert);
      return -1;

    case ssl_open_record_partial:
      /* Impossible in DTLS. */
      break;
  }

  assert(0);
  OPENSSL_PUT_ERROR(SSL, ERR_R_INTERNAL_ERROR);
  return -1;
}

int dtls1_read_app_data(SSL *ssl, uint8_t *buf, int len, int peek) {
  return dtls1_read_bytes(ssl, SSL3_RT_APPLICATION_DATA, buf, len, peek);
}

void dtls1_read_close_notify(SSL *ssl) {
  /* Bidirectional shutdown doesn't make sense for an unordered transport. DTLS
   * alerts also aren't delivered reliably, so we may even time out because the
   * peer never received our close_notify. Report to the caller that the channel
   * has fully shut down. */
  ssl->shutdown |= SSL_RECEIVED_SHUTDOWN;
}

/* Return up to 'len' payload bytes received in 'type' records.
 * 'type' is one of the following:
 *
 *   -  SSL3_RT_HANDSHAKE (when ssl3_get_message calls us)
 *   -  SSL3_RT_APPLICATION_DATA (when ssl3_read calls us)
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
int dtls1_read_bytes(SSL *s, int type, unsigned char *buf, int len, int peek) {
  int al, i, ret;
  unsigned int n;
  SSL3_RECORD *rr;
  void (*cb)(const SSL *ssl, int type2, int val) = NULL;

  if ((type != SSL3_RT_APPLICATION_DATA && type != SSL3_RT_HANDSHAKE) ||
      (peek && type != SSL3_RT_APPLICATION_DATA)) {
    OPENSSL_PUT_ERROR(SSL, ERR_R_INTERNAL_ERROR);
    return -1;
  }

  if (!s->in_handshake && SSL_in_init(s)) {
    /* type == SSL3_RT_APPLICATION_DATA */
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

  /* s->s3->rrec.type     - is the type of record
   * s->s3->rrec.data     - data
   * s->s3->rrec.off      - offset into 'data' for next read
   * s->s3->rrec.length   - number of bytes. */
  rr = &s->s3->rrec;

  /* Check for timeout */
  if (DTLSv1_handle_timeout(s) > 0) {
    goto start;
  }

  /* get new packet if necessary */
  if (rr->length == 0) {
    ret = dtls1_get_record(s);
    if (ret <= 0) {
      ret = dtls1_read_failed(s, ret);
      /* anything other than a timeout is an error */
      if (ret <= 0) {
        return ret;
      } else {
        goto start;
      }
    }
  }

  /* we now have a packet which can be read and processed */

  /* |change_cipher_spec is set when we receive a ChangeCipherSpec and reset by
   * ssl3_get_finished. */
  if (s->s3->change_cipher_spec && rr->type != SSL3_RT_HANDSHAKE &&
      rr->type != SSL3_RT_ALERT) {
    /* We now have an unexpected record between CCS and Finished. Most likely
     * the packets were reordered on their way. DTLS is unreliable, so drop the
     * packet and expect the peer to retransmit. */
    rr->length = 0;
    goto start;
  }

  /* If the other end has shut down, throw anything we read away (even in
   * 'peek' mode) */
  if (s->shutdown & SSL_RECEIVED_SHUTDOWN) {
    rr->length = 0;
    s->rwstate = SSL_NOTHING;
    return 0;
  }


  if (type == rr->type) { /* SSL3_RT_APPLICATION_DATA or SSL3_RT_HANDSHAKE */
    /* make sure that we are not getting application data when we
     * are doing a handshake for the first time */
    if (SSL_in_init(s) && (type == SSL3_RT_APPLICATION_DATA) &&
        (s->aead_read_ctx == NULL)) {
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

  /* If we get here, then type != rr->type. */

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
    } else if (alert_level == SSL3_AL_FATAL) {
      char tmp[16];

      s->rwstate = SSL_NOTHING;
      s->s3->fatal_alert = alert_descr;
      OPENSSL_PUT_ERROR(SSL, SSL_AD_REASON_OFFSET + alert_descr);
      BIO_snprintf(tmp, sizeof tmp, "%d", alert_descr);
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

  if (rr->type == SSL3_RT_CHANGE_CIPHER_SPEC) {
    /* 'Change Cipher Spec' is just a single byte, so we know exactly what the
     * record payload has to look like */
    if (rr->length != 1 || rr->off != 0 || rr->data[0] != SSL3_MT_CCS) {
      al = SSL_AD_ILLEGAL_PARAMETER;
      OPENSSL_PUT_ERROR(SSL, SSL_R_BAD_CHANGE_CIPHER_SPEC);
      goto f_err;
    }

    rr->length = 0;

    if (s->msg_callback) {
      s->msg_callback(0, s->version, SSL3_RT_CHANGE_CIPHER_SPEC, rr->data, 1, s,
                      s->msg_callback_arg);
    }

    /* We can't process a CCS now, because previous handshake
     * messages are still missing, so just drop it.
     */
    if (!s->d1->change_cipher_spec_ok) {
      goto start;
    }

    s->d1->change_cipher_spec_ok = 0;

    s->s3->change_cipher_spec = 1;
    if (!ssl3_do_change_cipher_spec(s)) {
      goto err;
    }

    /* do this whenever CCS is processed */
    dtls1_reset_seq_numbers(s, SSL3_CC_READ);

    goto start;
  }

  /* Unexpected handshake message. It may be a retransmitted Finished (the only
   * post-CCS message). Otherwise, it's a pre-CCS handshake message from an
   * unsupported renegotiation attempt. */
  if (rr->type == SSL3_RT_HANDSHAKE && !s->in_handshake) {
    if (rr->length < DTLS1_HM_HEADER_LENGTH) {
      al = SSL_AD_DECODE_ERROR;
      OPENSSL_PUT_ERROR(SSL, SSL_R_BAD_HANDSHAKE_RECORD);
      goto f_err;
    }
    struct hm_header_st msg_hdr;
    dtls1_get_message_header(&rr->data[rr->off], &msg_hdr);

    /* Ignore a stray Finished from the previous handshake. */
    if (msg_hdr.type == SSL3_MT_FINISHED) {
      if (msg_hdr.frag_off == 0) {
        /* Retransmit our last flight of messages. If the peer sends the second
         * Finished, they may not have received ours. Only do this for the
         * first fragment, in case the Finished was fragmented. */
        if (dtls1_check_timeout_num(s) < 0) {
          return -1;
        }

        dtls1_retransmit_buffered_messages(s);
      }

      rr->length = 0;
      goto start;
    }
  }

  /* We already handled these. */
  assert(rr->type != SSL3_RT_CHANGE_CIPHER_SPEC && rr->type != SSL3_RT_ALERT);

  al = SSL_AD_UNEXPECTED_MESSAGE;
  OPENSSL_PUT_ERROR(SSL, SSL_R_UNEXPECTED_RECORD);

f_err:
  ssl3_send_alert(s, SSL3_AL_FATAL, al);
err:
  return -1;
}

int dtls1_write_app_data(SSL *s, const void *buf_, int len) {
  int i;

  if (SSL_in_init(s) && !s->in_handshake) {
    i = s->handshake_func(s);
    if (i < 0) {
      return i;
    }
    if (i == 0) {
      OPENSSL_PUT_ERROR(SSL, SSL_R_SSL_HANDSHAKE_FAILURE);
      return -1;
    }
  }

  if (len > SSL3_RT_MAX_PLAIN_LENGTH) {
    OPENSSL_PUT_ERROR(SSL, SSL_R_DTLS_MESSAGE_TOO_BIG);
    return -1;
  }

  i = dtls1_write_bytes(s, SSL3_RT_APPLICATION_DATA, buf_, len,
                        dtls1_use_current_epoch);
  return i;
}

/* Call this to write data in records of type 'type' It will return <= 0 if not
 * all data has been sent or non-blocking IO. */
int dtls1_write_bytes(SSL *s, int type, const void *buf, int len,
                      enum dtls1_use_epoch_t use_epoch) {
  int i;

  assert(len <= SSL3_RT_MAX_PLAIN_LENGTH);
  s->rwstate = SSL_NOTHING;
  i = do_dtls1_write(s, type, buf, len, use_epoch);
  return i;
}

static int do_dtls1_write(SSL *s, int type, const uint8_t *buf,
                          unsigned int len, enum dtls1_use_epoch_t use_epoch) {
  /* ssl3_write_pending drops the write if |BIO_write| fails in DTLS, so there
   * is never pending data. */
  assert(!ssl_write_buffer_is_pending(s));

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
  uint8_t *out;
  size_t ciphertext_len;
  if (!ssl_write_buffer_init(s, &out, max_out) ||
      !dtls_seal_record(s, out, &ciphertext_len, max_out, type, buf, len,
                        use_epoch)) {
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

int dtls1_dispatch_alert(SSL *s) {
  int i, j;
  void (*cb)(const SSL *ssl, int type, int val) = NULL;
  uint8_t buf[DTLS1_AL_HEADER_LENGTH];
  uint8_t *ptr = &buf[0];

  s->s3->alert_dispatch = 0;

  memset(buf, 0x00, sizeof(buf));
  *ptr++ = s->s3->send_alert[0];
  *ptr++ = s->s3->send_alert[1];

  i = do_dtls1_write(s, SSL3_RT_ALERT, &buf[0], sizeof(buf),
                     dtls1_use_current_epoch);
  if (i <= 0) {
    s->s3->alert_dispatch = 1;
  } else {
    if (s->s3->send_alert[0] == SSL3_AL_FATAL) {
      (void)BIO_flush(s->wbio);
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

void dtls1_reset_seq_numbers(SSL *s, int rw) {
  uint8_t *seq;
  unsigned int seq_bytes = sizeof(s->s3->read_sequence);

  if (rw & SSL3_CC_READ) {
    seq = s->s3->read_sequence;
    s->d1->r_epoch++;
    memset(&s->d1->bitmap, 0, sizeof(DTLS1_BITMAP));
  } else {
    seq = s->s3->write_sequence;
    memcpy(s->d1->last_write_sequence, seq, sizeof(s->s3->write_sequence));
    s->d1->w_epoch++;
  }

  memset(seq, 0x00, seq_bytes);
}
