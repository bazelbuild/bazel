/*
 * DTLS implementation written by Nagendra Modadugu
 * (nagendra@cs.stanford.edu) for the OpenSSL project 2005.
 */
/* ====================================================================
 * Copyright (c) 1999-2005 The OpenSSL Project.  All rights reserved.
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
 * Hudson (tjh@cryptsoft.com). */

#include <openssl/ssl.h>

#include <limits.h>
#include <stdio.h>
#include <string.h>

#include <openssl/err.h>
#include <openssl/mem.h>
#include <openssl/obj.h>

#include "internal.h"

#if defined(OPENSSL_WINDOWS)
#include <sys/timeb.h>
#else
#include <sys/socket.h>
#include <sys/time.h>
#endif


/* DTLS1_MTU_TIMEOUTS is the maximum number of timeouts to expire
 * before starting to decrease the MTU. */
#define DTLS1_MTU_TIMEOUTS                     2

/* DTLS1_MAX_TIMEOUTS is the maximum number of timeouts to expire
 * before failing the DTLS handshake. */
#define DTLS1_MAX_TIMEOUTS                     12

static void get_current_time(const SSL *ssl, struct timeval *out_clock);

int dtls1_new(SSL *s) {
  DTLS1_STATE *d1;

  if (!ssl3_new(s)) {
    return 0;
  }
  d1 = OPENSSL_malloc(sizeof *d1);
  if (d1 == NULL) {
    ssl3_free(s);
    return 0;
  }
  memset(d1, 0, sizeof *d1);

  d1->buffered_messages = pqueue_new();
  d1->sent_messages = pqueue_new();

  if (!d1->buffered_messages || !d1->sent_messages) {
    pqueue_free(d1->buffered_messages);
    pqueue_free(d1->sent_messages);
    OPENSSL_free(d1);
    ssl3_free(s);
    return 0;
  }

  s->d1 = d1;

  /* Set the version to the highest version for DTLS. This controls the initial
   * state of |s->enc_method| and what the API reports as the version prior to
   * negotiation.
   *
   * TODO(davidben): This is fragile and confusing. */
  s->version = DTLS1_2_VERSION;
  return 1;
}

static void dtls1_clear_queues(SSL *s) {
  pitem *item = NULL;
  hm_fragment *frag = NULL;

  while ((item = pqueue_pop(s->d1->buffered_messages)) != NULL) {
    frag = (hm_fragment *)item->data;
    dtls1_hm_fragment_free(frag);
    pitem_free(item);
  }

  while ((item = pqueue_pop(s->d1->sent_messages)) != NULL) {
    frag = (hm_fragment *)item->data;
    dtls1_hm_fragment_free(frag);
    pitem_free(item);
  }
}

void dtls1_free(SSL *s) {
  ssl3_free(s);

  if (s == NULL || s->d1 == NULL) {
    return;
  }

  dtls1_clear_queues(s);

  pqueue_free(s->d1->buffered_messages);
  pqueue_free(s->d1->sent_messages);

  OPENSSL_free(s->d1);
  s->d1 = NULL;
}

int dtls1_supports_cipher(const SSL_CIPHER *cipher) {
  /* DTLS does not support stream ciphers. The NULL cipher is rejected because
   * it's not needed. */
  return cipher->algorithm_enc != SSL_RC4 && cipher->algorithm_enc != SSL_eNULL;
}

void dtls1_start_timer(SSL *s) {
  /* If timer is not set, initialize duration with 1 second */
  if (s->d1->next_timeout.tv_sec == 0 && s->d1->next_timeout.tv_usec == 0) {
    s->d1->timeout_duration = 1;
  }

  /* Set timeout to current time */
  get_current_time(s, &s->d1->next_timeout);

  /* Add duration to current time */
  s->d1->next_timeout.tv_sec += s->d1->timeout_duration;
  BIO_ctrl(SSL_get_rbio(s), BIO_CTRL_DGRAM_SET_NEXT_TIMEOUT, 0,
           &s->d1->next_timeout);
}

int DTLSv1_get_timeout(const SSL *ssl, struct timeval *out) {
  if (!SSL_IS_DTLS(ssl)) {
    return 0;
  }

  /* If no timeout is set, just return NULL */
  if (ssl->d1->next_timeout.tv_sec == 0 && ssl->d1->next_timeout.tv_usec == 0) {
    return 0;
  }

  /* Get current time */
  struct timeval timenow;
  get_current_time(ssl, &timenow);

  /* If timer already expired, set remaining time to 0 */
  if (ssl->d1->next_timeout.tv_sec < timenow.tv_sec ||
      (ssl->d1->next_timeout.tv_sec == timenow.tv_sec &&
       ssl->d1->next_timeout.tv_usec <= timenow.tv_usec)) {
    memset(out, 0, sizeof(struct timeval));
    return 1;
  }

  /* Calculate time left until timer expires */
  memcpy(out, &ssl->d1->next_timeout, sizeof(struct timeval));
  out->tv_sec -= timenow.tv_sec;
  out->tv_usec -= timenow.tv_usec;
  if (out->tv_usec < 0) {
    out->tv_sec--;
    out->tv_usec += 1000000;
  }

  /* If remaining time is less than 15 ms, set it to 0 to prevent issues
   * because of small devergences with socket timeouts. */
  if (out->tv_sec == 0 && out->tv_usec < 15000) {
    memset(out, 0, sizeof(struct timeval));
  }

  return 1;
}

int dtls1_is_timer_expired(SSL *s) {
  struct timeval timeleft;

  /* Get time left until timeout, return false if no timer running */
  if (!DTLSv1_get_timeout(s, &timeleft)) {
    return 0;
  }

  /* Return false if timer is not expired yet */
  if (timeleft.tv_sec > 0 || timeleft.tv_usec > 0) {
    return 0;
  }

  /* Timer expired, so return true */
  return 1;
}

void dtls1_double_timeout(SSL *s) {
  s->d1->timeout_duration *= 2;
  if (s->d1->timeout_duration > 60) {
    s->d1->timeout_duration = 60;
  }
  dtls1_start_timer(s);
}

void dtls1_stop_timer(SSL *s) {
  /* Reset everything */
  s->d1->num_timeouts = 0;
  memset(&s->d1->next_timeout, 0, sizeof(struct timeval));
  s->d1->timeout_duration = 1;
  BIO_ctrl(SSL_get_rbio(s), BIO_CTRL_DGRAM_SET_NEXT_TIMEOUT, 0,
           &s->d1->next_timeout);
  /* Clear retransmission buffer */
  dtls1_clear_record_buffer(s);
}

int dtls1_check_timeout_num(SSL *s) {
  s->d1->num_timeouts++;

  /* Reduce MTU after 2 unsuccessful retransmissions */
  if (s->d1->num_timeouts > DTLS1_MTU_TIMEOUTS &&
      !(SSL_get_options(s) & SSL_OP_NO_QUERY_MTU)) {
    long mtu = BIO_ctrl(SSL_get_wbio(s), BIO_CTRL_DGRAM_GET_FALLBACK_MTU, 0,
                        NULL);
    if (mtu >= 0 && mtu <= (1 << 30) && (unsigned)mtu >= dtls1_min_mtu()) {
      s->d1->mtu = (unsigned)mtu;
    }
  }

  if (s->d1->num_timeouts > DTLS1_MAX_TIMEOUTS) {
    /* fail the connection, enough alerts have been sent */
    OPENSSL_PUT_ERROR(SSL, SSL_R_READ_TIMEOUT_EXPIRED);
    return -1;
  }

  return 0;
}

int DTLSv1_handle_timeout(SSL *ssl) {
  if (!SSL_IS_DTLS(ssl)) {
    return -1;
  }

  /* if no timer is expired, don't do anything */
  if (!dtls1_is_timer_expired(ssl)) {
    return 0;
  }

  dtls1_double_timeout(ssl);

  if (dtls1_check_timeout_num(ssl) < 0) {
    return -1;
  }

  dtls1_start_timer(ssl);
  return dtls1_retransmit_buffered_messages(ssl);
}

static void get_current_time(const SSL *ssl, struct timeval *out_clock) {
  if (ssl->ctx->current_time_cb != NULL) {
    ssl->ctx->current_time_cb(ssl, out_clock);
    return;
  }

#if defined(OPENSSL_WINDOWS)
  struct _timeb time;
  _ftime(&time);
  out_clock->tv_sec = time.time;
  out_clock->tv_usec = time.millitm * 1000;
#else
  gettimeofday(out_clock, NULL);
#endif
}

int dtls1_set_handshake_header(SSL *s, int htype, unsigned long len) {
  uint8_t *message = (uint8_t *)s->init_buf->data;
  const struct hm_header_st *msg_hdr = &s->d1->w_msg_hdr;
  uint8_t serialised_header[DTLS1_HM_HEADER_LENGTH];
  uint8_t *p = serialised_header;

  s->d1->handshake_write_seq = s->d1->next_handshake_write_seq;
  s->d1->next_handshake_write_seq++;

  dtls1_set_message_header(s, htype, len, s->d1->handshake_write_seq, 0, len);
  s->init_num = (int)len + DTLS1_HM_HEADER_LENGTH;
  s->init_off = 0;

  /* Buffer the message to handle re-xmits */
  dtls1_buffer_message(s, 0);

  /* Add the new message to the handshake hash. Serialize the message
   * header as if it were a single fragment. */
  *p++ = msg_hdr->type;
  l2n3(msg_hdr->msg_len, p);
  s2n(msg_hdr->seq, p);
  l2n3(0, p);
  l2n3(msg_hdr->msg_len, p);
  return ssl3_update_handshake_hash(s, serialised_header,
                                    sizeof(serialised_header)) &&
         ssl3_update_handshake_hash(s, message + DTLS1_HM_HEADER_LENGTH, len);
}

int dtls1_handshake_write(SSL *s) {
  return dtls1_do_write(s, SSL3_RT_HANDSHAKE, dtls1_use_current_epoch);
}
