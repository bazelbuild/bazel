/*
 * DTLS implementation written by Nagendra Modadugu
 * (nagendra@cs.stanford.edu) for the OpenSSL project 2005. 
 */
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
#include <limits.h>
#include <stdio.h>
#include <string.h>

#include <openssl/buf.h>
#include <openssl/err.h>
#include <openssl/evp.h>
#include <openssl/mem.h>
#include <openssl/obj.h>
#include <openssl/rand.h>
#include <openssl/x509.h>

#include "internal.h"


/* TODO(davidben): 28 comes from the size of IP + UDP header. Is this reasonable
 * for these values? Notably, why is kMinMTU a function of the transport
 * protocol's overhead rather than, say, what's needed to hold a minimally-sized
 * handshake fragment plus protocol overhead. */

/* kMinMTU is the minimum acceptable MTU value. */
static const unsigned int kMinMTU = 256 - 28;

/* kDefaultMTU is the default MTU value to use if neither the user nor
 * the underlying BIO supplies one. */
static const unsigned int kDefaultMTU = 1500 - 28;

/* kMaxHandshakeBuffer is the maximum number of handshake messages ahead of the
 * current one to buffer. */
static const unsigned int kHandshakeBufferSize = 10;

static void dtls1_fix_message_header(SSL *s, unsigned long frag_off,
                                     unsigned long frag_len);
static unsigned char *dtls1_write_message_header(SSL *s, unsigned char *p);

static hm_fragment *dtls1_hm_fragment_new(size_t frag_len, int reassembly) {
  hm_fragment *frag = OPENSSL_malloc(sizeof(hm_fragment));
  if (frag == NULL) {
    OPENSSL_PUT_ERROR(SSL, ERR_R_MALLOC_FAILURE);
    return NULL;
  }
  memset(frag, 0, sizeof(hm_fragment));

  /* If the handshake message is empty, |frag->fragment| and |frag->reassembly|
   * are NULL. */
  if (frag_len > 0) {
    frag->fragment = OPENSSL_malloc(frag_len);
    if (frag->fragment == NULL) {
      OPENSSL_PUT_ERROR(SSL, ERR_R_MALLOC_FAILURE);
      goto err;
    }

    if (reassembly) {
      /* Initialize reassembly bitmask. */
      if (frag_len + 7 < frag_len) {
        OPENSSL_PUT_ERROR(SSL, ERR_R_OVERFLOW);
        goto err;
      }
      size_t bitmask_len = (frag_len + 7) / 8;
      frag->reassembly = OPENSSL_malloc(bitmask_len);
      if (frag->reassembly == NULL) {
        OPENSSL_PUT_ERROR(SSL, ERR_R_MALLOC_FAILURE);
        goto err;
      }
      memset(frag->reassembly, 0, bitmask_len);
    }
  }

  return frag;

err:
  dtls1_hm_fragment_free(frag);
  return NULL;
}

void dtls1_hm_fragment_free(hm_fragment *frag) {
  if (frag == NULL) {
    return;
  }
  OPENSSL_free(frag->fragment);
  OPENSSL_free(frag->reassembly);
  OPENSSL_free(frag);
}

#if !defined(inline)
#define inline __inline
#endif

/* bit_range returns a |uint8_t| with bits |start|, inclusive, to |end|,
 * exclusive, set. */
static inline uint8_t bit_range(size_t start, size_t end) {
  return (uint8_t)(~((1u << start) - 1) & ((1u << end) - 1));
}

/* dtls1_hm_fragment_mark marks bytes |start|, inclusive, to |end|, exclusive,
 * as received in |frag|. If |frag| becomes complete, it clears
 * |frag->reassembly|. The range must be within the bounds of |frag|'s message
 * and |frag->reassembly| must not be NULL. */
static void dtls1_hm_fragment_mark(hm_fragment *frag, size_t start,
                                   size_t end) {
  size_t i;
  size_t msg_len = frag->msg_header.msg_len;

  if (frag->reassembly == NULL || start > end || end > msg_len) {
    assert(0);
    return;
  }
  /* A zero-length message will never have a pending reassembly. */
  assert(msg_len > 0);

  if ((start >> 3) == (end >> 3)) {
    frag->reassembly[start >> 3] |= bit_range(start & 7, end & 7);
  } else {
    frag->reassembly[start >> 3] |= bit_range(start & 7, 8);
    for (i = (start >> 3) + 1; i < (end >> 3); i++) {
      frag->reassembly[i] = 0xff;
    }
    if ((end & 7) != 0) {
      frag->reassembly[end >> 3] |= bit_range(0, end & 7);
    }
  }

  /* Check if the fragment is complete. */
  for (i = 0; i < (msg_len >> 3); i++) {
    if (frag->reassembly[i] != 0xff) {
      return;
    }
  }
  if ((msg_len & 7) != 0 &&
      frag->reassembly[msg_len >> 3] != bit_range(0, msg_len & 7)) {
    return;
  }

  OPENSSL_free(frag->reassembly);
  frag->reassembly = NULL;
}

/* send s->init_buf in records of type 'type' (SSL3_RT_HANDSHAKE or
 * SSL3_RT_CHANGE_CIPHER_SPEC) */
int dtls1_do_write(SSL *s, int type, enum dtls1_use_epoch_t use_epoch) {
  int ret;
  int curr_mtu;
  unsigned int len, frag_off;

  /* AHA!  Figure out the MTU, and stick to the right size */
  if (s->d1->mtu < dtls1_min_mtu() &&
      !(SSL_get_options(s) & SSL_OP_NO_QUERY_MTU)) {
    long mtu = BIO_ctrl(SSL_get_wbio(s), BIO_CTRL_DGRAM_QUERY_MTU, 0, NULL);
    if (mtu >= 0 && mtu <= (1 << 30) && (unsigned)mtu >= dtls1_min_mtu()) {
      s->d1->mtu = (unsigned)mtu;
    } else {
      s->d1->mtu = kDefaultMTU;
      BIO_ctrl(SSL_get_wbio(s), BIO_CTRL_DGRAM_SET_MTU, s->d1->mtu, NULL);
    }
  }

  /* should have something reasonable now */
  assert(s->d1->mtu >= dtls1_min_mtu());

  if (s->init_off == 0 && type == SSL3_RT_HANDSHAKE) {
    assert(s->init_num ==
           (int)s->d1->w_msg_hdr.msg_len + DTLS1_HM_HEADER_LENGTH);
  }

  /* Determine the maximum overhead of the current cipher. */
  size_t max_overhead = SSL_AEAD_CTX_max_overhead(s->aead_write_ctx);

  frag_off = 0;
  while (s->init_num) {
    /* Account for data in the buffering BIO; multiple records may be packed
     * into a single packet during the handshake.
     *
     * TODO(davidben): This is buggy; if the MTU is larger than the buffer size,
     * the large record will be split across two packets. Moreover, in that
     * case, the |dtls1_write_bytes| call may not return synchronously. This
     * will break on retry as the |s->init_off| and |s->init_num| adjustment
     * will run a second time. */
    curr_mtu = s->d1->mtu - BIO_wpending(SSL_get_wbio(s)) -
        DTLS1_RT_HEADER_LENGTH - max_overhead;

    if (curr_mtu <= DTLS1_HM_HEADER_LENGTH) {
      /* Flush the buffer and continue with a fresh packet.
       *
       * TODO(davidben): If |BIO_flush| is not synchronous and requires multiple
       * calls to |dtls1_do_write|, |frag_off| will be wrong. */
      ret = BIO_flush(SSL_get_wbio(s));
      if (ret <= 0) {
        return ret;
      }
      assert(BIO_wpending(SSL_get_wbio(s)) == 0);
      curr_mtu = s->d1->mtu - DTLS1_RT_HEADER_LENGTH - max_overhead;
    }

    /* XDTLS: this function is too long.  split out the CCS part */
    if (type == SSL3_RT_HANDSHAKE) {
      /* If this isn't the first fragment, reserve space to prepend a new
       * fragment header. This will override the body of a previous fragment. */
      if (s->init_off != 0) {
        assert(s->init_off > DTLS1_HM_HEADER_LENGTH);
        s->init_off -= DTLS1_HM_HEADER_LENGTH;
        s->init_num += DTLS1_HM_HEADER_LENGTH;
      }

      if (curr_mtu <= DTLS1_HM_HEADER_LENGTH) {
        /* To make forward progress, the MTU must, at minimum, fit the handshake
         * header and one byte of handshake body. */
        OPENSSL_PUT_ERROR(SSL, SSL_R_MTU_TOO_SMALL);
        return -1;
      }

      if (s->init_num > curr_mtu) {
        len = curr_mtu;
      } else {
        len = s->init_num;
      }
      assert(len >= DTLS1_HM_HEADER_LENGTH);

      dtls1_fix_message_header(s, frag_off, len - DTLS1_HM_HEADER_LENGTH);
      dtls1_write_message_header(
          s, (uint8_t *)&s->init_buf->data[s->init_off]);
    } else {
      assert(type == SSL3_RT_CHANGE_CIPHER_SPEC);
      /* ChangeCipherSpec cannot be fragmented. */
      if (s->init_num > curr_mtu) {
        OPENSSL_PUT_ERROR(SSL, SSL_R_MTU_TOO_SMALL);
        return -1;
      }
      len = s->init_num;
    }

    ret = dtls1_write_bytes(s, type, &s->init_buf->data[s->init_off], len,
                            use_epoch);
    if (ret < 0) {
      return -1;
    }

    /* bad if this assert fails, only part of the handshake message got sent.
     * But why would this happen? */
    assert(len == (unsigned int)ret);

    if (ret == s->init_num) {
      if (s->msg_callback) {
        s->msg_callback(1, s->version, type, s->init_buf->data,
                        (size_t)(s->init_off + s->init_num), s,
                        s->msg_callback_arg);
      }

      s->init_off = 0; /* done writing this message */
      s->init_num = 0;

      return 1;
    }
    s->init_off += ret;
    s->init_num -= ret;
    frag_off += (ret -= DTLS1_HM_HEADER_LENGTH);
  }

  return 0;
}

/* dtls1_is_next_message_complete returns one if the next handshake message is
 * complete and zero otherwise. */
static int dtls1_is_next_message_complete(SSL *s) {
  pitem *item = pqueue_peek(s->d1->buffered_messages);
  if (item == NULL) {
    return 0;
  }

  hm_fragment *frag = (hm_fragment *)item->data;
  assert(s->d1->handshake_read_seq <= frag->msg_header.seq);

  return s->d1->handshake_read_seq == frag->msg_header.seq &&
      frag->reassembly == NULL;
}

/* dtls1_discard_fragment_body discards a handshake fragment body of length
 * |frag_len|. It returns one on success and zero on error.
 *
 * TODO(davidben): This function will go away when ssl_read_bytes is gone from
 * the DTLS side. */
static int dtls1_discard_fragment_body(SSL *s, size_t frag_len) {
  uint8_t discard[256];
  while (frag_len > 0) {
    size_t chunk = frag_len < sizeof(discard) ? frag_len : sizeof(discard);
    int ret = dtls1_read_bytes(s, SSL3_RT_HANDSHAKE, discard, chunk, 0);
    if (ret != chunk) {
      return 0;
    }
    frag_len -= chunk;
  }
  return 1;
}

/* dtls1_get_buffered_message returns the buffered message corresponding to
 * |msg_hdr|. If none exists, it creates a new one and inserts it in the
 * queue. Otherwise, it checks |msg_hdr| is consistent with the existing one. It
 * returns NULL on failure. The caller does not take ownership of the result. */
static hm_fragment *dtls1_get_buffered_message(
    SSL *s, const struct hm_header_st *msg_hdr) {
  uint8_t seq64be[8];
  memset(seq64be, 0, sizeof(seq64be));
  seq64be[6] = (uint8_t)(msg_hdr->seq >> 8);
  seq64be[7] = (uint8_t)msg_hdr->seq;
  pitem *item = pqueue_find(s->d1->buffered_messages, seq64be);

  hm_fragment *frag;
  if (item == NULL) {
    /* This is the first fragment from this message. */
    frag = dtls1_hm_fragment_new(msg_hdr->msg_len,
                                 1 /* reassembly buffer needed */);
    if (frag == NULL) {
      return NULL;
    }
    memcpy(&frag->msg_header, msg_hdr, sizeof(*msg_hdr));
    item = pitem_new(seq64be, frag);
    if (item == NULL) {
      dtls1_hm_fragment_free(frag);
      return NULL;
    }
    item = pqueue_insert(s->d1->buffered_messages, item);
    /* |pqueue_insert| fails iff a duplicate item is inserted, but |item| cannot
     * be a duplicate. */
    assert(item != NULL);
  } else {
    frag = item->data;
    assert(frag->msg_header.seq == msg_hdr->seq);
    if (frag->msg_header.type != msg_hdr->type ||
        frag->msg_header.msg_len != msg_hdr->msg_len) {
      /* The new fragment must be compatible with the previous fragments from
       * this message. */
      OPENSSL_PUT_ERROR(SSL, SSL_R_FRAGMENT_MISMATCH);
      ssl3_send_alert(s, SSL3_AL_FATAL, SSL_AD_ILLEGAL_PARAMETER);
      return NULL;
    }
  }
  return frag;
}

/* dtls1_max_handshake_message_len returns the maximum number of bytes
 * permitted in a DTLS handshake message for |s|. The minimum is 16KB, but may
 * be greater if the maximum certificate list size requires it. */
static size_t dtls1_max_handshake_message_len(const SSL *s) {
  size_t max_len = DTLS1_HM_HEADER_LENGTH + SSL3_RT_MAX_ENCRYPTED_LENGTH;
  if (max_len < s->max_cert_list) {
    return s->max_cert_list;
  }
  return max_len;
}

/* dtls1_process_fragment reads a handshake fragment and processes it. It
 * returns one if a fragment was successfully processed and 0 or -1 on error. */
static int dtls1_process_fragment(SSL *s) {
  /* Read handshake message header. */
  uint8_t header[DTLS1_HM_HEADER_LENGTH];
  int ret = dtls1_read_bytes(s, SSL3_RT_HANDSHAKE, header,
                             DTLS1_HM_HEADER_LENGTH, 0);
  if (ret <= 0) {
    return ret;
  }
  if (ret != DTLS1_HM_HEADER_LENGTH) {
    OPENSSL_PUT_ERROR(SSL, SSL_R_UNEXPECTED_MESSAGE);
    ssl3_send_alert(s, SSL3_AL_FATAL, SSL_AD_UNEXPECTED_MESSAGE);
    return -1;
  }

  /* Parse the message fragment header. */
  struct hm_header_st msg_hdr;
  dtls1_get_message_header(header, &msg_hdr);

  /* TODO(davidben): dtls1_read_bytes is the wrong abstraction for DTLS. There
   * should be no need to reach into |s->s3->rrec.length|. */
  const size_t frag_off = msg_hdr.frag_off;
  const size_t frag_len = msg_hdr.frag_len;
  const size_t msg_len = msg_hdr.msg_len;
  if (frag_off > msg_len || frag_off + frag_len < frag_off ||
      frag_off + frag_len > msg_len ||
      msg_len > dtls1_max_handshake_message_len(s) ||
      frag_len > s->s3->rrec.length) {
    OPENSSL_PUT_ERROR(SSL, SSL_R_EXCESSIVE_MESSAGE_SIZE);
    ssl3_send_alert(s, SSL3_AL_FATAL, SSL_AD_ILLEGAL_PARAMETER);
    return -1;
  }

  if (msg_hdr.seq < s->d1->handshake_read_seq ||
      msg_hdr.seq > (unsigned)s->d1->handshake_read_seq +
                    kHandshakeBufferSize) {
    /* Ignore fragments from the past, or ones too far in the future. */
    if (!dtls1_discard_fragment_body(s, frag_len)) {
      return -1;
    }
    return 1;
  }

  hm_fragment *frag = dtls1_get_buffered_message(s, &msg_hdr);
  if (frag == NULL) {
    return -1;
  }
  assert(frag->msg_header.msg_len == msg_len);

  if (frag->reassembly == NULL) {
    /* The message is already assembled. */
    if (!dtls1_discard_fragment_body(s, frag_len)) {
      return -1;
    }
    return 1;
  }
  assert(msg_len > 0);

  /* Read the body of the fragment. */
  ret = dtls1_read_bytes(s, SSL3_RT_HANDSHAKE, frag->fragment + frag_off,
                         frag_len, 0);
  if (ret != frag_len) {
    OPENSSL_PUT_ERROR(SSL, ERR_R_INTERNAL_ERROR);
    ssl3_send_alert(s, SSL3_AL_FATAL, SSL_AD_INTERNAL_ERROR);
    return -1;
  }
  dtls1_hm_fragment_mark(frag, frag_off, frag_off + frag_len);

  return 1;
}

/* dtls1_get_message reads a handshake message of message type |msg_type| (any
 * if |msg_type| == -1), maximum acceptable body length |max|. Read an entire
 * handshake message. Handshake messages arrive in fragments. */
long dtls1_get_message(SSL *s, int st1, int stn, int msg_type, long max,
                       enum ssl_hash_message_t hash_message, int *ok) {
  pitem *item = NULL;
  hm_fragment *frag = NULL;
  int al;

  /* s3->tmp is used to store messages that are unexpected, caused
   * by the absence of an optional handshake message */
  if (s->s3->tmp.reuse_message) {
    /* A ssl_dont_hash_message call cannot be combined with reuse_message; the
     * ssl_dont_hash_message would have to have been applied to the previous
     * call. */
    assert(hash_message == ssl_hash_message);
    s->s3->tmp.reuse_message = 0;
    if (msg_type >= 0 && s->s3->tmp.message_type != msg_type) {
      al = SSL_AD_UNEXPECTED_MESSAGE;
      OPENSSL_PUT_ERROR(SSL, SSL_R_UNEXPECTED_MESSAGE);
      goto f_err;
    }
    *ok = 1;
    s->init_msg = (uint8_t *)s->init_buf->data + DTLS1_HM_HEADER_LENGTH;
    s->init_num = (int)s->s3->tmp.message_size;
    return s->init_num;
  }

  /* Process fragments until one is found. */
  while (!dtls1_is_next_message_complete(s)) {
    int ret = dtls1_process_fragment(s);
    if (ret <= 0) {
      *ok = 0;
      return ret;
    }
  }

  /* Read out the next complete handshake message. */
  item = pqueue_pop(s->d1->buffered_messages);
  assert(item != NULL);
  frag = (hm_fragment *)item->data;
  assert(s->d1->handshake_read_seq == frag->msg_header.seq);
  assert(frag->reassembly == NULL);

  if (frag->msg_header.msg_len > (size_t)max) {
    OPENSSL_PUT_ERROR(SSL, SSL_R_EXCESSIVE_MESSAGE_SIZE);
    goto err;
  }

  /* Reconstruct the assembled message. */
  size_t len;
  CBB cbb;
  CBB_zero(&cbb);
  if (!BUF_MEM_grow(s->init_buf,
                    (size_t)frag->msg_header.msg_len +
                    DTLS1_HM_HEADER_LENGTH) ||
      !CBB_init_fixed(&cbb, (uint8_t *)s->init_buf->data, s->init_buf->max) ||
      !CBB_add_u8(&cbb, frag->msg_header.type) ||
      !CBB_add_u24(&cbb, frag->msg_header.msg_len) ||
      !CBB_add_u16(&cbb, frag->msg_header.seq) ||
      !CBB_add_u24(&cbb, 0 /* frag_off */) ||
      !CBB_add_u24(&cbb, frag->msg_header.msg_len) ||
      !CBB_add_bytes(&cbb, frag->fragment, frag->msg_header.msg_len) ||
      !CBB_finish(&cbb, NULL, &len)) {
    CBB_cleanup(&cbb);
    OPENSSL_PUT_ERROR(SSL, ERR_R_MALLOC_FAILURE);
    goto err;
  }
  assert(len == (size_t)frag->msg_header.msg_len + DTLS1_HM_HEADER_LENGTH);

  s->d1->handshake_read_seq++;

  /* TODO(davidben): This function has a lot of implicit outputs. Simplify the
   * |ssl_get_message| API. */
  s->s3->tmp.message_type = frag->msg_header.type;
  s->s3->tmp.message_size = frag->msg_header.msg_len;
  s->init_msg = (uint8_t *)s->init_buf->data + DTLS1_HM_HEADER_LENGTH;
  s->init_num = frag->msg_header.msg_len;

  if (msg_type >= 0 && s->s3->tmp.message_type != msg_type) {
    al = SSL_AD_UNEXPECTED_MESSAGE;
    OPENSSL_PUT_ERROR(SSL, SSL_R_UNEXPECTED_MESSAGE);
    goto f_err;
  }
  if (hash_message == ssl_hash_message && !ssl3_hash_current_message(s)) {
    goto err;
  }
  if (s->msg_callback) {
    s->msg_callback(0, s->version, SSL3_RT_HANDSHAKE, s->init_buf->data,
                    s->init_num + DTLS1_HM_HEADER_LENGTH, s,
                    s->msg_callback_arg);
  }

  pitem_free(item);
  dtls1_hm_fragment_free(frag);

  s->state = stn;
  *ok = 1;
  return s->init_num;

f_err:
  ssl3_send_alert(s, SSL3_AL_FATAL, al);
err:
  pitem_free(item);
  dtls1_hm_fragment_free(frag);
  *ok = 0;
  return -1;
}

/* for these 2 messages, we need to
 * ssl->enc_read_ctx			re-init
 * ssl->s3->read_sequence		zero
 * ssl->s3->read_mac_secret		re-init
 * ssl->session->read_sym_enc		assign
 * ssl->session->read_compression	assign
 * ssl->session->read_hash		assign */
int dtls1_send_change_cipher_spec(SSL *s, int a, int b) {
  uint8_t *p;

  if (s->state == a) {
    p = (uint8_t *)s->init_buf->data;
    *p++ = SSL3_MT_CCS;
    s->d1->handshake_write_seq = s->d1->next_handshake_write_seq;
    s->init_num = DTLS1_CCS_HEADER_LENGTH;

    s->init_off = 0;

    dtls1_set_message_header(s, SSL3_MT_CCS, 0, s->d1->handshake_write_seq, 0,
                             0);

    /* buffer the message to handle re-xmits */
    dtls1_buffer_message(s, 1);

    s->state = b;
  }

  /* SSL3_ST_CW_CHANGE_B */
  return dtls1_do_write(s, SSL3_RT_CHANGE_CIPHER_SPEC, dtls1_use_current_epoch);
}

int dtls1_read_failed(SSL *s, int code) {
  if (code > 0) {
    assert(0);
    return 1;
  }

  if (!dtls1_is_timer_expired(s)) {
    /* not a timeout, none of our business, let higher layers handle this. In
     * fact, it's probably an error */
    return code;
  }

  if (!SSL_in_init(s)) {
    /* done, no need to send a retransmit */
    BIO_set_flags(SSL_get_rbio(s), BIO_FLAGS_READ);
    return code;
  }

  return DTLSv1_handle_timeout(s);
}

int dtls1_get_queue_priority(unsigned short seq, int is_ccs) {
  /* The index of the retransmission queue actually is the message sequence
   * number, since the queue only contains messages of a single handshake.
   * However, the ChangeCipherSpec has no message sequence number and so using
   * only the sequence will result in the CCS and Finished having the same
   * index. To prevent this, the sequence number is multiplied by 2. In case of
   * a CCS 1 is subtracted. This does not only differ CSS and Finished, it also
   * maintains the order of the index (important for priority queues) and fits
   * in the unsigned short variable. */
  return seq * 2 - is_ccs;
}

static int dtls1_retransmit_message(SSL *s, hm_fragment *frag) {
  int ret;
  /* XDTLS: for now assuming that read/writes are blocking */
  unsigned long header_length;

  /* assert(s->init_num == 0);
     assert(s->init_off == 0); */

  if (frag->msg_header.is_ccs) {
    header_length = DTLS1_CCS_HEADER_LENGTH;
  } else {
    header_length = DTLS1_HM_HEADER_LENGTH;
  }

  memcpy(s->init_buf->data, frag->fragment,
         frag->msg_header.msg_len + header_length);
  s->init_num = frag->msg_header.msg_len + header_length;

  dtls1_set_message_header(s, frag->msg_header.type,
                           frag->msg_header.msg_len, frag->msg_header.seq,
                           0, frag->msg_header.frag_len);

  /* DTLS renegotiation is unsupported, so only epochs 0 (NULL cipher) and 1
   * (negotiated cipher) exist. */
  assert(s->d1->w_epoch == 0 || s->d1->w_epoch == 1);
  assert(frag->msg_header.epoch <= s->d1->w_epoch);
  enum dtls1_use_epoch_t use_epoch = dtls1_use_current_epoch;
  if (s->d1->w_epoch == 1 && frag->msg_header.epoch == 0) {
    use_epoch = dtls1_use_previous_epoch;
  }

  ret = dtls1_do_write(s, frag->msg_header.is_ccs ? SSL3_RT_CHANGE_CIPHER_SPEC
                                                  : SSL3_RT_HANDSHAKE,
                       use_epoch);

  (void)BIO_flush(SSL_get_wbio(s));
  return ret;
}


int dtls1_retransmit_buffered_messages(SSL *s) {
  pqueue sent = s->d1->sent_messages;
  piterator iter = pqueue_iterator(sent);
  pitem *item;

  for (item = pqueue_next(&iter); item != NULL; item = pqueue_next(&iter)) {
    hm_fragment *frag = (hm_fragment *)item->data;
    if (dtls1_retransmit_message(s, frag) <= 0) {
      return -1;
    }
  }

  return 1;
}

int dtls1_buffer_message(SSL *s, int is_ccs) {
  pitem *item;
  hm_fragment *frag;
  uint8_t seq64be[8];

  /* this function is called immediately after a message has
   * been serialized */
  assert(s->init_off == 0);

  frag = dtls1_hm_fragment_new(s->init_num, 0);
  if (!frag) {
    return 0;
  }

  memcpy(frag->fragment, s->init_buf->data, s->init_num);

  if (is_ccs) {
    assert(s->d1->w_msg_hdr.msg_len + DTLS1_CCS_HEADER_LENGTH ==
           (unsigned int)s->init_num);
  } else {
    assert(s->d1->w_msg_hdr.msg_len + DTLS1_HM_HEADER_LENGTH ==
           (unsigned int)s->init_num);
  }

  frag->msg_header.msg_len = s->d1->w_msg_hdr.msg_len;
  frag->msg_header.seq = s->d1->w_msg_hdr.seq;
  frag->msg_header.type = s->d1->w_msg_hdr.type;
  frag->msg_header.frag_off = 0;
  frag->msg_header.frag_len = s->d1->w_msg_hdr.msg_len;
  frag->msg_header.is_ccs = is_ccs;
  frag->msg_header.epoch = s->d1->w_epoch;

  memset(seq64be, 0, sizeof(seq64be));
  seq64be[6] = (uint8_t)(
      dtls1_get_queue_priority(frag->msg_header.seq, frag->msg_header.is_ccs) >>
      8);
  seq64be[7] = (uint8_t)(
      dtls1_get_queue_priority(frag->msg_header.seq, frag->msg_header.is_ccs));

  item = pitem_new(seq64be, frag);
  if (item == NULL) {
    dtls1_hm_fragment_free(frag);
    return 0;
  }

  pqueue_insert(s->d1->sent_messages, item);
  return 1;
}

/* call this function when the buffered messages are no longer needed */
void dtls1_clear_record_buffer(SSL *s) {
  pitem *item;

  for (item = pqueue_pop(s->d1->sent_messages); item != NULL;
       item = pqueue_pop(s->d1->sent_messages)) {
    dtls1_hm_fragment_free((hm_fragment *)item->data);
    pitem_free(item);
  }
}

/* don't actually do the writing, wait till the MTU has been retrieved */
void dtls1_set_message_header(SSL *s, uint8_t mt, unsigned long len,
                              unsigned short seq_num, unsigned long frag_off,
                              unsigned long frag_len) {
  struct hm_header_st *msg_hdr = &s->d1->w_msg_hdr;

  msg_hdr->type = mt;
  msg_hdr->msg_len = len;
  msg_hdr->seq = seq_num;
  msg_hdr->frag_off = frag_off;
  msg_hdr->frag_len = frag_len;
}

static void dtls1_fix_message_header(SSL *s, unsigned long frag_off,
                                     unsigned long frag_len) {
  struct hm_header_st *msg_hdr = &s->d1->w_msg_hdr;

  msg_hdr->frag_off = frag_off;
  msg_hdr->frag_len = frag_len;
}

static uint8_t *dtls1_write_message_header(SSL *s, uint8_t *p) {
  struct hm_header_st *msg_hdr = &s->d1->w_msg_hdr;

  *p++ = msg_hdr->type;
  l2n3(msg_hdr->msg_len, p);

  s2n(msg_hdr->seq, p);
  l2n3(msg_hdr->frag_off, p);
  l2n3(msg_hdr->frag_len, p);

  return p;
}

unsigned int dtls1_min_mtu(void) {
  return kMinMTU;
}

void dtls1_get_message_header(uint8_t *data,
                              struct hm_header_st *msg_hdr) {
  memset(msg_hdr, 0x00, sizeof(struct hm_header_st));
  msg_hdr->type = *(data++);
  n2l3(data, msg_hdr->msg_len);

  n2s(data, msg_hdr->seq);
  n2l3(data, msg_hdr->frag_off);
  n2l3(data, msg_hdr->frag_len);
}
