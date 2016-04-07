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

#include <openssl/pqueue.h>

#include <assert.h>
#include <string.h>

#include <openssl/mem.h>


typedef struct _pqueue {
  pitem *items;
  unsigned count;
} pqueue_s;


pitem *pitem_new(uint8_t prio64be[8], void *data) {
  pitem *item = (pitem *)OPENSSL_malloc(sizeof(pitem));
  if (item == NULL) {
    return NULL;
  }

  memcpy(item->priority, prio64be, sizeof(item->priority));

  item->data = data;
  item->next = NULL;

  return item;
}

void pitem_free(pitem *item) {
  if (item == NULL) {
    return;
  }

  OPENSSL_free(item);
}

pqueue pqueue_new(void) {
  pqueue_s *pq = (pqueue_s *)OPENSSL_malloc(sizeof(pqueue_s));
  if (pq == NULL) {
    return NULL;
  }

  memset(pq, 0, sizeof(pqueue_s));
  return pq;
}

void pqueue_free(pqueue_s *pq) {
  if (pq == NULL) {
    return;
  }

  /* The queue must be empty. */
  assert(pq->items == NULL);
  OPENSSL_free(pq);
}

pitem *pqueue_peek(pqueue_s *pq) { return pq->items; }

pitem *pqueue_find(pqueue_s *pq, uint8_t *prio64be) {
  pitem *curr;

  for (curr = pq->items; curr; curr = curr->next) {
    if (memcmp(curr->priority, prio64be, sizeof(curr->priority)) == 0) {
      return curr;
    }
  }

  return NULL;
}

size_t pqueue_size(pqueue_s *pq) {
  pitem *item = pq->items;
  size_t count = 0;

  while (item != NULL) {
    count++;
    item = item->next;
  }
  return count;
}

piterator pqueue_iterator(pqueue_s *pq) { return pq->items; }

pitem *pqueue_next(piterator *item) {
  pitem *ret;

  if (item == NULL || *item == NULL) {
    return NULL;
  }

  ret = *item;
  *item = (*item)->next;

  return ret;
}

pitem *pqueue_insert(pqueue_s *pq, pitem *item) {
  pitem *curr, *next;

  if (pq->items == NULL) {
    pq->items = item;
    return item;
  }

  for (curr = NULL, next = pq->items; next != NULL;
       curr = next, next = next->next) {
    /* we can compare 64-bit value in big-endian encoding with memcmp. */
    int cmp = memcmp(next->priority, item->priority, sizeof(item->priority));
    if (cmp > 0) {
      /* next > item */
      item->next = next;

      if (curr == NULL) {
        pq->items = item;
      } else {
        curr->next = item;
      }

      return item;
    } else if (cmp == 0) {
      /* duplicates not allowed */
      return NULL;
    }
  }

  item->next = NULL;
  curr->next = item;

  return item;
}


pitem *pqueue_pop(pqueue_s *pq) {
  pitem *item = pq->items;

  if (pq->items != NULL) {
    pq->items = pq->items->next;
  }

  return item;
}
