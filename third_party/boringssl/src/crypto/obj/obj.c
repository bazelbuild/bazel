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

#include <openssl/obj.h>

#include <limits.h>
#include <string.h>

#include <openssl/asn1.h>
#include <openssl/buf.h>
#include <openssl/bytestring.h>
#include <openssl/err.h>
#include <openssl/lhash.h>
#include <openssl/mem.h>
#include <openssl/thread.h>

#include "obj_dat.h"
#include "../internal.h"


static struct CRYPTO_STATIC_MUTEX global_added_lock = CRYPTO_STATIC_MUTEX_INIT;
/* These globals are protected by |global_added_lock|. */
static LHASH_OF(ASN1_OBJECT) *global_added_by_data = NULL;
static LHASH_OF(ASN1_OBJECT) *global_added_by_nid = NULL;
static LHASH_OF(ASN1_OBJECT) *global_added_by_short_name = NULL;
static LHASH_OF(ASN1_OBJECT) *global_added_by_long_name = NULL;

static struct CRYPTO_STATIC_MUTEX global_next_nid_lock =
    CRYPTO_STATIC_MUTEX_INIT;
static unsigned global_next_nid = NUM_NID;

static int obj_next_nid(void) {
  int ret;

  CRYPTO_STATIC_MUTEX_lock_write(&global_next_nid_lock);
  ret = global_next_nid++;
  CRYPTO_STATIC_MUTEX_unlock(&global_next_nid_lock);

  return ret;
}

ASN1_OBJECT *OBJ_dup(const ASN1_OBJECT *o) {
  ASN1_OBJECT *r;
  unsigned char *data = NULL;
  char *sn = NULL, *ln = NULL;

  if (o == NULL) {
    return NULL;
  }

  if (!(o->flags & ASN1_OBJECT_FLAG_DYNAMIC)) {
    /* TODO(fork): this is a little dangerous. */
    return (ASN1_OBJECT *)o;
  }

  r = ASN1_OBJECT_new();
  if (r == NULL) {
    OPENSSL_PUT_ERROR(OBJ, ERR_R_ASN1_LIB);
    return NULL;
  }
  r->ln = r->sn = NULL;

  data = OPENSSL_malloc(o->length);
  if (data == NULL) {
    goto err;
  }
  if (o->data != NULL) {
    memcpy(data, o->data, o->length);
  }

  /* once data is attached to an object, it remains const */
  r->data = data;
  r->length = o->length;
  r->nid = o->nid;

  if (o->ln != NULL) {
    ln = OPENSSL_strdup(o->ln);
    if (ln == NULL) {
      goto err;
    }
  }

  if (o->sn != NULL) {
    sn = OPENSSL_strdup(o->sn);
    if (sn == NULL) {
      goto err;
    }
  }

  r->sn = sn;
  r->ln = ln;

  r->flags =
      o->flags | (ASN1_OBJECT_FLAG_DYNAMIC | ASN1_OBJECT_FLAG_DYNAMIC_STRINGS |
                  ASN1_OBJECT_FLAG_DYNAMIC_DATA);
  return r;

err:
  OPENSSL_PUT_ERROR(OBJ, ERR_R_MALLOC_FAILURE);
  OPENSSL_free(ln);
  OPENSSL_free(sn);
  OPENSSL_free(data);
  OPENSSL_free(r);
  return NULL;
}

int OBJ_cmp(const ASN1_OBJECT *a, const ASN1_OBJECT *b) {
  int ret;

  ret = a->length - b->length;
  if (ret) {
    return ret;
  }
  return memcmp(a->data, b->data, a->length);
}

/* obj_cmp is called to search the kNIDsInOIDOrder array. The |key| argument is
 * an |ASN1_OBJECT|* that we're looking for and |element| is a pointer to an
 * unsigned int in the array. */
static int obj_cmp(const void *key, const void *element) {
  unsigned nid = *((const unsigned*) element);
  const ASN1_OBJECT *a = key;
  const ASN1_OBJECT *b = &kObjects[nid];

  if (a->length < b->length) {
    return -1;
  } else if (a->length > b->length) {
    return 1;
  }
  return memcmp(a->data, b->data, a->length);
}

int OBJ_obj2nid(const ASN1_OBJECT *obj) {
  const unsigned int *nid_ptr;

  if (obj == NULL) {
    return NID_undef;
  }

  if (obj->nid != 0) {
    return obj->nid;
  }

  CRYPTO_STATIC_MUTEX_lock_read(&global_added_lock);
  if (global_added_by_data != NULL) {
    ASN1_OBJECT *match;

    match = lh_ASN1_OBJECT_retrieve(global_added_by_data, obj);
    if (match != NULL) {
      CRYPTO_STATIC_MUTEX_unlock(&global_added_lock);
      return match->nid;
    }
  }
  CRYPTO_STATIC_MUTEX_unlock(&global_added_lock);

  nid_ptr = bsearch(obj, kNIDsInOIDOrder, NUM_OBJ, sizeof(unsigned), obj_cmp);
  if (nid_ptr == NULL) {
    return NID_undef;
  }

  return kObjects[*nid_ptr].nid;
}

int OBJ_cbs2nid(const CBS *cbs) {
  ASN1_OBJECT obj;
  memset(&obj, 0, sizeof(obj));
  obj.data = CBS_data(cbs);
  obj.length = CBS_len(cbs);

  return OBJ_obj2nid(&obj);
}

/* short_name_cmp is called to search the kNIDsInShortNameOrder array. The
 * |key| argument is name that we're looking for and |element| is a pointer to
 * an unsigned int in the array. */
static int short_name_cmp(const void *key, const void *element) {
  const char *name = (const char *) key;
  unsigned nid = *((unsigned*) element);

  return strcmp(name, kObjects[nid].sn);
}

int OBJ_sn2nid(const char *short_name) {
  const unsigned int *nid_ptr;

  CRYPTO_STATIC_MUTEX_lock_read(&global_added_lock);
  if (global_added_by_short_name != NULL) {
    ASN1_OBJECT *match, template;

    template.sn = short_name;
    match = lh_ASN1_OBJECT_retrieve(global_added_by_short_name, &template);
    if (match != NULL) {
      CRYPTO_STATIC_MUTEX_unlock(&global_added_lock);
      return match->nid;
    }
  }
  CRYPTO_STATIC_MUTEX_unlock(&global_added_lock);

  nid_ptr = bsearch(short_name, kNIDsInShortNameOrder, NUM_SN, sizeof(unsigned), short_name_cmp);
  if (nid_ptr == NULL) {
    return NID_undef;
  }

  return kObjects[*nid_ptr].nid;
}

/* long_name_cmp is called to search the kNIDsInLongNameOrder array. The
 * |key| argument is name that we're looking for and |element| is a pointer to
 * an unsigned int in the array. */
static int long_name_cmp(const void *key, const void *element) {
  const char *name = (const char *) key;
  unsigned nid = *((unsigned*) element);

  return strcmp(name, kObjects[nid].ln);
}

int OBJ_ln2nid(const char *long_name) {
  const unsigned int *nid_ptr;

  CRYPTO_STATIC_MUTEX_lock_read(&global_added_lock);
  if (global_added_by_long_name != NULL) {
    ASN1_OBJECT *match, template;

    template.ln = long_name;
    match = lh_ASN1_OBJECT_retrieve(global_added_by_long_name, &template);
    if (match != NULL) {
      CRYPTO_STATIC_MUTEX_unlock(&global_added_lock);
      return match->nid;
    }
  }
  CRYPTO_STATIC_MUTEX_unlock(&global_added_lock);

  nid_ptr = bsearch(long_name, kNIDsInLongNameOrder, NUM_LN, sizeof(unsigned), long_name_cmp);
  if (nid_ptr == NULL) {
    return NID_undef;
  }

  return kObjects[*nid_ptr].nid;
}

int OBJ_txt2nid(const char *s) {
  ASN1_OBJECT *obj;
  int nid;

  obj = OBJ_txt2obj(s, 0 /* search names */);
  nid = OBJ_obj2nid(obj);
  ASN1_OBJECT_free(obj);
  return nid;
}

OPENSSL_EXPORT int OBJ_nid2cbb(CBB *out, int nid) {
  const ASN1_OBJECT *obj = OBJ_nid2obj(nid);
  CBB oid;

  if (obj == NULL ||
      !CBB_add_asn1(out, &oid, CBS_ASN1_OBJECT) ||
      !CBB_add_bytes(&oid, obj->data, obj->length) ||
      !CBB_flush(out)) {
    return 0;
  }

  return 1;
}

const ASN1_OBJECT *OBJ_nid2obj(int nid) {
  if (nid >= 0 && nid < NUM_NID) {
    if (nid != NID_undef && kObjects[nid].nid == NID_undef) {
      goto err;
    }
    return &kObjects[nid];
  }

  CRYPTO_STATIC_MUTEX_lock_read(&global_added_lock);
  if (global_added_by_nid != NULL) {
    ASN1_OBJECT *match, template;

    template.nid = nid;
    match = lh_ASN1_OBJECT_retrieve(global_added_by_nid, &template);
    if (match != NULL) {
      CRYPTO_STATIC_MUTEX_unlock(&global_added_lock);
      return match;
    }
  }
  CRYPTO_STATIC_MUTEX_unlock(&global_added_lock);

err:
  OPENSSL_PUT_ERROR(OBJ, OBJ_R_UNKNOWN_NID);
  return NULL;
}

const char *OBJ_nid2sn(int nid) {
  const ASN1_OBJECT *obj = OBJ_nid2obj(nid);
  if (obj == NULL) {
    return NULL;
  }

  return obj->sn;
}

const char *OBJ_nid2ln(int nid) {
  const ASN1_OBJECT *obj = OBJ_nid2obj(nid);
  if (obj == NULL) {
    return NULL;
  }

  return obj->ln;
}

ASN1_OBJECT *OBJ_txt2obj(const char *s, int dont_search_names) {
  int nid = NID_undef;
  ASN1_OBJECT *op = NULL;
  unsigned char *buf;
  unsigned char *p;
  const unsigned char *bufp;
  int contents_len, total_len;

  if (!dont_search_names) {
    nid = OBJ_sn2nid(s);
    if (nid == NID_undef) {
      nid = OBJ_ln2nid(s);
    }

    if (nid != NID_undef) {
      return (ASN1_OBJECT*) OBJ_nid2obj(nid);
    }
  }

  /* Work out size of content octets */
  contents_len = a2d_ASN1_OBJECT(NULL, 0, s, -1);
  if (contents_len <= 0) {
    return NULL;
  }
  /* Work out total size */
  total_len = ASN1_object_size(0, contents_len, V_ASN1_OBJECT);

  buf = OPENSSL_malloc(total_len);
  if (buf == NULL) {
    OPENSSL_PUT_ERROR(OBJ, ERR_R_MALLOC_FAILURE);
    return NULL;
  }

  p = buf;
  /* Write out tag+length */
  ASN1_put_object(&p, 0, contents_len, V_ASN1_OBJECT, V_ASN1_UNIVERSAL);
  /* Write out contents */
  a2d_ASN1_OBJECT(p, contents_len, s, -1);

  bufp = buf;
  op = d2i_ASN1_OBJECT(NULL, &bufp, total_len);
  OPENSSL_free(buf);

  return op;
}

int OBJ_obj2txt(char *out, int out_len, const ASN1_OBJECT *obj, int dont_return_name) {
  int i, n = 0, len, nid, first, use_bn;
  BIGNUM *bl;
  unsigned long l;
  const unsigned char *p;
  char tbuf[DECIMAL_SIZE(i) + DECIMAL_SIZE(l) + 2];

  if (out && out_len > 0) {
    out[0] = 0;
  }

  if (obj == NULL || obj->data == NULL) {
    return 0;
  }

  if (!dont_return_name && (nid = OBJ_obj2nid(obj)) != NID_undef) {
    const char *s;
    s = OBJ_nid2ln(nid);
    if (s == NULL) {
      s = OBJ_nid2sn(nid);
    }
    if (s) {
      if (out) {
        BUF_strlcpy(out, s, out_len);
      }
      return strlen(s);
    }
  }

  len = obj->length;
  p = obj->data;

  first = 1;
  bl = NULL;

  while (len > 0) {
    l = 0;
    use_bn = 0;
    for (;;) {
      unsigned char c = *p++;
      len--;
      if (len == 0 && (c & 0x80)) {
        goto err;
      }
      if (use_bn) {
        if (!BN_add_word(bl, c & 0x7f)) {
          goto err;
        }
      } else {
        l |= c & 0x7f;
      }
      if (!(c & 0x80)) {
        break;
      }
      if (!use_bn && (l > (ULONG_MAX >> 7L))) {
        if (!bl && !(bl = BN_new())) {
          goto err;
        }
        if (!BN_set_word(bl, l)) {
          goto err;
        }
        use_bn = 1;
      }
      if (use_bn) {
        if (!BN_lshift(bl, bl, 7)) {
          goto err;
        }
      } else {
        l <<= 7L;
      }
    }

    if (first) {
      first = 0;
      if (l >= 80) {
        i = 2;
        if (use_bn) {
          if (!BN_sub_word(bl, 80)) {
            goto err;
          }
        } else {
          l -= 80;
        }
      } else {
        i = (int)(l / 40);
        l -= (long)(i * 40);
      }
      if (out && out_len > 1) {
        *out++ = i + '0';
        *out = '0';
        out_len--;
      }
      n++;
    }

    if (use_bn) {
      char *bndec;
      bndec = BN_bn2dec(bl);
      if (!bndec) {
        goto err;
      }
      i = strlen(bndec);
      if (out) {
        if (out_len > 1) {
          *out++ = '.';
          *out = 0;
          out_len--;
        }
        BUF_strlcpy(out, bndec, out_len);
        if (i > out_len) {
          out += out_len;
          out_len = 0;
        } else {
          out += i;
          out_len -= i;
        }
      }
      n++;
      n += i;
      OPENSSL_free(bndec);
    } else {
      BIO_snprintf(tbuf, sizeof(tbuf), ".%lu", l);
      i = strlen(tbuf);
      if (out && out_len > 0) {
        BUF_strlcpy(out, tbuf, out_len);
        if (i > out_len) {
          out += out_len;
          out_len = 0;
        } else {
          out += i;
          out_len -= i;
        }
      }
      n += i;
    }
  }

  BN_free(bl);
  return n;

err:
  BN_free(bl);
  return -1;
}

static uint32_t hash_nid(const ASN1_OBJECT *obj) {
  return obj->nid;
}

static int cmp_nid(const ASN1_OBJECT *a, const ASN1_OBJECT *b) {
  return a->nid - b->nid;
}

static uint32_t hash_data(const ASN1_OBJECT *obj) {
  return OPENSSL_hash32(obj->data, obj->length);
}

static int cmp_data(const ASN1_OBJECT *a, const ASN1_OBJECT *b) {
  int i = a->length - b->length;
  if (i) {
    return i;
  }
  return memcmp(a->data, b->data, a->length);
}

static uint32_t hash_short_name(const ASN1_OBJECT *obj) {
  return lh_strhash(obj->sn);
}

static int cmp_short_name(const ASN1_OBJECT *a, const ASN1_OBJECT *b) {
  return strcmp(a->sn, b->sn);
}

static uint32_t hash_long_name(const ASN1_OBJECT *obj) {
  return lh_strhash(obj->ln);
}

static int cmp_long_name(const ASN1_OBJECT *a, const ASN1_OBJECT *b) {
  return strcmp(a->ln, b->ln);
}

/* obj_add_object inserts |obj| into the various global hashes for run-time
 * added objects. It returns one on success or zero otherwise. */
static int obj_add_object(ASN1_OBJECT *obj) {
  int ok;
  ASN1_OBJECT *old_object;

  obj->flags &= ~(ASN1_OBJECT_FLAG_DYNAMIC | ASN1_OBJECT_FLAG_DYNAMIC_STRINGS |
                  ASN1_OBJECT_FLAG_DYNAMIC_DATA);

  CRYPTO_STATIC_MUTEX_lock_write(&global_added_lock);
  if (global_added_by_nid == NULL) {
    global_added_by_nid = lh_ASN1_OBJECT_new(hash_nid, cmp_nid);
    global_added_by_data = lh_ASN1_OBJECT_new(hash_data, cmp_data);
    global_added_by_short_name = lh_ASN1_OBJECT_new(hash_short_name, cmp_short_name);
    global_added_by_long_name = lh_ASN1_OBJECT_new(hash_long_name, cmp_long_name);
  }

  /* We don't pay attention to |old_object| (which contains any previous object
   * that was evicted from the hashes) because we don't have a reference count
   * on ASN1_OBJECT values. Also, we should never have duplicates nids and so
   * should always have objects in |global_added_by_nid|. */

  ok = lh_ASN1_OBJECT_insert(global_added_by_nid, &old_object, obj);
  if (obj->length != 0 && obj->data != NULL) {
    ok &= lh_ASN1_OBJECT_insert(global_added_by_data, &old_object, obj);
  }
  if (obj->sn != NULL) {
    ok &= lh_ASN1_OBJECT_insert(global_added_by_short_name, &old_object, obj);
  }
  if (obj->ln != NULL) {
    ok &= lh_ASN1_OBJECT_insert(global_added_by_long_name, &old_object, obj);
  }
  CRYPTO_STATIC_MUTEX_unlock(&global_added_lock);

  return ok;
}

int OBJ_create(const char *oid, const char *short_name, const char *long_name) {
  int ret = NID_undef;
  ASN1_OBJECT *op = NULL;
  unsigned char *buf = NULL;
  int len;

  len = a2d_ASN1_OBJECT(NULL, 0, oid, -1);
  if (len <= 0) {
    goto err;
  }

  buf = OPENSSL_malloc(len);
  if (buf == NULL) {
    OPENSSL_PUT_ERROR(OBJ, ERR_R_MALLOC_FAILURE);
    goto err;
  }

  len = a2d_ASN1_OBJECT(buf, len, oid, -1);
  if (len == 0) {
    goto err;
  }

  op = (ASN1_OBJECT *)ASN1_OBJECT_create(obj_next_nid(), buf, len, short_name,
                                         long_name);
  if (op == NULL) {
    goto err;
  }

  if (obj_add_object(op)) {
    ret = op->nid;
  }
  op = NULL;

err:
  ASN1_OBJECT_free(op);
  OPENSSL_free(buf);

  return ret;
}
