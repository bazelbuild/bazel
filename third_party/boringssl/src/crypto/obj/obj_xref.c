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

#include <stdlib.h>

#include "obj_xref.h"


static int nid_triple_cmp_by_sign_id(const void *in_a, const void *in_b) {
  const nid_triple *a = in_a;
  const nid_triple *b = in_b;

  return a->sign_id - b->sign_id;
}

int OBJ_find_sigid_algs(int sign_nid, int *out_digest_nid, int *out_pkey_nid) {
  nid_triple key;
  const nid_triple *triple;

  key.sign_id = sign_nid;

  triple = bsearch(&key, sigoid_srt, sizeof(sigoid_srt) / sizeof(nid_triple),
                   sizeof(nid_triple), nid_triple_cmp_by_sign_id);

  if (triple == NULL) {
    return 0;
  }
  if (out_digest_nid) {
    *out_digest_nid = triple->hash_id;
  }
  if (out_pkey_nid) {
    *out_pkey_nid = triple->pkey_id;
  }

  return 1;
}

static int nid_triple_cmp_by_digest_and_hash(const void *in_a,
                                             const void *in_b) {
  const nid_triple *a = *((nid_triple**) in_a);
  const nid_triple *b = *((nid_triple**) in_b);

  int ret = a->hash_id - b->hash_id;
  if (ret) {
    return ret;
  }
  return a->pkey_id - b->pkey_id;
}

int OBJ_find_sigid_by_algs(int *out_sign_nid, int digest_nid, int pkey_nid) {
  nid_triple key, *pkey;
  const nid_triple **triple;

  key.hash_id = digest_nid;
  key.pkey_id = pkey_nid;
  pkey = &key;

  triple = bsearch(&pkey, sigoid_srt_xref,
                   sizeof(sigoid_srt_xref) / sizeof(nid_triple *),
                   sizeof(nid_triple *), nid_triple_cmp_by_digest_and_hash);

  if (triple == NULL) {
    return 0;
  }
  if (out_sign_nid) {
    *out_sign_nid = (*triple)->sign_id;
  }
  return 1;
}
