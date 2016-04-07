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
 * Hudson (tjh@cryptsoft.com). */

#include <openssl/ssl.h>

#include <assert.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <openssl/bytestring.h>
#include <openssl/digest.h>
#include <openssl/err.h>
#include <openssl/evp.h>
#include <openssl/hmac.h>
#include <openssl/mem.h>
#include <openssl/obj.h>
#include <openssl/rand.h>

#include "internal.h"


static int ssl_check_clienthello_tlsext(SSL *s);
static int ssl_check_serverhello_tlsext(SSL *s);

const SSL3_ENC_METHOD TLSv1_enc_data = {
    tls1_prf,
    tls1_setup_key_block,
    tls1_generate_master_secret,
    tls1_change_cipher_state,
    tls1_final_finish_mac,
    tls1_cert_verify_mac,
    TLS_MD_CLIENT_FINISH_CONST,TLS_MD_CLIENT_FINISH_CONST_SIZE,
    TLS_MD_SERVER_FINISH_CONST,TLS_MD_SERVER_FINISH_CONST_SIZE,
    tls1_alert_code,
    tls1_export_keying_material,
    0,
};

const SSL3_ENC_METHOD TLSv1_1_enc_data = {
    tls1_prf,
    tls1_setup_key_block,
    tls1_generate_master_secret,
    tls1_change_cipher_state,
    tls1_final_finish_mac,
    tls1_cert_verify_mac,
    TLS_MD_CLIENT_FINISH_CONST,TLS_MD_CLIENT_FINISH_CONST_SIZE,
    TLS_MD_SERVER_FINISH_CONST,TLS_MD_SERVER_FINISH_CONST_SIZE,
    tls1_alert_code,
    tls1_export_keying_material,
    SSL_ENC_FLAG_EXPLICIT_IV,
};

const SSL3_ENC_METHOD TLSv1_2_enc_data = {
    tls1_prf,
    tls1_setup_key_block,
    tls1_generate_master_secret,
    tls1_change_cipher_state,
    tls1_final_finish_mac,
    tls1_cert_verify_mac,
    TLS_MD_CLIENT_FINISH_CONST,TLS_MD_CLIENT_FINISH_CONST_SIZE,
    TLS_MD_SERVER_FINISH_CONST,TLS_MD_SERVER_FINISH_CONST_SIZE,
    tls1_alert_code,
    tls1_export_keying_material,
    SSL_ENC_FLAG_EXPLICIT_IV|SSL_ENC_FLAG_SIGALGS|SSL_ENC_FLAG_SHA256_PRF
            |SSL_ENC_FLAG_TLS1_2_CIPHERS,
};

static int compare_uint16_t(const void *p1, const void *p2) {
  uint16_t u1 = *((const uint16_t *)p1);
  uint16_t u2 = *((const uint16_t *)p2);
  if (u1 < u2) {
    return -1;
  } else if (u1 > u2) {
    return 1;
  } else {
    return 0;
  }
}

/* Per http://tools.ietf.org/html/rfc5246#section-7.4.1.4, there may not be
 * more than one extension of the same type in a ClientHello or ServerHello.
 * This function does an initial scan over the extensions block to filter those
 * out. */
static int tls1_check_duplicate_extensions(const CBS *cbs) {
  CBS extensions = *cbs;
  size_t num_extensions = 0, i = 0;
  uint16_t *extension_types = NULL;
  int ret = 0;

  /* First pass: count the extensions. */
  while (CBS_len(&extensions) > 0) {
    uint16_t type;
    CBS extension;

    if (!CBS_get_u16(&extensions, &type) ||
        !CBS_get_u16_length_prefixed(&extensions, &extension)) {
      goto done;
    }

    num_extensions++;
  }

  if (num_extensions == 0) {
    return 1;
  }

  extension_types =
      (uint16_t *)OPENSSL_malloc(sizeof(uint16_t) * num_extensions);
  if (extension_types == NULL) {
    OPENSSL_PUT_ERROR(SSL, ERR_R_MALLOC_FAILURE);
    goto done;
  }

  /* Second pass: gather the extension types. */
  extensions = *cbs;
  for (i = 0; i < num_extensions; i++) {
    CBS extension;

    if (!CBS_get_u16(&extensions, &extension_types[i]) ||
        !CBS_get_u16_length_prefixed(&extensions, &extension)) {
      /* This should not happen. */
      goto done;
    }
  }
  assert(CBS_len(&extensions) == 0);

  /* Sort the extensions and make sure there are no duplicates. */
  qsort(extension_types, num_extensions, sizeof(uint16_t), compare_uint16_t);
  for (i = 1; i < num_extensions; i++) {
    if (extension_types[i - 1] == extension_types[i]) {
      goto done;
    }
  }

  ret = 1;

done:
  OPENSSL_free(extension_types);
  return ret;
}

char ssl_early_callback_init(struct ssl_early_callback_ctx *ctx) {
  CBS client_hello, session_id, cipher_suites, compression_methods, extensions;

  CBS_init(&client_hello, ctx->client_hello, ctx->client_hello_len);

  if (/* Skip client version. */
      !CBS_skip(&client_hello, 2) ||
      /* Skip client nonce. */
      !CBS_skip(&client_hello, 32) ||
      /* Extract session_id. */
      !CBS_get_u8_length_prefixed(&client_hello, &session_id)) {
    return 0;
  }

  ctx->session_id = CBS_data(&session_id);
  ctx->session_id_len = CBS_len(&session_id);

  /* Skip past DTLS cookie */
  if (SSL_IS_DTLS(ctx->ssl)) {
    CBS cookie;

    if (!CBS_get_u8_length_prefixed(&client_hello, &cookie)) {
      return 0;
    }
  }

  /* Extract cipher_suites. */
  if (!CBS_get_u16_length_prefixed(&client_hello, &cipher_suites) ||
      CBS_len(&cipher_suites) < 2 || (CBS_len(&cipher_suites) & 1) != 0) {
    return 0;
  }
  ctx->cipher_suites = CBS_data(&cipher_suites);
  ctx->cipher_suites_len = CBS_len(&cipher_suites);

  /* Extract compression_methods. */
  if (!CBS_get_u8_length_prefixed(&client_hello, &compression_methods) ||
      CBS_len(&compression_methods) < 1) {
    return 0;
  }
  ctx->compression_methods = CBS_data(&compression_methods);
  ctx->compression_methods_len = CBS_len(&compression_methods);

  /* If the ClientHello ends here then it's valid, but doesn't have any
   * extensions. (E.g. SSLv3.) */
  if (CBS_len(&client_hello) == 0) {
    ctx->extensions = NULL;
    ctx->extensions_len = 0;
    return 1;
  }

  /* Extract extensions and check it is valid. */
  if (!CBS_get_u16_length_prefixed(&client_hello, &extensions) ||
      !tls1_check_duplicate_extensions(&extensions) ||
      CBS_len(&client_hello) != 0) {
    return 0;
  }
  ctx->extensions = CBS_data(&extensions);
  ctx->extensions_len = CBS_len(&extensions);

  return 1;
}

char SSL_early_callback_ctx_extension_get(
    const struct ssl_early_callback_ctx *ctx, uint16_t extension_type,
    const uint8_t **out_data, size_t *out_len) {
  CBS extensions;

  CBS_init(&extensions, ctx->extensions, ctx->extensions_len);

  while (CBS_len(&extensions) != 0) {
    uint16_t type;
    CBS extension;

    /* Decode the next extension. */
    if (!CBS_get_u16(&extensions, &type) ||
        !CBS_get_u16_length_prefixed(&extensions, &extension)) {
      return 0;
    }

    if (type == extension_type) {
      *out_data = CBS_data(&extension);
      *out_len = CBS_len(&extension);
      return 1;
    }
  }

  return 0;
}

struct tls_curve {
  uint16_t curve_id;
  int nid;
  const char curve_name[8];
};

/* ECC curves from RFC4492. */
static const struct tls_curve tls_curves[] = {
    {21, NID_secp224r1, "P-224"},
    {23, NID_X9_62_prime256v1, "P-256"},
    {24, NID_secp384r1, "P-384"},
    {25, NID_secp521r1, "P-521"},
};

static const uint16_t eccurves_default[] = {
    23, /* X9_62_prime256v1 */
    24, /* secp384r1 */
#if defined(BORINGSSL_ANDROID_SYSTEM)
    25, /* secp521r1 */
#endif
};

int tls1_ec_curve_id2nid(uint16_t curve_id) {
  size_t i;
  for (i = 0; i < sizeof(tls_curves) / sizeof(tls_curves[0]); i++) {
    if (curve_id == tls_curves[i].curve_id) {
      return tls_curves[i].nid;
    }
  }
  return NID_undef;
}

int tls1_ec_nid2curve_id(uint16_t *out_curve_id, int nid) {
  size_t i;
  for (i = 0; i < sizeof(tls_curves) / sizeof(tls_curves[0]); i++) {
    if (nid == tls_curves[i].nid) {
      *out_curve_id = tls_curves[i].curve_id;
      return 1;
    }
  }
  return 0;
}

const char* tls1_ec_curve_id2name(uint16_t curve_id) {
  size_t i;
  for (i = 0; i < sizeof(tls_curves) / sizeof(tls_curves[0]); i++) {
    if (curve_id == tls_curves[i].curve_id) {
      return tls_curves[i].curve_name;
    }
  }
  return NULL;
}

/* tls1_get_curvelist sets |*out_curve_ids| and |*out_curve_ids_len| to the
 * list of allowed curve IDs. If |get_peer_curves| is non-zero, return the
 * peer's curve list. Otherwise, return the preferred list. */
static void tls1_get_curvelist(SSL *s, int get_peer_curves,
                               const uint16_t **out_curve_ids,
                               size_t *out_curve_ids_len) {
  if (get_peer_curves) {
    /* Only clients send a curve list, so this function is only called
     * on the server. */
    assert(s->server);
    *out_curve_ids = s->s3->tmp.peer_ellipticcurvelist;
    *out_curve_ids_len = s->s3->tmp.peer_ellipticcurvelist_length;
    return;
  }

  *out_curve_ids = s->tlsext_ellipticcurvelist;
  *out_curve_ids_len = s->tlsext_ellipticcurvelist_length;
  if (!*out_curve_ids) {
    *out_curve_ids = eccurves_default;
    *out_curve_ids_len = sizeof(eccurves_default) / sizeof(eccurves_default[0]);
  }
}

int tls1_check_curve(SSL *s, CBS *cbs, uint16_t *out_curve_id) {
  uint8_t curve_type;
  uint16_t curve_id;
  const uint16_t *curves;
  size_t curves_len, i;

  /* Only support named curves. */
  if (!CBS_get_u8(cbs, &curve_type) ||
      curve_type != NAMED_CURVE_TYPE ||
      !CBS_get_u16(cbs, &curve_id)) {
    return 0;
  }

  tls1_get_curvelist(s, 0, &curves, &curves_len);
  for (i = 0; i < curves_len; i++) {
    if (curve_id == curves[i]) {
      *out_curve_id = curve_id;
      return 1;
    }
  }

  return 0;
}

int tls1_get_shared_curve(SSL *s) {
  const uint16_t *curves, *peer_curves, *pref, *supp;
  size_t curves_len, peer_curves_len, pref_len, supp_len, i, j;

  /* Can't do anything on client side */
  if (s->server == 0) {
    return NID_undef;
  }

  tls1_get_curvelist(s, 0 /* local curves */, &curves, &curves_len);
  tls1_get_curvelist(s, 1 /* peer curves */, &peer_curves, &peer_curves_len);

  if (peer_curves_len == 0) {
    /* Clients are not required to send a supported_curves extension. In this
     * case, the server is free to pick any curve it likes. See RFC 4492,
     * section 4, paragraph 3. */
    return (curves_len == 0) ? NID_undef : tls1_ec_curve_id2nid(curves[0]);
  }

  if (s->options & SSL_OP_CIPHER_SERVER_PREFERENCE) {
    pref = curves;
    pref_len = curves_len;
    supp = peer_curves;
    supp_len = peer_curves_len;
  } else {
    pref = peer_curves;
    pref_len = peer_curves_len;
    supp = curves;
    supp_len = curves_len;
  }

  for (i = 0; i < pref_len; i++) {
    for (j = 0; j < supp_len; j++) {
      if (pref[i] == supp[j]) {
        return tls1_ec_curve_id2nid(pref[i]);
      }
    }
  }

  return NID_undef;
}

int tls1_set_curves(uint16_t **out_curve_ids, size_t *out_curve_ids_len,
                    const int *curves, size_t ncurves) {
  uint16_t *curve_ids;
  size_t i;

  curve_ids = (uint16_t *)OPENSSL_malloc(ncurves * sizeof(uint16_t));
  if (curve_ids == NULL) {
    return 0;
  }

  for (i = 0; i < ncurves; i++) {
    if (!tls1_ec_nid2curve_id(&curve_ids[i], curves[i])) {
      OPENSSL_free(curve_ids);
      return 0;
    }
  }

  OPENSSL_free(*out_curve_ids);
  *out_curve_ids = curve_ids;
  *out_curve_ids_len = ncurves;

  return 1;
}

/* tls1_curve_params_from_ec_key sets |*out_curve_id| and |*out_comp_id| to the
 * TLS curve ID and point format, respectively, for |ec|. It returns one on
 * success and zero on failure. */
static int tls1_curve_params_from_ec_key(uint16_t *out_curve_id,
                                         uint8_t *out_comp_id, EC_KEY *ec) {
  int nid;
  uint16_t id;
  const EC_GROUP *grp;

  if (ec == NULL) {
    return 0;
  }

  grp = EC_KEY_get0_group(ec);
  if (grp == NULL) {
    return 0;
  }

  /* Determine curve ID */
  nid = EC_GROUP_get_curve_name(grp);
  if (!tls1_ec_nid2curve_id(&id, nid)) {
    return 0;
  }

  /* Set the named curve ID. Arbitrary explicit curves are not supported. */
  *out_curve_id = id;

  if (out_comp_id) {
    if (EC_KEY_get0_public_key(ec) == NULL) {
      return 0;
    }
    if (EC_KEY_get_conv_form(ec) == POINT_CONVERSION_COMPRESSED) {
      *out_comp_id = TLSEXT_ECPOINTFORMAT_ansiX962_compressed_prime;
    } else {
      *out_comp_id = TLSEXT_ECPOINTFORMAT_uncompressed;
    }
  }

  return 1;
}

/* tls1_check_curve_id returns one if |curve_id| is consistent with both our
 * and the peer's curve preferences. Note: if called as the client, only our
 * preferences are checked; the peer (the server) does not send preferences. */
static int tls1_check_curve_id(SSL *s, uint16_t curve_id) {
  const uint16_t *curves;
  size_t curves_len, i, get_peer_curves;

  /* Check against our list, then the peer's list. */
  for (get_peer_curves = 0; get_peer_curves <= 1; get_peer_curves++) {
    if (get_peer_curves && !s->server) {
      /* Servers do not present a preference list so, if we are a client, only
       * check our list. */
      continue;
    }

    tls1_get_curvelist(s, get_peer_curves, &curves, &curves_len);
    if (get_peer_curves && curves_len == 0) {
      /* Clients are not required to send a supported_curves extension. In this
       * case, the server is free to pick any curve it likes. See RFC 4492,
       * section 4, paragraph 3. */
      continue;
    }
    for (i = 0; i < curves_len; i++) {
      if (curves[i] == curve_id) {
        break;
      }
    }

    if (i == curves_len) {
      return 0;
    }
  }

  return 1;
}

int tls1_check_ec_cert(SSL *s, X509 *x) {
  int ret = 0;
  EVP_PKEY *pkey = X509_get_pubkey(x);
  uint16_t curve_id;
  uint8_t comp_id;

  if (!pkey ||
      pkey->type != EVP_PKEY_EC ||
      !tls1_curve_params_from_ec_key(&curve_id, &comp_id, pkey->pkey.ec) ||
      !tls1_check_curve_id(s, curve_id) ||
      comp_id != TLSEXT_ECPOINTFORMAT_uncompressed) {
    goto done;
  }

  ret = 1;

done:
  EVP_PKEY_free(pkey);
  return ret;
}

int tls1_check_ec_tmp_key(SSL *s) {
  if (s->cert->ecdh_nid != NID_undef) {
    /* If the curve is preconfigured, ECDH is acceptable iff the peer supports
     * the curve. */
    uint16_t curve_id;
    return tls1_ec_nid2curve_id(&curve_id, s->cert->ecdh_nid) &&
           tls1_check_curve_id(s, curve_id);
  }

  if (s->cert->ecdh_tmp_cb != NULL) {
    /* Assume the callback will provide an acceptable curve. */
    return 1;
  }

  /* Otherwise, the curve gets selected automatically. ECDH is acceptable iff
   * there is a shared curve. */
  return tls1_get_shared_curve(s) != NID_undef;
}

/* List of supported signature algorithms and hashes. Should make this
 * customisable at some point, for now include everything we support. */

#define tlsext_sigalg_rsa(md) md, TLSEXT_signature_rsa,

#define tlsext_sigalg_ecdsa(md) md, TLSEXT_signature_ecdsa,

#define tlsext_sigalg(md) tlsext_sigalg_rsa(md) tlsext_sigalg_ecdsa(md)

static const uint8_t tls12_sigalgs[] = {
    tlsext_sigalg(TLSEXT_hash_sha512)
    tlsext_sigalg(TLSEXT_hash_sha384)
    tlsext_sigalg(TLSEXT_hash_sha256)
    tlsext_sigalg(TLSEXT_hash_sha224)
    tlsext_sigalg(TLSEXT_hash_sha1)
};

size_t tls12_get_psigalgs(SSL *s, const uint8_t **psigs) {
  *psigs = tls12_sigalgs;
  return sizeof(tls12_sigalgs);
}

/* tls12_check_peer_sigalg parses a SignatureAndHashAlgorithm out of |cbs|. It
 * checks it is consistent with |s|'s sent supported signature algorithms and,
 * if so, writes the relevant digest into |*out_md| and returns 1. Otherwise it
 * returns 0 and writes an alert into |*out_alert|. */
int tls12_check_peer_sigalg(const EVP_MD **out_md, int *out_alert, SSL *s,
                            CBS *cbs, EVP_PKEY *pkey) {
  const uint8_t *sent_sigs;
  size_t sent_sigslen, i;
  int sigalg = tls12_get_sigid(pkey->type);
  uint8_t hash, signature;

  /* Should never happen */
  if (sigalg == -1) {
    OPENSSL_PUT_ERROR(SSL, ERR_R_INTERNAL_ERROR);
    *out_alert = SSL_AD_INTERNAL_ERROR;
    return 0;
  }

  if (!CBS_get_u8(cbs, &hash) ||
      !CBS_get_u8(cbs, &signature)) {
    OPENSSL_PUT_ERROR(SSL, SSL_R_DECODE_ERROR);
    *out_alert = SSL_AD_DECODE_ERROR;
    return 0;
  }

  /* Check key type is consistent with signature */
  if (sigalg != signature) {
    OPENSSL_PUT_ERROR(SSL, SSL_R_WRONG_SIGNATURE_TYPE);
    *out_alert = SSL_AD_ILLEGAL_PARAMETER;
    return 0;
  }

  if (pkey->type == EVP_PKEY_EC) {
    uint16_t curve_id;
    uint8_t comp_id;
    /* Check compression and curve matches extensions */
    if (!tls1_curve_params_from_ec_key(&curve_id, &comp_id, pkey->pkey.ec)) {
      *out_alert = SSL_AD_INTERNAL_ERROR;
      return 0;
    }

    if (s->server && (!tls1_check_curve_id(s, curve_id) ||
                      comp_id != TLSEXT_ECPOINTFORMAT_uncompressed)) {
      OPENSSL_PUT_ERROR(SSL, SSL_R_WRONG_CURVE);
      *out_alert = SSL_AD_ILLEGAL_PARAMETER;
      return 0;
    }
  }

  /* Check signature matches a type we sent */
  sent_sigslen = tls12_get_psigalgs(s, &sent_sigs);
  for (i = 0; i < sent_sigslen; i += 2, sent_sigs += 2) {
    if (hash == sent_sigs[0] && signature == sent_sigs[1]) {
      break;
    }
  }

  /* Allow fallback to SHA-1. */
  if (i == sent_sigslen && hash != TLSEXT_hash_sha1) {
    OPENSSL_PUT_ERROR(SSL, SSL_R_WRONG_SIGNATURE_TYPE);
    *out_alert = SSL_AD_ILLEGAL_PARAMETER;
    return 0;
  }

  *out_md = tls12_get_hash(hash);
  if (*out_md == NULL) {
    OPENSSL_PUT_ERROR(SSL, SSL_R_UNKNOWN_DIGEST);
    *out_alert = SSL_AD_ILLEGAL_PARAMETER;
    return 0;
  }

  return 1;
}

/* Get a mask of disabled algorithms: an algorithm is disabled if it isn't
 * supported or doesn't appear in supported signature algorithms. Unlike
 * ssl_cipher_get_disabled this applies to a specific session and not global
 * settings. */
void ssl_set_client_disabled(SSL *s) {
  CERT *c = s->cert;
  const uint8_t *sigalgs;
  size_t i, sigalgslen;
  int have_rsa = 0, have_ecdsa = 0;
  c->mask_a = 0;
  c->mask_k = 0;

  /* Don't allow TLS 1.2 only ciphers if we don't suppport them */
  if (!SSL_CLIENT_USE_TLS1_2_CIPHERS(s)) {
    c->mask_ssl = SSL_TLSV1_2;
  } else {
    c->mask_ssl = 0;
  }

  /* Now go through all signature algorithms seeing if we support any for RSA,
   * DSA, ECDSA. Do this for all versions not just TLS 1.2. */
  sigalgslen = tls12_get_psigalgs(s, &sigalgs);
  for (i = 0; i < sigalgslen; i += 2, sigalgs += 2) {
    switch (sigalgs[1]) {
      case TLSEXT_signature_rsa:
        have_rsa = 1;
        break;

      case TLSEXT_signature_ecdsa:
        have_ecdsa = 1;
        break;
    }
  }

  /* Disable auth if we don't include any appropriate signature algorithms. */
  if (!have_rsa) {
    c->mask_a |= SSL_aRSA;
  }
  if (!have_ecdsa) {
    c->mask_a |= SSL_aECDSA;
  }

  /* with PSK there must be client callback set */
  if (!s->psk_client_callback) {
    c->mask_a |= SSL_aPSK;
    c->mask_k |= SSL_kPSK;
  }
}

/* tls_extension represents a TLS extension that is handled internally. The
 * |init| function is called for each handshake, before any other functions of
 * the extension. Then the add and parse callbacks are called as needed.
 *
 * The parse callbacks receive a |CBS| that contains the contents of the
 * extension (i.e. not including the type and length bytes). If an extension is
 * not received then the parse callbacks will be called with a NULL CBS so that
 * they can do any processing needed to handle the absence of an extension.
 *
 * The add callbacks receive a |CBB| to which the extension can be appended but
 * the function is responsible for appending the type and length bytes too.
 *
 * All callbacks return one for success and zero for error. If a parse function
 * returns zero then a fatal alert with value |*out_alert| will be sent. If
 * |*out_alert| isn't set, then a |decode_error| alert will be sent. */
struct tls_extension {
  uint16_t value;
  void (*init)(SSL *ssl);

  int (*add_clienthello)(SSL *ssl, CBB *out);
  int (*parse_serverhello)(SSL *ssl, uint8_t *out_alert, CBS *contents);

  int (*parse_clienthello)(SSL *ssl, uint8_t *out_alert, CBS *contents);
  int (*add_serverhello)(SSL *ssl, CBB *out);
};


/* Server name indication (SNI).
 *
 * https://tools.ietf.org/html/rfc6066#section-3. */

static void ext_sni_init(SSL *ssl) {
  ssl->s3->tmp.should_ack_sni = 0;
}

static int ext_sni_add_clienthello(SSL *ssl, CBB *out) {
  if (ssl->tlsext_hostname == NULL) {
    return 1;
  }

  CBB contents, server_name_list, name;
  if (!CBB_add_u16(out, TLSEXT_TYPE_server_name) ||
      !CBB_add_u16_length_prefixed(out, &contents) ||
      !CBB_add_u16_length_prefixed(&contents, &server_name_list) ||
      !CBB_add_u8(&server_name_list, TLSEXT_NAMETYPE_host_name) ||
      !CBB_add_u16_length_prefixed(&server_name_list, &name) ||
      !CBB_add_bytes(&name, (const uint8_t *)ssl->tlsext_hostname,
                     strlen(ssl->tlsext_hostname)) ||
      !CBB_flush(out)) {
    return 0;
  }

  return 1;
}

static int ext_sni_parse_serverhello(SSL *ssl, uint8_t *out_alert, CBS *contents) {
  if (contents == NULL) {
    return 1;
  }

  if (CBS_len(contents) != 0) {
    return 0;
  }

  assert(ssl->tlsext_hostname != NULL);

  if (!ssl->hit) {
    assert(ssl->session->tlsext_hostname == NULL);
    ssl->session->tlsext_hostname = BUF_strdup(ssl->tlsext_hostname);
    if (!ssl->session->tlsext_hostname) {
      *out_alert = SSL_AD_INTERNAL_ERROR;
      return 0;
    }
  }

  return 1;
}

static int ext_sni_parse_clienthello(SSL *ssl, uint8_t *out_alert, CBS *contents) {
  if (contents == NULL) {
    return 1;
  }

  /* The servername extension is treated as follows:
   *
   * - Only the hostname type is supported with a maximum length of 255.
   * - The servername is rejected if too long or if it contains zeros, in
   *   which case an fatal alert is generated.
   * - The servername field is maintained together with the session cache.
   * - When a session is resumed, the servername callback is invoked in order
   *   to allow the application to position itself to the right context.
   * - The servername is acknowledged if it is new for a session or when
   *   it is identical to a previously used for the same session.
   *   Applications can control the behaviour.  They can at any time
   *   set a 'desirable' servername for a new SSL object. This can be the
   *   case for example with HTTPS when a Host: header field is received and
   *   a renegotiation is requested. In this case, a possible servername
   *   presented in the new client hello is only acknowledged if it matches
   *   the value of the Host: field.
   * - Applications must  use SSL_OP_NO_SESSION_RESUMPTION_ON_RENEGOTIATION
   *   if they provide for changing an explicit servername context for the
   *   session,
   *   i.e. when the session has been established with a servername extension.
   */

  CBS server_name_list;
  char have_seen_host_name = 0;

  if (!CBS_get_u16_length_prefixed(contents, &server_name_list) ||
      CBS_len(&server_name_list) == 0 ||
      CBS_len(contents) != 0) {
    return 0;
  }

  /* Decode each ServerName in the extension. */
  while (CBS_len(&server_name_list) > 0) {
    uint8_t name_type;
    CBS host_name;

    if (!CBS_get_u8(&server_name_list, &name_type) ||
        !CBS_get_u16_length_prefixed(&server_name_list, &host_name)) {
      return 0;
    }

    /* Only host_name is supported. */
    if (name_type != TLSEXT_NAMETYPE_host_name) {
      continue;
    }

    if (have_seen_host_name) {
      /* The ServerNameList MUST NOT contain more than one name of the same
       * name_type. */
      return 0;
    }

    have_seen_host_name = 1;

    if (CBS_len(&host_name) == 0 ||
        CBS_len(&host_name) > TLSEXT_MAXLEN_host_name ||
        CBS_contains_zero_byte(&host_name)) {
      *out_alert = SSL_AD_UNRECOGNIZED_NAME;
      return 0;
    }

    if (!ssl->hit) {
      assert(ssl->session->tlsext_hostname == NULL);
      if (ssl->session->tlsext_hostname) {
        /* This should be impossible. */
        return 0;
      }

      /* Copy the hostname as a string. */
      if (!CBS_strdup(&host_name, &ssl->session->tlsext_hostname)) {
        *out_alert = SSL_AD_INTERNAL_ERROR;
        return 0;
      }

      ssl->s3->tmp.should_ack_sni = 1;
    }
  }

  return 1;
}

static int ext_sni_add_serverhello(SSL *ssl, CBB *out) {
  if (ssl->hit ||
      !ssl->s3->tmp.should_ack_sni ||
      ssl->session->tlsext_hostname == NULL) {
    return 1;
  }

  if (!CBB_add_u16(out, TLSEXT_TYPE_server_name) ||
      !CBB_add_u16(out, 0 /* length */)) {
    return 0;
  }

  return 1;
}


/* Renegotiation indication.
 *
 * https://tools.ietf.org/html/rfc5746 */

static int ext_ri_add_clienthello(SSL *ssl, CBB *out) {
  CBB contents, prev_finished;
  if (!CBB_add_u16(out, TLSEXT_TYPE_renegotiate) ||
      !CBB_add_u16_length_prefixed(out, &contents) ||
      !CBB_add_u8_length_prefixed(&contents, &prev_finished) ||
      !CBB_add_bytes(&prev_finished, ssl->s3->previous_client_finished,
                     ssl->s3->previous_client_finished_len) ||
      !CBB_flush(out)) {
    return 0;
  }

  return 1;
}

static int ext_ri_parse_serverhello(SSL *ssl, uint8_t *out_alert,
                                    CBS *contents) {
  if (contents == NULL) {
    /* No renegotiation extension received.
     *
     * Strictly speaking if we want to avoid an attack we should *always* see
     * RI even on initial ServerHello because the client doesn't see any
     * renegotiation during an attack. However this would mean we could not
     * connect to any server which doesn't support RI.
     *
     * A lack of the extension is allowed if SSL_OP_LEGACY_SERVER_CONNECT is
     * defined. */
    if (ssl->options & SSL_OP_LEGACY_SERVER_CONNECT) {
      return 1;
    }

    *out_alert = SSL_AD_HANDSHAKE_FAILURE;
    OPENSSL_PUT_ERROR(SSL, SSL_R_UNSAFE_LEGACY_RENEGOTIATION_DISABLED);
    return 0;
  }

  const size_t expected_len = ssl->s3->previous_client_finished_len +
                              ssl->s3->previous_server_finished_len;

  /* Check for logic errors */
  assert(!expected_len || ssl->s3->previous_client_finished_len);
  assert(!expected_len || ssl->s3->previous_server_finished_len);

  /* Parse out the extension contents. */
  CBS renegotiated_connection;
  if (!CBS_get_u8_length_prefixed(contents, &renegotiated_connection) ||
      CBS_len(contents) != 0) {
    OPENSSL_PUT_ERROR(SSL, SSL_R_RENEGOTIATION_ENCODING_ERR);
    *out_alert = SSL_AD_ILLEGAL_PARAMETER;
    return 0;
  }

  /* Check that the extension matches. */
  if (CBS_len(&renegotiated_connection) != expected_len) {
    OPENSSL_PUT_ERROR(SSL, SSL_R_RENEGOTIATION_MISMATCH);
    *out_alert = SSL_AD_HANDSHAKE_FAILURE;
    return 0;
  }

  const uint8_t *d = CBS_data(&renegotiated_connection);
  if (CRYPTO_memcmp(d, ssl->s3->previous_client_finished,
        ssl->s3->previous_client_finished_len)) {
    OPENSSL_PUT_ERROR(SSL, SSL_R_RENEGOTIATION_MISMATCH);
    *out_alert = SSL_AD_HANDSHAKE_FAILURE;
    return 0;
  }
  d += ssl->s3->previous_client_finished_len;

  if (CRYPTO_memcmp(d, ssl->s3->previous_server_finished,
        ssl->s3->previous_server_finished_len)) {
    OPENSSL_PUT_ERROR(SSL, SSL_R_RENEGOTIATION_MISMATCH);
    *out_alert = SSL_AD_ILLEGAL_PARAMETER;
    return 0;
  }
  ssl->s3->send_connection_binding = 1;

  return 1;
}

static int ext_ri_parse_clienthello(SSL *ssl, uint8_t *out_alert,
                                    CBS *contents) {
  /* Renegotiation isn't supported as a server so this function should never be
   * called after the initial handshake. */
  assert(!ssl->s3->initial_handshake_complete);

  CBS fake_contents;
  static const uint8_t kFakeExtension[] = {0};

  if (contents == NULL) {
    if (ssl->s3->send_connection_binding) {
      /* The renegotiation SCSV was received so pretend that we received a
       * renegotiation extension. */
      CBS_init(&fake_contents, kFakeExtension, sizeof(kFakeExtension));
      contents = &fake_contents;
      /* We require that the renegotiation extension is at index zero of
       * kExtensions. */
      ssl->s3->tmp.extensions.received |= (1u << 0);
    } else {
      return 1;
    }
  }

  CBS renegotiated_connection;

  if (!CBS_get_u8_length_prefixed(contents, &renegotiated_connection) ||
      CBS_len(contents) != 0) {
    OPENSSL_PUT_ERROR(SSL, SSL_R_RENEGOTIATION_ENCODING_ERR);
    return 0;
  }

  /* Check that the extension matches */
  if (!CBS_mem_equal(&renegotiated_connection, ssl->s3->previous_client_finished,
                     ssl->s3->previous_client_finished_len)) {
    OPENSSL_PUT_ERROR(SSL, SSL_R_RENEGOTIATION_MISMATCH);
    *out_alert = SSL_AD_HANDSHAKE_FAILURE;
    return 0;
  }

  ssl->s3->send_connection_binding = 1;

  return 1;
}

static int ext_ri_add_serverhello(SSL *ssl, CBB *out) {
  CBB contents, prev_finished;
  if (!CBB_add_u16(out, TLSEXT_TYPE_renegotiate) ||
      !CBB_add_u16_length_prefixed(out, &contents) ||
      !CBB_add_u8_length_prefixed(&contents, &prev_finished) ||
      !CBB_add_bytes(&prev_finished, ssl->s3->previous_client_finished,
                     ssl->s3->previous_client_finished_len) ||
      !CBB_add_bytes(&prev_finished, ssl->s3->previous_server_finished,
                     ssl->s3->previous_server_finished_len) ||
      !CBB_flush(out)) {
    return 0;
  }

  return 1;
}


/* Extended Master Secret.
 *
 * https://tools.ietf.org/html/draft-ietf-tls-session-hash-05 */

static void ext_ems_init(SSL *ssl) {
  ssl->s3->tmp.extended_master_secret = 0;
}

static int ext_ems_add_clienthello(SSL *ssl, CBB *out) {
  if (ssl->version == SSL3_VERSION) {
    return 1;
  }

  if (!CBB_add_u16(out, TLSEXT_TYPE_extended_master_secret) ||
      !CBB_add_u16(out, 0 /* length */)) {
    return 0;
  }

  return 1;
}

static int ext_ems_parse_serverhello(SSL *ssl, uint8_t *out_alert,
                                     CBS *contents) {
  if (contents == NULL) {
    return 1;
  }

  if (ssl->version == SSL3_VERSION || CBS_len(contents) != 0) {
    return 0;
  }

  ssl->s3->tmp.extended_master_secret = 1;
  return 1;
}

static int ext_ems_parse_clienthello(SSL *ssl, uint8_t *out_alert, CBS *contents) {
  if (ssl->version == SSL3_VERSION || contents == NULL) {
    return 1;
  }

  if (CBS_len(contents) != 0) {
    return 0;
  }

  ssl->s3->tmp.extended_master_secret = 1;
  return 1;
}

static int ext_ems_add_serverhello(SSL *ssl, CBB *out) {
  if (!ssl->s3->tmp.extended_master_secret) {
    return 1;
  }

  if (!CBB_add_u16(out, TLSEXT_TYPE_extended_master_secret) ||
      !CBB_add_u16(out, 0 /* length */)) {
    return 0;
  }

  return 1;
}


/* Session tickets.
 *
 * https://tools.ietf.org/html/rfc5077 */

static int ext_ticket_add_clienthello(SSL *ssl, CBB *out) {
  if (SSL_get_options(ssl) & SSL_OP_NO_TICKET) {
    return 1;
  }

  const uint8_t *ticket_data = NULL;
  int ticket_len = 0;

  /* Renegotiation does not participate in session resumption. However, still
   * advertise the extension to avoid potentially breaking servers which carry
   * over the state from the previous handshake, such as OpenSSL servers
   * without upstream's 3c3f0259238594d77264a78944d409f2127642c4. */
  if (!ssl->s3->initial_handshake_complete &&
      ssl->session != NULL &&
      ssl->session->tlsext_tick != NULL) {
    ticket_data = ssl->session->tlsext_tick;
    ticket_len = ssl->session->tlsext_ticklen;
  }

  CBB ticket;
  if (!CBB_add_u16(out, TLSEXT_TYPE_session_ticket) ||
      !CBB_add_u16_length_prefixed(out, &ticket) ||
      !CBB_add_bytes(&ticket, ticket_data, ticket_len) ||
      !CBB_flush(out)) {
    return 0;
  }

  return 1;
}

static int ext_ticket_parse_serverhello(SSL *ssl, uint8_t *out_alert,
                                        CBS *contents) {
  ssl->tlsext_ticket_expected = 0;

  if (contents == NULL) {
    return 1;
  }

  /* If |SSL_OP_NO_TICKET| is set then no extension will have been sent and
   * this function should never be called, even if the server tries to send the
   * extension. */
  assert((SSL_get_options(ssl) & SSL_OP_NO_TICKET) == 0);

  if (CBS_len(contents) != 0) {
    return 0;
  }

  ssl->tlsext_ticket_expected = 1;
  return 1;
}

static int ext_ticket_parse_clienthello(SSL *ssl, uint8_t *out_alert, CBS *contents) {
  /* This function isn't used because the ticket extension from the client is
   * handled in ssl_sess.c. */
  return 1;
}

static int ext_ticket_add_serverhello(SSL *ssl, CBB *out) {
  if (!ssl->tlsext_ticket_expected) {
    return 1;
  }

  /* If |SSL_OP_NO_TICKET| is set, |tlsext_ticket_expected| should never be
   * true. */
  assert((SSL_get_options(ssl) & SSL_OP_NO_TICKET) == 0);

  if (!CBB_add_u16(out, TLSEXT_TYPE_session_ticket) ||
      !CBB_add_u16(out, 0 /* length */)) {
    return 0;
  }

  return 1;
}


/* Signature Algorithms.
 *
 * https://tools.ietf.org/html/rfc5246#section-7.4.1.4.1 */

static int ext_sigalgs_add_clienthello(SSL *ssl, CBB *out) {
  if (ssl3_version_from_wire(ssl, ssl->client_version) < TLS1_2_VERSION) {
    return 1;
  }

  const uint8_t *sigalgs_data;
  const size_t sigalgs_len = tls12_get_psigalgs(ssl, &sigalgs_data);

  CBB contents, sigalgs;
  if (!CBB_add_u16(out, TLSEXT_TYPE_signature_algorithms) ||
      !CBB_add_u16_length_prefixed(out, &contents) ||
      !CBB_add_u16_length_prefixed(&contents, &sigalgs) ||
      !CBB_add_bytes(&sigalgs, sigalgs_data, sigalgs_len) ||
      !CBB_flush(out)) {
    return 0;
  }

  return 1;
}

static int ext_sigalgs_parse_serverhello(SSL *ssl, uint8_t *out_alert,
                                         CBS *contents) {
  if (contents != NULL) {
    /* Servers MUST NOT send this extension. */
    *out_alert = SSL_AD_UNSUPPORTED_EXTENSION;
    OPENSSL_PUT_ERROR(SSL, SSL_R_SIGNATURE_ALGORITHMS_EXTENSION_SENT_BY_SERVER);
    return 0;
  }

  return 1;
}

static int ext_sigalgs_parse_clienthello(SSL *ssl, uint8_t *out_alert,
                                         CBS *contents) {
  OPENSSL_free(ssl->cert->peer_sigalgs);
  ssl->cert->peer_sigalgs = NULL;
  ssl->cert->peer_sigalgslen = 0;

  if (contents == NULL) {
    return 1;
  }

  CBS supported_signature_algorithms;
  if (!CBS_get_u16_length_prefixed(contents, &supported_signature_algorithms) ||
      CBS_len(contents) != 0 ||
      CBS_len(&supported_signature_algorithms) == 0 ||
      !tls1_parse_peer_sigalgs(ssl, &supported_signature_algorithms)) {
    return 0;
  }

  return 1;
}

static int ext_sigalgs_add_serverhello(SSL *ssl, CBB *out) {
  /* Servers MUST NOT send this extension. */
  return 1;
}


/* OCSP Stapling.
 *
 * https://tools.ietf.org/html/rfc6066#section-8 */

static void ext_ocsp_init(SSL *ssl) {
  ssl->s3->tmp.certificate_status_expected = 0;
}

static int ext_ocsp_add_clienthello(SSL *ssl, CBB *out) {
  if (!ssl->ocsp_stapling_enabled) {
    return 1;
  }

  CBB contents;
  if (!CBB_add_u16(out, TLSEXT_TYPE_status_request) ||
      !CBB_add_u16_length_prefixed(out, &contents) ||
      !CBB_add_u8(&contents, TLSEXT_STATUSTYPE_ocsp) ||
      !CBB_add_u16(&contents, 0 /* empty responder ID list */) ||
      !CBB_add_u16(&contents, 0 /* empty request extensions */) ||
      !CBB_flush(out)) {
    return 0;
  }

  return 1;
}

static int ext_ocsp_parse_serverhello(SSL *ssl, uint8_t *out_alert,
                                      CBS *contents) {
  if (contents == NULL) {
    return 1;
  }

  if (CBS_len(contents) != 0) {
    return 0;
  }

  ssl->s3->tmp.certificate_status_expected = 1;
  return 1;
}

static int ext_ocsp_parse_clienthello(SSL *ssl, uint8_t *out_alert,
                                      CBS *contents) {
  if (contents == NULL) {
    return 1;
  }

  uint8_t status_type;
  if (!CBS_get_u8(contents, &status_type)) {
    return 0;
  }

  /* We cannot decide whether OCSP stapling will occur yet because the correct
   * SSL_CTX might not have been selected. */
  ssl->s3->tmp.ocsp_stapling_requested = status_type == TLSEXT_STATUSTYPE_ocsp;

  return 1;
}

static int ext_ocsp_add_serverhello(SSL *ssl, CBB *out) {
  /* The extension shouldn't be sent when resuming sessions. */
  if (ssl->hit ||
      !ssl->s3->tmp.ocsp_stapling_requested ||
      ssl->ctx->ocsp_response_length == 0) {
    return 1;
  }

  ssl->s3->tmp.certificate_status_expected = 1;

  return CBB_add_u16(out, TLSEXT_TYPE_status_request) &&
         CBB_add_u16(out, 0 /* length */);
}


/* Next protocol negotiation.
 *
 * https://htmlpreview.github.io/?https://github.com/agl/technotes/blob/master/nextprotoneg.html */

static void ext_npn_init(SSL *ssl) {
  ssl->s3->next_proto_neg_seen = 0;
}

static int ext_npn_add_clienthello(SSL *ssl, CBB *out) {
  if (ssl->s3->initial_handshake_complete ||
      ssl->ctx->next_proto_select_cb == NULL ||
      SSL_IS_DTLS(ssl)) {
    return 1;
  }

  if (!CBB_add_u16(out, TLSEXT_TYPE_next_proto_neg) ||
      !CBB_add_u16(out, 0 /* length */)) {
    return 0;
  }

  return 1;
}

static int ext_npn_parse_serverhello(SSL *ssl, uint8_t *out_alert,
                                     CBS *contents) {
  if (contents == NULL) {
    return 1;
  }

  /* If any of these are false then we should never have sent the NPN
   * extension in the ClientHello and thus this function should never have been
   * called. */
  assert(!ssl->s3->initial_handshake_complete);
  assert(!SSL_IS_DTLS(ssl));
  assert(ssl->ctx->next_proto_select_cb != NULL);

  if (ssl->s3->alpn_selected != NULL) {
    /* NPN and ALPN may not be negotiated in the same connection. */
    *out_alert = SSL_AD_ILLEGAL_PARAMETER;
    OPENSSL_PUT_ERROR(SSL, SSL_R_NEGOTIATED_BOTH_NPN_AND_ALPN);
    return 0;
  }

  const uint8_t *const orig_contents = CBS_data(contents);
  const size_t orig_len = CBS_len(contents);

  while (CBS_len(contents) != 0) {
    CBS proto;
    if (!CBS_get_u8_length_prefixed(contents, &proto) ||
        CBS_len(&proto) == 0) {
      return 0;
    }
  }

  uint8_t *selected;
  uint8_t selected_len;
  if (ssl->ctx->next_proto_select_cb(
          ssl, &selected, &selected_len, orig_contents, orig_len,
          ssl->ctx->next_proto_select_cb_arg) != SSL_TLSEXT_ERR_OK) {
    *out_alert = SSL_AD_INTERNAL_ERROR;
    return 0;
  }

  OPENSSL_free(ssl->next_proto_negotiated);
  ssl->next_proto_negotiated = BUF_memdup(selected, selected_len);
  if (ssl->next_proto_negotiated == NULL) {
    *out_alert = SSL_AD_INTERNAL_ERROR;
    return 0;
  }

  ssl->next_proto_negotiated_len = selected_len;
  ssl->s3->next_proto_neg_seen = 1;

  return 1;
}

static int ext_npn_parse_clienthello(SSL *ssl, uint8_t *out_alert,
                                     CBS *contents) {
  if (contents != NULL && CBS_len(contents) != 0) {
    return 0;
  }

  if (contents == NULL ||
      ssl->s3->initial_handshake_complete ||
      /* If the ALPN extension is seen before NPN, ignore it. (If ALPN is seen
       * afterwards, parsing the ALPN extension will clear
       * |next_proto_neg_seen|. */
      ssl->s3->alpn_selected != NULL ||
      ssl->ctx->next_protos_advertised_cb == NULL ||
      SSL_IS_DTLS(ssl)) {
    return 1;
  }

  ssl->s3->next_proto_neg_seen = 1;
  return 1;
}

static int ext_npn_add_serverhello(SSL *ssl, CBB *out) {
  /* |next_proto_neg_seen| might have been cleared when an ALPN extension was
   * parsed. */
  if (!ssl->s3->next_proto_neg_seen) {
    return 1;
  }

  const uint8_t *npa;
  unsigned npa_len;

  if (ssl->ctx->next_protos_advertised_cb(
          ssl, &npa, &npa_len, ssl->ctx->next_protos_advertised_cb_arg) !=
      SSL_TLSEXT_ERR_OK) {
    ssl->s3->next_proto_neg_seen = 0;
    return 1;
  }

  CBB contents;
  if (!CBB_add_u16(out, TLSEXT_TYPE_next_proto_neg) ||
      !CBB_add_u16_length_prefixed(out, &contents) ||
      !CBB_add_bytes(&contents, npa, npa_len) ||
      !CBB_flush(out)) {
    return 0;
  }

  return 1;
}


/* Signed certificate timestamps.
 *
 * https://tools.ietf.org/html/rfc6962#section-3.3.1 */

static int ext_sct_add_clienthello(SSL *ssl, CBB *out) {
  if (!ssl->signed_cert_timestamps_enabled) {
    return 1;
  }

  if (!CBB_add_u16(out, TLSEXT_TYPE_certificate_timestamp) ||
      !CBB_add_u16(out, 0 /* length */)) {
    return 0;
  }

  return 1;
}

static int ext_sct_parse_serverhello(SSL *ssl, uint8_t *out_alert,
                                     CBS *contents) {
  if (contents == NULL) {
    return 1;
  }

  /* If this is false then we should never have sent the SCT extension in the
   * ClientHello and thus this function should never have been called. */
  assert(ssl->signed_cert_timestamps_enabled);

  if (CBS_len(contents) == 0) {
    *out_alert = SSL_AD_DECODE_ERROR;
    return 0;
  }

  /* Session resumption uses the original session information. */
  if (!ssl->hit &&
      !CBS_stow(contents, &ssl->session->tlsext_signed_cert_timestamp_list,
                &ssl->session->tlsext_signed_cert_timestamp_list_length)) {
    *out_alert = SSL_AD_INTERNAL_ERROR;
    return 0;
  }

  return 1;
}

static int ext_sct_parse_clienthello(SSL *ssl, uint8_t *out_alert,
                                     CBS *contents) {
  return contents == NULL || CBS_len(contents) == 0;
}

static int ext_sct_add_serverhello(SSL *ssl, CBB *out) {
  /* The extension shouldn't be sent when resuming sessions. */
  if (ssl->hit ||
      ssl->ctx->signed_cert_timestamp_list_length == 0) {
    return 1;
  }

  CBB contents;
  return CBB_add_u16(out, TLSEXT_TYPE_certificate_timestamp) &&
         CBB_add_u16_length_prefixed(out, &contents) &&
         CBB_add_bytes(&contents, ssl->ctx->signed_cert_timestamp_list,
                       ssl->ctx->signed_cert_timestamp_list_length) &&
         CBB_flush(out);
}


/* Application-level Protocol Negotiation.
 *
 * https://tools.ietf.org/html/rfc7301 */

static void ext_alpn_init(SSL *ssl) {
  OPENSSL_free(ssl->s3->alpn_selected);
  ssl->s3->alpn_selected = NULL;
}

static int ext_alpn_add_clienthello(SSL *ssl, CBB *out) {
  if (ssl->alpn_client_proto_list == NULL ||
      ssl->s3->initial_handshake_complete) {
    return 1;
  }

  CBB contents, proto_list;
  if (!CBB_add_u16(out, TLSEXT_TYPE_application_layer_protocol_negotiation) ||
      !CBB_add_u16_length_prefixed(out, &contents) ||
      !CBB_add_u16_length_prefixed(&contents, &proto_list) ||
      !CBB_add_bytes(&proto_list, ssl->alpn_client_proto_list,
                     ssl->alpn_client_proto_list_len) ||
      !CBB_flush(out)) {
    return 0;
  }

  return 1;
}

static int ext_alpn_parse_serverhello(SSL *ssl, uint8_t *out_alert,
                                      CBS *contents) {
  if (contents == NULL) {
    return 1;
  }

  assert(!ssl->s3->initial_handshake_complete);
  assert(ssl->alpn_client_proto_list != NULL);

  if (ssl->s3->next_proto_neg_seen) {
    /* NPN and ALPN may not be negotiated in the same connection. */
    *out_alert = SSL_AD_ILLEGAL_PARAMETER;
    OPENSSL_PUT_ERROR(SSL, SSL_R_NEGOTIATED_BOTH_NPN_AND_ALPN);
    return 0;
  }

  /* The extension data consists of a ProtocolNameList which must have
   * exactly one ProtocolName. Each of these is length-prefixed. */
  CBS protocol_name_list, protocol_name;
  if (!CBS_get_u16_length_prefixed(contents, &protocol_name_list) ||
      CBS_len(contents) != 0 ||
      !CBS_get_u8_length_prefixed(&protocol_name_list, &protocol_name) ||
      /* Empty protocol names are forbidden. */
      CBS_len(&protocol_name) == 0 ||
      CBS_len(&protocol_name_list) != 0) {
    return 0;
  }

  if (!CBS_stow(&protocol_name, &ssl->s3->alpn_selected,
                &ssl->s3->alpn_selected_len)) {
    *out_alert = SSL_AD_INTERNAL_ERROR;
    return 0;
  }

  return 1;
}

static int ext_alpn_parse_clienthello(SSL *ssl, uint8_t *out_alert,
                                      CBS *contents) {
  if (contents == NULL) {
    return 1;
  }

  if (ssl->ctx->alpn_select_cb == NULL ||
      ssl->s3->initial_handshake_complete) {
    return 1;
  }

  /* ALPN takes precedence over NPN. */
  ssl->s3->next_proto_neg_seen = 0;

  CBS protocol_name_list;
  if (!CBS_get_u16_length_prefixed(contents, &protocol_name_list) ||
      CBS_len(contents) != 0 ||
      CBS_len(&protocol_name_list) < 2) {
    return 0;
  }

  /* Validate the protocol list. */
  CBS protocol_name_list_copy = protocol_name_list;
  while (CBS_len(&protocol_name_list_copy) > 0) {
    CBS protocol_name;

    if (!CBS_get_u8_length_prefixed(&protocol_name_list_copy, &protocol_name) ||
        /* Empty protocol names are forbidden. */
        CBS_len(&protocol_name) == 0) {
      return 0;
    }
  }

  const uint8_t *selected;
  uint8_t selected_len;
  if (ssl->ctx->alpn_select_cb(
          ssl, &selected, &selected_len, CBS_data(&protocol_name_list),
          CBS_len(&protocol_name_list),
          ssl->ctx->alpn_select_cb_arg) == SSL_TLSEXT_ERR_OK) {
    OPENSSL_free(ssl->s3->alpn_selected);
    ssl->s3->alpn_selected = BUF_memdup(selected, selected_len);
    if (ssl->s3->alpn_selected == NULL) {
      *out_alert = SSL_AD_INTERNAL_ERROR;
      return 0;
    }
    ssl->s3->alpn_selected_len = selected_len;
  }

  return 1;
}

static int ext_alpn_add_serverhello(SSL *ssl, CBB *out) {
  if (ssl->s3->alpn_selected == NULL) {
    return 1;
  }

  CBB contents, proto_list, proto;
  if (!CBB_add_u16(out, TLSEXT_TYPE_application_layer_protocol_negotiation) ||
      !CBB_add_u16_length_prefixed(out, &contents) ||
      !CBB_add_u16_length_prefixed(&contents, &proto_list) ||
      !CBB_add_u8_length_prefixed(&proto_list, &proto) ||
      !CBB_add_bytes(&proto, ssl->s3->alpn_selected, ssl->s3->alpn_selected_len) ||
      !CBB_flush(out)) {
    return 0;
  }

  return 1;
}


/* Channel ID.
 *
 * https://tools.ietf.org/html/draft-balfanz-tls-channelid-01 */

static void ext_channel_id_init(SSL *ssl) {
  ssl->s3->tlsext_channel_id_valid = 0;
}

static int ext_channel_id_add_clienthello(SSL *ssl, CBB *out) {
  if (!ssl->tlsext_channel_id_enabled ||
      SSL_IS_DTLS(ssl)) {
    return 1;
  }

  if (!CBB_add_u16(out, TLSEXT_TYPE_channel_id) ||
      !CBB_add_u16(out, 0 /* length */)) {
    return 0;
  }

  return 1;
}

static int ext_channel_id_parse_serverhello(SSL *ssl, uint8_t *out_alert,
                                            CBS *contents) {
  if (contents == NULL) {
    return 1;
  }

  assert(!SSL_IS_DTLS(ssl));
  assert(ssl->tlsext_channel_id_enabled);

  if (CBS_len(contents) != 0) {
    return 0;
  }

  ssl->s3->tlsext_channel_id_valid = 1;
  return 1;
}

static int ext_channel_id_parse_clienthello(SSL *ssl, uint8_t *out_alert,
                                            CBS *contents) {
  if (contents == NULL ||
      !ssl->tlsext_channel_id_enabled ||
      SSL_IS_DTLS(ssl)) {
    return 1;
  }

  if (CBS_len(contents) != 0) {
    return 0;
  }

  ssl->s3->tlsext_channel_id_valid = 1;
  return 1;
}

static int ext_channel_id_add_serverhello(SSL *ssl, CBB *out) {
  if (!ssl->s3->tlsext_channel_id_valid) {
    return 1;
  }

  if (!CBB_add_u16(out, TLSEXT_TYPE_channel_id) ||
      !CBB_add_u16(out, 0 /* length */)) {
    return 0;
  }

  return 1;
}


/* Secure Real-time Transport Protocol (SRTP) extension.
 *
 * https://tools.ietf.org/html/rfc5764 */


static void ext_srtp_init(SSL *ssl) {
  ssl->srtp_profile = NULL;
}

static int ext_srtp_add_clienthello(SSL *ssl, CBB *out) {
  STACK_OF(SRTP_PROTECTION_PROFILE) *profiles = SSL_get_srtp_profiles(ssl);
  if (profiles == NULL) {
    return 1;
  }
  const size_t num_profiles = sk_SRTP_PROTECTION_PROFILE_num(profiles);
  if (num_profiles == 0) {
    return 1;
  }

  CBB contents, profile_ids;
  if (!CBB_add_u16(out, TLSEXT_TYPE_srtp) ||
      !CBB_add_u16_length_prefixed(out, &contents) ||
      !CBB_add_u16_length_prefixed(&contents, &profile_ids)) {
    return 0;
  }

  size_t i;
  for (i = 0; i < num_profiles; i++) {
    if (!CBB_add_u16(&profile_ids,
                     sk_SRTP_PROTECTION_PROFILE_value(profiles, i)->id)) {
      return 0;
    }
  }

  if (!CBB_add_u8(&contents, 0 /* empty use_mki value */) ||
      !CBB_flush(out)) {
    return 0;
  }

  return 1;
}

static int ext_srtp_parse_serverhello(SSL *ssl, uint8_t *out_alert,
                                      CBS *contents) {
  if (contents == NULL) {
    return 1;
  }

  /* The extension consists of a u16-prefixed profile ID list containing a
   * single uint16_t profile ID, then followed by a u8-prefixed srtp_mki field.
   *
   * See https://tools.ietf.org/html/rfc5764#section-4.1.1 */
  CBS profile_ids, srtp_mki;
  uint16_t profile_id;
  if (!CBS_get_u16_length_prefixed(contents, &profile_ids) ||
      !CBS_get_u16(&profile_ids, &profile_id) ||
      CBS_len(&profile_ids) != 0 ||
      !CBS_get_u8_length_prefixed(contents, &srtp_mki) ||
      CBS_len(contents) != 0) {
    OPENSSL_PUT_ERROR(SSL, SSL_R_BAD_SRTP_PROTECTION_PROFILE_LIST);
    return 0;
  }

  if (CBS_len(&srtp_mki) != 0) {
    /* Must be no MKI, since we never offer one. */
    OPENSSL_PUT_ERROR(SSL, SSL_R_BAD_SRTP_MKI_VALUE);
    *out_alert = SSL_AD_ILLEGAL_PARAMETER;
    return 0;
  }

  STACK_OF(SRTP_PROTECTION_PROFILE) *profiles = SSL_get_srtp_profiles(ssl);

  /* Check to see if the server gave us something we support (and presumably
   * offered). */
  size_t i;
  for (i = 0; i < sk_SRTP_PROTECTION_PROFILE_num(profiles); i++) {
    const SRTP_PROTECTION_PROFILE *profile =
        sk_SRTP_PROTECTION_PROFILE_value(profiles, i);

    if (profile->id == profile_id) {
      ssl->srtp_profile = profile;
      return 1;
    }
  }

  OPENSSL_PUT_ERROR(SSL, SSL_R_BAD_SRTP_PROTECTION_PROFILE_LIST);
  *out_alert = SSL_AD_ILLEGAL_PARAMETER;
  return 0;
}

static int ext_srtp_parse_clienthello(SSL *ssl, uint8_t *out_alert,
                                      CBS *contents) {
  if (contents == NULL) {
    return 1;
  }

  CBS profile_ids, srtp_mki;
  if (!CBS_get_u16_length_prefixed(contents, &profile_ids) ||
      CBS_len(&profile_ids) < 2 ||
      !CBS_get_u8_length_prefixed(contents, &srtp_mki) ||
      CBS_len(contents) != 0) {
    OPENSSL_PUT_ERROR(SSL, SSL_R_BAD_SRTP_PROTECTION_PROFILE_LIST);
    return 0;
  }
  /* Discard the MKI value for now. */

  const STACK_OF(SRTP_PROTECTION_PROFILE) *server_profiles =
      SSL_get_srtp_profiles(ssl);

  /* Pick the server's most preferred profile. */
  size_t i;
  for (i = 0; i < sk_SRTP_PROTECTION_PROFILE_num(server_profiles); i++) {
    const SRTP_PROTECTION_PROFILE *server_profile =
        sk_SRTP_PROTECTION_PROFILE_value(server_profiles, i);

    CBS profile_ids_tmp;
    CBS_init(&profile_ids_tmp, CBS_data(&profile_ids), CBS_len(&profile_ids));

    while (CBS_len(&profile_ids_tmp) > 0) {
      uint16_t profile_id;
      if (!CBS_get_u16(&profile_ids_tmp, &profile_id)) {
        return 0;
      }

      if (server_profile->id == profile_id) {
        ssl->srtp_profile = server_profile;
        return 1;
      }
    }
  }

  return 1;
}

static int ext_srtp_add_serverhello(SSL *ssl, CBB *out) {
  if (ssl->srtp_profile == NULL) {
    return 1;
  }

  CBB contents, profile_ids;
  if (!CBB_add_u16(out, TLSEXT_TYPE_srtp) ||
      !CBB_add_u16_length_prefixed(out, &contents) ||
      !CBB_add_u16_length_prefixed(&contents, &profile_ids) ||
      !CBB_add_u16(&profile_ids, ssl->srtp_profile->id) ||
      !CBB_add_u8(&contents, 0 /* empty MKI */) ||
      !CBB_flush(out)) {
    return 0;
  }

  return 1;
}


/* EC point formats.
 *
 * https://tools.ietf.org/html/rfc4492#section-5.1.2 */

static int ssl_any_ec_cipher_suites_enabled(const SSL *ssl) {
  if (ssl->version < TLS1_VERSION && !SSL_IS_DTLS(ssl)) {
    return 0;
  }

  const STACK_OF(SSL_CIPHER) *cipher_stack = SSL_get_ciphers(ssl);

  size_t i;
  for (i = 0; i < sk_SSL_CIPHER_num(cipher_stack); i++) {
    const SSL_CIPHER *cipher = sk_SSL_CIPHER_value(cipher_stack, i);

    const uint32_t alg_k = cipher->algorithm_mkey;
    const uint32_t alg_a = cipher->algorithm_auth;
    if ((alg_k & SSL_kECDHE) || (alg_a & SSL_aECDSA)) {
      return 1;
    }
  }

  return 0;
}

static int ext_ec_point_add_extension(SSL *ssl, CBB *out) {
  CBB contents, formats;
  if (!CBB_add_u16(out, TLSEXT_TYPE_ec_point_formats) ||
      !CBB_add_u16_length_prefixed(out, &contents) ||
      !CBB_add_u8_length_prefixed(&contents, &formats) ||
      !CBB_add_u8(&formats, TLSEXT_ECPOINTFORMAT_uncompressed) ||
      !CBB_flush(out)) {
    return 0;
  }

  return 1;
}

static int ext_ec_point_add_clienthello(SSL *ssl, CBB *out) {
  if (!ssl_any_ec_cipher_suites_enabled(ssl)) {
    return 1;
  }

  return ext_ec_point_add_extension(ssl, out);
}

static int ext_ec_point_parse_serverhello(SSL *ssl, uint8_t *out_alert,
                                          CBS *contents) {
  if (contents == NULL) {
    return 1;
  }

  CBS ec_point_format_list;
  if (!CBS_get_u8_length_prefixed(contents, &ec_point_format_list) ||
      CBS_len(contents) != 0) {
    return 0;
  }

  /* Per RFC 4492, section 5.1.2, implementations MUST support the uncompressed
   * point format. */
  if (memchr(CBS_data(&ec_point_format_list), TLSEXT_ECPOINTFORMAT_uncompressed,
             CBS_len(&ec_point_format_list)) == NULL) {
    *out_alert = SSL_AD_ILLEGAL_PARAMETER;
    return 0;
  }

  return 1;
}

static int ext_ec_point_parse_clienthello(SSL *ssl, uint8_t *out_alert,
                                          CBS *contents) {
  return ext_ec_point_parse_serverhello(ssl, out_alert, contents);
}

static int ext_ec_point_add_serverhello(SSL *ssl, CBB *out) {
  const uint32_t alg_k = ssl->s3->tmp.new_cipher->algorithm_mkey;
  const uint32_t alg_a = ssl->s3->tmp.new_cipher->algorithm_auth;
  const int using_ecc = (alg_k & SSL_kECDHE) || (alg_a & SSL_aECDSA);

  if (!using_ecc) {
    return 1;
  }

  return ext_ec_point_add_extension(ssl, out);
}


/* EC supported curves.
 *
 * https://tools.ietf.org/html/rfc4492#section-5.1.2 */

static void ext_ec_curves_init(SSL *ssl) {
  OPENSSL_free(ssl->s3->tmp.peer_ellipticcurvelist);
  ssl->s3->tmp.peer_ellipticcurvelist = NULL;
  ssl->s3->tmp.peer_ellipticcurvelist_length = 0;
}

static int ext_ec_curves_add_clienthello(SSL *ssl, CBB *out) {
  if (!ssl_any_ec_cipher_suites_enabled(ssl)) {
    return 1;
  }

  CBB contents, curves_bytes;
  if (!CBB_add_u16(out, TLSEXT_TYPE_elliptic_curves) ||
      !CBB_add_u16_length_prefixed(out, &contents) ||
      !CBB_add_u16_length_prefixed(&contents, &curves_bytes)) {
    return 0;
  }

  const uint16_t *curves;
  size_t curves_len;
  tls1_get_curvelist(ssl, 0, &curves, &curves_len);

  size_t i;
  for (i = 0; i < curves_len; i++) {
    if (!CBB_add_u16(&curves_bytes, curves[i])) {
      return 0;
    }
  }

  return CBB_flush(out);
}

static int ext_ec_curves_parse_serverhello(SSL *ssl, uint8_t *out_alert,
                                           CBS *contents) {
  /* This extension is not expected to be echoed by servers and is ignored. */
  return 1;
}

static int ext_ec_curves_parse_clienthello(SSL *ssl, uint8_t *out_alert,
                                           CBS *contents) {
  if (contents == NULL) {
    return 1;
  }

  CBS elliptic_curve_list;
  if (!CBS_get_u16_length_prefixed(contents, &elliptic_curve_list) ||
      CBS_len(&elliptic_curve_list) == 0 ||
      (CBS_len(&elliptic_curve_list) & 1) != 0 ||
      CBS_len(contents) != 0) {
    return 0;
  }

  ssl->s3->tmp.peer_ellipticcurvelist =
      (uint16_t *)OPENSSL_malloc(CBS_len(&elliptic_curve_list));

  if (ssl->s3->tmp.peer_ellipticcurvelist == NULL) {
    *out_alert = SSL_AD_INTERNAL_ERROR;
    return 0;
  }

  const size_t num_curves = CBS_len(&elliptic_curve_list) / 2;
  size_t i;
  for (i = 0; i < num_curves; i++) {
    if (!CBS_get_u16(&elliptic_curve_list,
                     &ssl->s3->tmp.peer_ellipticcurvelist[i])) {
      goto err;
    }
  }

  assert(CBS_len(&elliptic_curve_list) == 0);
  ssl->s3->tmp.peer_ellipticcurvelist_length = num_curves;

  return 1;

err:
  OPENSSL_free(ssl->s3->tmp.peer_ellipticcurvelist);
  ssl->s3->tmp.peer_ellipticcurvelist = NULL;
  *out_alert = SSL_AD_INTERNAL_ERROR;
  return 0;
}

static int ext_ec_curves_add_serverhello(SSL *ssl, CBB *out) {
  /* Servers don't echo this extension. */
  return 1;
}


/* kExtensions contains all the supported extensions. */
static const struct tls_extension kExtensions[] = {
  {
    /* The renegotiation extension must always be at index zero because the
     * |received| and |sent| bitsets need to be tweaked when the "extension" is
     * sent as an SCSV. */
    TLSEXT_TYPE_renegotiate,
    NULL,
    ext_ri_add_clienthello,
    ext_ri_parse_serverhello,
    ext_ri_parse_clienthello,
    ext_ri_add_serverhello,
  },
  {
    TLSEXT_TYPE_server_name,
    ext_sni_init,
    ext_sni_add_clienthello,
    ext_sni_parse_serverhello,
    ext_sni_parse_clienthello,
    ext_sni_add_serverhello,
  },
  {
    TLSEXT_TYPE_extended_master_secret,
    ext_ems_init,
    ext_ems_add_clienthello,
    ext_ems_parse_serverhello,
    ext_ems_parse_clienthello,
    ext_ems_add_serverhello,
  },
  {
    TLSEXT_TYPE_session_ticket,
    NULL,
    ext_ticket_add_clienthello,
    ext_ticket_parse_serverhello,
    ext_ticket_parse_clienthello,
    ext_ticket_add_serverhello,
  },
  {
    TLSEXT_TYPE_signature_algorithms,
    NULL,
    ext_sigalgs_add_clienthello,
    ext_sigalgs_parse_serverhello,
    ext_sigalgs_parse_clienthello,
    ext_sigalgs_add_serverhello,
  },
  {
    TLSEXT_TYPE_status_request,
    ext_ocsp_init,
    ext_ocsp_add_clienthello,
    ext_ocsp_parse_serverhello,
    ext_ocsp_parse_clienthello,
    ext_ocsp_add_serverhello,
  },
  {
    TLSEXT_TYPE_next_proto_neg,
    ext_npn_init,
    ext_npn_add_clienthello,
    ext_npn_parse_serverhello,
    ext_npn_parse_clienthello,
    ext_npn_add_serverhello,
  },
  {
    TLSEXT_TYPE_certificate_timestamp,
    NULL,
    ext_sct_add_clienthello,
    ext_sct_parse_serverhello,
    ext_sct_parse_clienthello,
    ext_sct_add_serverhello,
  },
  {
    TLSEXT_TYPE_application_layer_protocol_negotiation,
    ext_alpn_init,
    ext_alpn_add_clienthello,
    ext_alpn_parse_serverhello,
    ext_alpn_parse_clienthello,
    ext_alpn_add_serverhello,
  },
  {
    TLSEXT_TYPE_channel_id,
    ext_channel_id_init,
    ext_channel_id_add_clienthello,
    ext_channel_id_parse_serverhello,
    ext_channel_id_parse_clienthello,
    ext_channel_id_add_serverhello,
  },
  {
    TLSEXT_TYPE_srtp,
    ext_srtp_init,
    ext_srtp_add_clienthello,
    ext_srtp_parse_serverhello,
    ext_srtp_parse_clienthello,
    ext_srtp_add_serverhello,
  },
  {
    TLSEXT_TYPE_ec_point_formats,
    NULL,
    ext_ec_point_add_clienthello,
    ext_ec_point_parse_serverhello,
    ext_ec_point_parse_clienthello,
    ext_ec_point_add_serverhello,
  },
  {
    TLSEXT_TYPE_elliptic_curves,
    ext_ec_curves_init,
    ext_ec_curves_add_clienthello,
    ext_ec_curves_parse_serverhello,
    ext_ec_curves_parse_clienthello,
    ext_ec_curves_add_serverhello,
  },
};

#define kNumExtensions (sizeof(kExtensions) / sizeof(struct tls_extension))

OPENSSL_COMPILE_ASSERT(kNumExtensions <=
                           sizeof(((SSL *)NULL)->s3->tmp.extensions.sent) * 8,
                       too_many_extensions_for_sent_bitset);
OPENSSL_COMPILE_ASSERT(kNumExtensions <=
                           sizeof(((SSL *)NULL)->s3->tmp.extensions.received) *
                               8,
                       too_many_extensions_for_received_bitset);

static const struct tls_extension *tls_extension_find(uint32_t *out_index,
                                                      uint16_t value) {
  unsigned i;
  for (i = 0; i < kNumExtensions; i++) {
    if (kExtensions[i].value == value) {
      *out_index = i;
      return &kExtensions[i];
    }
  }

  return NULL;
}

int SSL_extension_supported(unsigned extension_value) {
  uint32_t index;
  return extension_value == TLSEXT_TYPE_padding ||
         tls_extension_find(&index, extension_value) != NULL;
}

/* header_len is the length of the ClientHello header written so far, used to
 * compute padding. It does not include the record header. Pass 0 if no padding
 * is to be done. */
uint8_t *ssl_add_clienthello_tlsext(SSL *s, uint8_t *const buf,
                                    uint8_t *const limit, size_t header_len) {
  /* don't add extensions for SSLv3 unless doing secure renegotiation */
  if (s->client_version == SSL3_VERSION && !s->s3->send_connection_binding) {
    return buf;
  }

  CBB cbb, extensions;
  CBB_zero(&cbb);
  if (!CBB_init_fixed(&cbb, buf, limit - buf) ||
      !CBB_add_u16_length_prefixed(&cbb, &extensions)) {
    goto err;
  }

  s->s3->tmp.extensions.sent = 0;
  s->s3->tmp.custom_extensions.sent = 0;

  size_t i;
  for (i = 0; i < kNumExtensions; i++) {
    if (kExtensions[i].init != NULL) {
      kExtensions[i].init(s);
    }
  }

  for (i = 0; i < kNumExtensions; i++) {
    const size_t len_before = CBB_len(&extensions);
    if (!kExtensions[i].add_clienthello(s, &extensions)) {
      OPENSSL_PUT_ERROR(SSL, SSL_R_ERROR_ADDING_EXTENSION);
      ERR_add_error_dataf("extension: %u", (unsigned)kExtensions[i].value);
      goto err;
    }

    if (CBB_len(&extensions) != len_before) {
      s->s3->tmp.extensions.sent |= (1u << i);
    }
  }

  if (!custom_ext_add_clienthello(s, &extensions)) {
    goto err;
  }

  if (header_len > 0) {
    header_len += CBB_len(&extensions);
    if (header_len > 0xff && header_len < 0x200) {
      /* Add padding to workaround bugs in F5 terminators. See
       * https://tools.ietf.org/html/draft-agl-tls-padding-03
       *
       * NB: because this code works out the length of all existing extensions
       * it MUST always appear last. */
      size_t padding_len = 0x200 - header_len;
      /* Extensions take at least four bytes to encode. Always include least
       * one byte of data if including the extension. WebSphere Application
       * Server 7.0 is intolerant to the last extension being zero-length. */
      if (padding_len >= 4 + 1) {
        padding_len -= 4;
      } else {
        padding_len = 1;
      }

      uint8_t *padding_bytes;
      if (!CBB_add_u16(&extensions, TLSEXT_TYPE_padding) ||
          !CBB_add_u16(&extensions, padding_len) ||
          !CBB_add_space(&extensions, &padding_bytes, padding_len)) {
        goto err;
      }

      memset(padding_bytes, 0, padding_len);
    }
  }

  if (!CBB_flush(&cbb)) {
    goto err;
  }

  uint8_t *ret = buf;
  const size_t cbb_len = CBB_len(&cbb);
  /* If only two bytes have been written then the extensions are actually empty
   * and those two bytes are the zero length. In that case, we don't bother
   * sending the extensions length. */
  if (cbb_len > 2) {
    ret += cbb_len;
  }

  CBB_cleanup(&cbb);
  return ret;

err:
  CBB_cleanup(&cbb);
  OPENSSL_PUT_ERROR(SSL, ERR_R_INTERNAL_ERROR);
  return NULL;
}

uint8_t *ssl_add_serverhello_tlsext(SSL *s, uint8_t *const buf,
                                    uint8_t *const limit) {
  /* don't add extensions for SSLv3, unless doing secure renegotiation */
  if (s->version == SSL3_VERSION && !s->s3->send_connection_binding) {
    return buf;
  }

  CBB cbb, extensions;
  CBB_zero(&cbb);
  if (!CBB_init_fixed(&cbb, buf, limit - buf) ||
      !CBB_add_u16_length_prefixed(&cbb, &extensions)) {
    goto err;
  }

  unsigned i;
  for (i = 0; i < kNumExtensions; i++) {
    if (!(s->s3->tmp.extensions.received & (1u << i))) {
      /* Don't send extensions that were not received. */
      continue;
    }

    if (!kExtensions[i].add_serverhello(s, &extensions)) {
      OPENSSL_PUT_ERROR(SSL, SSL_R_ERROR_ADDING_EXTENSION);
      ERR_add_error_dataf("extension: %u", (unsigned)kExtensions[i].value);
      goto err;
    }
  }

  if (!custom_ext_add_serverhello(s, &extensions)) {
    goto err;
  }

  if (!CBB_flush(&cbb)) {
    goto err;
  }

  uint8_t *ret = buf;
  const size_t cbb_len = CBB_len(&cbb);
  /* If only two bytes have been written then the extensions are actually empty
   * and those two bytes are the zero length. In that case, we don't bother
   * sending the extensions length. */
  if (cbb_len > 2) {
    ret += cbb_len;
  }

  CBB_cleanup(&cbb);
  return ret;

err:
  CBB_cleanup(&cbb);
  OPENSSL_PUT_ERROR(SSL, ERR_R_INTERNAL_ERROR);
  return NULL;
}

static int ssl_scan_clienthello_tlsext(SSL *s, CBS *cbs, int *out_alert) {
  size_t i;
  for (i = 0; i < kNumExtensions; i++) {
    if (kExtensions[i].init != NULL) {
      kExtensions[i].init(s);
    }
  }

  s->s3->tmp.extensions.received = 0;
  s->s3->tmp.custom_extensions.received = 0;
  /* The renegotiation extension must always be at index zero because the
   * |received| and |sent| bitsets need to be tweaked when the "extension" is
   * sent as an SCSV. */
  assert(kExtensions[0].value == TLSEXT_TYPE_renegotiate);

  /* There may be no extensions. */
  if (CBS_len(cbs) != 0) {
    /* Decode the extensions block and check it is valid. */
    CBS extensions;
    if (!CBS_get_u16_length_prefixed(cbs, &extensions) ||
        !tls1_check_duplicate_extensions(&extensions)) {
      *out_alert = SSL_AD_DECODE_ERROR;
      return 0;
    }

    while (CBS_len(&extensions) != 0) {
      uint16_t type;
      CBS extension;

      /* Decode the next extension. */
      if (!CBS_get_u16(&extensions, &type) ||
          !CBS_get_u16_length_prefixed(&extensions, &extension)) {
        *out_alert = SSL_AD_DECODE_ERROR;
        return 0;
      }

      unsigned ext_index;
      const struct tls_extension *const ext =
          tls_extension_find(&ext_index, type);

      if (ext == NULL) {
        if (!custom_ext_parse_clienthello(s, out_alert, type, &extension)) {
          OPENSSL_PUT_ERROR(SSL, SSL_R_ERROR_PARSING_EXTENSION);
          return 0;
        }
        continue;
      }

      s->s3->tmp.extensions.received |= (1u << ext_index);
      uint8_t alert = SSL_AD_DECODE_ERROR;
      if (!ext->parse_clienthello(s, &alert, &extension)) {
        *out_alert = alert;
        OPENSSL_PUT_ERROR(SSL, SSL_R_ERROR_PARSING_EXTENSION);
        ERR_add_error_dataf("extension: %u", (unsigned)type);
        return 0;
      }
    }
  }

  for (i = 0; i < kNumExtensions; i++) {
    if (!(s->s3->tmp.extensions.received & (1u << i))) {
      /* Extension wasn't observed so call the callback with a NULL
       * parameter. */
      uint8_t alert = SSL_AD_DECODE_ERROR;
      if (!kExtensions[i].parse_clienthello(s, &alert, NULL)) {
        OPENSSL_PUT_ERROR(SSL, SSL_R_MISSING_EXTENSION);
        ERR_add_error_dataf("extension: %u", (unsigned)kExtensions[i].value);
        *out_alert = alert;
        return 0;
      }
    }
  }

  return 1;
}

int ssl_parse_clienthello_tlsext(SSL *s, CBS *cbs) {
  int alert = -1;
  if (ssl_scan_clienthello_tlsext(s, cbs, &alert) <= 0) {
    ssl3_send_alert(s, SSL3_AL_FATAL, alert);
    return 0;
  }

  if (ssl_check_clienthello_tlsext(s) <= 0) {
    OPENSSL_PUT_ERROR(SSL, SSL_R_CLIENTHELLO_TLSEXT);
    return 0;
  }

  return 1;
}

static int ssl_scan_serverhello_tlsext(SSL *s, CBS *cbs, int *out_alert) {
  uint32_t received = 0;
  assert(kNumExtensions <= sizeof(received) * 8);

  if (CBS_len(cbs) != 0) {
    /* Decode the extensions block and check it is valid. */
    CBS extensions;
    if (!CBS_get_u16_length_prefixed(cbs, &extensions) ||
        !tls1_check_duplicate_extensions(&extensions)) {
      *out_alert = SSL_AD_DECODE_ERROR;
      return 0;
    }


    while (CBS_len(&extensions) != 0) {
      uint16_t type;
      CBS extension;

      /* Decode the next extension. */
      if (!CBS_get_u16(&extensions, &type) ||
          !CBS_get_u16_length_prefixed(&extensions, &extension)) {
        *out_alert = SSL_AD_DECODE_ERROR;
        return 0;
      }

      unsigned ext_index;
      const struct tls_extension *const ext =
          tls_extension_find(&ext_index, type);

      if (ext == NULL) {
        if (!custom_ext_parse_serverhello(s, out_alert, type, &extension)) {
          return 0;
        }
        continue;
      }

      if (!(s->s3->tmp.extensions.sent & (1u << ext_index))) {
        /* If the extension was never sent then it is illegal. */
        OPENSSL_PUT_ERROR(SSL, SSL_R_UNEXPECTED_EXTENSION);
        ERR_add_error_dataf("extension :%u", (unsigned)type);
        *out_alert = SSL_AD_DECODE_ERROR;
        return 0;
      }

      received |= (1u << ext_index);

      uint8_t alert = SSL_AD_DECODE_ERROR;
      if (!ext->parse_serverhello(s, &alert, &extension)) {
        OPENSSL_PUT_ERROR(SSL, SSL_R_ERROR_PARSING_EXTENSION);
        ERR_add_error_dataf("extension: %u", (unsigned)type);
        *out_alert = alert;
        return 0;
      }
    }
  }

  size_t i;
  for (i = 0; i < kNumExtensions; i++) {
    if (!(received & (1u << i))) {
      /* Extension wasn't observed so call the callback with a NULL
       * parameter. */
      uint8_t alert = SSL_AD_DECODE_ERROR;
      if (!kExtensions[i].parse_serverhello(s, &alert, NULL)) {
        OPENSSL_PUT_ERROR(SSL, SSL_R_MISSING_EXTENSION);
        ERR_add_error_dataf("extension: %u", (unsigned)kExtensions[i].value);
        *out_alert = alert;
        return 0;
      }
    }
  }

  return 1;
}

static int ssl_check_clienthello_tlsext(SSL *s) {
  int ret = SSL_TLSEXT_ERR_NOACK;
  int al = SSL_AD_UNRECOGNIZED_NAME;

  /* The handling of the ECPointFormats extension is done elsewhere, namely in
   * ssl3_choose_cipher in s3_lib.c. */

  if (s->ctx != NULL && s->ctx->tlsext_servername_callback != 0) {
    ret = s->ctx->tlsext_servername_callback(s, &al,
                                             s->ctx->tlsext_servername_arg);
  } else if (s->initial_ctx != NULL &&
             s->initial_ctx->tlsext_servername_callback != 0) {
    ret = s->initial_ctx->tlsext_servername_callback(
        s, &al, s->initial_ctx->tlsext_servername_arg);
  }

  switch (ret) {
    case SSL_TLSEXT_ERR_ALERT_FATAL:
      ssl3_send_alert(s, SSL3_AL_FATAL, al);
      return -1;

    case SSL_TLSEXT_ERR_ALERT_WARNING:
      ssl3_send_alert(s, SSL3_AL_WARNING, al);
      return 1;

    case SSL_TLSEXT_ERR_NOACK:
      s->s3->tmp.should_ack_sni = 0;
      return 1;

    default:
      return 1;
  }
}

static int ssl_check_serverhello_tlsext(SSL *s) {
  int ret = SSL_TLSEXT_ERR_OK;
  int al = SSL_AD_UNRECOGNIZED_NAME;

  if (s->ctx != NULL && s->ctx->tlsext_servername_callback != 0) {
    ret = s->ctx->tlsext_servername_callback(s, &al,
                                             s->ctx->tlsext_servername_arg);
  } else if (s->initial_ctx != NULL &&
             s->initial_ctx->tlsext_servername_callback != 0) {
    ret = s->initial_ctx->tlsext_servername_callback(
        s, &al, s->initial_ctx->tlsext_servername_arg);
  }

  switch (ret) {
    case SSL_TLSEXT_ERR_ALERT_FATAL:
      ssl3_send_alert(s, SSL3_AL_FATAL, al);
      return -1;

    case SSL_TLSEXT_ERR_ALERT_WARNING:
      ssl3_send_alert(s, SSL3_AL_WARNING, al);
      return 1;

    default:
      return 1;
  }
}

int ssl_parse_serverhello_tlsext(SSL *s, CBS *cbs) {
  int alert = -1;
  if (s->version < SSL3_VERSION) {
    return 1;
  }

  if (ssl_scan_serverhello_tlsext(s, cbs, &alert) <= 0) {
    ssl3_send_alert(s, SSL3_AL_FATAL, alert);
    return 0;
  }

  if (ssl_check_serverhello_tlsext(s) <= 0) {
    OPENSSL_PUT_ERROR(SSL, SSL_R_SERVERHELLO_TLSEXT);
    return 0;
  }

  return 1;
}

int tls_process_ticket(SSL *ssl, SSL_SESSION **out_session,
                       int *out_send_ticket, const uint8_t *ticket,
                       size_t ticket_len, const uint8_t *session_id,
                       size_t session_id_len) {
  int ret = 1; /* Most errors are non-fatal. */
  SSL_CTX *ssl_ctx = ssl->initial_ctx;
  uint8_t *plaintext = NULL;

  HMAC_CTX hmac_ctx;
  HMAC_CTX_init(&hmac_ctx);
  EVP_CIPHER_CTX cipher_ctx;
  EVP_CIPHER_CTX_init(&cipher_ctx);

  *out_send_ticket = 0;
  *out_session = NULL;

  if (session_id_len > SSL_MAX_SSL_SESSION_ID_LENGTH) {
    goto done;
  }

  if (ticket_len == 0) {
    /* The client will accept a ticket but doesn't currently have one. */
    *out_send_ticket = 1;
    goto done;
  }

  /* Ensure there is room for the key name and the largest IV
   * |tlsext_ticket_key_cb| may try to consume. The real limit may be lower, but
   * the maximum IV length should be well under the minimum size for the
   * session material and HMAC. */
  if (ticket_len < SSL_TICKET_KEY_NAME_LEN + EVP_MAX_IV_LENGTH) {
    goto done;
  }
  const uint8_t *iv = ticket + SSL_TICKET_KEY_NAME_LEN;

  if (ssl_ctx->tlsext_ticket_key_cb != NULL) {
    int cb_ret = ssl_ctx->tlsext_ticket_key_cb(ssl, (uint8_t*)ticket /* name */,
                                               (uint8_t*)iv, &cipher_ctx, &hmac_ctx,
                                               0 /* decrypt */);
    if (cb_ret < 0) {
      ret = 0;
      goto done;
    }
    if (cb_ret == 0) {
      goto done;
    }
    if (cb_ret == 2) {
      *out_send_ticket = 1;
    }
  } else {
    /* Check the key name matches. */
    if (memcmp(ticket, ssl_ctx->tlsext_tick_key_name,
               SSL_TICKET_KEY_NAME_LEN) != 0) {
      goto done;
    }
    if (!HMAC_Init_ex(&hmac_ctx, ssl_ctx->tlsext_tick_hmac_key,
                      sizeof(ssl_ctx->tlsext_tick_hmac_key), tlsext_tick_md(),
                      NULL) ||
        !EVP_DecryptInit_ex(&cipher_ctx, EVP_aes_128_cbc(), NULL,
                            ssl_ctx->tlsext_tick_aes_key, iv)) {
      ret = 0;
      goto done;
    }
  }
  size_t iv_len = EVP_CIPHER_CTX_iv_length(&cipher_ctx);

  /* Check the MAC at the end of the ticket. */
  uint8_t mac[EVP_MAX_MD_SIZE];
  size_t mac_len = HMAC_size(&hmac_ctx);
  if (ticket_len < SSL_TICKET_KEY_NAME_LEN + iv_len + 1 + mac_len) {
    /* The ticket must be large enough for key name, IV, data, and MAC. */
    goto done;
  }
  HMAC_Update(&hmac_ctx, ticket, ticket_len - mac_len);
  HMAC_Final(&hmac_ctx, mac, NULL);
  if (CRYPTO_memcmp(mac, ticket + (ticket_len - mac_len), mac_len) != 0) {
    goto done;
  }

  /* Decrypt the session data. */
  const uint8_t *ciphertext = ticket + SSL_TICKET_KEY_NAME_LEN + iv_len;
  size_t ciphertext_len = ticket_len - SSL_TICKET_KEY_NAME_LEN - iv_len -
                          mac_len;
  plaintext = OPENSSL_malloc(ciphertext_len);
  if (plaintext == NULL) {
    ret = 0;
    goto done;
  }
  if (ciphertext_len >= INT_MAX) {
    goto done;
  }
  int len1, len2;
  if (!EVP_DecryptUpdate(&cipher_ctx, plaintext, &len1, ciphertext,
                         (int)ciphertext_len) ||
      !EVP_DecryptFinal_ex(&cipher_ctx, plaintext + len1, &len2)) {
    ERR_clear_error(); /* Don't leave an error on the queue. */
    goto done;
  }

  /* Decode the session. */
  SSL_SESSION *session = SSL_SESSION_from_bytes(plaintext, len1 + len2);
  if (session == NULL) {
    ERR_clear_error(); /* Don't leave an error on the queue. */
    goto done;
  }

  /* Copy the client's session ID into the new session, to denote the ticket has
   * been accepted. */
  memcpy(session->session_id, session_id, session_id_len);
  session->session_id_length = session_id_len;

  *out_session = session;

done:
  OPENSSL_free(plaintext);
  HMAC_CTX_cleanup(&hmac_ctx);
  EVP_CIPHER_CTX_cleanup(&cipher_ctx);
  return ret;
}

/* Tables to translate from NIDs to TLS v1.2 ids */
typedef struct {
  int nid;
  int id;
} tls12_lookup;

static const tls12_lookup tls12_md[] = {{NID_md5, TLSEXT_hash_md5},
                                        {NID_sha1, TLSEXT_hash_sha1},
                                        {NID_sha224, TLSEXT_hash_sha224},
                                        {NID_sha256, TLSEXT_hash_sha256},
                                        {NID_sha384, TLSEXT_hash_sha384},
                                        {NID_sha512, TLSEXT_hash_sha512}};

static const tls12_lookup tls12_sig[] = {{EVP_PKEY_RSA, TLSEXT_signature_rsa},
                                         {EVP_PKEY_EC, TLSEXT_signature_ecdsa}};

static int tls12_find_id(int nid, const tls12_lookup *table, size_t tlen) {
  size_t i;
  for (i = 0; i < tlen; i++) {
    if (table[i].nid == nid) {
      return table[i].id;
    }
  }

  return -1;
}

int tls12_get_sigid(int pkey_type) {
  return tls12_find_id(pkey_type, tls12_sig,
                       sizeof(tls12_sig) / sizeof(tls12_lookup));
}

int tls12_get_sigandhash(SSL *ssl, uint8_t *p, const EVP_MD *md) {
  int sig_id, md_id;

  if (!md) {
    return 0;
  }

  md_id = tls12_find_id(EVP_MD_type(md), tls12_md,
                        sizeof(tls12_md) / sizeof(tls12_lookup));
  if (md_id == -1) {
    return 0;
  }

  sig_id = tls12_get_sigid(ssl_private_key_type(ssl));
  if (sig_id == -1) {
    return 0;
  }

  p[0] = (uint8_t)md_id;
  p[1] = (uint8_t)sig_id;
  return 1;
}

const EVP_MD *tls12_get_hash(uint8_t hash_alg) {
  switch (hash_alg) {
    case TLSEXT_hash_md5:
      return EVP_md5();

    case TLSEXT_hash_sha1:
      return EVP_sha1();

    case TLSEXT_hash_sha224:
      return EVP_sha224();

    case TLSEXT_hash_sha256:
      return EVP_sha256();

    case TLSEXT_hash_sha384:
      return EVP_sha384();

    case TLSEXT_hash_sha512:
      return EVP_sha512();

    default:
      return NULL;
  }
}

/* tls12_get_pkey_type returns the EVP_PKEY type corresponding to TLS signature
 * algorithm |sig_alg|. It returns -1 if the type is unknown. */
static int tls12_get_pkey_type(uint8_t sig_alg) {
  switch (sig_alg) {
    case TLSEXT_signature_rsa:
      return EVP_PKEY_RSA;

    case TLSEXT_signature_ecdsa:
      return EVP_PKEY_EC;

    default:
      return -1;
  }
}

OPENSSL_COMPILE_ASSERT(sizeof(TLS_SIGALGS) == 2,
    sizeof_tls_sigalgs_is_not_two);

int tls1_parse_peer_sigalgs(SSL *ssl, const CBS *in_sigalgs) {
  /* Extension ignored for inappropriate versions */
  if (!SSL_USE_SIGALGS(ssl)) {
    return 1;
  }

  CERT *const cert = ssl->cert;
  OPENSSL_free(cert->peer_sigalgs);
  cert->peer_sigalgs = NULL;
  cert->peer_sigalgslen = 0;

  size_t num_sigalgs = CBS_len(in_sigalgs);

  if (num_sigalgs % 2 != 0) {
    return 0;
  }
  num_sigalgs /= 2;

  /* supported_signature_algorithms in the certificate request is
   * allowed to be empty. */
  if (num_sigalgs == 0) {
    return 1;
  }

  /* This multiplication doesn't overflow because sizeof(TLS_SIGALGS) is two
   * (statically asserted above) and we just divided |num_sigalgs| by two. */
  cert->peer_sigalgs = OPENSSL_malloc(num_sigalgs * sizeof(TLS_SIGALGS));
  if (cert->peer_sigalgs == NULL) {
    return 0;
  }
  cert->peer_sigalgslen = num_sigalgs;

  CBS sigalgs;
  CBS_init(&sigalgs, CBS_data(in_sigalgs), CBS_len(in_sigalgs));

  size_t i;
  for (i = 0; i < num_sigalgs; i++) {
    TLS_SIGALGS *const sigalg = &cert->peer_sigalgs[i];
    if (!CBS_get_u8(&sigalgs, &sigalg->rhash) ||
        !CBS_get_u8(&sigalgs, &sigalg->rsign)) {
      return 0;
    }
  }

  return 1;
}

const EVP_MD *tls1_choose_signing_digest(SSL *ssl) {
  CERT *cert = ssl->cert;
  int type = ssl_private_key_type(ssl);
  size_t i, j;

  static const int kDefaultDigestList[] = {NID_sha256, NID_sha384, NID_sha512,
                                           NID_sha224, NID_sha1};

  const int *digest_nids = kDefaultDigestList;
  size_t num_digest_nids =
      sizeof(kDefaultDigestList) / sizeof(kDefaultDigestList[0]);
  if (cert->digest_nids != NULL) {
    digest_nids = cert->digest_nids;
    num_digest_nids = cert->num_digest_nids;
  }

  for (i = 0; i < num_digest_nids; i++) {
    const int digest_nid = digest_nids[i];
    for (j = 0; j < cert->peer_sigalgslen; j++) {
      const EVP_MD *md = tls12_get_hash(cert->peer_sigalgs[j].rhash);
      if (md == NULL ||
          digest_nid != EVP_MD_type(md) ||
          tls12_get_pkey_type(cert->peer_sigalgs[j].rsign) != type) {
        continue;
      }

      return md;
    }
  }

  /* If no suitable digest may be found, default to SHA-1. */
  return EVP_sha1();
}

int tls1_channel_id_hash(SSL *ssl, uint8_t *out, size_t *out_len) {
  int ret = 0;
  EVP_MD_CTX ctx;

  EVP_MD_CTX_init(&ctx);
  if (!EVP_DigestInit_ex(&ctx, EVP_sha256(), NULL)) {
    goto err;
  }

  static const char kClientIDMagic[] = "TLS Channel ID signature";
  EVP_DigestUpdate(&ctx, kClientIDMagic, sizeof(kClientIDMagic));

  if (ssl->hit) {
    static const char kResumptionMagic[] = "Resumption";
    EVP_DigestUpdate(&ctx, kResumptionMagic, sizeof(kResumptionMagic));
    if (ssl->session->original_handshake_hash_len == 0) {
      OPENSSL_PUT_ERROR(SSL, ERR_R_INTERNAL_ERROR);
      goto err;
    }
    EVP_DigestUpdate(&ctx, ssl->session->original_handshake_hash,
                     ssl->session->original_handshake_hash_len);
  }

  uint8_t handshake_hash[EVP_MAX_MD_SIZE];
  int handshake_hash_len = tls1_handshake_digest(ssl, handshake_hash,
                                                 sizeof(handshake_hash));
  if (handshake_hash_len < 0) {
    goto err;
  }
  EVP_DigestUpdate(&ctx, handshake_hash, (size_t)handshake_hash_len);
  unsigned len_u;
  EVP_DigestFinal_ex(&ctx, out, &len_u);
  *out_len = len_u;

  ret = 1;

err:
  EVP_MD_CTX_cleanup(&ctx);
  return ret;
}

/* tls1_record_handshake_hashes_for_channel_id records the current handshake
 * hashes in |s->session| so that Channel ID resumptions can sign that data. */
int tls1_record_handshake_hashes_for_channel_id(SSL *s) {
  int digest_len;
  /* This function should never be called for a resumed session because the
   * handshake hashes that we wish to record are for the original, full
   * handshake. */
  if (s->hit) {
    return -1;
  }

  digest_len =
      tls1_handshake_digest(s, s->session->original_handshake_hash,
                            sizeof(s->session->original_handshake_hash));
  if (digest_len < 0) {
    return -1;
  }

  s->session->original_handshake_hash_len = digest_len;

  return 1;
}
