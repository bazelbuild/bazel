/* Written by Dr Stephen N Henson (steve@openssl.org) for the OpenSSL
 * project 1999-2004.
 */
/* ====================================================================
 * Copyright (c) 1999 The OpenSSL Project.  All rights reserved.
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
 *    licensing@OpenSSL.org.
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

#include <assert.h>
#include <limits.h>
#include <string.h>

#include <openssl/asn1t.h>
#include <openssl/cipher.h>
#include <openssl/err.h>
#include <openssl/mem.h>
#include <openssl/pkcs8.h>
#include <openssl/rand.h>
#include <openssl/x509.h>

#include "internal.h"


/* PKCS#5 v2.0 password based encryption structures */

ASN1_SEQUENCE(PBE2PARAM) = {
	ASN1_SIMPLE(PBE2PARAM, keyfunc, X509_ALGOR),
	ASN1_SIMPLE(PBE2PARAM, encryption, X509_ALGOR)
} ASN1_SEQUENCE_END(PBE2PARAM)

IMPLEMENT_ASN1_FUNCTIONS(PBE2PARAM)

ASN1_SEQUENCE(PBKDF2PARAM) = {
	ASN1_SIMPLE(PBKDF2PARAM, salt, ASN1_ANY),
	ASN1_SIMPLE(PBKDF2PARAM, iter, ASN1_INTEGER),
	ASN1_OPT(PBKDF2PARAM, keylength, ASN1_INTEGER),
	ASN1_OPT(PBKDF2PARAM, prf, X509_ALGOR)
} ASN1_SEQUENCE_END(PBKDF2PARAM)

IMPLEMENT_ASN1_FUNCTIONS(PBKDF2PARAM);

static int ASN1_TYPE_set_octetstring(ASN1_TYPE *a, unsigned char *data, int len)
	{
	ASN1_STRING *os;

	if ((os=M_ASN1_OCTET_STRING_new()) == NULL) return(0);
	if (!M_ASN1_OCTET_STRING_set(os,data,len))
		{
		M_ASN1_OCTET_STRING_free(os);
		return 0;
		}
	ASN1_TYPE_set(a,V_ASN1_OCTET_STRING,os);
	return(1);
	}

static int param_to_asn1(EVP_CIPHER_CTX *c, ASN1_TYPE *type)
	{
	unsigned iv_len;

	iv_len = EVP_CIPHER_CTX_iv_length(c);
	return ASN1_TYPE_set_octetstring(type, c->oiv, iv_len);
	}

/* Return an algorithm identifier for a PKCS#5 v2.0 PBE algorithm:
 * yes I know this is horrible!
 *
 * Extended version to allow application supplied PRF NID and IV. */

X509_ALGOR *PKCS5_pbe2_set_iv(const EVP_CIPHER *cipher, int iter,
				 unsigned char *salt, int saltlen,
				 unsigned char *aiv, int prf_nid)
{
	X509_ALGOR *scheme = NULL, *kalg = NULL, *ret = NULL;
	int alg_nid, keylen;
	EVP_CIPHER_CTX ctx;
	unsigned char iv[EVP_MAX_IV_LENGTH];
	PBE2PARAM *pbe2 = NULL;
	const ASN1_OBJECT *obj;

	alg_nid = EVP_CIPHER_nid(cipher);
	if(alg_nid == NID_undef) {
		OPENSSL_PUT_ERROR(PKCS8, PKCS8_R_CIPHER_HAS_NO_OBJECT_IDENTIFIER);
		goto err;
	}
	obj = OBJ_nid2obj(alg_nid);

	if(!(pbe2 = PBE2PARAM_new())) goto merr;

	/* Setup the AlgorithmIdentifier for the encryption scheme */
	scheme = pbe2->encryption;

	scheme->algorithm = (ASN1_OBJECT*) obj;
	if(!(scheme->parameter = ASN1_TYPE_new())) goto merr;

	/* Create random IV */
	if (EVP_CIPHER_iv_length(cipher))
		{
		if (aiv)
			memcpy(iv, aiv, EVP_CIPHER_iv_length(cipher));
		else if (!RAND_bytes(iv, EVP_CIPHER_iv_length(cipher)))
  			goto err;
		}

	EVP_CIPHER_CTX_init(&ctx);

	/* Dummy cipherinit to just setup the IV, and PRF */
	if (!EVP_CipherInit_ex(&ctx, cipher, NULL, NULL, iv, 0))
		goto err;
	if(param_to_asn1(&ctx, scheme->parameter) < 0) {
		OPENSSL_PUT_ERROR(PKCS8, PKCS8_R_ERROR_SETTING_CIPHER_PARAMS);
		EVP_CIPHER_CTX_cleanup(&ctx);
		goto err;
	}
	/* If prf NID unspecified see if cipher has a preference.
	 * An error is OK here: just means use default PRF.
	 */
	if ((prf_nid == -1) && 
	EVP_CIPHER_CTX_ctrl(&ctx, EVP_CTRL_PBE_PRF_NID, 0, &prf_nid) <= 0)
		{
		ERR_clear_error();
		prf_nid = NID_hmacWithSHA1;
		}
	EVP_CIPHER_CTX_cleanup(&ctx);

	/* If its RC2 then we'd better setup the key length */

	if(alg_nid == NID_rc2_cbc)
		keylen = EVP_CIPHER_key_length(cipher);
	else
		keylen = -1;

	/* Setup keyfunc */

	X509_ALGOR_free(pbe2->keyfunc);

	pbe2->keyfunc = PKCS5_pbkdf2_set(iter, salt, saltlen, prf_nid, keylen);

	if (!pbe2->keyfunc)
		goto merr;

	/* Now set up top level AlgorithmIdentifier */

	if(!(ret = X509_ALGOR_new())) goto merr;
	if(!(ret->parameter = ASN1_TYPE_new())) goto merr;

	ret->algorithm = (ASN1_OBJECT*) OBJ_nid2obj(NID_pbes2);

	/* Encode PBE2PARAM into parameter */

	if(!ASN1_item_pack(pbe2, ASN1_ITEM_rptr(PBE2PARAM),
				 &ret->parameter->value.sequence)) goto merr;
	ret->parameter->type = V_ASN1_SEQUENCE;

	PBE2PARAM_free(pbe2);
	pbe2 = NULL;

	return ret;

	merr:
	OPENSSL_PUT_ERROR(PKCS8, ERR_R_MALLOC_FAILURE);

	err:
	PBE2PARAM_free(pbe2);
	/* Note 'scheme' is freed as part of pbe2 */
	X509_ALGOR_free(kalg);
	X509_ALGOR_free(ret);

	return NULL;

}

X509_ALGOR *PKCS5_pbe2_set(const EVP_CIPHER *cipher, int iter,
				 unsigned char *salt, int saltlen)
	{
	return PKCS5_pbe2_set_iv(cipher, iter, salt, saltlen, NULL, -1);
	}

X509_ALGOR *PKCS5_pbkdf2_set(int iter, unsigned char *salt, int saltlen,
				int prf_nid, int keylen)
	{
	X509_ALGOR *keyfunc = NULL;
	PBKDF2PARAM *kdf = NULL;
	ASN1_OCTET_STRING *osalt = NULL;

	if(!(kdf = PBKDF2PARAM_new()))
		goto merr;
	if(!(osalt = M_ASN1_OCTET_STRING_new()))
		goto merr;

	kdf->salt->value.octet_string = osalt;
	kdf->salt->type = V_ASN1_OCTET_STRING;

	if (!saltlen)
		saltlen = PKCS5_SALT_LEN;
	if (!(osalt->data = OPENSSL_malloc (saltlen)))
		goto merr;

	osalt->length = saltlen;

	if (salt)
		memcpy (osalt->data, salt, saltlen);
	else if (!RAND_bytes(osalt->data, saltlen))
		goto merr;

	if(iter <= 0)
		iter = PKCS5_DEFAULT_ITERATIONS;

	if(!ASN1_INTEGER_set(kdf->iter, iter))
		goto merr;

	/* If have a key len set it up */

	if(keylen > 0) 
		{
		if(!(kdf->keylength = M_ASN1_INTEGER_new()))
			goto merr;
		if(!ASN1_INTEGER_set (kdf->keylength, keylen))
			goto merr;
		}

	/* prf can stay NULL if we are using hmacWithSHA1 */
	if (prf_nid > 0 && prf_nid != NID_hmacWithSHA1)
		{
		kdf->prf = X509_ALGOR_new();
		if (!kdf->prf)
			goto merr;
		X509_ALGOR_set0(kdf->prf, OBJ_nid2obj(prf_nid),
					V_ASN1_NULL, NULL);
		}

	/* Finally setup the keyfunc structure */

	keyfunc = X509_ALGOR_new();
	if (!keyfunc)
		goto merr;

	keyfunc->algorithm = (ASN1_OBJECT*) OBJ_nid2obj(NID_id_pbkdf2);

	/* Encode PBKDF2PARAM into parameter of pbe2 */

	if(!(keyfunc->parameter = ASN1_TYPE_new()))
		goto merr;

	if(!ASN1_item_pack(kdf, ASN1_ITEM_rptr(PBKDF2PARAM),
			 &keyfunc->parameter->value.sequence))
		goto merr;
	keyfunc->parameter->type = V_ASN1_SEQUENCE;

	PBKDF2PARAM_free(kdf);
	return keyfunc;

	merr:
	OPENSSL_PUT_ERROR(PKCS8, ERR_R_MALLOC_FAILURE);
	PBKDF2PARAM_free(kdf);
	X509_ALGOR_free(keyfunc);
	return NULL;
	}

static int PKCS5_v2_PBKDF2_keyivgen(EVP_CIPHER_CTX *ctx,
                                    const uint8_t *pass_raw,
                                    size_t pass_raw_len, const ASN1_TYPE *param,
                                    const ASN1_TYPE *iv, int enc) {
  int rv = 0;
  PBKDF2PARAM *pbkdf2param = NULL;

  if (EVP_CIPHER_CTX_cipher(ctx) == NULL) {
    OPENSSL_PUT_ERROR(PKCS8, CIPHER_R_NO_CIPHER_SET);
    goto err;
  }

  /* Decode parameters. */
  if (param == NULL || param->type != V_ASN1_SEQUENCE) {
    OPENSSL_PUT_ERROR(PKCS8, PKCS8_R_DECODE_ERROR);
    goto err;
  }

  const uint8_t *pbuf = param->value.sequence->data;
  int plen = param->value.sequence->length;
  pbkdf2param = d2i_PBKDF2PARAM(NULL, &pbuf, plen);
  if (pbkdf2param == NULL || pbuf != param->value.sequence->data + plen) {
    OPENSSL_PUT_ERROR(PKCS8, PKCS8_R_DECODE_ERROR);
    goto err;
  }

  /* Now check the parameters. */
  uint8_t key[EVP_MAX_KEY_LENGTH];
  const size_t key_len = EVP_CIPHER_CTX_key_length(ctx);
  assert(key_len <= sizeof(key));

  if (pbkdf2param->keylength != NULL &&
      ASN1_INTEGER_get(pbkdf2param->keylength) != (int) key_len) {
    OPENSSL_PUT_ERROR(PKCS8, PKCS8_R_UNSUPPORTED_KEYLENGTH);
    goto err;
  }

  if (pbkdf2param->prf != NULL &&
      OBJ_obj2nid(pbkdf2param->prf->algorithm) != NID_hmacWithSHA1) {
    OPENSSL_PUT_ERROR(PKCS8, PKCS8_R_UNSUPPORTED_PRF);
    goto err;
  }

  if (pbkdf2param->salt->type != V_ASN1_OCTET_STRING) {
    OPENSSL_PUT_ERROR(PKCS8, PKCS8_R_UNSUPPORTED_SALT_TYPE);
    goto err;
  }

  if (pbkdf2param->iter->type != V_ASN1_INTEGER) {
    OPENSSL_PUT_ERROR(PKCS8, PKCS8_R_BAD_ITERATION_COUNT);
    goto err;
  }
  long iterations = ASN1_INTEGER_get(pbkdf2param->iter);
  if (iterations <= 0 || iterations > UINT_MAX) {
    OPENSSL_PUT_ERROR(PKCS8, PKCS8_R_BAD_ITERATION_COUNT);
    goto err;
  }

  if (iv->type != V_ASN1_OCTET_STRING || iv->value.octet_string == NULL) {
    OPENSSL_PUT_ERROR(PKCS8, PKCS8_R_ERROR_SETTING_CIPHER_PARAMS);
    goto err;
  }

  const size_t iv_len = EVP_CIPHER_CTX_iv_length(ctx);
  if (iv->value.octet_string->length != iv_len) {
    OPENSSL_PUT_ERROR(PKCS8, PKCS8_R_ERROR_SETTING_CIPHER_PARAMS);
    goto err;
  }

  if (!PKCS5_PBKDF2_HMAC_SHA1((const char *) pass_raw, pass_raw_len,
                              pbkdf2param->salt->value.octet_string->data,
                              pbkdf2param->salt->value.octet_string->length,
                              iterations, key_len, key)) {
    goto err;
  }

  rv = EVP_CipherInit_ex(ctx, NULL /* cipher */, NULL /* engine */, key,
                         iv->value.octet_string->data, enc);

 err:
  PBKDF2PARAM_free(pbkdf2param);
  return rv;
}

int PKCS5_v2_PBE_keyivgen(EVP_CIPHER_CTX *ctx, const uint8_t *pass_raw,
                          size_t pass_raw_len, ASN1_TYPE *param,
                          const EVP_CIPHER *unused, const EVP_MD *unused2,
                          int enc) {
  PBE2PARAM *pbe2param = NULL;
  int rv = 0;

  if (param == NULL ||
      param->type != V_ASN1_SEQUENCE ||
      param->value.sequence == NULL) {
    OPENSSL_PUT_ERROR(PKCS8, PKCS8_R_DECODE_ERROR);
    goto err;
  }

  const uint8_t *pbuf = param->value.sequence->data;
  int plen = param->value.sequence->length;
  pbe2param = d2i_PBE2PARAM(NULL, &pbuf, plen);
  if (pbe2param == NULL || pbuf != param->value.sequence->data + plen) {
    OPENSSL_PUT_ERROR(PKCS8, PKCS8_R_DECODE_ERROR);
    goto err;
  }

  /* Check that the key derivation function is PBKDF2. */
  if (OBJ_obj2nid(pbe2param->keyfunc->algorithm) != NID_id_pbkdf2) {
    OPENSSL_PUT_ERROR(PKCS8, PKCS8_R_UNSUPPORTED_KEY_DERIVATION_FUNCTION);
    goto err;
  }

  /* See if we recognise the encryption algorithm. */
  const EVP_CIPHER *cipher =
      EVP_get_cipherbynid(OBJ_obj2nid(pbe2param->encryption->algorithm));
  if (cipher == NULL) {
    OPENSSL_PUT_ERROR(PKCS8, PKCS8_R_UNSUPPORTED_CIPHER);
    goto err;
  }

  /* Fixup cipher based on AlgorithmIdentifier. */
  if (!EVP_CipherInit_ex(ctx, cipher, NULL /* engine */, NULL /* key */,
                         NULL /* iv */, enc)) {
    goto err;
  }

  rv = PKCS5_v2_PBKDF2_keyivgen(ctx, pass_raw, pass_raw_len,
                                pbe2param->keyfunc->parameter,
                                pbe2param->encryption->parameter, enc);

 err:
  PBE2PARAM_free(pbe2param);
  return rv;
}
