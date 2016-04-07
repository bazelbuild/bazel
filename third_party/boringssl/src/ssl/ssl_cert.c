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
 * Hudson (tjh@cryptsoft.com).
 *
 */
/* ====================================================================
 * Copyright 2002 Sun Microsystems, Inc. ALL RIGHTS RESERVED.
 * ECC cipher suite support in OpenSSL originally developed by
 * SUN MICROSYSTEMS, INC., and contributed to the OpenSSL project. */

#include <openssl/ssl.h>

#include <string.h>

#include <openssl/bn.h>
#include <openssl/buf.h>
#include <openssl/ec_key.h>
#include <openssl/dh.h>
#include <openssl/err.h>
#include <openssl/mem.h>
#include <openssl/x509.h>
#include <openssl/x509v3.h>

#include "../crypto/dh/internal.h"
#include "../crypto/internal.h"
#include "internal.h"


int SSL_get_ex_data_X509_STORE_CTX_idx(void) {
  /* The ex_data index to go from |X509_STORE_CTX| to |SSL| always uses the
   * reserved app_data slot. Before ex_data was introduced, app_data was used.
   * Avoid breaking any software which assumes |X509_STORE_CTX_get_app_data|
   * works. */
  return 0;
}

CERT *ssl_cert_new(void) {
  CERT *ret = (CERT *)OPENSSL_malloc(sizeof(CERT));
  if (ret == NULL) {
    OPENSSL_PUT_ERROR(SSL, ERR_R_MALLOC_FAILURE);
    return NULL;
  }
  memset(ret, 0, sizeof(CERT));

  return ret;
}

CERT *ssl_cert_dup(CERT *cert) {
  CERT *ret = (CERT *)OPENSSL_malloc(sizeof(CERT));
  if (ret == NULL) {
    OPENSSL_PUT_ERROR(SSL, ERR_R_MALLOC_FAILURE);
    return NULL;
  }
  memset(ret, 0, sizeof(CERT));

  ret->mask_k = cert->mask_k;
  ret->mask_a = cert->mask_a;

  if (cert->dh_tmp != NULL) {
    ret->dh_tmp = DHparams_dup(cert->dh_tmp);
    if (ret->dh_tmp == NULL) {
      OPENSSL_PUT_ERROR(SSL, ERR_R_DH_LIB);
      goto err;
    }
    if (cert->dh_tmp->priv_key) {
      BIGNUM *b = BN_dup(cert->dh_tmp->priv_key);
      if (!b) {
        OPENSSL_PUT_ERROR(SSL, ERR_R_BN_LIB);
        goto err;
      }
      ret->dh_tmp->priv_key = b;
    }
    if (cert->dh_tmp->pub_key) {
      BIGNUM *b = BN_dup(cert->dh_tmp->pub_key);
      if (!b) {
        OPENSSL_PUT_ERROR(SSL, ERR_R_BN_LIB);
        goto err;
      }
      ret->dh_tmp->pub_key = b;
    }
  }
  ret->dh_tmp_cb = cert->dh_tmp_cb;

  ret->ecdh_nid = cert->ecdh_nid;
  ret->ecdh_tmp_cb = cert->ecdh_tmp_cb;

  if (cert->x509 != NULL) {
    ret->x509 = X509_up_ref(cert->x509);
  }

  if (cert->privatekey != NULL) {
    ret->privatekey = EVP_PKEY_up_ref(cert->privatekey);
  }

  if (cert->chain) {
    ret->chain = X509_chain_up_ref(cert->chain);
    if (!ret->chain) {
      OPENSSL_PUT_ERROR(SSL, ERR_R_MALLOC_FAILURE);
      goto err;
    }
  }

  ret->cert_cb = cert->cert_cb;
  ret->cert_cb_arg = cert->cert_cb_arg;

  return ret;

err:
  ssl_cert_free(ret);
  return NULL;
}

/* Free up and clear all certificates and chains */
void ssl_cert_clear_certs(CERT *cert) {
  if (cert == NULL) {
    return;
  }

  X509_free(cert->x509);
  cert->x509 = NULL;
  EVP_PKEY_free(cert->privatekey);
  cert->privatekey = NULL;
  sk_X509_pop_free(cert->chain, X509_free);
  cert->chain = NULL;
  cert->key_method = NULL;
}

void ssl_cert_free(CERT *c) {
  if (c == NULL) {
    return;
  }

  DH_free(c->dh_tmp);

  ssl_cert_clear_certs(c);
  OPENSSL_free(c->peer_sigalgs);
  OPENSSL_free(c->digest_nids);

  OPENSSL_free(c);
}

int ssl_cert_set0_chain(CERT *cert, STACK_OF(X509) *chain) {
  sk_X509_pop_free(cert->chain, X509_free);
  cert->chain = chain;
  return 1;
}

int ssl_cert_set1_chain(CERT *cert, STACK_OF(X509) *chain) {
  STACK_OF(X509) *dchain;
  if (chain == NULL) {
    return ssl_cert_set0_chain(cert, NULL);
  }

  dchain = X509_chain_up_ref(chain);
  if (dchain == NULL) {
    return 0;
  }

  if (!ssl_cert_set0_chain(cert, dchain)) {
    sk_X509_pop_free(dchain, X509_free);
    return 0;
  }

  return 1;
}

int ssl_cert_add0_chain_cert(CERT *cert, X509 *x509) {
  if (cert->chain == NULL) {
    cert->chain = sk_X509_new_null();
  }
  if (cert->chain == NULL || !sk_X509_push(cert->chain, x509)) {
    return 0;
  }

  return 1;
}

int ssl_cert_add1_chain_cert(CERT *cert, X509 *x509) {
  if (!ssl_cert_add0_chain_cert(cert, x509)) {
    return 0;
  }

  X509_up_ref(x509);
  return 1;
}

void ssl_cert_set_cert_cb(CERT *c, int (*cb)(SSL *ssl, void *arg), void *arg) {
  c->cert_cb = cb;
  c->cert_cb_arg = arg;
}

int ssl_verify_cert_chain(SSL *ssl, STACK_OF(X509) *cert_chain) {
  if (cert_chain == NULL || sk_X509_num(cert_chain) == 0) {
    return 0;
  }

  X509 *leaf = sk_X509_value(cert_chain, 0);
  int ret = 0;
  X509_STORE_CTX ctx;
  if (!X509_STORE_CTX_init(&ctx, ssl->ctx->cert_store, leaf, cert_chain)) {
    OPENSSL_PUT_ERROR(SSL, ERR_R_X509_LIB);
    return 0;
  }
  if (!X509_STORE_CTX_set_ex_data(&ctx, SSL_get_ex_data_X509_STORE_CTX_idx(),
                                  ssl)) {
    goto err;
  }

  /* We need to inherit the verify parameters. These can be determined by the
   * context: if its a server it will verify SSL client certificates or vice
   * versa. */
  X509_STORE_CTX_set_default(&ctx, ssl->server ? "ssl_client" : "ssl_server");

  /* Anything non-default in "param" should overwrite anything in the ctx. */
  X509_VERIFY_PARAM_set1(X509_STORE_CTX_get0_param(&ctx), ssl->param);

  if (ssl->verify_callback) {
    X509_STORE_CTX_set_verify_cb(&ctx, ssl->verify_callback);
  }

  if (ssl->ctx->app_verify_callback != NULL) {
    ret = ssl->ctx->app_verify_callback(&ctx, ssl->ctx->app_verify_arg);
  } else {
    ret = X509_verify_cert(&ctx);
  }

  ssl->verify_result = ctx.error;

err:
  X509_STORE_CTX_cleanup(&ctx);
  return ret;
}

static void set_client_CA_list(STACK_OF(X509_NAME) **ca_list,
                               STACK_OF(X509_NAME) *name_list) {
  sk_X509_NAME_pop_free(*ca_list, X509_NAME_free);
  *ca_list = name_list;
}

STACK_OF(X509_NAME) *SSL_dup_CA_list(STACK_OF(X509_NAME) *list) {
  STACK_OF(X509_NAME) *ret = sk_X509_NAME_new_null();
  if (ret == NULL) {
    return NULL;
  }

  size_t i;
  for (i = 0; i < sk_X509_NAME_num(list); i++) {
      X509_NAME *name = X509_NAME_dup(sk_X509_NAME_value(list, i));
    if (name == NULL || !sk_X509_NAME_push(ret, name)) {
      X509_NAME_free(name);
      sk_X509_NAME_pop_free(ret, X509_NAME_free);
      return NULL;
    }
  }

  return ret;
}

void SSL_set_client_CA_list(SSL *ssl, STACK_OF(X509_NAME) *name_list) {
  set_client_CA_list(&ssl->client_CA, name_list);
}

void SSL_CTX_set_client_CA_list(SSL_CTX *ctx, STACK_OF(X509_NAME) *name_list) {
  set_client_CA_list(&ctx->client_CA, name_list);
}

STACK_OF(X509_NAME) *SSL_CTX_get_client_CA_list(const SSL_CTX *ctx) {
  return ctx->client_CA;
}

STACK_OF(X509_NAME) *SSL_get_client_CA_list(const SSL *ssl) {
  /* For historical reasons, this function is used both to query configuration
   * state on a server as well as handshake state on a client. However, whether
   * |ssl| is a client or server is not known until explicitly configured with
   * |SSL_set_connect_state|. If |handshake_func| is NULL, |ssl| is in an
   * indeterminate mode and |ssl->server| is unset. */
  if (ssl->handshake_func != NULL && !ssl->server) {
    return ssl->s3->tmp.ca_names;
  }

  if (ssl->client_CA != NULL) {
    return ssl->client_CA;
  }
  return ssl->ctx->client_CA;
}

static int add_client_CA(STACK_OF(X509_NAME) **sk, X509 *x509) {
  X509_NAME *name;

  if (x509 == NULL) {
    return 0;
  }
  if (*sk == NULL) {
    *sk = sk_X509_NAME_new_null();
    if (*sk == NULL) {
      return 0;
    }
  }

  name = X509_NAME_dup(X509_get_subject_name(x509));
  if (name == NULL) {
    return 0;
  }

  if (!sk_X509_NAME_push(*sk, name)) {
    X509_NAME_free(name);
    return 0;
  }

  return 1;
}

int SSL_add_client_CA(SSL *ssl, X509 *x509) {
  return add_client_CA(&ssl->client_CA, x509);
}

int SSL_CTX_add_client_CA(SSL_CTX *ctx, X509 *x509) {
  return add_client_CA(&ctx->client_CA, x509);
}

/* Add a certificate to a BUF_MEM structure */
static int ssl_add_cert_to_buf(BUF_MEM *buf, unsigned long *l, X509 *x) {
  int n;
  uint8_t *p;

  n = i2d_X509(x, NULL);
  if (!BUF_MEM_grow_clean(buf, (int)(n + (*l) + 3))) {
    OPENSSL_PUT_ERROR(SSL, ERR_R_BUF_LIB);
    return 0;
  }
  p = (uint8_t *)&(buf->data[*l]);
  l2n3(n, p);
  i2d_X509(x, &p);
  *l += n + 3;

  return 1;
}

/* Add certificate chain to internal SSL BUF_MEM structure. */
int ssl_add_cert_chain(SSL *ssl, unsigned long *l) {
  CERT *cert = ssl->cert;
  BUF_MEM *buf = ssl->init_buf;
  int no_chain = 0;
  size_t i;

  X509 *x = cert->x509;
  STACK_OF(X509) *chain = cert->chain;

  if (x == NULL) {
    OPENSSL_PUT_ERROR(SSL, SSL_R_NO_CERTIFICATE_SET);
    return 0;
  }

  if ((ssl->mode & SSL_MODE_NO_AUTO_CHAIN) || chain != NULL) {
    no_chain = 1;
  }

  if (no_chain) {
    if (!ssl_add_cert_to_buf(buf, l, x)) {
      return 0;
    }

    for (i = 0; i < sk_X509_num(chain); i++) {
      x = sk_X509_value(chain, i);
      if (!ssl_add_cert_to_buf(buf, l, x)) {
        return 0;
      }
    }
  } else {
    X509_STORE_CTX xs_ctx;

    if (!X509_STORE_CTX_init(&xs_ctx, ssl->ctx->cert_store, x, NULL)) {
      OPENSSL_PUT_ERROR(SSL, ERR_R_X509_LIB);
      return 0;
    }
    X509_verify_cert(&xs_ctx);
    /* Don't leave errors in the queue */
    ERR_clear_error();
    for (i = 0; i < sk_X509_num(xs_ctx.chain); i++) {
      x = sk_X509_value(xs_ctx.chain, i);

      if (!ssl_add_cert_to_buf(buf, l, x)) {
        X509_STORE_CTX_cleanup(&xs_ctx);
        return 0;
      }
    }
    X509_STORE_CTX_cleanup(&xs_ctx);
  }

  return 1;
}

int SSL_CTX_set0_chain(SSL_CTX *ctx, STACK_OF(X509) *chain) {
  return ssl_cert_set0_chain(ctx->cert, chain);
}

int SSL_CTX_set1_chain(SSL_CTX *ctx, STACK_OF(X509) *chain) {
  return ssl_cert_set1_chain(ctx->cert, chain);
}

int SSL_set0_chain(SSL *ssl, STACK_OF(X509) *chain) {
  return ssl_cert_set0_chain(ssl->cert, chain);
}

int SSL_set1_chain(SSL *ssl, STACK_OF(X509) *chain) {
  return ssl_cert_set1_chain(ssl->cert, chain);
}

int SSL_CTX_add0_chain_cert(SSL_CTX *ctx, X509 *x509) {
  return ssl_cert_add0_chain_cert(ctx->cert, x509);
}

int SSL_CTX_add1_chain_cert(SSL_CTX *ctx, X509 *x509) {
  return ssl_cert_add1_chain_cert(ctx->cert, x509);
}

int SSL_CTX_add_extra_chain_cert(SSL_CTX *ctx, X509 *x509) {
  return SSL_CTX_add0_chain_cert(ctx, x509);
}

int SSL_add0_chain_cert(SSL *ssl, X509 *x509) {
  return ssl_cert_add0_chain_cert(ssl->cert, x509);
}

int SSL_add1_chain_cert(SSL *ssl, X509 *x509) {
  return ssl_cert_add1_chain_cert(ssl->cert, x509);
}

int SSL_CTX_clear_chain_certs(SSL_CTX *ctx) {
  return SSL_CTX_set0_chain(ctx, NULL);
}

int SSL_CTX_clear_extra_chain_certs(SSL_CTX *ctx) {
  return SSL_CTX_clear_chain_certs(ctx);
}

int SSL_clear_chain_certs(SSL *ssl) {
  return SSL_set0_chain(ssl, NULL);
}

int SSL_CTX_get0_chain_certs(const SSL_CTX *ctx, STACK_OF(X509) **out_chain) {
  *out_chain = ctx->cert->chain;
  return 1;
}

int SSL_CTX_get_extra_chain_certs(const SSL_CTX *ctx,
                                  STACK_OF(X509) **out_chain) {
  return SSL_CTX_get0_chain_certs(ctx, out_chain);
}

int SSL_get0_chain_certs(const SSL *ssl, STACK_OF(X509) **out_chain) {
  *out_chain = ssl->cert->chain;
  return 1;
}
