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
 * SUN MICROSYSTEMS, INC., and contributed to the OpenSSL project.
 */
/* ====================================================================
 * Copyright 2005 Nokia. All rights reserved.
 *
 * The portions of the attached software ("Contribution") is developed by
 * Nokia Corporation and is licensed pursuant to the OpenSSL open source
 * license.
 *
 * The Contribution, originally written by Mika Kousa and Pasi Eronen of
 * Nokia Corporation, consists of the "PSK" (Pre-Shared Key) ciphersuites
 * support (see RFC 4279) to OpenSSL.
 *
 * No patent licenses or other rights except those expressly stated in
 * the OpenSSL open source license shall be deemed granted or received
 * expressly, by implication, estoppel, or otherwise.
 *
 * No assurances are provided by Nokia that the Contribution does not
 * infringe the patent or other intellectual property rights of any third
 * party or that the license provides you with all the necessary rights
 * to make use of the Contribution.
 *
 * THE SOFTWARE IS PROVIDED "AS IS" WITHOUT WARRANTY OF ANY KIND. IN
 * ADDITION TO THE DISCLAIMERS INCLUDED IN THE LICENSE, NOKIA
 * SPECIFICALLY DISCLAIMS ANY LIABILITY FOR CLAIMS BROUGHT BY YOU OR ANY
 * OTHER ENTITY BASED ON INFRINGEMENT OF INTELLECTUAL PROPERTY RIGHTS OR
 * OTHERWISE. */

#include <openssl/ssl.h>

#include <assert.h>
#include <stdio.h>
#include <string.h>

#include <openssl/buf.h>
#include <openssl/err.h>
#include <openssl/md5.h>
#include <openssl/mem.h>
#include <openssl/sha.h>
#include <openssl/stack.h>

#include "internal.h"


/* kCiphers is an array of all supported ciphers, sorted by id. */
const SSL_CIPHER kCiphers[] = {
    /* The RSA ciphers */
    /* Cipher 02 */
    {
     SSL3_TXT_RSA_NULL_SHA, SSL3_CK_RSA_NULL_SHA, SSL_kRSA, SSL_aRSA,
     SSL_eNULL, SSL_SHA1, SSL_SSLV3, SSL_FIPS, SSL_HANDSHAKE_MAC_DEFAULT, 0, 0,
    },

    /* Cipher 04 */
    {
     SSL3_TXT_RSA_RC4_128_MD5, SSL3_CK_RSA_RC4_128_MD5, SSL_kRSA, SSL_aRSA,
     SSL_RC4, SSL_MD5, SSL_SSLV3, SSL_MEDIUM,
     SSL_HANDSHAKE_MAC_DEFAULT, 128, 128,
    },

    /* Cipher 05 */
    {
     SSL3_TXT_RSA_RC4_128_SHA, SSL3_CK_RSA_RC4_128_SHA, SSL_kRSA, SSL_aRSA,
     SSL_RC4, SSL_SHA1, SSL_SSLV3, SSL_MEDIUM,
     SSL_HANDSHAKE_MAC_DEFAULT, 128, 128,
    },

    /* Cipher 0A */
    {
     SSL3_TXT_RSA_DES_192_CBC3_SHA, SSL3_CK_RSA_DES_192_CBC3_SHA, SSL_kRSA,
     SSL_aRSA, SSL_3DES, SSL_SHA1, SSL_SSLV3, SSL_HIGH | SSL_FIPS,
     SSL_HANDSHAKE_MAC_DEFAULT, 112, 168,
    },


    /* New AES ciphersuites */

    /* Cipher 2F */
    {
     TLS1_TXT_RSA_WITH_AES_128_SHA, TLS1_CK_RSA_WITH_AES_128_SHA, SSL_kRSA,
     SSL_aRSA, SSL_AES128, SSL_SHA1, SSL_TLSV1, SSL_HIGH | SSL_FIPS,
     SSL_HANDSHAKE_MAC_DEFAULT, 128, 128,
    },

    /* Cipher 33 */
    {
     TLS1_TXT_DHE_RSA_WITH_AES_128_SHA, TLS1_CK_DHE_RSA_WITH_AES_128_SHA,
     SSL_kDHE, SSL_aRSA, SSL_AES128, SSL_SHA1, SSL_TLSV1, SSL_HIGH | SSL_FIPS,
     SSL_HANDSHAKE_MAC_DEFAULT, 128, 128,
    },

    /* Cipher 35 */
    {
     TLS1_TXT_RSA_WITH_AES_256_SHA, TLS1_CK_RSA_WITH_AES_256_SHA, SSL_kRSA,
     SSL_aRSA, SSL_AES256, SSL_SHA1, SSL_TLSV1, SSL_HIGH | SSL_FIPS,
     SSL_HANDSHAKE_MAC_DEFAULT, 256, 256,
    },

    /* Cipher 39 */
    {
     TLS1_TXT_DHE_RSA_WITH_AES_256_SHA, TLS1_CK_DHE_RSA_WITH_AES_256_SHA,
     SSL_kDHE, SSL_aRSA, SSL_AES256, SSL_SHA1, SSL_TLSV1, SSL_HIGH | SSL_FIPS,
     SSL_HANDSHAKE_MAC_DEFAULT, 256, 256,
    },


    /* TLS v1.2 ciphersuites */

    /* Cipher 3C */
    {
     TLS1_TXT_RSA_WITH_AES_128_SHA256, TLS1_CK_RSA_WITH_AES_128_SHA256,
     SSL_kRSA, SSL_aRSA, SSL_AES128, SSL_SHA256, SSL_TLSV1_2,
     SSL_HIGH | SSL_FIPS, SSL_HANDSHAKE_MAC_SHA256, 128, 128,
    },

    /* Cipher 3D */
    {
     TLS1_TXT_RSA_WITH_AES_256_SHA256, TLS1_CK_RSA_WITH_AES_256_SHA256,
     SSL_kRSA, SSL_aRSA, SSL_AES256, SSL_SHA256, SSL_TLSV1_2,
     SSL_HIGH | SSL_FIPS, SSL_HANDSHAKE_MAC_SHA256, 256, 256,
    },

    /* Cipher 67 */
    {
     TLS1_TXT_DHE_RSA_WITH_AES_128_SHA256,
     TLS1_CK_DHE_RSA_WITH_AES_128_SHA256, SSL_kDHE, SSL_aRSA, SSL_AES128,
     SSL_SHA256, SSL_TLSV1_2, SSL_HIGH | SSL_FIPS,
     SSL_HANDSHAKE_MAC_SHA256, 128, 128,
    },

    /* Cipher 6B */
    {
     TLS1_TXT_DHE_RSA_WITH_AES_256_SHA256,
     TLS1_CK_DHE_RSA_WITH_AES_256_SHA256, SSL_kDHE, SSL_aRSA, SSL_AES256,
     SSL_SHA256, SSL_TLSV1_2, SSL_HIGH | SSL_FIPS,
     SSL_HANDSHAKE_MAC_SHA256, 256, 256,
    },

    /* PSK cipher suites. */

    /* Cipher 8A */
    {
     TLS1_TXT_PSK_WITH_RC4_128_SHA, TLS1_CK_PSK_WITH_RC4_128_SHA, SSL_kPSK,
     SSL_aPSK, SSL_RC4, SSL_SHA1, SSL_TLSV1, SSL_MEDIUM,
     SSL_HANDSHAKE_MAC_DEFAULT, 128, 128,
    },

    /* Cipher 8C */
    {
     TLS1_TXT_PSK_WITH_AES_128_CBC_SHA, TLS1_CK_PSK_WITH_AES_128_CBC_SHA,
     SSL_kPSK, SSL_aPSK, SSL_AES128, SSL_SHA1, SSL_TLSV1, SSL_HIGH | SSL_FIPS,
     SSL_HANDSHAKE_MAC_DEFAULT, 128, 128,
    },

    /* Cipher 8D */
    {
     TLS1_TXT_PSK_WITH_AES_256_CBC_SHA, TLS1_CK_PSK_WITH_AES_256_CBC_SHA,
     SSL_kPSK, SSL_aPSK, SSL_AES256, SSL_SHA1, SSL_TLSV1, SSL_HIGH | SSL_FIPS,
     SSL_HANDSHAKE_MAC_DEFAULT, 256, 256,
    },

    /* GCM ciphersuites from RFC5288 */

    /* Cipher 9C */
    {
     TLS1_TXT_RSA_WITH_AES_128_GCM_SHA256,
     TLS1_CK_RSA_WITH_AES_128_GCM_SHA256, SSL_kRSA, SSL_aRSA, SSL_AES128GCM,
     SSL_AEAD, SSL_TLSV1_2, SSL_HIGH | SSL_FIPS,
     SSL_HANDSHAKE_MAC_SHA256,
     128, 128,
    },

    /* Cipher 9D */
    {
     TLS1_TXT_RSA_WITH_AES_256_GCM_SHA384,
     TLS1_CK_RSA_WITH_AES_256_GCM_SHA384, SSL_kRSA, SSL_aRSA, SSL_AES256GCM,
     SSL_AEAD, SSL_TLSV1_2, SSL_HIGH | SSL_FIPS,
     SSL_HANDSHAKE_MAC_SHA384,
     256, 256,
    },

    /* Cipher 9E */
    {
     TLS1_TXT_DHE_RSA_WITH_AES_128_GCM_SHA256,
     TLS1_CK_DHE_RSA_WITH_AES_128_GCM_SHA256, SSL_kDHE, SSL_aRSA, SSL_AES128GCM,
     SSL_AEAD, SSL_TLSV1_2, SSL_HIGH | SSL_FIPS,
     SSL_HANDSHAKE_MAC_SHA256,
     128, 128,
    },

    /* Cipher 9F */
    {
     TLS1_TXT_DHE_RSA_WITH_AES_256_GCM_SHA384,
     TLS1_CK_DHE_RSA_WITH_AES_256_GCM_SHA384, SSL_kDHE, SSL_aRSA, SSL_AES256GCM,
     SSL_AEAD, SSL_TLSV1_2, SSL_HIGH | SSL_FIPS,
     SSL_HANDSHAKE_MAC_SHA384,
     256, 256,
    },

    /* Cipher C007 */
    {
     TLS1_TXT_ECDHE_ECDSA_WITH_RC4_128_SHA,
     TLS1_CK_ECDHE_ECDSA_WITH_RC4_128_SHA, SSL_kECDHE, SSL_aECDSA, SSL_RC4,
     SSL_SHA1, SSL_TLSV1, SSL_MEDIUM, SSL_HANDSHAKE_MAC_DEFAULT, 128,
     128,
    },

    /* Cipher C009 */
    {
     TLS1_TXT_ECDHE_ECDSA_WITH_AES_128_CBC_SHA,
     TLS1_CK_ECDHE_ECDSA_WITH_AES_128_CBC_SHA, SSL_kECDHE, SSL_aECDSA,
     SSL_AES128, SSL_SHA1, SSL_TLSV1, SSL_HIGH | SSL_FIPS,
     SSL_HANDSHAKE_MAC_DEFAULT, 128, 128,
    },

    /* Cipher C00A */
    {
     TLS1_TXT_ECDHE_ECDSA_WITH_AES_256_CBC_SHA,
     TLS1_CK_ECDHE_ECDSA_WITH_AES_256_CBC_SHA, SSL_kECDHE, SSL_aECDSA,
     SSL_AES256, SSL_SHA1, SSL_TLSV1, SSL_HIGH | SSL_FIPS,
     SSL_HANDSHAKE_MAC_DEFAULT, 256, 256,
    },

    /* Cipher C011 */
    {
     TLS1_TXT_ECDHE_RSA_WITH_RC4_128_SHA, TLS1_CK_ECDHE_RSA_WITH_RC4_128_SHA,
     SSL_kECDHE, SSL_aRSA, SSL_RC4, SSL_SHA1, SSL_TLSV1, SSL_MEDIUM,
     SSL_HANDSHAKE_MAC_DEFAULT, 128, 128,
    },

    /* Cipher C013 */
    {
     TLS1_TXT_ECDHE_RSA_WITH_AES_128_CBC_SHA,
     TLS1_CK_ECDHE_RSA_WITH_AES_128_CBC_SHA, SSL_kECDHE, SSL_aRSA, SSL_AES128,
     SSL_SHA1, SSL_TLSV1, SSL_HIGH | SSL_FIPS,
     SSL_HANDSHAKE_MAC_DEFAULT, 128, 128,
    },

    /* Cipher C014 */
    {
     TLS1_TXT_ECDHE_RSA_WITH_AES_256_CBC_SHA,
     TLS1_CK_ECDHE_RSA_WITH_AES_256_CBC_SHA, SSL_kECDHE, SSL_aRSA, SSL_AES256,
     SSL_SHA1, SSL_TLSV1, SSL_HIGH | SSL_FIPS,
     SSL_HANDSHAKE_MAC_DEFAULT, 256, 256,
    },


    /* HMAC based TLS v1.2 ciphersuites from RFC5289 */

    /* Cipher C023 */
    {
     TLS1_TXT_ECDHE_ECDSA_WITH_AES_128_SHA256,
     TLS1_CK_ECDHE_ECDSA_WITH_AES_128_SHA256, SSL_kECDHE, SSL_aECDSA,
     SSL_AES128, SSL_SHA256, SSL_TLSV1_2, SSL_HIGH | SSL_FIPS,
     SSL_HANDSHAKE_MAC_SHA256, 128, 128,
    },

    /* Cipher C024 */
    {
     TLS1_TXT_ECDHE_ECDSA_WITH_AES_256_SHA384,
     TLS1_CK_ECDHE_ECDSA_WITH_AES_256_SHA384, SSL_kECDHE, SSL_aECDSA,
     SSL_AES256, SSL_SHA384, SSL_TLSV1_2, SSL_HIGH | SSL_FIPS,
     SSL_HANDSHAKE_MAC_SHA384, 256, 256,
    },

    /* Cipher C027 */
    {
     TLS1_TXT_ECDHE_RSA_WITH_AES_128_SHA256,
     TLS1_CK_ECDHE_RSA_WITH_AES_128_SHA256, SSL_kECDHE, SSL_aRSA, SSL_AES128,
     SSL_SHA256, SSL_TLSV1_2, SSL_HIGH | SSL_FIPS,
     SSL_HANDSHAKE_MAC_SHA256, 128, 128,
    },

    /* Cipher C028 */
    {
     TLS1_TXT_ECDHE_RSA_WITH_AES_256_SHA384,
     TLS1_CK_ECDHE_RSA_WITH_AES_256_SHA384, SSL_kECDHE, SSL_aRSA, SSL_AES256,
     SSL_SHA384, SSL_TLSV1_2, SSL_HIGH | SSL_FIPS,
     SSL_HANDSHAKE_MAC_SHA384, 256, 256,
    },


    /* GCM based TLS v1.2 ciphersuites from RFC5289 */

    /* Cipher C02B */
    {
     TLS1_TXT_ECDHE_ECDSA_WITH_AES_128_GCM_SHA256,
     TLS1_CK_ECDHE_ECDSA_WITH_AES_128_GCM_SHA256, SSL_kECDHE, SSL_aECDSA,
     SSL_AES128GCM, SSL_AEAD, SSL_TLSV1_2, SSL_HIGH | SSL_FIPS,
     SSL_HANDSHAKE_MAC_SHA256,
     128, 128,
    },

    /* Cipher C02C */
    {
     TLS1_TXT_ECDHE_ECDSA_WITH_AES_256_GCM_SHA384,
     TLS1_CK_ECDHE_ECDSA_WITH_AES_256_GCM_SHA384, SSL_kECDHE, SSL_aECDSA,
     SSL_AES256GCM, SSL_AEAD, SSL_TLSV1_2, SSL_HIGH | SSL_FIPS,
     SSL_HANDSHAKE_MAC_SHA384,
     256, 256,
    },

    /* Cipher C02F */
    {
     TLS1_TXT_ECDHE_RSA_WITH_AES_128_GCM_SHA256,
     TLS1_CK_ECDHE_RSA_WITH_AES_128_GCM_SHA256, SSL_kECDHE, SSL_aRSA,
     SSL_AES128GCM, SSL_AEAD, SSL_TLSV1_2, SSL_HIGH | SSL_FIPS,
     SSL_HANDSHAKE_MAC_SHA256,
     128, 128,
    },

    /* Cipher C030 */
    {
     TLS1_TXT_ECDHE_RSA_WITH_AES_256_GCM_SHA384,
     TLS1_CK_ECDHE_RSA_WITH_AES_256_GCM_SHA384, SSL_kECDHE, SSL_aRSA,
     SSL_AES256GCM, SSL_AEAD, SSL_TLSV1_2, SSL_HIGH | SSL_FIPS,
     SSL_HANDSHAKE_MAC_SHA384,
     256, 256,
    },

    /* ECDHE-PSK cipher suites. */

    /* Cipher C035 */
    {
     TLS1_TXT_ECDHE_PSK_WITH_AES_128_CBC_SHA,
     TLS1_CK_ECDHE_PSK_WITH_AES_128_CBC_SHA,
     SSL_kECDHE, SSL_aPSK, SSL_AES128, SSL_SHA1, SSL_TLSV1, SSL_HIGH | SSL_FIPS,
     SSL_HANDSHAKE_MAC_DEFAULT, 128, 128,
    },

    /* Cipher C036 */
    {
     TLS1_TXT_ECDHE_PSK_WITH_AES_256_CBC_SHA,
     TLS1_CK_ECDHE_PSK_WITH_AES_256_CBC_SHA,
     SSL_kECDHE, SSL_aPSK, SSL_AES256, SSL_SHA1, SSL_TLSV1, SSL_HIGH | SSL_FIPS,
     SSL_HANDSHAKE_MAC_DEFAULT, 256, 256,
    },

#if !defined(BORINGSSL_ANDROID_SYSTEM)
    /* ChaCha20-Poly1305 cipher suites. */

    {
     TLS1_TXT_ECDHE_RSA_WITH_CHACHA20_POLY1305,
     TLS1_CK_ECDHE_RSA_CHACHA20_POLY1305, SSL_kECDHE, SSL_aRSA,
     SSL_CHACHA20POLY1305, SSL_AEAD, SSL_TLSV1_2, SSL_HIGH,
     SSL_HANDSHAKE_MAC_SHA256,
     256, 256,
    },

    {
     TLS1_TXT_ECDHE_ECDSA_WITH_CHACHA20_POLY1305,
     TLS1_CK_ECDHE_ECDSA_CHACHA20_POLY1305, SSL_kECDHE, SSL_aECDSA,
     SSL_CHACHA20POLY1305, SSL_AEAD, SSL_TLSV1_2, SSL_HIGH,
     SSL_HANDSHAKE_MAC_SHA256,
     256, 256,
    },
#endif
};

static const size_t kCiphersLen = sizeof(kCiphers) / sizeof(kCiphers[0]);

#define CIPHER_ADD 1
#define CIPHER_KILL 2
#define CIPHER_DEL 3
#define CIPHER_ORD 4
#define CIPHER_SPECIAL 5

typedef struct cipher_order_st {
  const SSL_CIPHER *cipher;
  int active;
  int in_group;
  struct cipher_order_st *next, *prev;
} CIPHER_ORDER;

typedef struct cipher_alias_st {
  /* name is the name of the cipher alias. */
  const char *name;

  /* The following fields are bitmasks for the corresponding fields on
   * |SSL_CIPHER|. A cipher matches a cipher alias iff, for each bitmask, the
   * bit corresponding to the cipher's value is set to 1. If any bitmask is
   * all zeroes, the alias matches nothing. Use |~0u| for the default value. */
  uint32_t algorithm_mkey;
  uint32_t algorithm_auth;
  uint32_t algorithm_enc;
  uint32_t algorithm_mac;
  uint32_t algorithm_ssl;
  uint32_t algo_strength;
} CIPHER_ALIAS;

static const CIPHER_ALIAS kCipherAliases[] = {
    /* "ALL" doesn't include eNULL (must be specifically enabled) */
    {SSL_TXT_ALL, ~0u, ~0u, ~SSL_eNULL, ~0u, ~0u, ~0u},

    /* The "COMPLEMENTOFDEFAULT" rule is omitted. It matches nothing. */

    /* key exchange aliases
     * (some of those using only a single bit here combine
     * multiple key exchange algs according to the RFCs,
     * e.g. kEDH combines DHE_DSS and DHE_RSA) */
    {SSL_TXT_kRSA, SSL_kRSA, ~0u, ~0u, ~0u, ~0u, ~0u},

    {SSL_TXT_kDHE, SSL_kDHE, ~0u, ~0u, ~0u, ~0u, ~0u},
    {SSL_TXT_kEDH, SSL_kDHE, ~0u, ~0u, ~0u, ~0u, ~0u},
    {SSL_TXT_DH, SSL_kDHE, ~0u, ~0u, ~0u, ~0u, ~0u},

    {SSL_TXT_kECDHE, SSL_kECDHE, ~0u, ~0u, ~0u, ~0u, ~0u},
    {SSL_TXT_kEECDH, SSL_kECDHE, ~0u, ~0u, ~0u, ~0u, ~0u},
    {SSL_TXT_ECDH, SSL_kECDHE, ~0u, ~0u, ~0u, ~0u, ~0u},

    {SSL_TXT_kPSK, SSL_kPSK, ~0u, ~0u, ~0u, ~0u, ~0u},

    /* server authentication aliases */
    {SSL_TXT_aRSA, ~0u, SSL_aRSA, ~SSL_eNULL, ~0u, ~0u, ~0u},
    {SSL_TXT_aECDSA, ~0u, SSL_aECDSA, ~0u, ~0u, ~0u, ~0u},
    {SSL_TXT_ECDSA, ~0u, SSL_aECDSA, ~0u, ~0u, ~0u, ~0u},
    {SSL_TXT_aPSK, ~0u, SSL_aPSK, ~0u, ~0u, ~0u, ~0u},

    /* aliases combining key exchange and server authentication */
    {SSL_TXT_DHE, SSL_kDHE, ~0u, ~0u, ~0u, ~0u, ~0u},
    {SSL_TXT_EDH, SSL_kDHE, ~0u, ~0u, ~0u, ~0u, ~0u},
    {SSL_TXT_ECDHE, SSL_kECDHE, ~0u, ~0u, ~0u, ~0u, ~0u},
    {SSL_TXT_EECDH, SSL_kECDHE, ~0u, ~0u, ~0u, ~0u, ~0u},
    {SSL_TXT_RSA, SSL_kRSA, SSL_aRSA, ~SSL_eNULL, ~0u, ~0u, ~0u},
    {SSL_TXT_PSK, SSL_kPSK, SSL_aPSK, ~0u, ~0u, ~0u, ~0u},

    /* symmetric encryption aliases */
    {SSL_TXT_3DES, ~0u, ~0u, SSL_3DES, ~0u, ~0u, ~0u},
    {SSL_TXT_RC4, ~0u, ~0u, SSL_RC4, ~0u, ~0u, ~0u},
    {SSL_TXT_AES128, ~0u, ~0u, SSL_AES128 | SSL_AES128GCM, ~0u, ~0u, ~0u},
    {SSL_TXT_AES256, ~0u, ~0u, SSL_AES256 | SSL_AES256GCM, ~0u, ~0u, ~0u},
    {SSL_TXT_AES, ~0u, ~0u, SSL_AES, ~0u, ~0u, ~0u},
    {SSL_TXT_AES_GCM, ~0u, ~0u, SSL_AES128GCM | SSL_AES256GCM, ~0u, ~0u, ~0u},
    {SSL_TXT_CHACHA20, ~0u, ~0u, SSL_CHACHA20POLY1305, ~0u, ~0u, ~0u},

    /* MAC aliases */
    {SSL_TXT_MD5, ~0u, ~0u, ~0u, SSL_MD5, ~0u, ~0u},
    {SSL_TXT_SHA1, ~0u, ~0u, ~SSL_eNULL, SSL_SHA1, ~0u, ~0u},
    {SSL_TXT_SHA, ~0u, ~0u, ~SSL_eNULL, SSL_SHA1, ~0u, ~0u},
    {SSL_TXT_SHA256, ~0u, ~0u, ~0u, SSL_SHA256, ~0u, ~0u},
    {SSL_TXT_SHA384, ~0u, ~0u, ~0u, SSL_SHA384, ~0u, ~0u},

    /* protocol version aliases */
    {SSL_TXT_SSLV3, ~0u, ~0u, ~SSL_eNULL, ~0u, SSL_SSLV3, ~0u},
    {SSL_TXT_TLSV1, ~0u, ~0u, ~SSL_eNULL, ~0u, SSL_TLSV1, ~0u},
    {SSL_TXT_TLSV1_2, ~0u, ~0u, ~SSL_eNULL, ~0u, SSL_TLSV1_2, ~0u},

    /* strength classes */
    {SSL_TXT_MEDIUM, ~0u, ~0u, ~0u, ~0u, ~0u, SSL_MEDIUM},
    {SSL_TXT_HIGH, ~0u, ~0u, ~0u, ~0u, ~0u, SSL_HIGH},
    /* FIPS 140-2 approved ciphersuite */
    {SSL_TXT_FIPS, ~0u, ~0u, ~SSL_eNULL, ~0u, ~0u, SSL_FIPS},
};

static const size_t kCipherAliasesLen =
    sizeof(kCipherAliases) / sizeof(kCipherAliases[0]);

static int ssl_cipher_id_cmp(const void *in_a, const void *in_b) {
  const SSL_CIPHER *a = in_a;
  const SSL_CIPHER *b = in_b;

  if (a->id > b->id) {
    return 1;
  } else if (a->id < b->id) {
    return -1;
  } else {
    return 0;
  }
}

static int ssl_cipher_ptr_id_cmp(const SSL_CIPHER **a, const SSL_CIPHER **b) {
  return ssl_cipher_id_cmp(*a, *b);
}

const SSL_CIPHER *SSL_get_cipher_by_value(uint16_t value) {
  SSL_CIPHER c;

  c.id = 0x03000000L | value;
  return bsearch(&c, kCiphers, kCiphersLen, sizeof(SSL_CIPHER),
                 ssl_cipher_id_cmp);
}

int ssl_cipher_get_evp_aead(const EVP_AEAD **out_aead,
                            size_t *out_mac_secret_len,
                            size_t *out_fixed_iv_len,
                            const SSL_CIPHER *cipher, uint16_t version) {
  *out_aead = NULL;
  *out_mac_secret_len = 0;
  *out_fixed_iv_len = 0;

  switch (cipher->algorithm_enc) {
    case SSL_AES128GCM:
      *out_aead = EVP_aead_aes_128_gcm();
      *out_fixed_iv_len = 4;
      return 1;

    case SSL_AES256GCM:
      *out_aead = EVP_aead_aes_256_gcm();
      *out_fixed_iv_len = 4;
      return 1;

#if !defined(BORINGSSL_ANDROID_SYSTEM)
    case SSL_CHACHA20POLY1305:
      *out_aead = EVP_aead_chacha20_poly1305();
      *out_fixed_iv_len = 0;
      return 1;
#endif

    case SSL_RC4:
      switch (cipher->algorithm_mac) {
        case SSL_MD5:
          if (version == SSL3_VERSION) {
            *out_aead = EVP_aead_rc4_md5_ssl3();
          } else {
            *out_aead = EVP_aead_rc4_md5_tls();
          }
          *out_mac_secret_len = MD5_DIGEST_LENGTH;
          return 1;
        case SSL_SHA1:
          if (version == SSL3_VERSION) {
            *out_aead = EVP_aead_rc4_sha1_ssl3();
          } else {
            *out_aead = EVP_aead_rc4_sha1_tls();
          }
          *out_mac_secret_len = SHA_DIGEST_LENGTH;
          return 1;
        default:
          return 0;
      }

    case SSL_AES128:
      switch (cipher->algorithm_mac) {
        case SSL_SHA1:
          if (version == SSL3_VERSION) {
            *out_aead = EVP_aead_aes_128_cbc_sha1_ssl3();
            *out_fixed_iv_len = 16;
          } else if (version == TLS1_VERSION) {
            *out_aead = EVP_aead_aes_128_cbc_sha1_tls_implicit_iv();
            *out_fixed_iv_len = 16;
          } else {
            *out_aead = EVP_aead_aes_128_cbc_sha1_tls();
          }
          *out_mac_secret_len = SHA_DIGEST_LENGTH;
          return 1;
        case SSL_SHA256:
          *out_aead = EVP_aead_aes_128_cbc_sha256_tls();
          *out_mac_secret_len = SHA256_DIGEST_LENGTH;
          return 1;
        default:
          return 0;
      }

    case SSL_AES256:
      switch (cipher->algorithm_mac) {
        case SSL_SHA1:
          if (version == SSL3_VERSION) {
            *out_aead = EVP_aead_aes_256_cbc_sha1_ssl3();
            *out_fixed_iv_len = 16;
          } else if (version == TLS1_VERSION) {
            *out_aead = EVP_aead_aes_256_cbc_sha1_tls_implicit_iv();
            *out_fixed_iv_len = 16;
          } else {
            *out_aead = EVP_aead_aes_256_cbc_sha1_tls();
          }
          *out_mac_secret_len = SHA_DIGEST_LENGTH;
          return 1;
        case SSL_SHA256:
          *out_aead = EVP_aead_aes_256_cbc_sha256_tls();
          *out_mac_secret_len = SHA256_DIGEST_LENGTH;
          return 1;
        case SSL_SHA384:
          *out_aead = EVP_aead_aes_256_cbc_sha384_tls();
          *out_mac_secret_len = SHA384_DIGEST_LENGTH;
          return 1;
        default:
          return 0;
      }

    case SSL_3DES:
      switch (cipher->algorithm_mac) {
        case SSL_SHA1:
          if (version == SSL3_VERSION) {
            *out_aead = EVP_aead_des_ede3_cbc_sha1_ssl3();
            *out_fixed_iv_len = 8;
          } else if (version == TLS1_VERSION) {
            *out_aead = EVP_aead_des_ede3_cbc_sha1_tls_implicit_iv();
            *out_fixed_iv_len = 8;
          } else {
            *out_aead = EVP_aead_des_ede3_cbc_sha1_tls();
          }
          *out_mac_secret_len = SHA_DIGEST_LENGTH;
          return 1;
        default:
          return 0;
      }

    case SSL_eNULL:
      switch (cipher->algorithm_mac) {
        case SSL_SHA1:
          if (version == SSL3_VERSION) {
            *out_aead = EVP_aead_null_sha1_ssl3();
          } else {
            *out_aead = EVP_aead_null_sha1_tls();
          }
          *out_mac_secret_len = SHA_DIGEST_LENGTH;
          return 1;
        default:
          return 0;
      }

    default:
      return 0;
  }
}

const EVP_MD *ssl_get_handshake_digest(uint32_t algorithm_prf) {
  switch (algorithm_prf) {
    case SSL_HANDSHAKE_MAC_DEFAULT:
      return EVP_sha1();
    case SSL_HANDSHAKE_MAC_SHA256:
      return EVP_sha256();
    case SSL_HANDSHAKE_MAC_SHA384:
      return EVP_sha384();
    default:
      return NULL;
  }
}

#define ITEM_SEP(a) \
  (((a) == ':') || ((a) == ' ') || ((a) == ';') || ((a) == ','))

/* rule_equals returns one iff the NUL-terminated string |rule| is equal to the
 * |buf_len| bytes at |buf|. */
static int rule_equals(const char *rule, const char *buf, size_t buf_len) {
  /* |strncmp| alone only checks that |buf| is a prefix of |rule|. */
  return strncmp(rule, buf, buf_len) == 0 && rule[buf_len] == '\0';
}

static void ll_append_tail(CIPHER_ORDER **head, CIPHER_ORDER *curr,
                           CIPHER_ORDER **tail) {
  if (curr == *tail) {
    return;
  }
  if (curr == *head) {
    *head = curr->next;
  }
  if (curr->prev != NULL) {
    curr->prev->next = curr->next;
  }
  if (curr->next != NULL) {
    curr->next->prev = curr->prev;
  }
  (*tail)->next = curr;
  curr->prev = *tail;
  curr->next = NULL;
  *tail = curr;
}

static void ll_append_head(CIPHER_ORDER **head, CIPHER_ORDER *curr,
                           CIPHER_ORDER **tail) {
  if (curr == *head) {
    return;
  }
  if (curr == *tail) {
    *tail = curr->prev;
  }
  if (curr->next != NULL) {
    curr->next->prev = curr->prev;
  }
  if (curr->prev != NULL) {
    curr->prev->next = curr->next;
  }
  (*head)->prev = curr;
  curr->next = *head;
  curr->prev = NULL;
  *head = curr;
}

static void ssl_cipher_collect_ciphers(const SSL_PROTOCOL_METHOD *ssl_method,
                                       CIPHER_ORDER *co_list,
                                       CIPHER_ORDER **head_p,
                                       CIPHER_ORDER **tail_p) {
  /* The set of ciphers is static, but some subset may be unsupported by
   * |ssl_method|, so the list may be smaller. */
  size_t co_list_num = 0;
  size_t i;
  for (i = 0; i < kCiphersLen; i++) {
    const SSL_CIPHER *cipher = &kCiphers[i];
    if (ssl_method->supports_cipher(cipher)) {
      co_list[co_list_num].cipher = cipher;
      co_list[co_list_num].next = NULL;
      co_list[co_list_num].prev = NULL;
      co_list[co_list_num].active = 0;
      co_list[co_list_num].in_group = 0;
      co_list_num++;
    }
  }

  /* Prepare linked list from list entries. */
  if (co_list_num > 0) {
    co_list[0].prev = NULL;

    if (co_list_num > 1) {
      co_list[0].next = &co_list[1];

      for (i = 1; i < co_list_num - 1; i++) {
        co_list[i].prev = &co_list[i - 1];
        co_list[i].next = &co_list[i + 1];
      }

      co_list[co_list_num - 1].prev = &co_list[co_list_num - 2];
    }

    co_list[co_list_num - 1].next = NULL;

    *head_p = &co_list[0];
    *tail_p = &co_list[co_list_num - 1];
  }
}

/* ssl_cipher_apply_rule applies the rule type |rule| to ciphers matching its
 * parameters in the linked list from |*head_p| to |*tail_p|. It writes the new
 * head and tail of the list to |*head_p| and |*tail_p|, respectively.
 *
 * - If |cipher_id| is non-zero, only that cipher is selected.
 * - Otherwise, if |strength_bits| is non-negative, it selects ciphers
 *   of that strength.
 * - Otherwise, it selects ciphers that match each bitmasks in |alg_*| and
 *   |algo_strength|. */
static void ssl_cipher_apply_rule(
    uint32_t cipher_id, uint32_t alg_mkey, uint32_t alg_auth,
    uint32_t alg_enc, uint32_t alg_mac, uint32_t alg_ssl,
    uint32_t algo_strength, int rule, int strength_bits, int in_group,
    CIPHER_ORDER **head_p, CIPHER_ORDER **tail_p) {
  CIPHER_ORDER *head, *tail, *curr, *next, *last;
  const SSL_CIPHER *cp;
  int reverse = 0;

  if (cipher_id == 0 && strength_bits == -1 &&
      (alg_mkey == 0 || alg_auth == 0 || alg_enc == 0 || alg_mac == 0 ||
       alg_ssl == 0 || algo_strength == 0)) {
    /* The rule matches nothing, so bail early. */
    return;
  }

  if (rule == CIPHER_DEL) {
    /* needed to maintain sorting between currently deleted ciphers */
    reverse = 1;
  }

  head = *head_p;
  tail = *tail_p;

  if (reverse) {
    next = tail;
    last = head;
  } else {
    next = head;
    last = tail;
  }

  curr = NULL;
  for (;;) {
    if (curr == last) {
      break;
    }

    curr = next;
    if (curr == NULL) {
      break;
    }

    next = reverse ? curr->prev : curr->next;
    cp = curr->cipher;

    /* Selection criteria is either a specific cipher, the value of
     * |strength_bits|, or the algorithms used. */
    if (cipher_id != 0) {
      if (cipher_id != cp->id) {
        continue;
      }
    } else if (strength_bits >= 0) {
      if (strength_bits != cp->strength_bits) {
        continue;
      }
    } else if (!(alg_mkey & cp->algorithm_mkey) ||
               !(alg_auth & cp->algorithm_auth) ||
               !(alg_enc & cp->algorithm_enc) ||
               !(alg_mac & cp->algorithm_mac) ||
               !(alg_ssl & cp->algorithm_ssl) ||
               !(algo_strength & cp->algo_strength)) {
      continue;
    }

    /* add the cipher if it has not been added yet. */
    if (rule == CIPHER_ADD) {
      /* reverse == 0 */
      if (!curr->active) {
        ll_append_tail(&head, curr, &tail);
        curr->active = 1;
        curr->in_group = in_group;
      }
    }

    /* Move the added cipher to this location */
    else if (rule == CIPHER_ORD) {
      /* reverse == 0 */
      if (curr->active) {
        ll_append_tail(&head, curr, &tail);
        curr->in_group = 0;
      }
    } else if (rule == CIPHER_DEL) {
      /* reverse == 1 */
      if (curr->active) {
        /* most recently deleted ciphersuites get best positions
         * for any future CIPHER_ADD (note that the CIPHER_DEL loop
         * works in reverse to maintain the order) */
        ll_append_head(&head, curr, &tail);
        curr->active = 0;
        curr->in_group = 0;
      }
    } else if (rule == CIPHER_KILL) {
      /* reverse == 0 */
      if (head == curr) {
        head = curr->next;
      } else {
        curr->prev->next = curr->next;
      }

      if (tail == curr) {
        tail = curr->prev;
      }
      curr->active = 0;
      if (curr->next != NULL) {
        curr->next->prev = curr->prev;
      }
      if (curr->prev != NULL) {
        curr->prev->next = curr->next;
      }
      curr->next = NULL;
      curr->prev = NULL;
    }
  }

  *head_p = head;
  *tail_p = tail;
}

static int ssl_cipher_strength_sort(CIPHER_ORDER **head_p,
                                    CIPHER_ORDER **tail_p) {
  int max_strength_bits, i, *number_uses;
  CIPHER_ORDER *curr;

  /* This routine sorts the ciphers with descending strength. The sorting must
   * keep the pre-sorted sequence, so we apply the normal sorting routine as
   * '+' movement to the end of the list. */
  max_strength_bits = 0;
  curr = *head_p;
  while (curr != NULL) {
    if (curr->active && curr->cipher->strength_bits > max_strength_bits) {
      max_strength_bits = curr->cipher->strength_bits;
    }
    curr = curr->next;
  }

  number_uses = OPENSSL_malloc((max_strength_bits + 1) * sizeof(int));
  if (!number_uses) {
    OPENSSL_PUT_ERROR(SSL, ERR_R_MALLOC_FAILURE);
    return 0;
  }
  memset(number_uses, 0, (max_strength_bits + 1) * sizeof(int));

  /* Now find the strength_bits values actually used. */
  curr = *head_p;
  while (curr != NULL) {
    if (curr->active) {
      number_uses[curr->cipher->strength_bits]++;
    }
    curr = curr->next;
  }

  /* Go through the list of used strength_bits values in descending order. */
  for (i = max_strength_bits; i >= 0; i--) {
    if (number_uses[i] > 0) {
      ssl_cipher_apply_rule(0, 0, 0, 0, 0, 0, 0, CIPHER_ORD, i, 0, head_p,
                            tail_p);
    }
  }

  OPENSSL_free(number_uses);
  return 1;
}

static int ssl_cipher_process_rulestr(const SSL_PROTOCOL_METHOD *ssl_method,
                                      const char *rule_str,
                                      CIPHER_ORDER **head_p,
                                      CIPHER_ORDER **tail_p) {
  uint32_t alg_mkey, alg_auth, alg_enc, alg_mac, alg_ssl, algo_strength;
  const char *l, *buf;
  int multi, rule, retval, ok, in_group = 0, has_group = 0;
  size_t j, buf_len;
  uint32_t cipher_id;
  char ch;

  retval = 1;
  l = rule_str;
  for (;;) {
    ch = *l;

    if (ch == '\0') {
      break; /* done */
    }

    if (in_group) {
      if (ch == ']') {
        if (*tail_p) {
          (*tail_p)->in_group = 0;
        }
        in_group = 0;
        l++;
        continue;
      }

      if (ch == '|') {
        rule = CIPHER_ADD;
        l++;
        continue;
      } else if (!(ch >= 'a' && ch <= 'z') && !(ch >= 'A' && ch <= 'Z') &&
                 !(ch >= '0' && ch <= '9')) {
        OPENSSL_PUT_ERROR(SSL, SSL_R_UNEXPECTED_OPERATOR_IN_GROUP);
        retval = in_group = 0;
        break;
      } else {
        rule = CIPHER_ADD;
      }
    } else if (ch == '-') {
      rule = CIPHER_DEL;
      l++;
    } else if (ch == '+') {
      rule = CIPHER_ORD;
      l++;
    } else if (ch == '!') {
      rule = CIPHER_KILL;
      l++;
    } else if (ch == '@') {
      rule = CIPHER_SPECIAL;
      l++;
    } else if (ch == '[') {
      if (in_group) {
        OPENSSL_PUT_ERROR(SSL, SSL_R_NESTED_GROUP);
        retval = in_group = 0;
        break;
      }
      in_group = 1;
      has_group = 1;
      l++;
      continue;
    } else {
      rule = CIPHER_ADD;
    }

    /* If preference groups are enabled, the only legal operator is +.
     * Otherwise the in_group bits will get mixed up. */
    if (has_group && rule != CIPHER_ADD) {
      OPENSSL_PUT_ERROR(SSL, SSL_R_MIXED_SPECIAL_OPERATOR_WITH_GROUPS);
      retval = in_group = 0;
      break;
    }

    if (ITEM_SEP(ch)) {
      l++;
      continue;
    }

    multi = 0;
    cipher_id = 0;
    alg_mkey = ~0u;
    alg_auth = ~0u;
    alg_enc = ~0u;
    alg_mac = ~0u;
    alg_ssl = ~0u;
    algo_strength = ~0u;

    for (;;) {
      ch = *l;
      buf = l;
      buf_len = 0;
      while (((ch >= 'A') && (ch <= 'Z')) || ((ch >= '0') && (ch <= '9')) ||
             ((ch >= 'a') && (ch <= 'z')) || (ch == '-') || (ch == '.')) {
        ch = *(++l);
        buf_len++;
      }

      if (buf_len == 0) {
        /* We hit something we cannot deal with, it is no command or separator
         * nor alphanumeric, so we call this an error. */
        OPENSSL_PUT_ERROR(SSL, SSL_R_INVALID_COMMAND);
        retval = in_group = 0;
        l++;
        break;
      }

      if (rule == CIPHER_SPECIAL) {
        break;
      }

      /* Look for a matching exact cipher. These aren't allowed in multipart
       * rules. */
      if (!multi && ch != '+') {
        for (j = 0; j < kCiphersLen; j++) {
          const SSL_CIPHER *cipher = &kCiphers[j];
          if (rule_equals(cipher->name, buf, buf_len)) {
            cipher_id = cipher->id;
            break;
          }
        }
      }
      if (cipher_id == 0) {
        /* If not an exact cipher, look for a matching cipher alias. */
        for (j = 0; j < kCipherAliasesLen; j++) {
          if (rule_equals(kCipherAliases[j].name, buf, buf_len)) {
            alg_mkey &= kCipherAliases[j].algorithm_mkey;
            alg_auth &= kCipherAliases[j].algorithm_auth;
            alg_enc &= kCipherAliases[j].algorithm_enc;
            alg_mac &= kCipherAliases[j].algorithm_mac;
            alg_ssl &= kCipherAliases[j].algorithm_ssl;
            algo_strength &= kCipherAliases[j].algo_strength;
            break;
          }
        }
        if (j == kCipherAliasesLen) {
          alg_mkey = alg_auth = alg_enc = alg_mac = alg_ssl = algo_strength = 0;
        }
      }

      /* Check for a multipart rule. */
      if (ch != '+') {
        break;
      }
      l++;
      multi = 1;
    }

    /* Ok, we have the rule, now apply it. */
    if (rule == CIPHER_SPECIAL) {
      /* special command */
      ok = 0;
      if (buf_len == 8 && !strncmp(buf, "STRENGTH", 8)) {
        ok = ssl_cipher_strength_sort(head_p, tail_p);
      } else {
        OPENSSL_PUT_ERROR(SSL, SSL_R_INVALID_COMMAND);
      }

      if (ok == 0) {
        retval = 0;
      }

      /* We do not support any "multi" options together with "@", so throw away
       * the rest of the command, if any left, until end or ':' is found. */
      while (*l != '\0' && !ITEM_SEP(*l)) {
        l++;
      }
    } else {
      ssl_cipher_apply_rule(cipher_id, alg_mkey, alg_auth, alg_enc, alg_mac,
                            alg_ssl, algo_strength, rule, -1, in_group, head_p,
                            tail_p);
    }
  }

  if (in_group) {
    OPENSSL_PUT_ERROR(SSL, SSL_R_INVALID_COMMAND);
    retval = 0;
  }

  return retval;
}

STACK_OF(SSL_CIPHER) *
ssl_create_cipher_list(const SSL_PROTOCOL_METHOD *ssl_method,
                       struct ssl_cipher_preference_list_st **out_cipher_list,
                       STACK_OF(SSL_CIPHER) **out_cipher_list_by_id,
                       const char *rule_str) {
  int ok;
  STACK_OF(SSL_CIPHER) *cipherstack = NULL, *tmp_cipher_list = NULL;
  const char *rule_p;
  CIPHER_ORDER *co_list = NULL, *head = NULL, *tail = NULL, *curr;
  uint8_t *in_group_flags = NULL;
  unsigned int num_in_group_flags = 0;
  struct ssl_cipher_preference_list_st *pref_list = NULL;

  /* Return with error if nothing to do. */
  if (rule_str == NULL || out_cipher_list == NULL) {
    return NULL;
  }

  /* Now we have to collect the available ciphers from the compiled in ciphers.
   * We cannot get more than the number compiled in, so it is used for
   * allocation. */
  co_list = (CIPHER_ORDER *)OPENSSL_malloc(sizeof(CIPHER_ORDER) * kCiphersLen);
  if (co_list == NULL) {
    OPENSSL_PUT_ERROR(SSL, ERR_R_MALLOC_FAILURE);
    return NULL;
  }

  ssl_cipher_collect_ciphers(ssl_method, co_list, &head, &tail);

  /* Now arrange all ciphers by preference:
   * TODO(davidben): Compute this order once and copy it. */

  /* Everything else being equal, prefer ECDHE_ECDSA then ECDHE_RSA over other
   * key exchange mechanisms */
  ssl_cipher_apply_rule(0, SSL_kECDHE, SSL_aECDSA, ~0u, ~0u, ~0u, ~0u,
                        CIPHER_ADD, -1, 0, &head, &tail);
  ssl_cipher_apply_rule(0, SSL_kECDHE, ~0u, ~0u, ~0u, ~0u, ~0u, CIPHER_ADD, -1,
                        0, &head, &tail);
  ssl_cipher_apply_rule(0, SSL_kECDHE, ~0u, ~0u, ~0u, ~0u, ~0u, CIPHER_DEL, -1,
                        0, &head, &tail);

  /* Order the bulk ciphers. First the preferred AEAD ciphers. We prefer
   * CHACHA20 unless there is hardware support for fast and constant-time
   * AES_GCM. */
  if (EVP_has_aes_hardware()) {
    ssl_cipher_apply_rule(0, ~0u, ~0u, SSL_AES256GCM, ~0u, ~0u, ~0u, CIPHER_ADD,
                          -1, 0, &head, &tail);
    ssl_cipher_apply_rule(0, ~0u, ~0u, SSL_AES128GCM, ~0u, ~0u, ~0u, CIPHER_ADD,
                          -1, 0, &head, &tail);
    ssl_cipher_apply_rule(0, ~0u, ~0u, SSL_CHACHA20POLY1305, ~0u, ~0u, ~0u,
                          CIPHER_ADD, -1, 0, &head, &tail);
  } else {
    ssl_cipher_apply_rule(0, ~0u, ~0u, SSL_CHACHA20POLY1305, ~0u, ~0u, ~0u,
                          CIPHER_ADD, -1, 0, &head, &tail);
    ssl_cipher_apply_rule(0, ~0u, ~0u, SSL_AES256GCM, ~0u, ~0u, ~0u, CIPHER_ADD,
                          -1, 0, &head, &tail);
    ssl_cipher_apply_rule(0, ~0u, ~0u, SSL_AES128GCM, ~0u, ~0u, ~0u, CIPHER_ADD,
                          -1, 0, &head, &tail);
  }

  /* Then the legacy non-AEAD ciphers: AES_256_CBC, AES-128_CBC, RC4_128_SHA,
   * RC4_128_MD5, 3DES_EDE_CBC_SHA. */
  ssl_cipher_apply_rule(0, ~0u, ~0u, SSL_AES256, ~0u, ~0u, ~0u, CIPHER_ADD, -1,
                        0, &head, &tail);
  ssl_cipher_apply_rule(0, ~0u, ~0u, SSL_AES128, ~0u, ~0u, ~0u, CIPHER_ADD, -1,
                        0, &head, &tail);
  ssl_cipher_apply_rule(0, ~0u, ~0u, SSL_RC4, ~SSL_MD5, ~0u, ~0u, CIPHER_ADD,
                        -1, 0, &head, &tail);
  ssl_cipher_apply_rule(0, ~0u, ~0u, SSL_RC4, SSL_MD5, ~0u, ~0u, CIPHER_ADD, -1,
                        0, &head, &tail);
  ssl_cipher_apply_rule(0, ~0u, ~0u, SSL_3DES, ~0u, ~0u, ~0u, CIPHER_ADD, -1, 0,
                        &head, &tail);

  /* Temporarily enable everything else for sorting */
  ssl_cipher_apply_rule(0, ~0u, ~0u, ~0u, ~0u, ~0u, ~0u, CIPHER_ADD, -1, 0,
                        &head, &tail);

  /* Move ciphers without forward secrecy to the end. */
  ssl_cipher_apply_rule(0, ~(SSL_kDHE | SSL_kECDHE), ~0u, ~0u, ~0u, ~0u, ~0u,
                        CIPHER_ORD, -1, 0, &head, &tail);

  /* Now disable everything (maintaining the ordering!) */
  ssl_cipher_apply_rule(0, ~0u, ~0u, ~0u, ~0u, ~0u, ~0u, CIPHER_DEL, -1, 0,
                        &head, &tail);

  /* If the rule_string begins with DEFAULT, apply the default rule before
   * using the (possibly available) additional rules. */
  ok = 1;
  rule_p = rule_str;
  if (strncmp(rule_str, "DEFAULT", 7) == 0) {
    ok = ssl_cipher_process_rulestr(ssl_method, SSL_DEFAULT_CIPHER_LIST, &head,
                                    &tail);
    rule_p += 7;
    if (*rule_p == ':') {
      rule_p++;
    }
  }

  if (ok && strlen(rule_p) > 0) {
    ok = ssl_cipher_process_rulestr(ssl_method, rule_p, &head, &tail);
  }

  if (!ok) {
    goto err;
  }

  /* Allocate new "cipherstack" for the result, return with error
   * if we cannot get one. */
  cipherstack = sk_SSL_CIPHER_new_null();
  if (cipherstack == NULL) {
    goto err;
  }

  in_group_flags = OPENSSL_malloc(kCiphersLen);
  if (!in_group_flags) {
    goto err;
  }

  /* The cipher selection for the list is done. The ciphers are added
   * to the resulting precedence to the STACK_OF(SSL_CIPHER). */
  for (curr = head; curr != NULL; curr = curr->next) {
    if (curr->active) {
      if (!sk_SSL_CIPHER_push(cipherstack, curr->cipher)) {
        goto err;
      }
      in_group_flags[num_in_group_flags++] = curr->in_group;
    }
  }
  OPENSSL_free(co_list); /* Not needed any longer */
  co_list = NULL;

  tmp_cipher_list = sk_SSL_CIPHER_dup(cipherstack);
  if (tmp_cipher_list == NULL) {
    goto err;
  }
  pref_list = OPENSSL_malloc(sizeof(struct ssl_cipher_preference_list_st));
  if (!pref_list) {
    goto err;
  }
  pref_list->ciphers = cipherstack;
  pref_list->in_group_flags = OPENSSL_malloc(num_in_group_flags);
  if (!pref_list->in_group_flags) {
    goto err;
  }
  memcpy(pref_list->in_group_flags, in_group_flags, num_in_group_flags);
  OPENSSL_free(in_group_flags);
  in_group_flags = NULL;
  if (*out_cipher_list != NULL) {
    ssl_cipher_preference_list_free(*out_cipher_list);
  }
  *out_cipher_list = pref_list;
  pref_list = NULL;

  if (out_cipher_list_by_id != NULL) {
    sk_SSL_CIPHER_free(*out_cipher_list_by_id);
    *out_cipher_list_by_id = tmp_cipher_list;
    tmp_cipher_list = NULL;
    (void) sk_SSL_CIPHER_set_cmp_func(*out_cipher_list_by_id,
                                      ssl_cipher_ptr_id_cmp);

    sk_SSL_CIPHER_sort(*out_cipher_list_by_id);
  } else {
    sk_SSL_CIPHER_free(tmp_cipher_list);
    tmp_cipher_list = NULL;
  }

  return cipherstack;

err:
  OPENSSL_free(co_list);
  OPENSSL_free(in_group_flags);
  sk_SSL_CIPHER_free(cipherstack);
  sk_SSL_CIPHER_free(tmp_cipher_list);
  if (pref_list) {
    OPENSSL_free(pref_list->in_group_flags);
  }
  OPENSSL_free(pref_list);
  return NULL;
}

uint32_t SSL_CIPHER_get_id(const SSL_CIPHER *cipher) { return cipher->id; }

uint16_t ssl_cipher_get_value(const SSL_CIPHER *cipher) {
  uint32_t id = cipher->id;
  /* All ciphers are SSLv3. */
  assert((id & 0xff000000) == 0x03000000);
  return id & 0xffff;
}

int SSL_CIPHER_is_AES(const SSL_CIPHER *cipher) {
  return (cipher->algorithm_enc & SSL_AES) != 0;
}

int SSL_CIPHER_has_MD5_HMAC(const SSL_CIPHER *cipher) {
  return (cipher->algorithm_mac & SSL_MD5) != 0;
}

int SSL_CIPHER_is_AESGCM(const SSL_CIPHER *cipher) {
  return (cipher->algorithm_enc & (SSL_AES128GCM | SSL_AES256GCM)) != 0;
}

int SSL_CIPHER_is_CHACHA20POLY1305(const SSL_CIPHER *cipher) {
  return (cipher->algorithm_enc & SSL_CHACHA20POLY1305) != 0;
}

int SSL_CIPHER_is_NULL(const SSL_CIPHER *cipher) {
  return (cipher->algorithm_enc & SSL_eNULL) != 0;
}

int SSL_CIPHER_is_RC4(const SSL_CIPHER *cipher) {
  return (cipher->algorithm_enc & SSL_RC4) != 0;
}

int SSL_CIPHER_is_block_cipher(const SSL_CIPHER *cipher) {
  /* Neither stream cipher nor AEAD. */
  return (cipher->algorithm_enc & (SSL_RC4 | SSL_eNULL)) == 0 &&
      cipher->algorithm_mac != SSL_AEAD;
}

/* return the actual cipher being used */
const char *SSL_CIPHER_get_name(const SSL_CIPHER *cipher) {
  if (cipher != NULL) {
    return cipher->name;
  }

  return "(NONE)";
}

const char *SSL_CIPHER_get_kx_name(const SSL_CIPHER *cipher) {
  if (cipher == NULL) {
    return "";
  }

  switch (cipher->algorithm_mkey) {
    case SSL_kRSA:
      return "RSA";

    case SSL_kDHE:
      switch (cipher->algorithm_auth) {
        case SSL_aRSA:
          return "DHE_RSA";
        default:
          assert(0);
          return "UNKNOWN";
      }

    case SSL_kECDHE:
      switch (cipher->algorithm_auth) {
        case SSL_aECDSA:
          return "ECDHE_ECDSA";
        case SSL_aRSA:
          return "ECDHE_RSA";
        case SSL_aPSK:
          return "ECDHE_PSK";
        default:
          assert(0);
          return "UNKNOWN";
      }

    case SSL_kPSK:
      assert(cipher->algorithm_auth == SSL_aPSK);
      return "PSK";

    default:
      assert(0);
      return "UNKNOWN";
  }
}

static const char *ssl_cipher_get_enc_name(const SSL_CIPHER *cipher) {
  switch (cipher->algorithm_enc) {
    case SSL_3DES:
      return "3DES_EDE_CBC";
    case SSL_RC4:
      return "RC4";
    case SSL_AES128:
      return "AES_128_CBC";
    case SSL_AES256:
      return "AES_256_CBC";
    case SSL_AES128GCM:
      return "AES_128_GCM";
    case SSL_AES256GCM:
      return "AES_256_GCM";
    case SSL_CHACHA20POLY1305:
      return "CHACHA20_POLY1305";
      break;
    default:
      assert(0);
      return "UNKNOWN";
  }
}

static const char *ssl_cipher_get_prf_name(const SSL_CIPHER *cipher) {
  switch (cipher->algorithm_prf) {
    case SSL_HANDSHAKE_MAC_DEFAULT:
      /* Before TLS 1.2, the PRF component is the hash used in the HMAC, which is
       * only ever MD5 or SHA-1. */
      switch (cipher->algorithm_mac) {
        case SSL_MD5:
          return "MD5";
        case SSL_SHA1:
          return "SHA";
      }
      break;
    case SSL_HANDSHAKE_MAC_SHA256:
      return "SHA256";
    case SSL_HANDSHAKE_MAC_SHA384:
      return "SHA384";
  }
  assert(0);
  return "UNKNOWN";
}

char *SSL_CIPHER_get_rfc_name(const SSL_CIPHER *cipher) {
  if (cipher == NULL) {
    return NULL;
  }

  const char *kx_name = SSL_CIPHER_get_kx_name(cipher);
  const char *enc_name = ssl_cipher_get_enc_name(cipher);
  const char *prf_name = ssl_cipher_get_prf_name(cipher);

  /* The final name is TLS_{kx_name}_WITH_{enc_name}_{prf_name}. */
  size_t len = 4 + strlen(kx_name) + 6 + strlen(enc_name) + 1 +
      strlen(prf_name) + 1;
  char *ret = OPENSSL_malloc(len);
  if (ret == NULL) {
    return NULL;
  }
  if (BUF_strlcpy(ret, "TLS_", len) >= len ||
      BUF_strlcat(ret, kx_name, len) >= len ||
      BUF_strlcat(ret, "_WITH_", len) >= len ||
      BUF_strlcat(ret, enc_name, len) >= len ||
      BUF_strlcat(ret, "_", len) >= len ||
      BUF_strlcat(ret, prf_name, len) >= len) {
    assert(0);
    OPENSSL_free(ret);
    return NULL;
  }
  assert(strlen(ret) + 1 == len);
  return ret;
}

int SSL_CIPHER_get_bits(const SSL_CIPHER *cipher, int *out_alg_bits) {
  if (cipher == NULL) {
    return 0;
  }

  if (out_alg_bits != NULL) {
    *out_alg_bits = cipher->alg_bits;
  }
  return cipher->strength_bits;
}

const char *SSL_CIPHER_description(const SSL_CIPHER *cipher, char *buf,
                                   int len) {
  const char *ver;
  const char *kx, *au, *enc, *mac;
  uint32_t alg_mkey, alg_auth, alg_enc, alg_mac, alg_ssl;
  static const char *format = "%-23s %s Kx=%-8s Au=%-4s Enc=%-9s Mac=%-4s\n";

  alg_mkey = cipher->algorithm_mkey;
  alg_auth = cipher->algorithm_auth;
  alg_enc = cipher->algorithm_enc;
  alg_mac = cipher->algorithm_mac;
  alg_ssl = cipher->algorithm_ssl;

  if (alg_ssl & SSL_SSLV3) {
    ver = "SSLv3";
  } else if (alg_ssl & SSL_TLSV1_2) {
    ver = "TLSv1.2";
  } else {
    ver = "unknown";
  }

  switch (alg_mkey) {
    case SSL_kRSA:
      kx = "RSA";
      break;

    case SSL_kDHE:
      kx = "DH";
      break;

    case SSL_kECDHE:
      kx = "ECDH";
      break;

    case SSL_kPSK:
      kx = "PSK";
      break;

    default:
      kx = "unknown";
  }

  switch (alg_auth) {
    case SSL_aRSA:
      au = "RSA";
      break;

    case SSL_aECDSA:
      au = "ECDSA";
      break;

    case SSL_aPSK:
      au = "PSK";
      break;

    default:
      au = "unknown";
      break;
  }

  switch (alg_enc) {
    case SSL_3DES:
      enc = "3DES(168)";
      break;

    case SSL_RC4:
      enc = "RC4(128)";
      break;

    case SSL_AES128:
      enc = "AES(128)";
      break;

    case SSL_AES256:
      enc = "AES(256)";
      break;

    case SSL_AES128GCM:
      enc = "AESGCM(128)";
      break;

    case SSL_AES256GCM:
      enc = "AESGCM(256)";
      break;

    case SSL_CHACHA20POLY1305:
      enc = "ChaCha20-Poly1305";
      break;

    case SSL_eNULL:
      enc="None";
      break;

    default:
      enc = "unknown";
      break;
  }

  switch (alg_mac) {
    case SSL_MD5:
      mac = "MD5";
      break;

    case SSL_SHA1:
      mac = "SHA1";
      break;

    case SSL_SHA256:
      mac = "SHA256";
      break;

    case SSL_SHA384:
      mac = "SHA384";
      break;

    case SSL_AEAD:
      mac = "AEAD";
      break;

    default:
      mac = "unknown";
      break;
  }

  if (buf == NULL) {
    len = 128;
    buf = OPENSSL_malloc(len);
    if (buf == NULL) {
      return NULL;
    }
  } else if (len < 128) {
    return "Buffer too small";
  }

  BIO_snprintf(buf, len, format, cipher->name, ver, kx, au, enc, mac);
  return buf;
}

const char *SSL_CIPHER_get_version(const SSL_CIPHER *cipher) {
  return "TLSv1/SSLv3";
}

COMP_METHOD *SSL_COMP_get_compression_methods(void) { return NULL; }

int SSL_COMP_add_compression_method(int id, COMP_METHOD *cm) { return 1; }

const char *SSL_COMP_get_name(const COMP_METHOD *comp) { return NULL; }

int ssl_cipher_get_key_type(const SSL_CIPHER *cipher) {
  uint32_t alg_a = cipher->algorithm_auth;

  if (alg_a & SSL_aECDSA) {
    return EVP_PKEY_EC;
  } else if (alg_a & SSL_aRSA) {
    return EVP_PKEY_RSA;
  }

  return EVP_PKEY_NONE;
}

int ssl_cipher_has_server_public_key(const SSL_CIPHER *cipher) {
  /* PSK-authenticated ciphers do not use a certificate. (RSA_PSK is not
   * supported.) */
  if (cipher->algorithm_auth & SSL_aPSK) {
    return 0;
  }

  /* All other ciphers include it. */
  return 1;
}

int ssl_cipher_requires_server_key_exchange(const SSL_CIPHER *cipher) {
  /* Ephemeral Diffie-Hellman key exchanges require a ServerKeyExchange. */
  if (cipher->algorithm_mkey & SSL_kDHE || cipher->algorithm_mkey & SSL_kECDHE) {
    return 1;
  }

  /* It is optional in all others. */
  return 0;
}

size_t ssl_cipher_get_record_split_len(const SSL_CIPHER *cipher) {
  size_t block_size;
  switch (cipher->algorithm_enc) {
    case SSL_3DES:
      block_size = 8;
      break;
    case SSL_AES128:
    case SSL_AES256:
      block_size = 16;
      break;
    default:
      return 0;
  }

  size_t mac_len;
  switch (cipher->algorithm_mac) {
    case SSL_MD5:
      mac_len = MD5_DIGEST_LENGTH;
      break;
    case SSL_SHA1:
      mac_len = SHA_DIGEST_LENGTH;
      break;
    default:
      return 0;
  }

  size_t ret = 1 + mac_len;
  ret += block_size - (ret % block_size);
  return ret;
}
