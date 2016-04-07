/* ssl/ssl.h */
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
 * OTHERWISE.
 */

#ifndef OPENSSL_HEADER_SSL_H
#define OPENSSL_HEADER_SSL_H

#include <openssl/base.h>

#include <openssl/bio.h>
#include <openssl/buf.h>
#include <openssl/hmac.h>
#include <openssl/lhash.h>
#include <openssl/pem.h>
#include <openssl/thread.h>
#include <openssl/x509.h>

#if !defined(OPENSSL_WINDOWS)
#include <sys/time.h>
#endif

/* wpa_supplicant expects to get the version functions from ssl.h */
#include <openssl/crypto.h>

/* Forward-declare struct timeval. On Windows, it is defined in winsock2.h and
 * Windows headers define too many macros to be included in public headers.
 * However, only a forward declaration is needed. */
struct timeval;

#if defined(__cplusplus)
extern "C" {
#endif


/* SSL implementation. */


/* Initialization. */

/* SSL_library_init initializes the crypto and SSL libraries and returns one. */
OPENSSL_EXPORT int SSL_library_init(void);


/* SSL contexts.
 *
 * |SSL_CTX| objects manage shared state and configuration between multiple TLS
 * or DTLS connections. Whether the connections are TLS or DTLS is selected by
 * an |SSL_METHOD| on creation.
 *
 * |SSL_CTX| are reference-counted and may be shared by connections across
 * multiple threads. Once shared, functions which change the |SSL_CTX|'s
 * configuration may not be used. */

/* TLS_method is the |SSL_METHOD| used for TLS (and SSLv3) connections. */
OPENSSL_EXPORT const SSL_METHOD *TLS_method(void);

/* DTLS_method is the |SSL_METHOD| used for DTLS connections. */
OPENSSL_EXPORT const SSL_METHOD *DTLS_method(void);

/* SSL_CTX_new returns a newly-allocated |SSL_CTX| with default settings or NULL
 * on error. */
OPENSSL_EXPORT SSL_CTX *SSL_CTX_new(const SSL_METHOD *method);

/* SSL_CTX_free releases memory associated with |ctx|. */
OPENSSL_EXPORT void SSL_CTX_free(SSL_CTX *ctx);


/* SSL connections.
 *
 * An |SSL| object represents a single TLS or DTLS connection. Although the
 * shared |SSL_CTX| is thread-safe, an |SSL| is not thread-safe and may only be
 * used on one thread at a time. */

/* SSL_new returns a newly-allocated |SSL| using |ctx| or NULL on error. The new
 * connection inherits settings from |ctx| at the time of creation. Settings may
 * also be individually configured on the connection.
 *
 * On creation, an |SSL| is not configured to be either a client or server. Call
 * |SSL_set_connect_state| or |SSL_set_accept_state| to set this. */
OPENSSL_EXPORT SSL *SSL_new(SSL_CTX *ctx);

/* SSL_free releases memory associated with |ssl|. */
OPENSSL_EXPORT void SSL_free(SSL *ssl);

/* SSL_set_connect_state configures |ssl| to be a client. */
OPENSSL_EXPORT void SSL_set_connect_state(SSL *ssl);

/* SSL_set_accept_state configures |ssl| to be a server. */
OPENSSL_EXPORT void SSL_set_accept_state(SSL *ssl);

/* SSL_is_server returns one if |ssl| is configured as a server and zero
 * otherwise. */
OPENSSL_EXPORT int SSL_is_server(SSL *ssl);

/* SSL_set_bio configures |ssl| to read from |rbio| and write to |wbio|. |ssl|
 * takes ownership of the two |BIO|s. If |rbio| and |wbio| are the same, |ssl|
 * only takes ownership of one reference.
 *
 * Calling this function on an already-configured |ssl| is deprecated. */
OPENSSL_EXPORT void SSL_set_bio(SSL *ssl, BIO *rbio, BIO *wbio);

/* SSL_get_rbio returns the |BIO| that |ssl| reads from. */
OPENSSL_EXPORT BIO *SSL_get_rbio(const SSL *ssl);

/* SSL_get_wbio returns the |BIO| that |ssl| writes to. */
OPENSSL_EXPORT BIO *SSL_get_wbio(const SSL *ssl);

/* SSL_do_handshake continues the current handshake. If there is none or the
 * handshake has completed or False Started, it returns one. Otherwise, it
 * returns <= 0. The caller should pass the value into |SSL_get_error| to
 * determine how to proceed.
 *
 * TODO(davidben): Ensure 0 is only returned on transport EOF.
 * https://crbug.com/466303. */
OPENSSL_EXPORT int SSL_do_handshake(SSL *ssl);

/* SSL_connect configures |ssl| as a client, if unconfigured, and calls
 * |SSL_do_handshake|. */
OPENSSL_EXPORT int SSL_connect(SSL *ssl);

/* SSL_accept configures |ssl| as a server, if unconfigured, and calls
 * |SSL_do_handshake|. */
OPENSSL_EXPORT int SSL_accept(SSL *ssl);

/* SSL_read reads up to |num| bytes from |ssl| into |buf|. It implicitly runs
 * any pending handshakes, including renegotiations when enabled. On success, it
 * returns the number of bytes read. Otherwise, it returns <= 0. The caller
 * should pass the value into |SSL_get_error| to determine how to proceed.
 *
 * TODO(davidben): Ensure 0 is only returned on transport EOF.
 * https://crbug.com/466303. */
OPENSSL_EXPORT int SSL_read(SSL *ssl, void *buf, int num);

/* SSL_peek behaves like |SSL_read| but does not consume any bytes returned. */
OPENSSL_EXPORT int SSL_peek(SSL *ssl, void *buf, int num);

/* SSL_write writes up to |num| bytes from |buf| into |ssl|. It implicitly runs
 * any pending handshakes, including renegotiations when enabled. On success, it
 * returns the number of bytes read. Otherwise, it returns <= 0. The caller
 * should pass the value into |SSL_get_error| to determine how to proceed.
 *
 * A non-blocking |SSL_write| differs from non-blocking |write| in that a failed
 * |SSL_write| still commits to the data passed in. When retrying, the caller
 * must supply the original write buffer (or a larger one containing the
 * original as a prefix). By default, retries will fail if they also do not
 * reuse the same |buf| pointer. This may be relaxed with
 * |SSL_MODE_ACCEPT_MOVING_WRITE_BUFFER|, but the buffer contents still must be
 * unchanged.
 *
 * By default, |SSL_write| will not return success until all |num| bytes are
 * written. This may be relaxed with |SSL_MODE_ENABLE_PARTIAL_WRITE|. It allows
 * |SSL_write| to complete with a partial result when only part of the input was
 * written in a single record.
 *
 * TODO(davidben): Ensure 0 is only returned on transport EOF.
 * https://crbug.com/466303. */
OPENSSL_EXPORT int SSL_write(SSL *ssl, const void *buf, int num);

/* SSL_shutdown shuts down |ssl|. On success, it completes in two stages. First,
 * it returns 0 if |ssl| completed uni-directional shutdown; close_notify has
 * been sent, but the peer's close_notify has not been received. Most callers
 * may stop at this point. For bi-directional shutdown, call |SSL_shutdown|
 * again. It returns 1 if close_notify has been both sent and received.
 *
 * If the peer's close_notify arrived first, the first stage is skipped.
 * |SSL_shutdown| will return 1 once close_notify is sent and skip 0. Callers
 * only interested in uni-directional shutdown must therefore allow for the
 * first stage returning either 0 or 1.
 *
 * |SSL_shutdown| returns -1 on failure. The caller should pass the return value
 * into |SSL_get_error| to determine how to proceed. If the underlying |BIO| is
 * non-blocking, both stages may require retry.
 *
 * |SSL_shutdown| must be called to retain |ssl|'s session in the session
 * cache. Use |SSL_CTX_set_quiet_shutdown| to configure |SSL_shutdown| to
 * neither send nor wait for close_notify but still retain the session.
 *
 * TODO(davidben): Is there any point in the session cache interaction? Remove
 * it? */
OPENSSL_EXPORT int SSL_shutdown(SSL *ssl);

/* SSL_get_error returns a |SSL_ERROR_*| value for the most recent operation on
 * |ssl|. It should be called after an operation failed to determine. */
OPENSSL_EXPORT int SSL_get_error(const SSL *ssl, int ret_code);

/* SSL_ERROR_NONE indicates the operation succeeded. */
#define SSL_ERROR_NONE 0

/* SSL_ERROR_SSL indicates the operation failed within the library. The caller
 * may inspect the error queue for more information. */
#define SSL_ERROR_SSL 1

/* SSL_ERROR_WANT_READ indicates the operation failed attempting to read from
 * the transport. The caller may retry the operation when the transport is ready
 * for reading. */
#define SSL_ERROR_WANT_READ 2

/* SSL_ERROR_WANT_READ indicates the operation failed attempting to write to
 * the transport. The caller may retry the operation when the transport is ready
 * for writing. */
#define SSL_ERROR_WANT_WRITE 3

/* SSL_ERROR_WANT_X509_LOOKUP indicates the operation failed in calling the
 * |cert_cb| or |client_cert_cb|. The caller may retry the operation when the
 * callback is ready to return a certificate or one has been configured
 * externally.
 *
 * See also |SSL_CTX_set_cert_cb| and |SSL_CTX_set_client_cert_cb|. */
#define SSL_ERROR_WANT_X509_LOOKUP 4

/* SSL_ERROR_WANT_SYSCALL indicates the operation failed externally to the
 * library. The caller should consult the system-specific error mechanism. This
 * is typically |errno| but may be something custom if using a custom |BIO|. It
 * may also be signaled if the transport returned EOF, in which case the
 * operation's return value will be zero. */
#define SSL_ERROR_SYSCALL 5

/* SSL_ERROR_ZERO_RETURN indicates the operation failed because the connection
 * was cleanly shut down with a close_notify alert. */
#define SSL_ERROR_ZERO_RETURN 6

/* SSL_ERROR_WANT_CONNECT indicates the operation failed attempting to connect
 * the transport (the |BIO| signaled |BIO_RR_CONNECT|). The caller may retry the
 * operation when the transport is ready. */
#define SSL_ERROR_WANT_CONNECT 7

/* SSL_ERROR_WANT_ACCEPT indicates the operation failed attempting to accept a
 * connection from the transport (the |BIO| signaled |BIO_RR_ACCEPT|). The
 * caller may retry the operation when the transport is ready.
 *
 * TODO(davidben): Remove this. It's used by accept BIOs which are bizarre. */
#define SSL_ERROR_WANT_ACCEPT 8

/* SSL_ERROR_WANT_CHANNEL_ID_LOOKUP indicates the operation failed looking up
 * the Channel ID key. The caller may retry the operation when |channel_id_cb|
 * is ready to return a key or one has been configured with
 * |SSL_set1_tls_channel_id|.
 *
 * See also |SSL_CTX_set_channel_id_cb|. */
#define SSL_ERROR_WANT_CHANNEL_ID_LOOKUP 9

/* SSL_ERROR_PENDING_SESSION indicates the operation failed because the session
 * lookup callback indicated the session was unavailable. The caller may retry
 * the operation when lookup has completed.
 *
 * See also |SSL_CTX_sess_set_get_cb| and |SSL_magic_pending_session_ptr|. */
#define SSL_ERROR_PENDING_SESSION 11

/* SSL_ERROR_PENDING_CERTIFICATE indicates the operation failed because the
 * early callback indicated certificate lookup was incomplete. The caller may
 * retry the operation when lookup has completed. Note: when the operation is
 * retried, the early callback will not be called a second time.
 *
 * See also |select_certificate_cb| on |SSL_CTX|. */
#define SSL_ERROR_PENDING_CERTIFICATE 12

/* SSL_ERROR_WANT_PRIVATE_KEY_OPERATION indicates the operation failed because
 * a private key operation was unfinished. The caller may retry the operation
 * when the private key operation is complete.
 *
 * See also |SSL_set_private_key_method|. */
#define SSL_ERROR_WANT_PRIVATE_KEY_OPERATION 13


/* Protocol versions. */

#define DTLS1_VERSION_MAJOR 0xfe
#define SSL3_VERSION_MAJOR 0x03

#define SSL3_VERSION 0x0300
#define TLS1_VERSION 0x0301
#define TLS1_1_VERSION 0x0302
#define TLS1_2_VERSION 0x0303

#define DTLS1_VERSION 0xfeff
#define DTLS1_2_VERSION 0xfefd

/* SSL_CTX_set_min_version sets the minimum protocol version for |ctx| to
 * |version|. */
OPENSSL_EXPORT void SSL_CTX_set_min_version(SSL_CTX *ctx, uint16_t version);

/* SSL_CTX_set_max_version sets the maximum protocol version for |ctx| to
 * |version|. */
OPENSSL_EXPORT void SSL_CTX_set_max_version(SSL_CTX *ctx, uint16_t version);

/* SSL_set_min_version sets the minimum protocol version for |ssl| to
 * |version|. */
OPENSSL_EXPORT void SSL_set_min_version(SSL *ssl, uint16_t version);

/* SSL_set_max_version sets the maximum protocol version for |ssl| to
 * |version|. */
OPENSSL_EXPORT void SSL_set_max_version(SSL *ssl, uint16_t version);

/* SSL_version returns the TLS or DTLS protocol version used by |ssl|, which is
 * one of the |*_VERSION| values. (E.g. |TLS1_2_VERSION|.) Before the version
 * is negotiated, the result is undefined. */
OPENSSL_EXPORT int SSL_version(const SSL *ssl);


/* Options.
 *
 * Options configure protocol behavior. */

/* SSL_OP_LEGACY_SERVER_CONNECT allows initial connections to servers that don't
 * support the renegotiation_info extension (RFC 5746). It is on by default. */
#define SSL_OP_LEGACY_SERVER_CONNECT 0x00000004L

/* SSL_OP_MICROSOFT_BIG_SSLV3_BUFFER allows for record sizes |SSL3_RT_MAX_EXTRA|
 * bytes above the maximum record size. */
#define SSL_OP_MICROSOFT_BIG_SSLV3_BUFFER 0x00000020L

/* SSL_OP_TLS_D5_BUG accepts an RSAClientKeyExchange in TLS encoded as in SSL3
 * (i.e. without a length prefix). */
#define SSL_OP_TLS_D5_BUG 0x00000100L

/* SSL_OP_ALL enables the above bug workarounds that are enabled by many
 * consumers.
 * TODO(davidben): Determine which of the remaining may be removed now. */
#define SSL_OP_ALL 0x00000BFFL

/* SSL_OP_NO_QUERY_MTU, in DTLS, disables querying the MTU from the underlying
 * |BIO|. Instead, the MTU is configured with |SSL_set_mtu|. */
#define SSL_OP_NO_QUERY_MTU 0x00001000L

/* SSL_OP_NO_TICKET disables session ticket support (RFC 5077). */
#define SSL_OP_NO_TICKET 0x00004000L

/* SSL_OP_CIPHER_SERVER_PREFERENCE configures servers to select ciphers and
 * ECDHE curves according to the server's preferences instead of the
 * client's. */
#define SSL_OP_CIPHER_SERVER_PREFERENCE 0x00400000L

/* SSL_CTX_set_options enables all options set in |options| (which should be one
 * or more of the |SSL_OP_*| values, ORed together) in |ctx|. It returns a
 * bitmask representing the resulting enabled options. */
OPENSSL_EXPORT uint32_t SSL_CTX_set_options(SSL_CTX *ctx, uint32_t options);

/* SSL_CTX_clear_options disables all options set in |options| (which should be
 * one or more of the |SSL_OP_*| values, ORed together) in |ctx|. It returns a
 * bitmask representing the resulting enabled options. */
OPENSSL_EXPORT uint32_t SSL_CTX_clear_options(SSL_CTX *ctx, uint32_t options);

/* SSL_CTX_get_options returns a bitmask of |SSL_OP_*| values that represent all
 * the options enabled for |ctx|. */
OPENSSL_EXPORT uint32_t SSL_CTX_get_options(const SSL_CTX *ctx);

/* SSL_set_options enables all options set in |options| (which should be one or
 * more of the |SSL_OP_*| values, ORed together) in |ssl|. It returns a bitmask
 * representing the resulting enabled options. */
OPENSSL_EXPORT uint32_t SSL_set_options(SSL *ssl, uint32_t options);

/* SSL_clear_options disables all options set in |options| (which should be one
 * or more of the |SSL_OP_*| values, ORed together) in |ssl|. It returns a
 * bitmask representing the resulting enabled options. */
OPENSSL_EXPORT uint32_t SSL_clear_options(SSL *ssl, uint32_t options);

/* SSL_get_options returns a bitmask of |SSL_OP_*| values that represent all the
 * options enabled for |ssl|. */
OPENSSL_EXPORT uint32_t SSL_get_options(const SSL *ssl);


/* Modes.
 *
 * Modes configure API behavior. */

/* SSL_MODE_ENABLE_PARTIAL_WRITE allows |SSL_write| to complete with a partial
 * result when the only part of the input was written in a single record. */
#define SSL_MODE_ENABLE_PARTIAL_WRITE 0x00000001L

/* SSL_MODE_ACCEPT_MOVING_WRITE_BUFFER allows retrying an incomplete |SSL_write|
 * with a different buffer. However, |SSL_write| still assumes the buffer
 * contents are unchanged. This is not the default to avoid the misconception
 * that non-blocking |SSL_write| behaves like non-blocking |write|. */
#define SSL_MODE_ACCEPT_MOVING_WRITE_BUFFER 0x00000002L

/* SSL_MODE_NO_AUTO_CHAIN disables automatically building a certificate chain
 * before sending certificates to the peer.
 * TODO(davidben): Remove this behavior. https://crbug.com/486295. */
#define SSL_MODE_NO_AUTO_CHAIN 0x00000008L

/* SSL_MODE_ENABLE_FALSE_START allows clients to send application data before
 * receipt of CCS and Finished. This mode enables full-handshakes to 'complete'
 * in one RTT. See draft-bmoeller-tls-falsestart-01. */
#define SSL_MODE_ENABLE_FALSE_START 0x00000080L

/* SSL_MODE_CBC_RECORD_SPLITTING causes multi-byte CBC records in SSL 3.0 and
 * TLS 1.0 to be split in two: the first record will contain a single byte and
 * the second will contain the remainder. This effectively randomises the IV and
 * prevents BEAST attacks. */
#define SSL_MODE_CBC_RECORD_SPLITTING 0x00000100L

/* SSL_MODE_NO_SESSION_CREATION will cause any attempts to create a session to
 * fail with SSL_R_SESSION_MAY_NOT_BE_CREATED. This can be used to enforce that
 * session resumption is used for a given SSL*. */
#define SSL_MODE_NO_SESSION_CREATION 0x00000200L

/* SSL_MODE_SEND_FALLBACK_SCSV sends TLS_FALLBACK_SCSV in the ClientHello.
 * To be set only by applications that reconnect with a downgraded protocol
 * version; see RFC 7507 for details.
 *
 * DO NOT ENABLE THIS if your application attempts a normal handshake. Only use
 * this in explicit fallback retries, following the guidance in RFC 7507. */
#define SSL_MODE_SEND_FALLBACK_SCSV 0x00000400L

/* SSL_CTX_set_mode enables all modes set in |mode| (which should be one or more
 * of the |SSL_MODE_*| values, ORed together) in |ctx|. It returns a bitmask
 * representing the resulting enabled modes. */
OPENSSL_EXPORT uint32_t SSL_CTX_set_mode(SSL_CTX *ctx, uint32_t mode);

/* SSL_CTX_clear_mode disables all modes set in |mode| (which should be one or
 * more of the |SSL_MODE_*| values, ORed together) in |ctx|. It returns a
 * bitmask representing the resulting enabled modes. */
OPENSSL_EXPORT uint32_t SSL_CTX_clear_mode(SSL_CTX *ctx, uint32_t mode);

/* SSL_CTX_get_mode returns a bitmask of |SSL_MODE_*| values that represent all
 * the modes enabled for |ssl|. */
OPENSSL_EXPORT uint32_t SSL_CTX_get_mode(const SSL_CTX *ctx);

/* SSL_set_mode enables all modes set in |mode| (which should be one or more of
 * the |SSL_MODE_*| values, ORed together) in |ssl|. It returns a bitmask
 * representing the resulting enabled modes. */
OPENSSL_EXPORT uint32_t SSL_set_mode(SSL *ssl, uint32_t mode);

/* SSL_clear_mode disables all modes set in |mode| (which should be one or more
 * of the |SSL_MODE_*| values, ORed together) in |ssl|. It returns a bitmask
 * representing the resulting enabled modes. */
OPENSSL_EXPORT uint32_t SSL_clear_mode(SSL *ssl, uint32_t mode);

/* SSL_get_mode returns a bitmask of |SSL_MODE_*| values that represent all the
 * modes enabled for |ssl|. */
OPENSSL_EXPORT uint32_t SSL_get_mode(const SSL *ssl);


/* Configuring certificates and private keys.
 *
 * These functions configure the connection's leaf certificate, private key, and
 * certificate chain. The certificate chain is ordered leaf to root (as sent on
 * the wire) but does not include the leaf. Both client and server certificates
 * use these functions.
 *
 * Certificates and keys may be configured before the handshake or dynamically
 * in the early callback and certificate callback. */

/* SSL_CTX_use_certificate sets |ctx|'s leaf certificate to |x509|. It returns
 * one on success and zero on failure. */
OPENSSL_EXPORT int SSL_CTX_use_certificate(SSL_CTX *ctx, X509 *x509);

/* SSL_use_certificate sets |ssl|'s leaf certificate to |x509|. It returns one
 * on success and zero on failure. */
OPENSSL_EXPORT int SSL_use_certificate(SSL *ssl, X509 *x509);

/* SSL_CTX_use_PrivateKey sets |ctx|'s private key to |pkey|. It returns one on
 * success and zero on failure. */
OPENSSL_EXPORT int SSL_CTX_use_PrivateKey(SSL_CTX *ctx, EVP_PKEY *pkey);

/* SSL_use_PrivateKey sets |ssl|'s private key to |pkey|. It returns one on
 * success and zero on failure. */
OPENSSL_EXPORT int SSL_use_PrivateKey(SSL *ssl, EVP_PKEY *pkey);

/* SSL_CTX_set0_chain sets |ctx|'s certificate chain, excluding the leaf, to
 * |chain|. On success, it returns one and takes ownership of |chain|.
 * Otherwise, it returns zero. */
OPENSSL_EXPORT int SSL_CTX_set0_chain(SSL_CTX *ctx, STACK_OF(X509) *chain);

/* SSL_CTX_set1_chain sets |ctx|'s certificate chain, excluding the leaf, to
 * |chain|. It returns one on success and zero on failure. The caller retains
 * ownership of |chain| and may release it freely. */
OPENSSL_EXPORT int SSL_CTX_set1_chain(SSL_CTX *ctx, STACK_OF(X509) *chain);

/* SSL_set0_chain sets |ssl|'s certificate chain, excluding the leaf, to
 * |chain|. On success, it returns one and takes ownership of |chain|.
 * Otherwise, it returns zero. */
OPENSSL_EXPORT int SSL_set0_chain(SSL *ssl, STACK_OF(X509) *chain);

/* SSL_set1_chain sets |ssl|'s certificate chain, excluding the leaf, to
 * |chain|. It returns one on success and zero on failure. The caller retains
 * ownership of |chain| and may release it freely. */
OPENSSL_EXPORT int SSL_set1_chain(SSL *ssl, STACK_OF(X509) *chain);

/* SSL_CTX_add0_chain_cert appends |x509| to |ctx|'s certificate chain. On
 * success, it returns one and takes ownership of |x509|. Otherwise, it returns
 * zero. */
OPENSSL_EXPORT int SSL_CTX_add0_chain_cert(SSL_CTX *ctx, X509 *x509);

/* SSL_CTX_add1_chain_cert appends |x509| to |ctx|'s certificate chain. It
 * returns one on success and zero on failure. The caller retains ownership of
 * |x509| and may release it freely. */
OPENSSL_EXPORT int SSL_CTX_add1_chain_cert(SSL_CTX *ctx, X509 *x509);

/* SSL_add0_chain_cert appends |x509| to |ctx|'s certificate chain. On success,
 * it returns one and takes ownership of |x509|. Otherwise, it returns zero. */
OPENSSL_EXPORT int SSL_add0_chain_cert(SSL *ssl, X509 *x509);

/* SSL_CTX_add_extra_chain_cert calls |SSL_CTX_add0_chain_cert|. */
OPENSSL_EXPORT int SSL_CTX_add_extra_chain_cert(SSL_CTX *ctx, X509 *x509);

/* SSL_add1_chain_cert appends |x509| to |ctx|'s certificate chain. It returns
 * one on success and zero on failure. The caller retains ownership of |x509|
 * and may release it freely. */
OPENSSL_EXPORT int SSL_add1_chain_cert(SSL *ssl, X509 *x509);

/* SSL_CTX_clear_chain_certs clears |ctx|'s certificate chain and returns
 * one. */
OPENSSL_EXPORT int SSL_CTX_clear_chain_certs(SSL_CTX *ctx);

/* SSL_CTX_clear_extra_chain_certs calls |SSL_CTX_clear_chain_certs|. */
OPENSSL_EXPORT int SSL_CTX_clear_extra_chain_certs(SSL_CTX *ctx);

/* SSL_clear_chain_certs clears |ssl|'s certificate chain and returns one. */
OPENSSL_EXPORT int SSL_clear_chain_certs(SSL *ssl);

/* SSL_CTX_set_cert_cb sets a callback that is called to select a certificate.
 * The callback returns one on success, zero on internal error, and a negative
 * number on failure or to pause the handshake. If the handshake is paused,
 * |SSL_get_error| will return |SSL_ERROR_WANT_X509_LOOKUP|. */
OPENSSL_EXPORT void SSL_CTX_set_cert_cb(SSL_CTX *ctx,
                                        int (*cb)(SSL *ssl, void *arg),
                                        void *arg);

/* SSL_set_cert_cb sets a callback that is called to select a certificate. The
 * callback returns one on success, zero on internal error, and a negative
 * number on failure or to pause the handshake. If the handshake is paused,
 * |SSL_get_error| will return |SSL_ERROR_WANT_X509_LOOKUP|. */
OPENSSL_EXPORT void SSL_set_cert_cb(SSL *ssl, int (*cb)(SSL *ssl, void *arg),
                                    void *arg);

/* SSL_certs_clear resets the private key, leaf certificate, and certificate
 * chain of |ssl|. */
OPENSSL_EXPORT void SSL_certs_clear(SSL *ssl);

/* SSL_CTX_check_private_key returns one if the certificate and private key
 * configured in |ctx| are consistent and zero otherwise. */
OPENSSL_EXPORT int SSL_CTX_check_private_key(const SSL_CTX *ctx);

/* SSL_check_private_key returns one if the certificate and private key
 * configured in |ssl| are consistent and zero otherwise. */
OPENSSL_EXPORT int SSL_check_private_key(const SSL *ssl);

/* SSL_CTX_get0_certificate returns |ctx|'s leaf certificate. */
OPENSSL_EXPORT X509 *SSL_CTX_get0_certificate(const SSL_CTX *ctx);

/* SSL_get_certificate returns |ssl|'s leaf certificate. */
OPENSSL_EXPORT X509 *SSL_get_certificate(const SSL *ssl);

/* SSL_CTX_get0_privatekey returns |ctx|'s private key. */
OPENSSL_EXPORT EVP_PKEY *SSL_CTX_get0_privatekey(const SSL_CTX *ctx);

/* SSL_get_privatekey returns |ssl|'s private key. */
OPENSSL_EXPORT EVP_PKEY *SSL_get_privatekey(const SSL *ssl);

/* SSL_CTX_get0_chain_certs sets |*out_chain| to |ctx|'s certificate chain and
 * returns one. */
OPENSSL_EXPORT int SSL_CTX_get0_chain_certs(const SSL_CTX *ctx,
                                            STACK_OF(X509) **out_chain);

/* SSL_CTX_get_extra_chain_certs calls |SSL_CTX_get0_chain_certs|. */
OPENSSL_EXPORT int SSL_CTX_get_extra_chain_certs(const SSL_CTX *ctx,
                                                 STACK_OF(X509) **out_chain);

/* SSL_get0_chain_certs sets |*out_chain| to |ssl|'s certificate chain and
 * returns one. */
OPENSSL_EXPORT int SSL_get0_chain_certs(const SSL *ssl,
                                        STACK_OF(X509) **out_chain);

/* SSL_CTX_set_signed_cert_timestamp_list sets the list of signed certificate
 * timestamps that is sent to clients that request it. The |list| argument must
 * contain one or more SCT structures serialised as a SignedCertificateTimestamp
 * List (see https://tools.ietf.org/html/rfc6962#section-3.3) – i.e. each SCT
 * is prefixed by a big-endian, uint16 length and the concatenation of one or
 * more such prefixed SCTs are themselves also prefixed by a uint16 length. It
 * returns one on success and zero on error. The caller retains ownership of
 * |list|. */
OPENSSL_EXPORT int SSL_CTX_set_signed_cert_timestamp_list(SSL_CTX *ctx,
                                                          const uint8_t *list,
                                                          size_t list_len);

/* SSL_CTX_set_ocsp_response sets the OCSP reponse that is sent to clients
 * which request it. It returns one on success and zero on error. The caller
 * retains ownership of |response|. */
OPENSSL_EXPORT int SSL_CTX_set_ocsp_response(SSL_CTX *ctx,
                                             const uint8_t *response,
                                             size_t response_len);

/* SSL_set_private_key_digest_prefs copies |num_digests| NIDs from |digest_nids|
 * into |ssl|. These digests will be used, in decreasing order of preference,
 * when signing with |ssl|'s private key. It returns one on success and zero on
 * error. */
OPENSSL_EXPORT int SSL_set_private_key_digest_prefs(SSL *ssl,
                                                    const int *digest_nids,
                                                    size_t num_digests);


/* Certificate and private key convenience functions. */

/* SSL_CTX_use_RSAPrivateKey sets |ctx|'s private key to |rsa|. It returns one
 * on success and zero on failure. */
OPENSSL_EXPORT int SSL_CTX_use_RSAPrivateKey(SSL_CTX *ctx, RSA *rsa);

/* SSL_use_RSAPrivateKey sets |ctx|'s private key to |rsa|. It returns one on
 * success and zero on failure. */
OPENSSL_EXPORT int SSL_use_RSAPrivateKey(SSL *ssl, RSA *rsa);

/* The following functions configure certificates or private keys but take as
 * input DER-encoded structures. They return one on success and zero on
 * failure. */

OPENSSL_EXPORT int SSL_CTX_use_certificate_ASN1(SSL_CTX *ctx, int len,
                                                const uint8_t *d);
OPENSSL_EXPORT int SSL_use_certificate_ASN1(SSL *ssl, const uint8_t *der,
                                            int len);

OPENSSL_EXPORT int SSL_CTX_use_PrivateKey_ASN1(int pk, SSL_CTX *ctx,
                                               const uint8_t *d, long len);
OPENSSL_EXPORT int SSL_use_PrivateKey_ASN1(int type, SSL *ssl,
                                           const uint8_t *d, long len);

OPENSSL_EXPORT int SSL_CTX_use_RSAPrivateKey_ASN1(SSL_CTX *ctx,
                                                  const uint8_t *der,
                                                  size_t der_len);
OPENSSL_EXPORT int SSL_use_RSAPrivateKey_ASN1(SSL *ssl, const uint8_t *der,
                                              size_t der_len);

/* The following functions configure certificates or private keys but take as
 * input files to read from. They return one on success and zero on failure. The
 * |type| parameter is one of the |SSL_FILETYPE_*| values and determines whether
 * the file's contents are read as PEM or DER. */

#define SSL_FILETYPE_ASN1 X509_FILETYPE_ASN1
#define SSL_FILETYPE_PEM X509_FILETYPE_PEM

OPENSSL_EXPORT int SSL_CTX_use_RSAPrivateKey_file(SSL_CTX *ctx,
                                                  const char *file,
                                                  int type);
OPENSSL_EXPORT int SSL_use_RSAPrivateKey_file(SSL *ssl, const char *file,
                                              int type);

OPENSSL_EXPORT int SSL_CTX_use_certificate_file(SSL_CTX *ctx, const char *file,
                                                int type);
OPENSSL_EXPORT int SSL_use_certificate_file(SSL *ssl, const char *file,
                                            int type);

OPENSSL_EXPORT int SSL_CTX_use_PrivateKey_file(SSL_CTX *ctx, const char *file,
                                               int type);
OPENSSL_EXPORT int SSL_use_PrivateKey_file(SSL *ssl, const char *file,
                                           int type);

/* SSL_CTX_use_certificate_chain_file configures certificates for |ctx|. It
 * reads the contents of |file| as a PEM-encoded leaf certificate followed
 * optionally by the certificate chain to send to the peer. It returns one on
 * success and zero on failure. */
OPENSSL_EXPORT int SSL_CTX_use_certificate_chain_file(SSL_CTX *ctx,
                                                      const char *file);

/* SSL_CTX_set_default_passwd_cb sets the password callback for PEM-based
 * convenience functions called on |ctx|. */
OPENSSL_EXPORT void SSL_CTX_set_default_passwd_cb(SSL_CTX *ctx,
                                                  pem_password_cb *cb);

/* SSL_CTX_set_default_passwd_cb_userdata sets the userdata parameter for
 * |ctx|'s password callback. */
OPENSSL_EXPORT void SSL_CTX_set_default_passwd_cb_userdata(SSL_CTX *ctx,
                                                           void *data);


/* Custom private keys. */

enum ssl_private_key_result_t {
  ssl_private_key_success,
  ssl_private_key_retry,
  ssl_private_key_failure,
};

/* SSL_PRIVATE_KEY_METHOD describes private key hooks. This is used to off-load
 * signing operations to a custom, potentially asynchronous, backend. */
typedef struct ssl_private_key_method_st {
  /* type returns either |EVP_PKEY_RSA| or |EVP_PKEY_EC| to denote the type of
   * key used by |ssl|. */
  int (*type)(SSL *ssl);

  /* max_signature_len returns the maximum length of a signature signed by the
   * key used by |ssl|. This must be a constant value for a given |ssl|. */
  size_t (*max_signature_len)(SSL *ssl);

  /* sign signs |in_len| bytes of digest from |in|. |md| is the hash function
   * used to calculate |in|. On success, it returns |ssl_private_key_success|
   * and writes at most |max_out| bytes of signature data to |out|. On failure,
   * it returns |ssl_private_key_failure|. If the operation has not completed,
   * it returns |ssl_private_key_retry|. |sign| should arrange for the
   * high-level operation on |ssl| to be retried when the operation is
   * completed. This will result in a call to |sign_complete|.
   *
   * If the key is an RSA key, implementations must use PKCS#1 padding. |in| is
   * the digest itself, so the DigestInfo prefix, if any, must be prepended by
   * |sign|. If |md| is |EVP_md5_sha1|, there is no prefix.
   *
   * It is an error to call |sign| while another private key operation is in
   * progress on |ssl|. */
  enum ssl_private_key_result_t (*sign)(SSL *ssl, uint8_t *out, size_t *out_len,
                                        size_t max_out, const EVP_MD *md,
                                        const uint8_t *in, size_t in_len);

  /* sign_complete completes a pending |sign| operation. If the operation has
   * completed, it returns |ssl_private_key_success| and writes the result to
   * |out| as in |sign|. Otherwise, it returns |ssl_private_key_failure| on
   * failure and |ssl_private_key_retry| if the operation is still in progress.
   *
   * |sign_complete| may be called arbitrarily many times before completion, but
   * it is an error to call |sign_complete| if there is no pending |sign|
   * operation in progress on |ssl|. */
  enum ssl_private_key_result_t (*sign_complete)(SSL *ssl, uint8_t *out,
                                                 size_t *out_len, size_t max_out);
} SSL_PRIVATE_KEY_METHOD;

/* SSL_set_private_key_method configures a custom private key on |ssl|.
 * |key_method| must remain valid for the lifetime of |ssl|. */
OPENSSL_EXPORT void SSL_set_private_key_method(
    SSL *ssl, const SSL_PRIVATE_KEY_METHOD *key_method);


/* Cipher suites.
 *
 * |SSL_CIPHER| objects represent cipher suites. */

DECLARE_STACK_OF(SSL_CIPHER)

/* SSL_get_cipher_by_value returns the structure representing a TLS cipher
 * suite based on its assigned number, or NULL if unknown. See
 * https://www.iana.org/assignments/tls-parameters/tls-parameters.xhtml#tls-parameters-4. */
OPENSSL_EXPORT const SSL_CIPHER *SSL_get_cipher_by_value(uint16_t value);

/* SSL_CIPHER_get_id returns |cipher|'s id. It may be cast to a |uint16_t| to
 * get the cipher suite value. */
OPENSSL_EXPORT uint32_t SSL_CIPHER_get_id(const SSL_CIPHER *cipher);

/* SSL_CIPHER_is_AES returns one if |cipher| uses AES (either GCM or CBC
 * mode). */
OPENSSL_EXPORT int SSL_CIPHER_is_AES(const SSL_CIPHER *cipher);

/* SSL_CIPHER_has_MD5_HMAC returns one if |cipher| uses HMAC-MD5. */
OPENSSL_EXPORT int SSL_CIPHER_has_MD5_HMAC(const SSL_CIPHER *cipher);

/* SSL_CIPHER_is_AESGCM returns one if |cipher| uses AES-GCM. */
OPENSSL_EXPORT int SSL_CIPHER_is_AESGCM(const SSL_CIPHER *cipher);

/* SSL_CIPHER_is_CHACHA20POLY1305 returns one if |cipher| uses
 * CHACHA20_POLY1305. */
OPENSSL_EXPORT int SSL_CIPHER_is_CHACHA20POLY1305(const SSL_CIPHER *cipher);

/* SSL_CIPHER_is_NULL returns one if |cipher| does not encrypt. */
OPENSSL_EXPORT int SSL_CIPHER_is_NULL(const SSL_CIPHER *cipher);

/* SSL_CIPHER_is_RC4 returns one if |cipher| uses RC4. */
OPENSSL_EXPORT int SSL_CIPHER_is_RC4(const SSL_CIPHER *cipher);

/* SSL_CIPHER_is_block_cipher returns one if |cipher| is a block cipher. */
OPENSSL_EXPORT int SSL_CIPHER_is_block_cipher(const SSL_CIPHER *cipher);

/* SSL_CIPHER_get_name returns the OpenSSL name of |cipher|. */
OPENSSL_EXPORT const char *SSL_CIPHER_get_name(const SSL_CIPHER *cipher);

/* SSL_CIPHER_get_kx_name returns a string that describes the key-exchange
 * method used by |cipher|. For example, "ECDHE_ECDSA". */
OPENSSL_EXPORT const char *SSL_CIPHER_get_kx_name(const SSL_CIPHER *cipher);

/* SSL_CIPHER_get_rfc_name returns a newly-allocated string with the standard
 * name for |cipher| or NULL on error. For example,
 * "TLS_ECDHE_RSA_WITH_AES_128_GCM_SHA256". The caller is responsible for
 * calling |OPENSSL_free| on the result. */
OPENSSL_EXPORT char *SSL_CIPHER_get_rfc_name(const SSL_CIPHER *cipher);

/* SSL_CIPHER_get_bits returns the strength, in bits, of |cipher|. If
 * |out_alg_bits| is not NULL, it writes the number of bits consumed by the
 * symmetric algorithm to |*out_alg_bits|. */
OPENSSL_EXPORT int SSL_CIPHER_get_bits(const SSL_CIPHER *cipher,
                                       int *out_alg_bits);


/* Cipher suite configuration.
 *
 * OpenSSL uses a mini-language to configure cipher suites. The language
 * maintains an ordered list of enabled ciphers, along with an ordered list of
 * disabled but available ciphers. Initially, all ciphers are disabled with a
 * default ordering. The cipher string is then interpreted as a sequence of
 * directives, separated by colons, each of which modifies this state.
 *
 * Most directives consist of a one character or empty opcode followed by a
 * selector which matches a subset of available ciphers.
 *
 * Available opcodes are:
 *
 *   The empty opcode enables and appends all matching disabled ciphers to the
 *   end of the enabled list. The newly appended ciphers are ordered relative to
 *   each other matching their order in the disabled list.
 *
 *   |-| disables all matching enabled ciphers and prepends them to the disabled
 *   list, with relative order from the enabled list preserved. This means the
 *   most recently disabled ciphers get highest preference relative to other
 *   disabled ciphers if re-enabled.
 *
 *   |+| moves all matching enabled ciphers to the end of the enabled list, with
 *   relative order preserved.
 *
 *   |!| deletes all matching ciphers, enabled or not, from either list. Deleted
 *   ciphers will not matched by future operations.
 *
 * A selector may be a specific cipher (using the OpenSSL name for the cipher)
 * or one or more rules separated by |+|. The final selector matches the
 * intersection of each rule. For instance, |AESGCM+aECDSA| matches
 * ECDSA-authenticated AES-GCM ciphers.
 *
 * Available cipher rules are:
 *
 *   |ALL| matches all ciphers.
 *
 *   |kRSA|, |kDHE|, |kECDHE|, and |kPSK| match ciphers using plain RSA, DHE,
 *   ECDHE, and plain PSK key exchanges, respectively. Note that ECDHE_PSK is
 *   matched by |kECDHE| and not |kPSK|.
 *
 *   |aRSA|, |aECDSA|, and |aPSK| match ciphers authenticated by RSA, ECDSA, and
 *   a pre-shared key, respectively.
 *
 *   |RSA|, |DHE|, |ECDHE|, |PSK|, |ECDSA|, and |PSK| are aliases for the
 *   corresponding |k*| or |a*| cipher rule. |RSA| is an alias for |kRSA|, not
 *   |aRSA|.
 *
 *   |3DES|, |RC4|, |AES128|, |AES256|, |AES|, |AESGCM|, |CHACHA20| match
 *   ciphers whose bulk cipher use the corresponding encryption scheme. Note
 *   that |AES|, |AES128|, and |AES256| match both CBC and GCM ciphers.
 *
 *   |MD5|, |SHA1|, |SHA256|, and |SHA384| match legacy cipher suites using the
 *   corresponding hash function in their MAC. AEADs are matched by none of
 *   these.
 *
 *   |SHA| is an alias for |SHA1|.
 *
 * Although implemented, authentication-only ciphers match no rules and must be
 * explicitly selected by name.
 *
 * Deprecated cipher rules:
 *
 *   |kEDH|, |EDH|, |kEECDH|, and |EECDH| are legacy aliases for |kDHE|, |DHE|,
 *   |kECDHE|, and |ECDHE|, respectively.
 *
 *   |MEDIUM| and |HIGH| match ciphers historically labeled by OpenSSL as
 *   'medium' and 'high', respectively.
 *
 *   |FIPS| matches ciphers historically FIPS-approved in OpenSSL.
 *
 *   |SSLv3| and |TLSv1| match ciphers available in TLS 1.1 or earlier.
 *   |TLSv1_2| matches ciphers new in TLS 1.2. This is confusing and should not
 *   be used.
 *
 * Unknown rules silently match nothing.
 *
 * The special |@STRENGTH| directive will sort all enabled ciphers by strength.
 *
 * The |DEFAULT| directive, when appearing at the front of the string, expands
 * to the default ordering of available ciphers.
 *
 * If configuring a server, one may also configure equal-preference groups to
 * partially respect the client's preferences when
 * |SSL_OP_CIPHER_SERVER_PREFERENCE| is enabled. Ciphers in an equal-preference
 * group have equal priority and use the client order. This may be used to
 * enforce that AEADs are preferred but select AES-GCM vs. ChaCha20-Poly1305
 * based on client preferences. An equal-preference is specified with square
 * brackets, combining multiple selectors separated by |. For example:
 *
 *   [ECDHE-ECDSA-CHACHA20-POLY1305|ECDHE-ECDSA-AES128-GCM-SHA256]
 *
 * Once an equal-preference group is used, future directives must be
 * opcode-less. */

/* SSL_DEFAULT_CIPHER_LIST is the default cipher suite configuration. It is
 * substituted when a cipher string starts with 'DEFAULT'. */
#define SSL_DEFAULT_CIPHER_LIST "ALL"

/* SSL_CTX_set_cipher_list configures the cipher list for |ctx|, evaluating
 * |str| as a cipher string. It returns one on success and zero on failure. */
OPENSSL_EXPORT int SSL_CTX_set_cipher_list(SSL_CTX *ctx, const char *str);

/* SSL_CTX_set_cipher_list configures the TLS 1.0+ cipher list for |ctx|,
 * evaluating |str| as a cipher string. It returns one on success and zero on
 * failure. If set, servers will use this cipher suite list for TLS 1.0 or
 * higher. */
OPENSSL_EXPORT int SSL_CTX_set_cipher_list_tls10(SSL_CTX *ctx, const char *str);

/* SSL_CTX_set_cipher_list configures the TLS 1.1+ cipher list for |ctx|,
 * evaluating |str| as a cipher string. It returns one on success and zero on
 * failure. If set, servers will use this cipher suite list for TLS 1.1 or
 * higher. */
OPENSSL_EXPORT int SSL_CTX_set_cipher_list_tls11(SSL_CTX *ctx, const char *str);

/* SSL_set_cipher_list configures the cipher list for |ssl|, evaluating |str| as
 * a cipher string. It returns one on success and zero on failure. */
OPENSSL_EXPORT int SSL_set_cipher_list(SSL *ssl, const char *str);

/* SSL_get_ciphers returns the cipher list for |ssl|, in order of preference. If
 * |SSL_CTX_set_cipher_list_tls10| or |SSL_CTX_set_cipher_list_tls11| has been
 * used, the corresponding list for the current version is returned. */
OPENSSL_EXPORT STACK_OF(SSL_CIPHER) *SSL_get_ciphers(const SSL *ssl);


/* Connection information. */

/* SSL_get_peer_certificate returns the peer's leaf certificate or NULL if the
 * peer did not use certificates. The caller must call |X509_free| on the
 * result to release it. */
OPENSSL_EXPORT X509 *SSL_get_peer_certificate(const SSL *ssl);

/* SSL_get_peer_cert_chain returns the peer's certificate chain or NULL if
 * unavailable or the peer did not use certificates. For historical reasons,
 * this may not be available if resuming a serialized |SSL_SESSION|. The caller
 * does not take ownership of the result.
 *
 * WARNING: This function behaves differently between client and server. If
 * |ssl| is a server, the returned chain does not include the leaf certificate.
 * If a client, it does. */
OPENSSL_EXPORT STACK_OF(X509) *SSL_get_peer_cert_chain(const SSL *ssl);

/* SSL_get_tls_unique writes at most |max_out| bytes of the tls-unique value
 * for |ssl| to |out| and sets |*out_len| to the number of bytes written. It
 * returns one on success or zero on error. In general |max_out| should be at
 * least 12.
 *
 * This function will always fail if the initial handshake has not completed.
 * The tls-unique value will change after a renegotiation but, since
 * renegotiations can be initiated by the server at any point, the higher-level
 * protocol must either leave them disabled or define states in which the
 * tls-unique value can be read.
 *
 * The tls-unique value is defined by
 * https://tools.ietf.org/html/rfc5929#section-3.1. Due to a weakness in the
 * TLS protocol, tls-unique is broken for resumed connections unless the
 * Extended Master Secret extension is negotiated. Thus this function will
 * return zero if |ssl| performed session resumption unless EMS was used when
 * negotiating the original session. */
OPENSSL_EXPORT int SSL_get_tls_unique(const SSL *ssl, uint8_t *out,
                                      size_t *out_len, size_t max_out);

/* SSL_get_extms_support returns one if the Extended Master Secret
 * extension was negotiated. Otherwise, it returns zero. */
OPENSSL_EXPORT int SSL_get_extms_support(const SSL *ssl);

/* SSL_get_current_cipher returns the cipher used in the current outgoing
 * connection state, or NULL if the null cipher is active. */
OPENSSL_EXPORT const SSL_CIPHER *SSL_get_current_cipher(const SSL *ssl);

/* SSL_session_reused returns one if |ssl| performed an abbreviated handshake
 * and zero otherwise.
 *
 * TODO(davidben): Hammer down the semantics of this API while a handshake,
 * initial or renego, is in progress. */
OPENSSL_EXPORT int SSL_session_reused(const SSL *ssl);

/* SSL_get_secure_renegotiation_support returns one if the peer supports secure
 * renegotiation (RFC 5746) and zero otherwise. */
OPENSSL_EXPORT int SSL_get_secure_renegotiation_support(const SSL *ssl);


/* Custom extensions.
 *
 * The custom extension functions allow TLS extensions to be added to
 * ClientHello and ServerHello messages. */

/* SSL_custom_ext_add_cb is a callback function that is called when the
 * ClientHello (for clients) or ServerHello (for servers) is constructed. In
 * the case of a server, this callback will only be called for a given
 * extension if the ClientHello contained that extension – it's not possible to
 * inject extensions into a ServerHello that the client didn't request.
 *
 * When called, |extension_value| will contain the extension number that is
 * being considered for addition (so that a single callback can handle multiple
 * extensions). If the callback wishes to include the extension, it must set
 * |*out| to point to |*out_len| bytes of extension contents and return one. In
 * this case, the corresponding |SSL_custom_ext_free_cb| callback will later be
 * called with the value of |*out| once that data has been copied.
 *
 * If the callback does not wish to add an extension it must return zero.
 *
 * Alternatively, the callback can abort the connection by setting
 * |*out_alert_value| to a TLS alert number and returning -1. */
typedef int (*SSL_custom_ext_add_cb)(SSL *ssl, unsigned extension_value,
                                     const uint8_t **out, size_t *out_len,
                                     int *out_alert_value, void *add_arg);

/* SSL_custom_ext_free_cb is a callback function that is called by OpenSSL iff
 * an |SSL_custom_ext_add_cb| callback previously returned one. In that case,
 * this callback is called and passed the |out| pointer that was returned by
 * the add callback. This is to free any dynamically allocated data created by
 * the add callback. */
typedef void (*SSL_custom_ext_free_cb)(SSL *ssl, unsigned extension_value,
                                       const uint8_t *out, void *add_arg);

/* SSL_custom_ext_parse_cb is a callback function that is called by OpenSSL to
 * parse an extension from the peer: that is from the ServerHello for a client
 * and from the ClientHello for a server.
 *
 * When called, |extension_value| will contain the extension number and the
 * contents of the extension are |contents_len| bytes at |contents|.
 *
 * The callback must return one to continue the handshake. Otherwise, if it
 * returns zero, a fatal alert with value |*out_alert_value| is sent and the
 * handshake is aborted. */
typedef int (*SSL_custom_ext_parse_cb)(SSL *ssl, unsigned extension_value,
                                       const uint8_t *contents,
                                       size_t contents_len,
                                       int *out_alert_value, void *parse_arg);

/* SSL_extension_supported returns one iff OpenSSL internally handles
 * extensions of type |extension_value|. This can be used to avoid registering
 * custom extension handlers for extensions that a future version of OpenSSL
 * may handle internally. */
OPENSSL_EXPORT int SSL_extension_supported(unsigned extension_value);

/* SSL_CTX_add_client_custom_ext registers callback functions for handling
 * custom TLS extensions for client connections.
 *
 * If |add_cb| is NULL then an empty extension will be added in each
 * ClientHello. Otherwise, see the comment for |SSL_custom_ext_add_cb| about
 * this callback.
 *
 * The |free_cb| may be NULL if |add_cb| doesn't dynamically allocate data that
 * needs to be freed.
 *
 * It returns one on success or zero on error. It's always an error to register
 * callbacks for the same extension twice, or to register callbacks for an
 * extension that OpenSSL handles internally. See |SSL_extension_supported| to
 * discover, at runtime, which extensions OpenSSL handles internally. */
OPENSSL_EXPORT int SSL_CTX_add_client_custom_ext(
    SSL_CTX *ctx, unsigned extension_value, SSL_custom_ext_add_cb add_cb,
    SSL_custom_ext_free_cb free_cb, void *add_arg,
    SSL_custom_ext_parse_cb parse_cb, void *parse_arg);

/* SSL_CTX_add_server_custom_ext is the same as
 * |SSL_CTX_add_client_custom_ext|, but for server connections.
 *
 * Unlike on the client side, if |add_cb| is NULL no extension will be added.
 * The |add_cb|, if any, will only be called if the ClientHello contained a
 * matching extension. */
OPENSSL_EXPORT int SSL_CTX_add_server_custom_ext(
    SSL_CTX *ctx, unsigned extension_value, SSL_custom_ext_add_cb add_cb,
    SSL_custom_ext_free_cb free_cb, void *add_arg,
    SSL_custom_ext_parse_cb parse_cb, void *parse_arg);


/* Sessions.
 *
 * An |SSL_SESSION| represents an SSL session that may be resumed in an
 * abbreviated handshake. It is reference-counted and immutable. Once
 * established, an |SSL_SESSION| may be shared by multiple |SSL| objects on
 * different threads and must not be modified. */

DECLARE_LHASH_OF(SSL_SESSION)

/* SSL_SESSION_new returns a newly-allocated blank |SSL_SESSION| or NULL on
 * error. This may be useful in writing tests but otherwise should not be
 * used outside the library. */
OPENSSL_EXPORT SSL_SESSION *SSL_SESSION_new(void);

/* SSL_SESSION_up_ref, if |session| is not NULL, increments the reference count
 * of |session|. It then returns |session|. */
OPENSSL_EXPORT SSL_SESSION *SSL_SESSION_up_ref(SSL_SESSION *session);

/* SSL_SESSION_free decrements the reference count of |session|. If it reaches
 * zero, all data referenced by |session| and |session| itself are released. */
OPENSSL_EXPORT void SSL_SESSION_free(SSL_SESSION *session);

/* SSL_SESSION_to_bytes serializes |in| into a newly allocated buffer and sets
 * |*out_data| to that buffer and |*out_len| to its length. The caller takes
 * ownership of the buffer and must call |OPENSSL_free| when done. It returns
 * one on success and zero on error. */
OPENSSL_EXPORT int SSL_SESSION_to_bytes(const SSL_SESSION *in,
                                        uint8_t **out_data, size_t *out_len);

/* SSL_SESSION_to_bytes_for_ticket serializes |in|, but excludes the session
 * identification information, namely the session ID and ticket. */
OPENSSL_EXPORT int SSL_SESSION_to_bytes_for_ticket(const SSL_SESSION *in,
                                                   uint8_t **out_data,
                                                   size_t *out_len);

/* SSL_SESSION_from_bytes parses |in_len| bytes from |in| as an SSL_SESSION. It
 * returns a newly-allocated |SSL_SESSION| on success or NULL on error. */
OPENSSL_EXPORT SSL_SESSION *SSL_SESSION_from_bytes(const uint8_t *in,
                                                   size_t in_len);

/* SSL_SESSION_get_version returns a string describing the TLS version |session|
 * was established at. For example, "TLSv1.2" or "SSLv3". */
OPENSSL_EXPORT const char *SSL_SESSION_get_version(const SSL_SESSION *session);

/* SSL_SESSION_get_id returns a pointer to a buffer containg |session|'s session
 * ID and sets |*out_len| to its length. */
OPENSSL_EXPORT const uint8_t *SSL_SESSION_get_id(const SSL_SESSION *session,
                                                 unsigned *out_len);

/* SSL_SESSION_get_time returns the time at which |session| was established in
 * seconds since the UNIX epoch. */
OPENSSL_EXPORT long SSL_SESSION_get_time(const SSL_SESSION *session);

/* SSL_SESSION_get_timeout returns the lifetime of |session| in seconds. */
OPENSSL_EXPORT long SSL_SESSION_get_timeout(const SSL_SESSION *session);

/* SSL_SESSION_get_key_exchange_info returns a value that describes the
 * strength of the asymmetric operation that provides confidentiality to
 * |session|. Its interpretation depends on the operation used. See the
 * documentation for this value in the |SSL_SESSION| structure. */
OPENSSL_EXPORT uint32_t SSL_SESSION_get_key_exchange_info(
    const SSL_SESSION *session);

/* SSL_SESSION_get0_peer return's the peer leaf certificate stored in
 * |session|.
 *
 * TODO(davidben): This should return a const X509 *. */
OPENSSL_EXPORT X509 *SSL_SESSION_get0_peer(const SSL_SESSION *session);

/* SSL_SESSION_set_time sets |session|'s creation time to |time| and returns
 * |time|. This function may be useful in writing tests but otherwise should not
 * be used. */
OPENSSL_EXPORT long SSL_SESSION_set_time(SSL_SESSION *session, long time);

/* SSL_SESSION_set_timeout sets |session|'s timeout to |timeout| and returns
 * one. This function may be useful in writing tests but otherwise should not
 * be used. */
OPENSSL_EXPORT long SSL_SESSION_set_timeout(SSL_SESSION *session, long timeout);

/* SSL_SESSION_set1_id_context sets |session|'s session ID context (see
 * |SSL_CTX_set_session_id_context|) to |sid_ctx|. It returns one on success and
 * zero on error. This function may be useful in writing tests but otherwise
 * should not be used. */
OPENSSL_EXPORT int SSL_SESSION_set1_id_context(SSL_SESSION *session,
                                               const uint8_t *sid_ctx,
                                               unsigned sid_ctx_len);


/* Session caching.
 *
 * Session caching allows clients to reconnect to a server based on saved
 * parameters from a previous connection.
 *
 * For a server, the library implements a built-in internal session cache as an
 * in-memory hash table. One may also register callbacks to implement a custom
 * external session cache. An external cache may be used in addition to or
 * instead of the internal one. Use |SSL_CTX_set_session_cache_mode| to toggle
 * the internal cache.
 *
 * For a client, the only option is an external session cache. Prior to
 * handshaking, the consumer should look up a session externally (keyed, for
 * instance, by hostname) and use |SSL_set_session| to configure which session
 * to offer. The callbacks may be used to determine when new sessions are
 * available.
 *
 * Note that offering or accepting a session short-circuits most parameter
 * negotiation. Resuming sessions across different configurations may result in
 * surprising behavor. So, for instance, a client implementing a version
 * fallback should shard its session cache by maximum protocol version. */

/* SSL_SESS_CACHE_OFF disables all session caching. */
#define SSL_SESS_CACHE_OFF 0x0000

/* SSL_SESS_CACHE_CLIENT enables session caching for a client.
 *
 * TODO(davidben): The internal cache is useless on the client. Always act as if
 * SSL_SESS_CACHE_NO_INTERNAL is set. https://crbug.com/531194. Also see TODO
 * attached to |SSL_CTX_sess_set_new_cb|. */
#define SSL_SESS_CACHE_CLIENT 0x0001

/* SSL_SESS_CACHE_SERVER enables session caching for a server. */
#define SSL_SESS_CACHE_SERVER 0x0002

/* SSL_SESS_CACHE_SERVER enables session caching for both client and server. */
#define SSL_SESS_CACHE_BOTH (SSL_SESS_CACHE_CLIENT | SSL_SESS_CACHE_SERVER)

/* SSL_SESS_CACHE_NO_AUTO_CLEAR disables automatically calling
 * |SSL_CTX_flush_sessions| every 255 connections. */
#define SSL_SESS_CACHE_NO_AUTO_CLEAR 0x0080

/* SSL_SESS_CACHE_NO_INTERNAL_LOOKUP disables looking up a session from the
 * internal session cache. */
#define SSL_SESS_CACHE_NO_INTERNAL_LOOKUP 0x0100

/* SSL_SESS_CACHE_NO_INTERNAL_STORE disables storing sessions in the internal
 * session cache. */
#define SSL_SESS_CACHE_NO_INTERNAL_STORE 0x0200

/* SSL_SESS_CACHE_NO_INTERNAL disables the internal session cache. */
#define SSL_SESS_CACHE_NO_INTERNAL \
    (SSL_SESS_CACHE_NO_INTERNAL_LOOKUP | SSL_SESS_CACHE_NO_INTERNAL_STORE)

/* SSL_CTX_set_session_cache_mode sets the session cache mode bits for |ctx| to
 * |mode|. It returns the previous value. */
OPENSSL_EXPORT int SSL_CTX_set_session_cache_mode(SSL_CTX *ctx, int mode);

/* SSL_CTX_get_session_cache_mode returns the session cache mode bits for
 * |ctx| */
OPENSSL_EXPORT int SSL_CTX_get_session_cache_mode(const SSL_CTX *ctx);

/* SSL_set_session, for a client, configures |ssl| to offer to resume |session|
 * in the initial handshake and returns one. The caller retains ownership of
 * |session|. */
OPENSSL_EXPORT int SSL_set_session(SSL *ssl, SSL_SESSION *session);

/* SSL_get_session returns a non-owning pointer to |ssl|'s session. Prior to the
 * initial handshake beginning, this is the session to be offered, set by
 * |SSL_set_session|. After a handshake has finished, this is the currently
 * active session. Its behavior is undefined while a handshake is progress. */
OPENSSL_EXPORT SSL_SESSION *SSL_get_session(const SSL *ssl);

/* SSL_get0_session is an alias for |SSL_get_session|. */
#define SSL_get0_session SSL_get_session

/* SSL_get1_session acts like |SSL_get_session| but returns a new reference to
 * the session. */
OPENSSL_EXPORT SSL_SESSION *SSL_get1_session(SSL *ssl);

/* SSL_DEFAULT_SESSION_TIMEOUT is the default lifetime, in seconds, of a
 * session. */
#define SSL_DEFAULT_SESSION_TIMEOUT (2 * 60 * 60)

/* SSL_CTX_set_timeout sets the lifetime, in seconds, of sessions created in
 * |ctx| to |timeout|. */
OPENSSL_EXPORT long SSL_CTX_set_timeout(SSL_CTX *ctx, long timeout);

/* SSL_CTX_get_timeout returns the lifetime, in seconds, of sessions created in
 * |ctx|. */
OPENSSL_EXPORT long SSL_CTX_get_timeout(const SSL_CTX *ctx);

/* SSL_CTX_set_session_id_context sets |ctx|'s session ID context to |sid_ctx|.
 * It returns one on success and zero on error. The session ID context is an
 * application-defined opaque byte string. A session will not be used in a
 * connection without a matching session ID context.
 *
 * For a server, if |SSL_VERIFY_PEER| is enabled, it is an error to not set a
 * session ID context.
 *
 * TODO(davidben): Is that check needed? That seems a special case of taking
 * care not to cross-resume across configuration changes, and this is only
 * relevant if a server requires client auth. */
OPENSSL_EXPORT int SSL_CTX_set_session_id_context(SSL_CTX *ctx,
                                                  const uint8_t *sid_ctx,
                                                  unsigned sid_ctx_len);

/* SSL_set_session_id_context sets |ssl|'s session ID context to |sid_ctx|. It
 * returns one on success and zero on error. See also
 * |SSL_CTX_set_session_id_context|. */
OPENSSL_EXPORT int SSL_set_session_id_context(SSL *ssl, const uint8_t *sid_ctx,
                                              unsigned sid_ctx_len);

/* SSL_SESSION_CACHE_MAX_SIZE_DEFAULT is the default maximum size of a session
 * cache. */
#define SSL_SESSION_CACHE_MAX_SIZE_DEFAULT (1024 * 20)

/* SSL_CTX_sess_set_cache_size sets the maximum size of |ctx|'s internal session
 * cache to |size|. It returns the previous value. */
OPENSSL_EXPORT unsigned long SSL_CTX_sess_set_cache_size(SSL_CTX *ctx,
                                                         unsigned long size);

/* SSL_CTX_sess_get_cache_size returns the maximum size of |ctx|'s internal
 * session cache. */
OPENSSL_EXPORT unsigned long SSL_CTX_sess_get_cache_size(const SSL_CTX *ctx);

/* SSL_CTX_sessions returns |ctx|'s internal session cache. */
OPENSSL_EXPORT LHASH_OF(SSL_SESSION) *SSL_CTX_sessions(SSL_CTX *ctx);

/* SSL_CTX_sess_number returns the number of sessions in |ctx|'s internal
 * session cache. */
OPENSSL_EXPORT size_t SSL_CTX_sess_number(const SSL_CTX *ctx);

/* SSL_CTX_add_session inserts |session| into |ctx|'s internal session cache. It
 * returns one on success and zero on error or if |ctx| already included a
 * session with that session ID. The caller retains its reference to
 * |session|. */
OPENSSL_EXPORT int SSL_CTX_add_session(SSL_CTX *ctx, SSL_SESSION *session);

/* SSL_CTX_remove_session removes |session| from |ctx|'s internal session cache.
 * It returns one on success and zero on error or if no session with a matching
 * ID was found. */
OPENSSL_EXPORT int SSL_CTX_remove_session(SSL_CTX *ctx, SSL_SESSION *session);

/* SSL_CTX_flush_sessions removes all sessions from |ctx| which have expired as
 * of time |time|. If |time| is zero, all sessions are removed. */
OPENSSL_EXPORT void SSL_CTX_flush_sessions(SSL_CTX *ctx, long time);

/* SSL_CTX_sess_set_new_cb sets the callback to be called when a new session is
 * established and ready to be cached. If the session cache is disabled (the
 * appropriate one of |SSL_SESS_CACHE_CLIENT| or |SSL_SESS_CACHE_SERVER| is
 * unset), the callback is not called.
 *
 * The callback is passed a reference to |session|. It returns one if it takes
 * ownership and zero otherwise.
 *
 * Note: For a client, the callback may be called on abbreviated handshakes if a
 * ticket is renewed. Further, it may not be called until some time after
 * |SSL_do_handshake| or |SSL_connect| completes if False Start is enabled. Thus
 * it's recommended to use this callback over checking |SSL_session_reused| on
 * handshake completion.
 *
 * TODO(davidben): Conditioning callbacks on |SSL_SESS_CACHE_CLIENT| or
 * |SSL_SESS_CACHE_SERVER| doesn't make any sense when one could just as easily
 * not supply the callbacks. Removing that condition and the client internal
 * cache would simplify things. */
OPENSSL_EXPORT void SSL_CTX_sess_set_new_cb(
    SSL_CTX *ctx, int (*new_session_cb)(SSL *ssl, SSL_SESSION *session));

/* SSL_CTX_sess_get_new_cb returns the callback set by
 * |SSL_CTX_sess_set_new_cb|. */
OPENSSL_EXPORT int (*SSL_CTX_sess_get_new_cb(SSL_CTX *ctx))(
    SSL *ssl, SSL_SESSION *session);

/* SSL_CTX_sess_set_remove_cb sets a callback which is called when a session is
 * removed from the internal session cache.
 *
 * TODO(davidben): What is the point of this callback? It seems useless since it
 * only fires on sessions in the internal cache. */
OPENSSL_EXPORT void SSL_CTX_sess_set_remove_cb(
    SSL_CTX *ctx,
    void (*remove_session_cb)(SSL_CTX *ctx, SSL_SESSION *session));

/* SSL_CTX_sess_get_remove_cb returns the callback set by
 * |SSL_CTX_sess_set_remove_cb|. */
OPENSSL_EXPORT void (*SSL_CTX_sess_get_remove_cb(SSL_CTX *ctx))(
    SSL_CTX *ctx, SSL_SESSION *session);

/* SSL_CTX_sess_set_get_cb sets a callback to look up a session by ID for a
 * server. The callback is passed the session ID and should return a matching
 * |SSL_SESSION| or NULL if not found. It should set |*out_copy| to zero and
 * return a new reference to the session. This callback is not used for a
 * client.
 *
 * For historical reasons, if |*out_copy| is set to one (default), the SSL
 * library will take a new reference to the returned |SSL_SESSION|, expecting
 * the callback to return a non-owning pointer. This is not recommended. If
 * |ctx| and thus the callback is used on multiple threads, the session may be
 * removed and invalidated before the SSL library calls |SSL_SESSION_up_ref|,
 * whereas the callback may synchronize internally.
 *
 * To look up a session asynchronously, the callback may return
 * |SSL_magic_pending_session_ptr|. See the documentation for that function and
 * |SSL_ERROR_PENDING_SESSION|.
 *
 * If the internal session cache is enabled, the callback is only consulted if
 * the internal cache does not return a match. */
OPENSSL_EXPORT void SSL_CTX_sess_set_get_cb(
    SSL_CTX *ctx,
    SSL_SESSION *(*get_session_cb)(SSL *ssl, uint8_t *id, int id_len,
                                   int *out_copy));

/* SSL_CTX_sess_get_get_cb returns the callback set by
 * |SSL_CTX_sess_set_get_cb|. */
OPENSSL_EXPORT SSL_SESSION *(*SSL_CTX_sess_get_get_cb(SSL_CTX *ctx))(
    SSL *ssl, uint8_t *id, int id_len, int *out_copy);

/* SSL_magic_pending_session_ptr returns a magic |SSL_SESSION|* which indicates
 * that the session isn't currently unavailable. |SSL_get_error| will then
 * return |SSL_ERROR_PENDING_SESSION| and the handshake can be retried later
 * when the lookup has completed. */
OPENSSL_EXPORT SSL_SESSION *SSL_magic_pending_session_ptr(void);

/* GEN_SESSION_CB is a callback to generate session IDs for |ssl|. It returns
 * one on success and zero on error. On success, the generated ID is written to
 * |id| and |*id_len| set to the length. On entry, |*id_len| is the maximum
 * length of the ID, but the callback may shorten it if desired. It is an error
 * for the callback to set the size to zero.
 *
 * Callbacks may use |SSL_has_matching_session_id| to check that the generated
 * ID is unique. */
typedef int (*GEN_SESSION_CB)(const SSL *ssl, uint8_t *id, unsigned *id_len);

/* SSL_CTX_set_generate_session_id sets the session ID callback of |ctx| to
 * |cb| and returns one. It will be called on the server when establishing a new
 * session. */
OPENSSL_EXPORT int SSL_CTX_set_generate_session_id(SSL_CTX *ctx,
                                                   GEN_SESSION_CB cb);

/* SSL_set_generate_session_id sets the session ID callback of |ssl| to |cb| and
 * returns one. It will be called on the server when establishing a new
 * session. */
OPENSSL_EXPORT int SSL_set_generate_session_id(SSL *ssl, GEN_SESSION_CB cb);

/* SSL_has_matching_session_id returns one if |ssl|'s session cache has a
 * session of value |id| and zero otherwise. */
OPENSSL_EXPORT int SSL_has_matching_session_id(const SSL *ssl,
                                               const uint8_t *id,
                                               unsigned id_len);


/* Session tickets.
 *
 * Session tickets, from RFC 5077, allow session resumption without server-side
 * state. Session tickets are supported in by default but may be disabled with
 * |SSL_OP_NO_TICKET|.
 *
 * On the client, ticket-based sessions use the same APIs as ID-based tickets.
 * Callers do not need to handle them differently.
 *
 * On the server, tickets are encrypted and authenticated with a secret key. By
 * default, an |SSL_CTX| generates a key on creation. Tickets are minted and
 * processed transparently. The following functions may be used to configure a
 * persistent key or implement more custom behavior. */

/* SSL_CTX_get_tlsext_ticket_keys writes |ctx|'s session ticket key material to
 * |len| bytes of |out|. It returns one on success and zero if |len| is not
 * 48. If |out| is NULL, it returns 48 instead. */
OPENSSL_EXPORT int SSL_CTX_get_tlsext_ticket_keys(SSL_CTX *ctx, void *out,
                                                  size_t len);

/* SSL_CTX_set_tlsext_ticket_keys sets |ctx|'s session ticket key material to
 * |len| bytes of |in|. It returns one on success and zero if |len| is not
 * 48. If |in| is NULL, it returns 48 instead. */
OPENSSL_EXPORT int SSL_CTX_set_tlsext_ticket_keys(SSL_CTX *ctx, const void *in,
                                                  size_t len);

/* SSL_TICKET_KEY_NAME_LEN is the length of the key name prefix of a session
 * ticket. */
#define SSL_TICKET_KEY_NAME_LEN 16

/* SSL_CTX_set_tlsext_ticket_key_cb sets the ticket callback to |callback| and
 * returns one. |callback| will be called when encrypting a new ticket and when
 * decrypting a ticket from the client.
 *
 * In both modes, |ctx| and |hmac_ctx| will already have been initialized with
 * |EVP_CIPHER_CTX_init| and |HMAC_CTX_init|, respectively. |callback|
 * configures |hmac_ctx| with an HMAC digest and key, and configures |ctx|
 * for encryption or decryption, based on the mode.
 *
 * When encrypting a new ticket, |encrypt| will be one. It writes a public
 * 16-byte key name to |key_name| and a fresh IV to |iv|. The output IV length
 * must match |EVP_CIPHER_CTX_iv_length| of the cipher selected. In this mode,
 * |callback| returns 1 on success and -1 on error.
 *
 * When decrypting a ticket, |encrypt| will be zero. |key_name| will point to a
 * 16-byte key name and |iv| points to an IV. The length of the IV consumed must
 * match |EVP_CIPHER_CTX_iv_length| of the cipher selected. In this mode,
 * |callback| returns -1 to abort the handshake, 0 if decrypting the ticket
 * failed, and 1 or 2 on success. If it returns 2, the ticket will be renewed.
 * This may be used to re-key the ticket.
 *
 * WARNING: |callback| wildly breaks the usual return value convention and is
 * called in two different modes. */
OPENSSL_EXPORT int SSL_CTX_set_tlsext_ticket_key_cb(
    SSL_CTX *ctx, int (*callback)(SSL *ssl, uint8_t *key_name, uint8_t *iv,
                                  EVP_CIPHER_CTX *ctx, HMAC_CTX *hmac_ctx,
                                  int encrypt));


/* Elliptic curve Diffie-Hellman.
 *
 * Cipher suites using an ECDHE key exchange perform Diffie-Hellman over an
 * elliptic curve negotiated by both endpoints. See RFC 4492. Only named curves
 * are supported. ECDHE is always enabled, but the curve preferences may be
 * configured with these functions.
 *
 * A client may use |SSL_SESSION_get_key_exchange_info| to determine the curve
 * selected. */

/* SSL_CTX_set1_curves sets the preferred curves for |ctx| to be |curves|. Each
 * element of |curves| should be a curve nid. It returns one on success and
 * zero on failure. */
OPENSSL_EXPORT int SSL_CTX_set1_curves(SSL_CTX *ctx, const int *curves,
                                       size_t curves_len);

/* SSL_set1_curves sets the preferred curves for |ssl| to be |curves|. Each
 * element of |curves| should be a curve nid. It returns one on success and
 * zero on failure. */
OPENSSL_EXPORT int SSL_set1_curves(SSL *ssl, const int *curves,
                                   size_t curves_len);

/* SSL_CTX_set_tmp_ecdh configures |ctx| to use the curve from |ecdh| as the
 * curve for ephemeral ECDH keys. For historical reasons, this API expects an
 * |EC_KEY|, but only the curve is used. It returns one on success and zero on
 * error. If unset, an appropriate curve will be chosen based on curve
 * preferences. (This is recommended.) */
OPENSSL_EXPORT int SSL_CTX_set_tmp_ecdh(SSL_CTX *ctx, const EC_KEY *ec_key);

/* SSL_set_tmp_ecdh configures |ssl| to use the curve from |ecdh| as the curve
 * for ephemeral ECDH keys. For historical reasons, this API expects an
 * |EC_KEY|, but only the curve is used. It returns one on success and zero on
 * error. If unset, an appropriate curve will be chosen based on curve
 * preferences. (This is recommended.) */
OPENSSL_EXPORT int SSL_set_tmp_ecdh(SSL *ssl, const EC_KEY *ec_key);

/* SSL_CTX_set_tmp_ecdh_callback configures |ctx| to use |callback| to determine
 * the curve for ephemeral ECDH keys. |callback| should ignore |is_export| and
 * |keylength| and return an |EC_KEY| of the selected curve or NULL on
 * error. Only the curve is used, so the |EC_KEY| needn't have a generated
 * keypair.
 *
 * If the callback is unset, an appropriate curve will be chosen based on curve
 * preferences. (This is recommended.)
 *
 * WARNING: The caller does not take ownership of the resulting |EC_KEY|, so
 * |callback| must save and release the object elsewhere. */
OPENSSL_EXPORT void SSL_CTX_set_tmp_ecdh_callback(
    SSL_CTX *ctx, EC_KEY *(*callback)(SSL *ssl, int is_export, int keylength));

/* SSL_set_tmp_ecdh_callback configures |ssl| to use |callback| to determine the
 * curve for ephemeral ECDH keys. |callback| should ignore |is_export| and
 * |keylength| and return an |EC_KEY| of the selected curve or NULL on
 * error. Only the curve is used, so the |EC_KEY| needn't have a generated
 * keypair.
 *
 * If the callback is unset, an appropriate curve will be chosen based on curve
 * preferences. (This is recommended.)
 *
 * WARNING: The caller does not take ownership of the resulting |EC_KEY|, so
 * |callback| must save and release the object elsewhere. */
OPENSSL_EXPORT void SSL_set_tmp_ecdh_callback(
    SSL *ssl, EC_KEY *(*callback)(SSL *ssl, int is_export, int keylength));

/* SSL_get_curve_name returns a human-readable name for the elliptic curve
 * specified by the given TLS curve id, or NULL if the curve if unknown. */
OPENSSL_EXPORT const char *SSL_get_curve_name(uint16_t curve_id);


/* Multiplicative Diffie-Hellman.
 *
 * Cipher suites using a DHE key exchange perform Diffie-Hellman over a
 * multiplicative group selected by the server. These ciphers are disabled for a
 * server unless a group is chosen with one of these functions.
 *
 * A client may use |SSL_SESSION_get_key_exchange_info| to determine the size of
 * the selected group's prime, but note that servers may select degenerate
 * groups. */

/* SSL_CTX_set_tmp_dh configures |ctx| to use the group from |dh| as the group
 * for DHE. Only the group is used, so |dh| needn't have a keypair. It returns
 * one on success and zero on error. */
OPENSSL_EXPORT int SSL_CTX_set_tmp_dh(SSL_CTX *ctx, const DH *dh);

/* SSL_set_tmp_dh configures |ssl| to use the group from |dh| as the group for
 * DHE. Only the group is used, so |dh| needn't have a keypair. It returns one
 * on success and zero on error. */
OPENSSL_EXPORT int SSL_set_tmp_dh(SSL *ssl, const DH *dh);

/* SSL_CTX_set_tmp_dh_callback configures |ctx| to use |callback| to determine
 * the group for DHE ciphers. |callback| should ignore |is_export| and
 * |keylength| and return a |DH| of the selected group or NULL on error. Only
 * the parameters are used, so the |DH| needn't have a generated keypair.
 *
 * WARNING: The caller does not take ownership of the resulting |DH|, so
 * |callback| must save and release the object elsewhere. */
OPENSSL_EXPORT void SSL_CTX_set_tmp_dh_callback(
    SSL_CTX *ctx, DH *(*callback)(SSL *ssl, int is_export, int keylength));

/* SSL_set_tmp_dh_callback configures |ssl| to use |callback| to determine the
 * group for DHE ciphers. |callback| should ignore |is_export| and |keylength|
 * and return a |DH| of the selected group or NULL on error. Only the
 * parameters are used, so the |DH| needn't have a generated keypair.
 *
 * WARNING: The caller does not take ownership of the resulting |DH|, so
 * |callback| must save and release the object elsewhere. */
OPENSSL_EXPORT void SSL_set_tmp_dh_callback(SSL *ssl,
                                            DH *(*dh)(SSL *ssl, int is_export,
                                                      int keylength));


/* Certificate verification.
 *
 * SSL may authenticate either endpoint with an X.509 certificate. Typically
 * this is used to authenticate the server to the client. These functions
 * configure certificate verification.
 *
 * WARNING: By default, certificate verification errors on a client are not
 * fatal. See |SSL_VERIFY_NONE| This may be configured with
 * |SSL_CTX_set_verify|.
 *
 * By default clients are anonymous but a server may request a certificate from
 * the client by setting |SSL_VERIFY_PEER|.
 *
 * Many of these functions use OpenSSL's legacy X.509 stack which is
 * underdocumented and deprecated, but the replacement isn't ready yet. For
 * now, consumers may use the existing stack or bypass it by performing
 * certificate verification externally. This may be done with
 * |SSL_CTX_set_cert_verify_callback| or by extracting the chain with
 * |SSL_get_peer_cert_chain| after the handshake. In the future, functions will
 * be added to use the SSL stack without dependency on any part of the legacy
 * X.509 and ASN.1 stack.
 *
 * To augment certificate verification, a client may also enable OCSP stapling
 * (RFC 6066) and Certificate Transparency (RFC 6962) extensions. */

/* SSL_VERIFY_NONE, on a client, verifies the server certificate but does not
 * make errors fatal. The result may be checked with |SSL_get_verify_result|. On
 * a server it does not request a client certificate. This is the default. */
#define SSL_VERIFY_NONE 0x00

/* SSL_VERIFY_PEER, on a client, makes server certificate errors fatal. On a
 * server it requests a client certificate and makes errors fatal. However,
 * anonymous clients are still allowed. See
 * |SSL_VERIFY_FAIL_IF_NO_PEER_CERT|. */
#define SSL_VERIFY_PEER 0x01

/* SSL_VERIFY_FAIL_IF_NO_PEER_CERT configures a server to reject connections if
 * the client declines to send a certificate. Otherwise |SSL_VERIFY_PEER| still
 * allows anonymous clients. */
#define SSL_VERIFY_FAIL_IF_NO_PEER_CERT 0x02

/* SSL_VERIFY_PEER_IF_NO_OBC configures a server to request a client certificate
 * if and only if Channel ID is not negotiated. */
#define SSL_VERIFY_PEER_IF_NO_OBC 0x04

/* SSL_CTX_set_verify configures certificate verification behavior. |mode| is
 * one of the |SSL_VERIFY_*| values defined above. |callback|, if not NULL, is
 * used to customize certificate verification. See the behavior of
 * |X509_STORE_CTX_set_verify_cb|.
 *
 * The callback may use |SSL_get_ex_data_X509_STORE_CTX_idx| with
 * |X509_STORE_CTX_get_ex_data| to look up the |SSL| from |store_ctx|. */
OPENSSL_EXPORT void SSL_CTX_set_verify(
    SSL_CTX *ctx, int mode, int (*callback)(int ok, X509_STORE_CTX *store_ctx));

/* SSL_set_verify configures certificate verification behavior. |mode| is one of
 * the |SSL_VERIFY_*| values defined above. |callback|, if not NULL, is used to
 * customize certificate verification. See the behavior of
 * |X509_STORE_CTX_set_verify_cb|.
 *
 * The callback may use |SSL_get_ex_data_X509_STORE_CTX_idx| with
 * |X509_STORE_CTX_get_ex_data| to look up the |SSL| from |store_ctx|. */
OPENSSL_EXPORT void SSL_set_verify(SSL *ssl, int mode,
                                   int (*callback)(int ok,
                                                   X509_STORE_CTX *store_ctx));

/* SSL_CTX_get_verify_mode returns |ctx|'s verify mode, set by
 * |SSL_CTX_set_verify|. */
OPENSSL_EXPORT int SSL_CTX_get_verify_mode(const SSL_CTX *ctx);

/* SSL_get_verify_mode returns |ssl|'s verify mode, set by |SSL_CTX_set_verify|
 * or |SSL_set_verify|. */
OPENSSL_EXPORT int SSL_get_verify_mode(const SSL *ssl);

/* SSL_CTX_get_verify_callback returns the callback set by
 * |SSL_CTX_set_verify|. */
OPENSSL_EXPORT int (*SSL_CTX_get_verify_callback(const SSL_CTX *ctx))(
    int ok, X509_STORE_CTX *store_ctx);

/* SSL_get_verify_callback returns the callback set by |SSL_CTX_set_verify| or
 * |SSL_set_verify|. */
OPENSSL_EXPORT int (*SSL_get_verify_callback(const SSL *ssl))(
    int ok, X509_STORE_CTX *store_ctx);

/* SSL_CTX_set_verify_depth sets the maximum depth of a certificate chain
 * accepted in verification. This number does not include the leaf, so a depth
 * of 1 allows the leaf and one CA certificate. */
OPENSSL_EXPORT void SSL_CTX_set_verify_depth(SSL_CTX *ctx, int depth);

/* SSL_set_verify_depth sets the maximum depth of a certificate chain accepted
 * in verification. This number does not include the leaf, so a depth of 1
 * allows the leaf and one CA certificate. */
OPENSSL_EXPORT void SSL_set_verify_depth(SSL *ssl, int depth);

/* SSL_CTX_get_verify_depth returns the maximum depth of a certificate accepted
 * in verification. */
OPENSSL_EXPORT int SSL_CTX_get_verify_depth(const SSL_CTX *ctx);

/* SSL_get_verify_depth returns the maximum depth of a certificate accepted in
 * verification. */
OPENSSL_EXPORT int SSL_get_verify_depth(const SSL *ssl);

/* SSL_CTX_set1_param sets verification parameters from |param|. It returns one
 * on success and zero on failure. The caller retains ownership of |param|. */
OPENSSL_EXPORT int SSL_CTX_set1_param(SSL_CTX *ctx,
                                      const X509_VERIFY_PARAM *param);

/* SSL_set1_param sets verification parameters from |param|. It returns one on
 * success and zero on failure. The caller retains ownership of |param|. */
OPENSSL_EXPORT int SSL_set1_param(SSL *ssl,
                                  const X509_VERIFY_PARAM *param);

/* SSL_CTX_get0_param returns |ctx|'s |X509_VERIFY_PARAM| for certificate
 * verification. The caller must not release the returned pointer but may call
 * functions on it to configure it. */
OPENSSL_EXPORT X509_VERIFY_PARAM *SSL_CTX_get0_param(SSL_CTX *ctx);

/* SSL_get0_param returns |ssl|'s |X509_VERIFY_PARAM| for certificate
 * verification. The caller must not release the returned pointer but may call
 * functions on it to configure it. */
OPENSSL_EXPORT X509_VERIFY_PARAM *SSL_get0_param(SSL *ssl);

/* SSL_CTX_set_purpose sets |ctx|'s |X509_VERIFY_PARAM|'s 'purpose' parameter to
 * |purpose|. It returns one on success and zero on error. */
OPENSSL_EXPORT int SSL_CTX_set_purpose(SSL_CTX *ctx, int purpose);

/* SSL_set_purpose sets |ssl|'s |X509_VERIFY_PARAM|'s 'purpose' parameter to
 * |purpose|. It returns one on success and zero on error. */
OPENSSL_EXPORT int SSL_set_purpose(SSL *ssl, int purpose);

/* SSL_CTX_set_trust sets |ctx|'s |X509_VERIFY_PARAM|'s 'trust' parameter to
 * |trust|. It returns one on success and zero on error. */
OPENSSL_EXPORT int SSL_CTX_set_trust(SSL_CTX *ctx, int trust);

/* SSL_set_trust sets |ssl|'s |X509_VERIFY_PARAM|'s 'trust' parameter to
 * |trust|. It returns one on success and zero on error. */
OPENSSL_EXPORT int SSL_set_trust(SSL *ssl, int trust);

/* SSL_CTX_set_cert_store sets |ctx|'s certificate store to |store|. It takes
 * ownership of |store|. The store is used for certificate verification.
 *
 * The store is also used for the auto-chaining feature, but this is deprecated.
 * See also |SSL_MODE_NO_AUTO_CHAIN|. */
OPENSSL_EXPORT void SSL_CTX_set_cert_store(SSL_CTX *ctx, X509_STORE *store);

/* SSL_CTX_get_cert_store returns |ctx|'s certificate store. */
OPENSSL_EXPORT X509_STORE *SSL_CTX_get_cert_store(const SSL_CTX *ctx);

/* SSL_CTX_set_default_verify_paths loads the OpenSSL system-default trust
 * anchors into |ctx|'s store. It returns one on success and zero on failure. */
OPENSSL_EXPORT int SSL_CTX_set_default_verify_paths(SSL_CTX *ctx);

/* SSL_CTX_load_verify_locations loads trust anchors into |ctx|'s store from
 * |ca_file| and |ca_dir|, either of which may be NULL. If |ca_file| is passed,
 * it is opened and PEM-encoded CA certificates are read. If |ca_dir| is passed,
 * it is treated as a directory in OpenSSL's hashed directory format. It returns
 * one on success and zero on failure.
 *
 * See
 * https://www.openssl.org/docs/manmaster/ssl/SSL_CTX_load_verify_locations.html
 * for documentation on the directory format. */
OPENSSL_EXPORT int SSL_CTX_load_verify_locations(SSL_CTX *ctx,
                                                 const char *ca_file,
                                                 const char *ca_dir);

/* SSL_get_verify_result returns the result of certificate verification. It is
 * either |X509_V_OK| or a |X509_V_ERR_*| value. */
OPENSSL_EXPORT long SSL_get_verify_result(const SSL *ssl);

/* SSL_set_verify_result overrides the result of certificate verification. */
OPENSSL_EXPORT void SSL_set_verify_result(SSL *ssl, long result);

/* SSL_get_ex_data_X509_STORE_CTX_idx returns the ex_data index used to look up
 * the |SSL| associated with an |X509_STORE_CTX| in the verify callback. */
OPENSSL_EXPORT int SSL_get_ex_data_X509_STORE_CTX_idx(void);

/* SSL_CTX_set_cert_verify_callback sets a custom callback to be called on
 * certificate verification rather than |X509_verify_cert|. |store_ctx| contains
 * the verification parameters. The callback should return one on success and
 * zero on fatal error. It may use |X509_STORE_CTX_set_error| to set a
 * verification result.
 *
 * The callback may use either the |arg| parameter or
 * |SSL_get_ex_data_X509_STORE_CTX_idx| to recover the associated |SSL|
 * object. */
OPENSSL_EXPORT void SSL_CTX_set_cert_verify_callback(
    SSL_CTX *ctx, int (*callback)(X509_STORE_CTX *store_ctx, void *arg),
    void *arg);

/* SSL_enable_signed_cert_timestamps causes |ssl| (which must be the client end
 * of a connection) to request SCTs from the server. See
 * https://tools.ietf.org/html/rfc6962. It returns one. */
OPENSSL_EXPORT int SSL_enable_signed_cert_timestamps(SSL *ssl);

/* SSL_CTX_enable_signed_cert_timestamps enables SCT requests on all client SSL
 * objects created from |ctx|. */
OPENSSL_EXPORT void SSL_CTX_enable_signed_cert_timestamps(SSL_CTX *ctx);

/* SSL_get0_signed_cert_timestamp_list sets |*out| and |*out_len| to point to
 * |*out_len| bytes of SCT information from the server. This is only valid if
 * |ssl| is a client. The SCT information is a SignedCertificateTimestampList
 * (including the two leading length bytes).
 * See https://tools.ietf.org/html/rfc6962#section-3.3
 * If no SCT was received then |*out_len| will be zero on return.
 *
 * WARNING: the returned data is not guaranteed to be well formed. */
OPENSSL_EXPORT void SSL_get0_signed_cert_timestamp_list(const SSL *ssl,
                                                        const uint8_t **out,
                                                        size_t *out_len);

/* SSL_enable_ocsp_stapling causes |ssl| (which must be the client end of a
 * connection) to request a stapled OCSP response from the server. It returns
 * one. */
OPENSSL_EXPORT int SSL_enable_ocsp_stapling(SSL *ssl);

/* SSL_CTX_enable_ocsp_stapling enables OCSP stapling on all client SSL objects
 * created from |ctx|. */
OPENSSL_EXPORT void SSL_CTX_enable_ocsp_stapling(SSL_CTX *ctx);

/* SSL_get0_ocsp_response sets |*out| and |*out_len| to point to |*out_len|
 * bytes of an OCSP response from the server. This is the DER encoding of an
 * OCSPResponse type as defined in RFC 2560.
 *
 * WARNING: the returned data is not guaranteed to be well formed. */
OPENSSL_EXPORT void SSL_get0_ocsp_response(const SSL *ssl, const uint8_t **out,
                                           size_t *out_len);


/* Client certificate CA list.
 *
 * When requesting a client certificate, a server may advertise a list of
 * certificate authorities which are accepted. These functions may be used to
 * configure this list. */

/* SSL_set_client_CA_list sets |ssl|'s client certificate CA list to
 * |name_list|. It takes ownership of |name_list|. */
OPENSSL_EXPORT void SSL_set_client_CA_list(SSL *ssl,
                                           STACK_OF(X509_NAME) *name_list);

/* SSL_CTX_set_client_CA_list sets |ctx|'s client certificate CA list to
 * |name_list|. It takes ownership of |name_list|. */
OPENSSL_EXPORT void SSL_CTX_set_client_CA_list(SSL_CTX *ctx,
                                               STACK_OF(X509_NAME) *name_list);

/* SSL_get_client_CA_list returns |ssl|'s client certificate CA list. */
OPENSSL_EXPORT STACK_OF(X509_NAME) *SSL_get_client_CA_list(const SSL *ssl);

/* SSL_CTX_get_client_CA_list returns |ctx|'s client certificate CA list. */
OPENSSL_EXPORT STACK_OF(X509_NAME) *
    SSL_CTX_get_client_CA_list(const SSL_CTX *ctx);

/* SSL_add_client_CA appends |x509|'s subject to the client certificate CA list.
 * It returns one on success or zero on error. The caller retains ownership of
 * |x509|. */
OPENSSL_EXPORT int SSL_add_client_CA(SSL *ssl, X509 *x509);

/* SSL_CTX_add_client_CA appends |x509|'s subject to the client certificate CA
 * list. It returns one on success or zero on error. The caller retains
 * ownership of |x509|. */
OPENSSL_EXPORT int SSL_CTX_add_client_CA(SSL_CTX *ctx, X509 *x509);

/* SSL_load_client_CA_file opens |file| and reads PEM-encoded certificates from
 * it. It returns a newly-allocated stack of the certificate subjects or NULL
 * on error. */
OPENSSL_EXPORT STACK_OF(X509_NAME) *SSL_load_client_CA_file(const char *file);

/* SSL_dup_CA_list makes a deep copy of |list|. It returns the new list on
 * success or NULL on allocation error. */
OPENSSL_EXPORT STACK_OF(X509_NAME) *SSL_dup_CA_list(STACK_OF(X509_NAME) *list);

/* SSL_add_file_cert_subjects_to_stack behaves like |SSL_load_client_CA_file|
 * but appends the result to |out|. It returns one on success or zero on
 * error. */
OPENSSL_EXPORT int SSL_add_file_cert_subjects_to_stack(STACK_OF(X509_NAME) *out,
                                                       const char *file);

/* SSL_add_dir_cert_subjects_to_stack lists files in directory |dir|. It calls
 * |SSL_add_file_cert_subjects_to_stack| on each file and returns one on success
 * or zero on error. */
OPENSSL_EXPORT int SSL_add_dir_cert_subjects_to_stack(STACK_OF(X509_NAME) *out,
                                                      const char *dir);


/* Application-layer protocol negotation.
 *
 * The ALPN extension (RFC 7301) allows negotiating different application-layer
 * protocols over a single port. This is used, for example, to negotiate
 * HTTP/2. */

/* SSL_CTX_set_alpn_protos sets the client ALPN protocol list on |ctx| to
 * |protos|. |protos| must be in wire-format (i.e. a series of non-empty, 8-bit
 * length-prefixed strings). It returns zero on success and one on failure.
 * Configuring this list enables ALPN on a client.
 *
 * WARNING: this function is dangerous because it breaks the usual return value
 * convention. */
OPENSSL_EXPORT int SSL_CTX_set_alpn_protos(SSL_CTX *ctx, const uint8_t *protos,
                                           unsigned protos_len);

/* SSL_set_alpn_protos sets the client ALPN protocol list on |ssl| to |protos|.
 * |protos| must be in wire-format (i.e. a series of non-empty, 8-bit
 * length-prefixed strings). It returns zero on success and one on failure.
 * Configuring this list enables ALPN on a client.
 *
 * WARNING: this function is dangerous because it breaks the usual return value
 * convention. */
OPENSSL_EXPORT int SSL_set_alpn_protos(SSL *ssl, const uint8_t *protos,
                                       unsigned protos_len);

/* SSL_CTX_set_alpn_select_cb sets a callback function on |ctx| that is called
 * during ClientHello processing in order to select an ALPN protocol from the
 * client's list of offered protocols. Configuring this callback enables ALPN on
 * a server.
 *
 * The callback is passed a wire-format (i.e. a series of non-empty, 8-bit
 * length-prefixed strings) ALPN protocol list in |in|. It should set |*out| and
 * |*out_len| to the selected protocol and return |SSL_TLSEXT_ERR_OK| on
 * success. It does not pass ownership of the buffer. Otherwise, it should
 * return |SSL_TLSEXT_ERR_NOACK|. Other |SSL_TLSEXT_ERR_*| values are
 * unimplemented and will be treated as |SSL_TLSEXT_ERR_NOACK|. */
OPENSSL_EXPORT void SSL_CTX_set_alpn_select_cb(
    SSL_CTX *ctx, int (*cb)(SSL *ssl, const uint8_t **out, uint8_t *out_len,
                            const uint8_t *in, unsigned in_len, void *arg),
    void *arg);

/* SSL_get0_alpn_selected gets the selected ALPN protocol (if any) from |ssl|.
 * On return it sets |*out_data| to point to |*out_len| bytes of protocol name
 * (not including the leading length-prefix byte). If the server didn't respond
 * with a negotiated protocol then |*out_len| will be zero. */
OPENSSL_EXPORT void SSL_get0_alpn_selected(const SSL *ssl,
                                           const uint8_t **out_data,
                                           unsigned *out_len);


/* Next protocol negotiation.
 *
 * The NPN extension (draft-agl-tls-nextprotoneg-03) is the predecessor to ALPN
 * and deprecated in favor of it. */

/* SSL_CTX_set_next_protos_advertised_cb sets a callback that is called when a
 * TLS server needs a list of supported protocols for Next Protocol
 * Negotiation. The returned list must be in wire format. The list is returned
 * by setting |*out| to point to it and |*out_len| to its length. This memory
 * will not be modified, but one should assume that |ssl| keeps a reference to
 * it.
 *
 * The callback should return |SSL_TLSEXT_ERR_OK| if it wishes to advertise.
 * Otherwise, no such extension will be included in the ServerHello. */
OPENSSL_EXPORT void SSL_CTX_set_next_protos_advertised_cb(
    SSL_CTX *ctx,
    int (*cb)(SSL *ssl, const uint8_t **out, unsigned *out_len, void *arg),
    void *arg);

/* SSL_CTX_set_next_proto_select_cb sets a callback that is called when a client
 * needs to select a protocol from the server's provided list. |*out| must be
 * set to point to the selected protocol (which may be within |in|). The length
 * of the protocol name must be written into |*out_len|. The server's advertised
 * protocols are provided in |in| and |in_len|. The callback can assume that
 * |in| is syntactically valid.
 *
 * The client must select a protocol. It is fatal to the connection if this
 * callback returns a value other than |SSL_TLSEXT_ERR_OK|.
 *
 * Configuring this callback enables NPN on a client. */
OPENSSL_EXPORT void SSL_CTX_set_next_proto_select_cb(
    SSL_CTX *ctx, int (*cb)(SSL *ssl, uint8_t **out, uint8_t *out_len,
                            const uint8_t *in, unsigned in_len, void *arg),
    void *arg);

/* SSL_get0_next_proto_negotiated sets |*out_data| and |*out_len| to point to
 * the client's requested protocol for this connection. If the client didn't
 * request any protocol, then |*out_data| is set to NULL.
 *
 * Note that the client can request any protocol it chooses. The value returned
 * from this function need not be a member of the list of supported protocols
 * provided by the server. */
OPENSSL_EXPORT void SSL_get0_next_proto_negotiated(const SSL *ssl,
                                                   const uint8_t **out_data,
                                                   unsigned *out_len);

/* SSL_select_next_proto implements the standard protocol selection. It is
 * expected that this function is called from the callback set by
 * |SSL_CTX_set_next_proto_select_cb|.
 *
 * The protocol data is assumed to be a vector of 8-bit, length prefixed byte
 * strings. The length byte itself is not included in the length. A byte
 * string of length 0 is invalid. No byte string may be truncated.
 *
 * The current, but experimental algorithm for selecting the protocol is:
 *
 * 1) If the server doesn't support NPN then this is indicated to the
 * callback. In this case, the client application has to abort the connection
 * or have a default application level protocol.
 *
 * 2) If the server supports NPN, but advertises an empty list then the
 * client selects the first protcol in its list, but indicates via the
 * API that this fallback case was enacted.
 *
 * 3) Otherwise, the client finds the first protocol in the server's list
 * that it supports and selects this protocol. This is because it's
 * assumed that the server has better information about which protocol
 * a client should use.
 *
 * 4) If the client doesn't support any of the server's advertised
 * protocols, then this is treated the same as case 2.
 *
 * It returns either |OPENSSL_NPN_NEGOTIATED| if a common protocol was found, or
 * |OPENSSL_NPN_NO_OVERLAP| if the fallback case was reached. */
OPENSSL_EXPORT int SSL_select_next_proto(uint8_t **out, uint8_t *out_len,
                                         const uint8_t *server,
                                         unsigned server_len,
                                         const uint8_t *client,
                                         unsigned client_len);

#define OPENSSL_NPN_UNSUPPORTED 0
#define OPENSSL_NPN_NEGOTIATED 1
#define OPENSSL_NPN_NO_OVERLAP 2


/* Channel ID.
 *
 * See draft-balfanz-tls-channelid-01. */

/* SSL_CTX_enable_tls_channel_id either configures a TLS server to accept TLS
 * Channel IDs from clients, or configures a client to send TLS Channel IDs to
 * a server. It returns one. */
OPENSSL_EXPORT int SSL_CTX_enable_tls_channel_id(SSL_CTX *ctx);

/* SSL_enable_tls_channel_id either configures a TLS server to accept TLS
 * Channel IDs from clients, or configures a client to send TLS Channel IDs to
 * server. It returns one. */
OPENSSL_EXPORT int SSL_enable_tls_channel_id(SSL *ssl);

/* SSL_CTX_set1_tls_channel_id configures a TLS client to send a TLS Channel ID
 * to compatible servers. |private_key| must be a P-256 EC key. It returns one
 * on success and zero on error. */
OPENSSL_EXPORT int SSL_CTX_set1_tls_channel_id(SSL_CTX *ctx,
                                               EVP_PKEY *private_key);

/* SSL_set1_tls_channel_id configures a TLS client to send a TLS Channel ID to
 * compatible servers. |private_key| must be a P-256 EC key. It returns one on
 * success and zero on error. */
OPENSSL_EXPORT int SSL_set1_tls_channel_id(SSL *ssl, EVP_PKEY *private_key);

/* SSL_get_tls_channel_id gets the client's TLS Channel ID from a server |SSL*|
 * and copies up to the first |max_out| bytes into |out|. The Channel ID
 * consists of the client's P-256 public key as an (x,y) pair where each is a
 * 32-byte, big-endian field element. It returns 0 if the client didn't offer a
 * Channel ID and the length of the complete Channel ID otherwise. */
OPENSSL_EXPORT size_t SSL_get_tls_channel_id(SSL *ssl, uint8_t *out,
                                             size_t max_out);

/* SSL_CTX_set_channel_id_cb sets a callback to be called when a TLS Channel ID
 * is requested. The callback may set |*out_pkey| to a key, passing a reference
 * to the caller. If none is returned, the handshake will pause and
 * |SSL_get_error| will return |SSL_ERROR_WANT_CHANNEL_ID_LOOKUP|.
 *
 * See also |SSL_ERROR_WANT_CHANNEL_ID_LOOKUP|. */
OPENSSL_EXPORT void SSL_CTX_set_channel_id_cb(
    SSL_CTX *ctx, void (*channel_id_cb)(SSL *ssl, EVP_PKEY **out_pkey));

/* SSL_CTX_get_channel_id_cb returns the callback set by
 * |SSL_CTX_set_channel_id_cb|. */
OPENSSL_EXPORT void (*SSL_CTX_get_channel_id_cb(SSL_CTX *ctx))(
    SSL *ssl, EVP_PKEY **out_pkey);


/* DTLS-SRTP.
 *
 * See RFC 5764. */

/* An SRTP_PROTECTION_PROFILE is an SRTP profile for use with the use_srtp
 * extension. */
struct srtp_protection_profile_st {
  const char *name;
  unsigned long id;
} /* SRTP_PROTECTION_PROFILE */;

DECLARE_STACK_OF(SRTP_PROTECTION_PROFILE)

/* SRTP_* define constants for SRTP profiles. */
#define SRTP_AES128_CM_SHA1_80 0x0001
#define SRTP_AES128_CM_SHA1_32 0x0002
#define SRTP_AES128_F8_SHA1_80 0x0003
#define SRTP_AES128_F8_SHA1_32 0x0004
#define SRTP_NULL_SHA1_80      0x0005
#define SRTP_NULL_SHA1_32      0x0006

/* SSL_CTX_set_srtp_profiles enables SRTP for all SSL objects created from
 * |ctx|. |profile| contains a colon-separated list of profile names. It returns
 * one on success and zero on failure. */
OPENSSL_EXPORT int SSL_CTX_set_srtp_profiles(SSL_CTX *ctx,
                                             const char *profiles);

/* SSL_set_srtp_profiles enables SRTP for |ssl|.  |profile| contains a
 * colon-separated list of profile names. It returns one on success and zero on
 * failure. */
OPENSSL_EXPORT int SSL_set_srtp_profiles(SSL *ssl, const char *profiles);

/* SSL_get_srtp_profiles returns the SRTP profiles supported by |ssl|. */
OPENSSL_EXPORT STACK_OF(SRTP_PROTECTION_PROFILE) *SSL_get_srtp_profiles(
    SSL *ssl);

/* SSL_get_selected_srtp_profile returns the selected SRTP profile, or NULL if
 * SRTP was not negotiated. */
OPENSSL_EXPORT const SRTP_PROTECTION_PROFILE *SSL_get_selected_srtp_profile(
    SSL *ssl);


/* Pre-shared keys.
 *
 * Connections may be configured with PSK (Pre-Shared Key) cipher suites. These
 * authenticate using out-of-band pre-shared keys rather than certificates. See
 * RFC 4279.
 *
 * This implementation uses NUL-terminated C strings for identities and identity
 * hints, so values with a NUL character are not supported. (RFC 4279 does not
 * specify the format of an identity.) */

/* PSK_MAX_IDENTITY_LEN is the maximum supported length of a PSK identity,
 * excluding the NUL terminator. */
#define PSK_MAX_IDENTITY_LEN 128

/* PSK_MAX_PSK_LEN is the maximum supported length of a pre-shared key. */
#define PSK_MAX_PSK_LEN 256

/* SSL_CTX_set_psk_client_callback sets the callback to be called when PSK is
 * negotiated on the client. This callback must be set to enable PSK cipher
 * suites on the client.
 *
 * The callback is passed the identity hint in |hint| or NULL if none was
 * provided. It should select a PSK identity and write the identity and the
 * corresponding PSK to |identity| and |psk|, respectively. The identity is
 * written as a NUL-terminated C string of length (excluding the NUL terminator)
 * at most |max_identity_len|. The PSK's length must be at most |max_psk_len|.
 * The callback returns the length of the PSK or 0 if no suitable identity was
 * found. */
OPENSSL_EXPORT void SSL_CTX_set_psk_client_callback(
    SSL_CTX *ctx,
    unsigned (*psk_client_callback)(
        SSL *ssl, const char *hint, char *identity,
        unsigned max_identity_len, uint8_t *psk, unsigned max_psk_len));

/* SSL_set_psk_client_callback sets the callback to be called when PSK is
 * negotiated on the client. This callback must be set to enable PSK cipher
 * suites on the client. See also |SSL_CTX_set_psk_client_callback|. */
OPENSSL_EXPORT void SSL_set_psk_client_callback(
    SSL *ssl, unsigned (*psk_client_callback)(SSL *ssl, const char *hint,
                                              char *identity,
                                              unsigned max_identity_len,
                                              uint8_t *psk,
                                              unsigned max_psk_len));

/* SSL_CTX_set_psk_server_callback sets the callback to be called when PSK is
 * negotiated on the server. This callback must be set to enable PSK cipher
 * suites on the server.
 *
 * The callback is passed the identity in |identity|. It should write a PSK of
 * length at most |max_psk_len| to |psk| and return the number of bytes written
 * or zero if the PSK identity is unknown. */
OPENSSL_EXPORT void SSL_CTX_set_psk_server_callback(
    SSL_CTX *ctx,
    unsigned (*psk_server_callback)(SSL *ssl, const char *identity,
                                    uint8_t *psk,
                                    unsigned max_psk_len));

/* SSL_set_psk_server_callback sets the callback to be called when PSK is
 * negotiated on the server. This callback must be set to enable PSK cipher
 * suites on the server. See also |SSL_CTX_set_psk_server_callback|. */
OPENSSL_EXPORT void SSL_set_psk_server_callback(
    SSL *ssl,
    unsigned (*psk_server_callback)(SSL *ssl, const char *identity,
                                    uint8_t *psk,
                                    unsigned max_psk_len));

/* SSL_CTX_use_psk_identity_hint configures server connections to advertise an
 * identity hint of |identity_hint|. It returns one on success and zero on
 * error. */
OPENSSL_EXPORT int SSL_CTX_use_psk_identity_hint(SSL_CTX *ctx,
                                                 const char *identity_hint);

/* SSL_use_psk_identity_hint configures server connections to advertise an
 * identity hint of |identity_hint|. It returns one on success and zero on
 * error. */
OPENSSL_EXPORT int SSL_use_psk_identity_hint(SSL *ssl,
                                             const char *identity_hint);

/* SSL_get_psk_identity_hint returns the PSK identity hint advertised for |ssl|
 * or NULL if there is none. */
OPENSSL_EXPORT const char *SSL_get_psk_identity_hint(const SSL *ssl);

/* SSL_get_psk_identity, after the handshake completes, returns the PSK identity
 * that was negotiated by |ssl| or NULL if PSK was not used. */
OPENSSL_EXPORT const char *SSL_get_psk_identity(const SSL *ssl);


/* ex_data functions.
 *
 * See |ex_data.h| for details. */

OPENSSL_EXPORT int SSL_set_ex_data(SSL *ssl, int idx, void *data);
OPENSSL_EXPORT void *SSL_get_ex_data(const SSL *ssl, int idx);
OPENSSL_EXPORT int SSL_get_ex_new_index(long argl, void *argp,
                                        CRYPTO_EX_new *new_func,
                                        CRYPTO_EX_dup *dup_func,
                                        CRYPTO_EX_free *free_func);

OPENSSL_EXPORT int SSL_SESSION_set_ex_data(SSL_SESSION *session, int idx,
                                           void *data);
OPENSSL_EXPORT void *SSL_SESSION_get_ex_data(const SSL_SESSION *session,
                                             int idx);
OPENSSL_EXPORT int SSL_SESSION_get_ex_new_index(long argl, void *argp,
                                                CRYPTO_EX_new *new_func,
                                                CRYPTO_EX_dup *dup_func,
                                                CRYPTO_EX_free *free_func);

OPENSSL_EXPORT int SSL_CTX_set_ex_data(SSL_CTX *ctx, int idx, void *data);
OPENSSL_EXPORT void *SSL_CTX_get_ex_data(const SSL_CTX *ctx, int idx);
OPENSSL_EXPORT int SSL_CTX_get_ex_new_index(long argl, void *argp,
                                            CRYPTO_EX_new *new_func,
                                            CRYPTO_EX_dup *dup_func,
                                            CRYPTO_EX_free *free_func);


/* Obscure functions. */

/* SSL_get_rc4_state sets |*read_key| and |*write_key| to the RC4 states for
 * the read and write directions. It returns one on success or zero if |ssl|
 * isn't using an RC4-based cipher suite. */
OPENSSL_EXPORT int SSL_get_rc4_state(const SSL *ssl, const RC4_KEY **read_key,
                                     const RC4_KEY **write_key);

/* SSL_get_structure_sizes returns the sizes of the SSL, SSL_CTX and
 * SSL_SESSION structures so that a test can ensure that outside code agrees on
 * these values. */
OPENSSL_EXPORT void SSL_get_structure_sizes(size_t *ssl_size,
                                            size_t *ssl_ctx_size,
                                            size_t *ssl_session_size);

/* SSL_CTX_set_msg_callback installs |cb| as the message callback for |ctx|.
 * This callback will be called when sending or receiving low-level record
 * headers, complete handshake messages, ChangeCipherSpec, and alerts.
 * |write_p| is one for outgoing messages and zero for incoming messages.
 *
 * For each record header, |cb| is called with |version| = 0 and |content_type|
 * = |SSL3_RT_HEADER|. The |len| bytes from |buf| contain the header. Note that
 * this does not include the record body. If the record is sealed, the length
 * in the header is the length of the ciphertext.
 *
 * For each handshake message, ChangeCipherSpec, and alert, |version| is the
 * protocol version and |content_type| is the corresponding record type. The
 * |len| bytes from |buf| contain the handshake message, one-byte
 * ChangeCipherSpec body, and two-byte alert, respectively. */
OPENSSL_EXPORT void SSL_CTX_set_msg_callback(
    SSL_CTX *ctx, void (*cb)(int write_p, int version, int content_type,
                             const void *buf, size_t len, SSL *ssl, void *arg));

/* SSL_CTX_set_msg_callback_arg sets the |arg| parameter of the message
 * callback. */
OPENSSL_EXPORT void SSL_CTX_set_msg_callback_arg(SSL_CTX *ctx, void *arg);

/* SSL_set_msg_callback installs |cb| as the message callback of |ssl|. See
 * |SSL_CTX_set_msg_callback| for when this callback is called. */
OPENSSL_EXPORT void SSL_set_msg_callback(
    SSL *ssl, void (*cb)(int write_p, int version, int content_type,
                         const void *buf, size_t len, SSL *ssl, void *arg));

/* SSL_set_msg_callback_arg sets the |arg| parameter of the message callback. */
OPENSSL_EXPORT void SSL_set_msg_callback_arg(SSL *ssl, void *arg);

/* SSL_CTX_set_keylog_bio sets configures all SSL objects attached to |ctx| to
 * log session material to |keylog_bio|. This is intended for debugging use
 * with tools like Wireshark. |ctx| takes ownership of |keylog_bio|.
 *
 * The format is described in
 * https://developer.mozilla.org/en-US/docs/Mozilla/Projects/NSS/Key_Log_Format. */
OPENSSL_EXPORT void SSL_CTX_set_keylog_bio(SSL_CTX *ctx, BIO *keylog_bio);

/* SSL_set_reject_peer_renegotiations controls whether renegotiation attempts by
 * the peer are rejected. It may be set at any point in a connection's lifetime
 * to control future renegotiations programmatically. By default, renegotiations
 * are rejected. (Renegotiations requested by a client are always rejected.) */
OPENSSL_EXPORT void SSL_set_reject_peer_renegotiations(SSL *ssl, int reject);


/* Underdocumented functions.
 *
 * Functions below here haven't been touched up and may be underdocumented. */

/* SSLeay version number for ASN.1 encoding of the session information */
/* Version 0 - initial version
 * Version 1 - added the optional peer certificate. */
#define SSL_SESSION_ASN1_VERSION 0x0001

#define SSL_MAX_SSL_SESSION_ID_LENGTH 32
#define SSL_MAX_SID_CTX_LENGTH 32
#define SSL_MAX_MASTER_KEY_LENGTH 48

/* These are used to specify which ciphers to use and not to use */

#define SSL_TXT_MEDIUM "MEDIUM"
#define SSL_TXT_HIGH "HIGH"
#define SSL_TXT_FIPS "FIPS"

#define SSL_TXT_kRSA "kRSA"
#define SSL_TXT_kDHE "kDHE"
#define SSL_TXT_kEDH "kEDH" /* same as "kDHE" */
#define SSL_TXT_kECDHE "kECDHE"
#define SSL_TXT_kEECDH "kEECDH" /* same as "kECDHE" */
#define SSL_TXT_kPSK "kPSK"

#define SSL_TXT_aRSA "aRSA"
#define SSL_TXT_aECDSA "aECDSA"
#define SSL_TXT_aPSK "aPSK"

#define SSL_TXT_DH "DH"
#define SSL_TXT_DHE "DHE" /* same as "kDHE" */
#define SSL_TXT_EDH "EDH" /* same as "DHE" */
#define SSL_TXT_RSA "RSA"
#define SSL_TXT_ECDH "ECDH"
#define SSL_TXT_ECDHE "ECDHE" /* same as "kECDHE" */
#define SSL_TXT_EECDH "EECDH" /* same as "ECDHE" */
#define SSL_TXT_ECDSA "ECDSA"
#define SSL_TXT_PSK "PSK"

#define SSL_TXT_3DES "3DES"
#define SSL_TXT_RC4 "RC4"
#define SSL_TXT_AES128 "AES128"
#define SSL_TXT_AES256 "AES256"
#define SSL_TXT_AES "AES"
#define SSL_TXT_AES_GCM "AESGCM"
#define SSL_TXT_CHACHA20 "CHACHA20"

#define SSL_TXT_MD5 "MD5"
#define SSL_TXT_SHA1 "SHA1"
#define SSL_TXT_SHA "SHA" /* same as "SHA1" */
#define SSL_TXT_SHA256 "SHA256"
#define SSL_TXT_SHA384 "SHA384"

#define SSL_TXT_SSLV3 "SSLv3"
#define SSL_TXT_TLSV1 "TLSv1"
#define SSL_TXT_TLSV1_1 "TLSv1.1"
#define SSL_TXT_TLSV1_2 "TLSv1.2"

#define SSL_TXT_ALL "ALL"

#define SSL_TXT_CMPDEF "COMPLEMENTOFDEFAULT"

/* Used in SSL_set_shutdown()/SSL_get_shutdown(); */
#define SSL_SENT_SHUTDOWN 1
#define SSL_RECEIVED_SHUTDOWN 2

typedef struct ssl_protocol_method_st SSL_PROTOCOL_METHOD;
typedef struct ssl_conf_ctx_st SSL_CONF_CTX;
typedef struct ssl3_enc_method SSL3_ENC_METHOD;

/* SSL_set_mtu sets the |ssl|'s MTU in DTLS to |mtu|. It returns one on success
 * and zero on failure. */
OPENSSL_EXPORT int SSL_set_mtu(SSL *ssl, unsigned mtu);

struct ssl_aead_ctx_st;
typedef struct ssl_aead_ctx_st SSL_AEAD_CTX;

#define SSL_MAX_CERT_LIST_DEFAULT 1024 * 100 /* 100k max cert list */

/* ssl_early_callback_ctx is passed to certain callbacks that are called very
 * early on during the server handshake. At this point, much of the SSL* hasn't
 * been filled out and only the ClientHello can be depended on. */
struct ssl_early_callback_ctx {
  SSL *ssl;
  const uint8_t *client_hello;
  size_t client_hello_len;
  const uint8_t *session_id;
  size_t session_id_len;
  const uint8_t *cipher_suites;
  size_t cipher_suites_len;
  const uint8_t *compression_methods;
  size_t compression_methods_len;
  const uint8_t *extensions;
  size_t extensions_len;
};

/* SSL_early_callback_ctx_extension_get searches the extensions in |ctx| for an
 * extension of the given type. If not found, it returns zero. Otherwise it
 * sets |out_data| to point to the extension contents (not including the type
 * and length bytes), sets |out_len| to the length of the extension contents
 * and returns one. */
OPENSSL_EXPORT char SSL_early_callback_ctx_extension_get(
    const struct ssl_early_callback_ctx *ctx, uint16_t extension_type,
    const uint8_t **out_data, size_t *out_len);

OPENSSL_EXPORT void SSL_CTX_set_info_callback(SSL_CTX *ctx,
                                              void (*cb)(const SSL *ssl,
                                                         int type, int val));
OPENSSL_EXPORT void (*SSL_CTX_get_info_callback(SSL_CTX *ctx))(const SSL *ssl,
                                                               int type,
                                                               int val);
OPENSSL_EXPORT void SSL_CTX_set_client_cert_cb(
    SSL_CTX *ctx,
    int (*client_cert_cb)(SSL *ssl, X509 **x509, EVP_PKEY **pkey));
OPENSSL_EXPORT int (*SSL_CTX_get_client_cert_cb(SSL_CTX *ctx))(SSL *ssl,
                                                               X509 **x509,
                                                               EVP_PKEY **pkey);

#define SSL_NOTHING 1
#define SSL_WRITING 2
#define SSL_READING 3
#define SSL_X509_LOOKUP 4
#define SSL_CHANNEL_ID_LOOKUP 5
#define SSL_PENDING_SESSION 7
#define SSL_CERTIFICATE_SELECTION_PENDING 8
#define SSL_PRIVATE_KEY_OPERATION 9

/* These will only be used when doing non-blocking IO */
#define SSL_want_nothing(s) (SSL_want(s) == SSL_NOTHING)
#define SSL_want_read(s) (SSL_want(s) == SSL_READING)
#define SSL_want_write(s) (SSL_want(s) == SSL_WRITING)
#define SSL_want_x509_lookup(s) (SSL_want(s) == SSL_X509_LOOKUP)
#define SSL_want_channel_id_lookup(s) (SSL_want(s) == SSL_CHANNEL_ID_LOOKUP)
#define SSL_want_session(s) (SSL_want(s) == SSL_PENDING_SESSION)
#define SSL_want_certificate(s) \
  (SSL_want(s) == SSL_CERTIFICATE_SELECTION_PENDING)
#define SSL_want_private_key_operation(s) \
  (SSL_want(s) == SSL_PRIVATE_KEY_OPERATION)

/* The following are the possible values for ssl->state are are used to
 * indicate where we are up to in the SSL connection establishment. The macros
 * that follow are about the only things you should need to use and even then,
 * only when using non-blocking IO. It can also be useful to work out where you
 * were when the connection failed */

#define SSL_ST_CONNECT 0x1000
#define SSL_ST_ACCEPT 0x2000
#define SSL_ST_MASK 0x0FFF
#define SSL_ST_INIT (SSL_ST_CONNECT | SSL_ST_ACCEPT)
#define SSL_ST_OK 0x03
#define SSL_ST_RENEGOTIATE (0x04 | SSL_ST_INIT)

#define SSL_CB_LOOP 0x01
#define SSL_CB_EXIT 0x02
#define SSL_CB_READ 0x04
#define SSL_CB_WRITE 0x08
#define SSL_CB_ALERT 0x4000 /* used in callback */
#define SSL_CB_READ_ALERT (SSL_CB_ALERT | SSL_CB_READ)
#define SSL_CB_WRITE_ALERT (SSL_CB_ALERT | SSL_CB_WRITE)
#define SSL_CB_ACCEPT_LOOP (SSL_ST_ACCEPT | SSL_CB_LOOP)
#define SSL_CB_ACCEPT_EXIT (SSL_ST_ACCEPT | SSL_CB_EXIT)
#define SSL_CB_CONNECT_LOOP (SSL_ST_CONNECT | SSL_CB_LOOP)
#define SSL_CB_CONNECT_EXIT (SSL_ST_CONNECT | SSL_CB_EXIT)
#define SSL_CB_HANDSHAKE_START 0x10
#define SSL_CB_HANDSHAKE_DONE 0x20

/* Is the SSL_connection established? */
#define SSL_get_state(a) SSL_state(a)
#define SSL_is_init_finished(a) (SSL_state(a) == SSL_ST_OK)
#define SSL_in_init(a) (SSL_state(a) & SSL_ST_INIT)
#define SSL_in_connect_init(a) (SSL_state(a) & SSL_ST_CONNECT)
#define SSL_in_accept_init(a) (SSL_state(a) & SSL_ST_ACCEPT)

/* SSL_in_false_start returns one if |s| has a pending unfinished handshake that
 * is in False Start. |SSL_write| may be called at this point without waiting
 * for the peer, but |SSL_read| will require the handshake to be completed. */
OPENSSL_EXPORT int SSL_in_false_start(const SSL *s);

/* Obtain latest Finished message
 *   -- that we sent (SSL_get_finished)
 *   -- that we expected from peer (SSL_get_peer_finished).
 * Returns length (0 == no Finished so far), copies up to 'count' bytes. */
OPENSSL_EXPORT size_t SSL_get_finished(const SSL *s, void *buf, size_t count);
OPENSSL_EXPORT size_t SSL_get_peer_finished(const SSL *s, void *buf, size_t count);

#define d2i_SSL_SESSION_bio(bp, s_id) \
  ASN1_d2i_bio_of(SSL_SESSION, SSL_SESSION_new, d2i_SSL_SESSION, bp, s_id)
#define i2d_SSL_SESSION_bio(bp, s_id) \
  ASN1_i2d_bio_of(SSL_SESSION, i2d_SSL_SESSION, bp, s_id)

DECLARE_PEM_rw(SSL_SESSION, SSL_SESSION)

/* make_errors.go reserves error codes above 1000 for manually-assigned errors.
 * This value must be kept in sync with reservedReasonCode in make_errors.h */
#define SSL_AD_REASON_OFFSET \
  1000 /* offset to get SSL_R_... value from SSL_AD_... */

/* These alert types are for SSLv3 and TLSv1 */
#define SSL_AD_CLOSE_NOTIFY SSL3_AD_CLOSE_NOTIFY
#define SSL_AD_UNEXPECTED_MESSAGE SSL3_AD_UNEXPECTED_MESSAGE /* fatal */
#define SSL_AD_BAD_RECORD_MAC SSL3_AD_BAD_RECORD_MAC         /* fatal */
#define SSL_AD_DECRYPTION_FAILED TLS1_AD_DECRYPTION_FAILED
#define SSL_AD_RECORD_OVERFLOW TLS1_AD_RECORD_OVERFLOW
#define SSL_AD_DECOMPRESSION_FAILURE SSL3_AD_DECOMPRESSION_FAILURE /* fatal */
#define SSL_AD_HANDSHAKE_FAILURE SSL3_AD_HANDSHAKE_FAILURE         /* fatal */
#define SSL_AD_NO_CERTIFICATE SSL3_AD_NO_CERTIFICATE /* Not for TLS */
#define SSL_AD_BAD_CERTIFICATE SSL3_AD_BAD_CERTIFICATE
#define SSL_AD_UNSUPPORTED_CERTIFICATE SSL3_AD_UNSUPPORTED_CERTIFICATE
#define SSL_AD_CERTIFICATE_REVOKED SSL3_AD_CERTIFICATE_REVOKED
#define SSL_AD_CERTIFICATE_EXPIRED SSL3_AD_CERTIFICATE_EXPIRED
#define SSL_AD_CERTIFICATE_UNKNOWN SSL3_AD_CERTIFICATE_UNKNOWN
#define SSL_AD_ILLEGAL_PARAMETER SSL3_AD_ILLEGAL_PARAMETER /* fatal */
#define SSL_AD_UNKNOWN_CA TLS1_AD_UNKNOWN_CA               /* fatal */
#define SSL_AD_ACCESS_DENIED TLS1_AD_ACCESS_DENIED         /* fatal */
#define SSL_AD_DECODE_ERROR TLS1_AD_DECODE_ERROR           /* fatal */
#define SSL_AD_DECRYPT_ERROR TLS1_AD_DECRYPT_ERROR
#define SSL_AD_EXPORT_RESTRICTION TLS1_AD_EXPORT_RESTRICTION       /* fatal */
#define SSL_AD_PROTOCOL_VERSION TLS1_AD_PROTOCOL_VERSION           /* fatal */
#define SSL_AD_INSUFFICIENT_SECURITY TLS1_AD_INSUFFICIENT_SECURITY /* fatal */
#define SSL_AD_INTERNAL_ERROR TLS1_AD_INTERNAL_ERROR               /* fatal */
#define SSL_AD_USER_CANCELLED TLS1_AD_USER_CANCELLED
#define SSL_AD_NO_RENEGOTIATION TLS1_AD_NO_RENEGOTIATION
#define SSL_AD_UNSUPPORTED_EXTENSION TLS1_AD_UNSUPPORTED_EXTENSION
#define SSL_AD_CERTIFICATE_UNOBTAINABLE TLS1_AD_CERTIFICATE_UNOBTAINABLE
#define SSL_AD_UNRECOGNIZED_NAME TLS1_AD_UNRECOGNIZED_NAME
#define SSL_AD_BAD_CERTIFICATE_STATUS_RESPONSE \
  TLS1_AD_BAD_CERTIFICATE_STATUS_RESPONSE
#define SSL_AD_BAD_CERTIFICATE_HASH_VALUE TLS1_AD_BAD_CERTIFICATE_HASH_VALUE
#define SSL_AD_UNKNOWN_PSK_IDENTITY TLS1_AD_UNKNOWN_PSK_IDENTITY     /* fatal */
#define SSL_AD_INAPPROPRIATE_FALLBACK SSL3_AD_INAPPROPRIATE_FALLBACK /* fatal */

/* DTLSv1_get_timeout queries the next DTLS handshake timeout. If there is a
 * timeout in progress, it sets |*out| to the time remaining and returns one.
 * Otherwise, it returns zero.
 *
 * When the timeout expires, call |DTLSv1_handle_timeout| to handle the
 * retransmit behavior.
 *
 * NOTE: This function must be queried again whenever the handshake state
 * machine changes, including when |DTLSv1_handle_timeout| is called. */
OPENSSL_EXPORT int DTLSv1_get_timeout(const SSL *ssl, struct timeval *out);

/* DTLSv1_handle_timeout is called when a DTLS handshake timeout expires. If no
 * timeout had expired, it returns 0. Otherwise, it retransmits the previous
 * flight of handshake messages and returns 1. If too many timeouts had expired
 * without progress or an error occurs, it returns -1.
 *
 * NOTE: The caller's external timer should be compatible with the one |ssl|
 * queries within some fudge factor. Otherwise, the call will be a no-op, but
 * |DTLSv1_get_timeout| will return an updated timeout.
 *
 * WARNING: This function breaks the usual return value convention. */
OPENSSL_EXPORT int DTLSv1_handle_timeout(SSL *ssl);

/* SSL_total_renegotiations returns the total number of renegotiation handshakes
 * peformed by |ssl|. This includes the pending renegotiation, if any. */
OPENSSL_EXPORT int SSL_total_renegotiations(const SSL *ssl);

/* SSL_get0_certificate_types, for a client, sets |*out_types| to an array
 * containing the client certificate types requested by a server. It returns the
 * length of the array. */
OPENSSL_EXPORT size_t SSL_get0_certificate_types(SSL *ssl,
                                                 const uint8_t **out_types);

OPENSSL_EXPORT int SSL_want(const SSL *s);

OPENSSL_EXPORT int SSL_get_fd(const SSL *s);
OPENSSL_EXPORT int SSL_get_rfd(const SSL *s);
OPENSSL_EXPORT int SSL_get_wfd(const SSL *s);
OPENSSL_EXPORT int SSL_pending(const SSL *s);
OPENSSL_EXPORT int SSL_set_fd(SSL *s, int fd);
OPENSSL_EXPORT int SSL_set_rfd(SSL *s, int fd);
OPENSSL_EXPORT int SSL_set_wfd(SSL *s, int fd);

OPENSSL_EXPORT const char *SSL_state_string(const SSL *s);
OPENSSL_EXPORT const char *SSL_state_string_long(const SSL *s);

OPENSSL_EXPORT int SSL_SESSION_print_fp(FILE *fp, const SSL_SESSION *ses);
OPENSSL_EXPORT int SSL_SESSION_print(BIO *fp, const SSL_SESSION *ses);

/* SSL_renegotiate_pending returns one if |ssl| is in the middle of a
 * renegotiation. */
OPENSSL_EXPORT int SSL_renegotiate_pending(SSL *ssl);

OPENSSL_EXPORT const char *SSL_alert_type_string_long(int value);
OPENSSL_EXPORT const char *SSL_alert_type_string(int value);
OPENSSL_EXPORT const char *SSL_alert_desc_string_long(int value);
OPENSSL_EXPORT const char *SSL_alert_desc_string(int value);

OPENSSL_EXPORT void SSL_CTX_set_quiet_shutdown(SSL_CTX *ctx, int mode);
OPENSSL_EXPORT int SSL_CTX_get_quiet_shutdown(const SSL_CTX *ctx);
OPENSSL_EXPORT void SSL_set_quiet_shutdown(SSL *ssl, int mode);
OPENSSL_EXPORT int SSL_get_quiet_shutdown(const SSL *ssl);
OPENSSL_EXPORT void SSL_set_shutdown(SSL *ssl, int mode);
OPENSSL_EXPORT int SSL_get_shutdown(const SSL *ssl);
OPENSSL_EXPORT SSL_CTX *SSL_get_SSL_CTX(const SSL *ssl);
OPENSSL_EXPORT SSL_CTX *SSL_set_SSL_CTX(SSL *ssl, SSL_CTX *ctx);
OPENSSL_EXPORT void SSL_set_info_callback(SSL *ssl,
                                          void (*cb)(const SSL *ssl, int type,
                                                     int val));
OPENSSL_EXPORT void (*SSL_get_info_callback(const SSL *ssl))(const SSL *ssl,
                                                             int type, int val);
OPENSSL_EXPORT int SSL_state(const SSL *ssl);

/* SSL_CTX_get_max_cert_list returns the maximum length, in bytes, of a peer
 * certificate chain accepted by |ctx|. */
OPENSSL_EXPORT size_t SSL_CTX_get_max_cert_list(const SSL_CTX *ctx);

/* SSL_CTX_set_max_cert_list sets the maximum length, in bytes, of a peer
 * certificate chain to |max_cert_list|. This affects how much memory may be
 * consumed during the handshake. */
OPENSSL_EXPORT void SSL_CTX_set_max_cert_list(SSL_CTX *ctx,
                                              size_t max_cert_list);

/* SSL_get_max_cert_list returns the maximum length, in bytes, of a peer
 * certificate chain accepted by |ssl|. */
OPENSSL_EXPORT size_t SSL_get_max_cert_list(const SSL *ssl);

/* SSL_set_max_cert_list sets the maximum length, in bytes, of a peer
 * certificate chain to |max_cert_list|. This affects how much memory may be
 * consumed during the handshake. */
OPENSSL_EXPORT void SSL_set_max_cert_list(SSL *ssl, size_t max_cert_list);

/* SSL_CTX_set_max_send_fragment sets the maximum length, in bytes, of records
 * sent by |ctx|. Beyond this length, handshake messages and application data
 * will be split into multiple records. */
OPENSSL_EXPORT void SSL_CTX_set_max_send_fragment(SSL_CTX *ctx,
                                                  size_t max_send_fragment);

/* SSL_set_max_send_fragment sets the maximum length, in bytes, of records
 * sent by |ssl|. Beyond this length, handshake messages and application data
 * will be split into multiple records. */
OPENSSL_EXPORT void SSL_set_max_send_fragment(SSL *ssl,
                                              size_t max_send_fragment);

/* SSL_CTX_set_dos_protection_cb sets a callback that is called once the
 * resumption decision for a ClientHello has been made. It can return 1 to
 * allow the handshake to continue or zero to cause the handshake to abort. */
OPENSSL_EXPORT void SSL_CTX_set_dos_protection_cb(
    SSL_CTX *ctx, int (*cb)(const struct ssl_early_callback_ctx *));


/* Deprecated functions. */

/* SSL_CIPHER_description writes a description of |cipher| into |buf| and
 * returns |buf|. If |buf| is NULL, it returns a newly allocated string, to be
 * freed with |OPENSSL_free|, or NULL on error.
 *
 * The description includes a trailing newline and has the form:
 * AES128-SHA              SSLv3 Kx=RSA      Au=RSA  Enc=AES(128)  Mac=SHA1
 *
 * Consider |SSL_CIPHER_get_name| or |SSL_CIPHER_get_rfc_name| instead. */
OPENSSL_EXPORT const char *SSL_CIPHER_description(const SSL_CIPHER *cipher,
                                                  char *buf, int len);

/* SSL_CIPHER_get_version returns the string "TLSv1/SSLv3". */
OPENSSL_EXPORT const char *SSL_CIPHER_get_version(const SSL_CIPHER *cipher);

typedef void COMP_METHOD;

/* SSL_COMP_get_compression_methods returns NULL. */
OPENSSL_EXPORT COMP_METHOD *SSL_COMP_get_compression_methods(void);

/* SSL_COMP_add_compression_method returns one. */
OPENSSL_EXPORT int SSL_COMP_add_compression_method(int id, COMP_METHOD *cm);

/* SSL_COMP_get_name returns NULL. */
OPENSSL_EXPORT const char *SSL_COMP_get_name(const COMP_METHOD *comp);

/* SSLv23_method calls |TLS_method|. */
OPENSSL_EXPORT const SSL_METHOD *SSLv23_method(void);

/* Version-specific methods behave exactly like |TLS_method| and |DTLS_method|
 * except they also call |SSL_CTX_set_min_version| and |SSL_CTX_set_max_version|
 * to lock connections to that protocol version. */
OPENSSL_EXPORT const SSL_METHOD *SSLv3_method(void);
OPENSSL_EXPORT const SSL_METHOD *TLSv1_method(void);
OPENSSL_EXPORT const SSL_METHOD *TLSv1_1_method(void);
OPENSSL_EXPORT const SSL_METHOD *TLSv1_2_method(void);
OPENSSL_EXPORT const SSL_METHOD *DTLSv1_method(void);
OPENSSL_EXPORT const SSL_METHOD *DTLSv1_2_method(void);

/* Client- and server-specific methods call their corresponding generic
 * methods. */
OPENSSL_EXPORT const SSL_METHOD *SSLv23_server_method(void);
OPENSSL_EXPORT const SSL_METHOD *SSLv23_client_method(void);
OPENSSL_EXPORT const SSL_METHOD *SSLv3_server_method(void);
OPENSSL_EXPORT const SSL_METHOD *SSLv3_client_method(void);
OPENSSL_EXPORT const SSL_METHOD *TLSv1_server_method(void);
OPENSSL_EXPORT const SSL_METHOD *TLSv1_client_method(void);
OPENSSL_EXPORT const SSL_METHOD *TLSv1_1_server_method(void);
OPENSSL_EXPORT const SSL_METHOD *TLSv1_1_client_method(void);
OPENSSL_EXPORT const SSL_METHOD *TLSv1_2_server_method(void);
OPENSSL_EXPORT const SSL_METHOD *TLSv1_2_client_method(void);
OPENSSL_EXPORT const SSL_METHOD *DTLS_server_method(void);
OPENSSL_EXPORT const SSL_METHOD *DTLS_client_method(void);
OPENSSL_EXPORT const SSL_METHOD *DTLSv1_server_method(void);
OPENSSL_EXPORT const SSL_METHOD *DTLSv1_client_method(void);
OPENSSL_EXPORT const SSL_METHOD *DTLSv1_2_server_method(void);
OPENSSL_EXPORT const SSL_METHOD *DTLSv1_2_client_method(void);

/* SSL_clear resets |ssl| to allow another connection and returns one on success
 * or zero on failure. It returns most configuration state but releases memory
 * associated with the current connection.
 *
 * Free |ssl| and create a new one instead. */
OPENSSL_EXPORT int SSL_clear(SSL *ssl);

/* SSL_CTX_set_tmp_rsa_callback does nothing. */
OPENSSL_EXPORT void SSL_CTX_set_tmp_rsa_callback(
    SSL_CTX *ctx, RSA *(*cb)(SSL *ssl, int is_export, int keylength));

/* SSL_set_tmp_rsa_callback does nothing. */
OPENSSL_EXPORT void SSL_set_tmp_rsa_callback(SSL *ssl,
                                             RSA *(*cb)(SSL *ssl, int is_export,
                                                        int keylength));

/* SSL_CTX_sess_connect returns zero. */
OPENSSL_EXPORT int SSL_CTX_sess_connect(const SSL_CTX *ctx);

/* SSL_CTX_sess_connect_good returns zero. */
OPENSSL_EXPORT int SSL_CTX_sess_connect_good(const SSL_CTX *ctx);

/* SSL_CTX_sess_connect_renegotiate returns zero. */
OPENSSL_EXPORT int SSL_CTX_sess_connect_renegotiate(const SSL_CTX *ctx);

/* SSL_CTX_sess_accept returns zero. */
OPENSSL_EXPORT int SSL_CTX_sess_accept(const SSL_CTX *ctx);

/* SSL_CTX_sess_accept_renegotiate returns zero. */
OPENSSL_EXPORT int SSL_CTX_sess_accept_renegotiate(const SSL_CTX *ctx);

/* SSL_CTX_sess_accept_good returns zero. */
OPENSSL_EXPORT int SSL_CTX_sess_accept_good(const SSL_CTX *ctx);

/* SSL_CTX_sess_hits returns zero. */
OPENSSL_EXPORT int SSL_CTX_sess_hits(const SSL_CTX *ctx);

/* SSL_CTX_sess_cb_hits returns zero. */
OPENSSL_EXPORT int SSL_CTX_sess_cb_hits(const SSL_CTX *ctx);

/* SSL_CTX_sess_misses returns zero. */
OPENSSL_EXPORT int SSL_CTX_sess_misses(const SSL_CTX *ctx);

/* SSL_CTX_sess_timeouts returns zero. */
OPENSSL_EXPORT int SSL_CTX_sess_timeouts(const SSL_CTX *ctx);

/* SSL_CTX_sess_cache_full returns zero. */
OPENSSL_EXPORT int SSL_CTX_sess_cache_full(const SSL_CTX *ctx);

/* SSL_cutthrough_complete calls |SSL_in_false_start|. */
OPENSSL_EXPORT int SSL_cutthrough_complete(const SSL *s);

/* SSL_num_renegotiations calls |SSL_total_renegotiations|. */
OPENSSL_EXPORT int SSL_num_renegotiations(const SSL *ssl);

/* SSL_CTX_need_tmp_RSA returns zero. */
OPENSSL_EXPORT int SSL_CTX_need_tmp_RSA(const SSL_CTX *ctx);

/* SSL_need_tmp_RSA returns zero. */
OPENSSL_EXPORT int SSL_need_tmp_RSA(const SSL *ssl);

/* SSL_CTX_set_tmp_rsa returns one. */
OPENSSL_EXPORT int SSL_CTX_set_tmp_rsa(SSL_CTX *ctx, const RSA *rsa);

/* SSL_set_tmp_rsa returns one. */
OPENSSL_EXPORT int SSL_set_tmp_rsa(SSL *ssl, const RSA *rsa);

/* SSL_CTX_get_read_ahead returns zero. */
OPENSSL_EXPORT int SSL_CTX_get_read_ahead(const SSL_CTX *ctx);

/* SSL_CTX_set_read_ahead does nothing. */
OPENSSL_EXPORT void SSL_CTX_set_read_ahead(SSL_CTX *ctx, int yes);

/* SSL_get_read_ahead returns zero. */
OPENSSL_EXPORT int SSL_get_read_ahead(const SSL *s);

/* SSL_set_read_ahead does nothing. */
OPENSSL_EXPORT void SSL_set_read_ahead(SSL *s, int yes);

/* SSL_renegotiate put an error on the error queue and returns zero. */
OPENSSL_EXPORT int SSL_renegotiate(SSL *ssl);

/* SSL_set_state does nothing. */
OPENSSL_EXPORT void SSL_set_state(SSL *ssl, int state);

/* SSL_MODE_HANDSHAKE_CUTTHROUGH is the same as SSL_MODE_ENABLE_FALSE_START. */
#define SSL_MODE_HANDSHAKE_CUTTHROUGH SSL_MODE_ENABLE_FALSE_START

/* i2d_SSL_SESSION serializes |in| to the bytes pointed to by |*pp|. On success,
 * it returns the number of bytes written and advances |*pp| by that many bytes.
 * On failure, it returns -1. If |pp| is NULL, no bytes are written and only the
 * length is returned.
 *
 * Use |SSL_SESSION_to_bytes| instead. */
OPENSSL_EXPORT int i2d_SSL_SESSION(SSL_SESSION *in, uint8_t **pp);

/* d2i_SSL_SESSION parses a serialized session from the |length| bytes pointed
 * to by |*pp|. It returns the new |SSL_SESSION| and advances |*pp| by the
 * number of bytes consumed on success and NULL on failure. The caller takes
 * ownership of the new session and must call |SSL_SESSION_free| when done.
 *
 * If |a| is non-NULL, |*a| is released and set the new |SSL_SESSION|.
 *
 * Use |SSL_SESSION_from_bytes| instead. */
OPENSSL_EXPORT SSL_SESSION *d2i_SSL_SESSION(SSL_SESSION **a, const uint8_t **pp,
                                            long length);

/* ERR_load_SSL_strings does nothing. */
OPENSSL_EXPORT void ERR_load_SSL_strings(void);

/* SSL_load_error_strings does nothing. */
OPENSSL_EXPORT void SSL_load_error_strings(void);

/* SSL_CTX_set_tlsext_use_srtp calls |SSL_CTX_set_srtp_profiles|. It returns
 * zero on success and one on failure.
 *
 * WARNING: this function is dangerous because it breaks the usual return value
 * convention. Use |SSL_CTX_set_srtp_profiles| instead. */
OPENSSL_EXPORT int SSL_CTX_set_tlsext_use_srtp(SSL_CTX *ctx,
                                               const char *profiles);

/* SSL_set_tlsext_use_srtp calls |SSL_set_srtp_profiles|. It returns zero on
 * success and one on failure.
 *
 * WARNING: this function is dangerous because it breaks the usual return value
 * convention. Use |SSL_set_srtp_profiles| instead. */
OPENSSL_EXPORT int SSL_set_tlsext_use_srtp(SSL *ssl, const char *profiles);

/* SSL_get_current_compression returns NULL. */
OPENSSL_EXPORT const COMP_METHOD *SSL_get_current_compression(SSL *s);

/* SSL_get_current_expansion returns NULL. */
OPENSSL_EXPORT const COMP_METHOD *SSL_get_current_expansion(SSL *s);

#define SSL_set_app_data(s, arg) (SSL_set_ex_data(s, 0, (char *)arg))
#define SSL_get_app_data(s) (SSL_get_ex_data(s, 0))
#define SSL_SESSION_set_app_data(s, a) \
  (SSL_SESSION_set_ex_data(s, 0, (char *)a))
#define SSL_SESSION_get_app_data(s) (SSL_SESSION_get_ex_data(s, 0))
#define SSL_CTX_get_app_data(ctx) (SSL_CTX_get_ex_data(ctx, 0))
#define SSL_CTX_set_app_data(ctx, arg) \
  (SSL_CTX_set_ex_data(ctx, 0, (char *)arg))

#define OpenSSL_add_ssl_algorithms() SSL_library_init()
#define SSLeay_add_ssl_algorithms() SSL_library_init()

#define SSL_get_cipher(ssl) SSL_CIPHER_get_name(SSL_get_current_cipher(ssl))
#define SSL_get_cipher_bits(ssl, out_alg_bits) \
	  SSL_CIPHER_get_bits(SSL_get_current_cipher(ssl), out_alg_bits)
#define SSL_get_cipher_version(ssl) \
	  SSL_CIPHER_get_version(SSL_get_current_cipher(ssl))
#define SSL_get_cipher_name(ssl) \
	  SSL_CIPHER_get_name(SSL_get_current_cipher(ssl))
#define SSL_get_time(session) SSL_SESSION_get_time(session)
#define SSL_set_time(session, time) SSL_SESSION_set_time((session), (time))
#define SSL_get_timeout(session) SSL_SESSION_get_timeout(session)
#define SSL_set_timeout(session, timeout) \
		SSL_SESSION_set_timeout((session), (timeout))

typedef struct ssl_comp_st SSL_COMP;

struct ssl_comp_st {
  int id;
  const char *name;
  char *method;
};

DECLARE_STACK_OF(SSL_COMP)

/* The following flags toggle individual protocol versions. This is deprecated.
 * Use |SSL_CTX_set_min_version| and |SSL_CTX_set_max_version| instead. */
#define SSL_OP_NO_SSLv3 0x02000000L
#define SSL_OP_NO_TLSv1 0x04000000L
#define SSL_OP_NO_TLSv1_2 0x08000000L
#define SSL_OP_NO_TLSv1_1 0x10000000L
#define SSL_OP_NO_DTLSv1 SSL_OP_NO_TLSv1
#define SSL_OP_NO_DTLSv1_2 SSL_OP_NO_TLSv1_2

/* The following flags do nothing and are included only to make it easier to
 * compile code with BoringSSL. */
#define SSL_MODE_AUTO_RETRY 0
#define SSL_MODE_RELEASE_BUFFERS 0
#define SSL_MODE_SEND_CLIENTHELLO_TIME 0
#define SSL_MODE_SEND_SERVERHELLO_TIME 0
#define SSL_OP_ALLOW_UNSAFE_LEGACY_RENEGOTIATION 0
#define SSL_OP_DONT_INSERT_EMPTY_FRAGMENTS 0
#define SSL_OP_EPHEMERAL_RSA 0
#define SSL_OP_MICROSOFT_SESS_ID_BUG 0
#define SSL_OP_MSIE_SSLV2_RSA_PADDING 0
#define SSL_OP_NETSCAPE_CA_DN_BUG 0
#define SSL_OP_NETSCAPE_CHALLENGE_BUG 0
#define SSL_OP_NETSCAPE_DEMO_CIPHER_CHANGE_BUG 0
#define SSL_OP_NETSCAPE_REUSE_CIPHER_CHANGE_BUG 0
#define SSL_OP_NO_COMPRESSION 0
#define SSL_OP_NO_SESSION_RESUMPTION_ON_RENEGOTIATION 0
#define SSL_OP_NO_SSLv2 0
#define SSL_OP_PKCS1_CHECK_1 0
#define SSL_OP_PKCS1_CHECK_2 0
#define SSL_OP_SINGLE_DH_USE 0
#define SSL_OP_SINGLE_ECDH_USE 0
#define SSL_OP_SSLEAY_080_CLIENT_DH_BUG 0
#define SSL_OP_SSLREF2_REUSE_CERT_TYPE_BUG 0
#define SSL_OP_TLS_BLOCK_PADDING_BUG 0
#define SSL_OP_TLS_ROLLBACK_BUG 0
#define SSL_VERIFY_CLIENT_ONCE 0

/* SSL_cache_hit calls |SSL_session_resumed|. */
OPENSSL_EXPORT int SSL_cache_hit(SSL *ssl);

/* SSL_get_default_timeout returns |SSL_DEFAULT_SESSION_TIMEOUT|. */
OPENSSL_EXPORT long SSL_get_default_timeout(const SSL *ssl);

/* SSL_get_version returns a string describing the TLS version used by |ssl|.
 * For example, "TLSv1.2" or "SSLv3". */
OPENSSL_EXPORT const char *SSL_get_version(const SSL *ssl);

/* SSL_get_cipher_list returns the name of the |n|th cipher in the output of
 * |SSL_get_ciphers| or NULL if out of range. Use |SSL_get_ciphers| insteads. */
OPENSSL_EXPORT const char *SSL_get_cipher_list(const SSL *ssl, int n);


/* Private structures.
 *
 * This structures are exposed for historical reasons, but access to them is
 * deprecated. */

struct ssl_cipher_st {
  /* name is the OpenSSL name for the cipher. */
  const char *name;
  /* id is the cipher suite value bitwise OR-d with 0x03000000. */
  uint32_t id;

  /* The following are internal fields. See ssl/internal.h for their values. */

  uint32_t algorithm_mkey;
  uint32_t algorithm_auth;
  uint32_t algorithm_enc;
  uint32_t algorithm_mac;
  uint32_t algorithm_ssl;
  uint32_t algo_strength;
  uint32_t algorithm_prf;

  /* strength_bits is the strength of the cipher in bits. */
  int strength_bits;
  /* alg_bits is the number of bits of key material used by the algorithm. */
  int alg_bits;
};

struct ssl_session_st {
  int ssl_version; /* what ssl version session info is being kept in here? */

  int master_key_length;
  uint8_t master_key[SSL_MAX_MASTER_KEY_LENGTH];
  /* session_id - valid? */
  unsigned int session_id_length;
  uint8_t session_id[SSL_MAX_SSL_SESSION_ID_LENGTH];
  /* this is used to determine whether the session is being reused in
   * the appropriate context. It is up to the application to set this,
   * via SSL_new */
  unsigned int sid_ctx_length;
  uint8_t sid_ctx[SSL_MAX_SID_CTX_LENGTH];

  char *psk_identity;
  /* Used to indicate that session resumption is not allowed. Applications can
   * also set this bit for a new session via not_resumable_session_cb to
   * disable session caching and tickets. */
  int not_resumable;

  /* peer is the peer's certificate. */
  X509 *peer;

  /* cert_chain is the certificate chain sent by the peer. NOTE: for historical
   * reasons, when a client (so the peer is a server), the chain includes
   * |peer|, but when a server it does not. */
  STACK_OF(X509) *cert_chain;

  /* when app_verify_callback accepts a session where the peer's certificate is
   * not ok, we must remember the error for session reuse: */
  long verify_result; /* only for servers */

  CRYPTO_refcount_t references;
  long timeout;
  long time;

  const SSL_CIPHER *cipher;

  /* key_exchange_info contains an indication of the size of the asymmetric
   * primitive used in the handshake that created this session. In the event
   * that two asymmetric operations are used, this value applies to the one
   * that controls the confidentiality of the connection. Its interpretation
   * depends on the primitive that was used; as specified by the cipher suite:
   *   DHE: the size, in bits, of the multiplicative group.
   *   RSA: the size, in bits, of the modulus.
   *   ECDHE: the TLS id for the curve.
   *
   * A zero indicates that the value is unknown. */
  uint32_t key_exchange_info;

  CRYPTO_EX_DATA ex_data; /* application specific data */

  /* These are used to make removal of session-ids more efficient and to
   * implement a maximum cache size. */
  SSL_SESSION *prev, *next;
  char *tlsext_hostname;
  /* RFC4507 info */
  uint8_t *tlsext_tick;               /* Session ticket */
  size_t tlsext_ticklen;              /* Session ticket length */
  uint32_t tlsext_tick_lifetime_hint; /* Session lifetime hint in seconds */

  size_t tlsext_signed_cert_timestamp_list_length;
  uint8_t *tlsext_signed_cert_timestamp_list; /* Server's list. */

  /* The OCSP response that came with the session. */
  size_t ocsp_response_length;
  uint8_t *ocsp_response;

  char peer_sha256_valid; /* Non-zero if peer_sha256 is valid */
  uint8_t
      peer_sha256[SHA256_DIGEST_LENGTH]; /* SHA256 of peer certificate */

  /* original_handshake_hash contains the handshake hash (either SHA-1+MD5 or
   * SHA-2, depending on TLS version) for the original, full handshake that
   * created a session. This is used by Channel IDs during resumption. */
  uint8_t original_handshake_hash[EVP_MAX_MD_SIZE];
  unsigned int original_handshake_hash_len;

  /* extended_master_secret is true if the master secret in this session was
   * generated using EMS and thus isn't vulnerable to the Triple Handshake
   * attack. */
  char extended_master_secret;
};

/* ssl_cipher_preference_list_st contains a list of SSL_CIPHERs with
 * equal-preference groups. For TLS clients, the groups are moot because the
 * server picks the cipher and groups cannot be expressed on the wire. However,
 * for servers, the equal-preference groups allow the client's preferences to
 * be partially respected. (This only has an effect with
 * SSL_OP_CIPHER_SERVER_PREFERENCE).
 *
 * The equal-preference groups are expressed by grouping SSL_CIPHERs together.
 * All elements of a group have the same priority: no ordering is expressed
 * within a group.
 *
 * The values in |ciphers| are in one-to-one correspondence with
 * |in_group_flags|. (That is, sk_SSL_CIPHER_num(ciphers) is the number of
 * bytes in |in_group_flags|.) The bytes in |in_group_flags| are either 1, to
 * indicate that the corresponding SSL_CIPHER is not the last element of a
 * group, or 0 to indicate that it is.
 *
 * For example, if |in_group_flags| contains all zeros then that indicates a
 * traditional, fully-ordered preference. Every SSL_CIPHER is the last element
 * of the group (i.e. they are all in a one-element group).
 *
 * For a more complex example, consider:
 *   ciphers:        A  B  C  D  E  F
 *   in_group_flags: 1  1  0  0  1  0
 *
 * That would express the following, order:
 *
 *    A         E
 *    B -> D -> F
 *    C
 */
struct ssl_cipher_preference_list_st {
  STACK_OF(SSL_CIPHER) *ciphers;
  uint8_t *in_group_flags;
};

struct ssl_ctx_st {
  const SSL_PROTOCOL_METHOD *method;

  /* lock is used to protect various operations on this object. */
  CRYPTO_MUTEX lock;

  /* max_version is the maximum acceptable protocol version. If zero, the
   * maximum supported version, currently (D)TLS 1.2, is used. */
  uint16_t max_version;

  /* min_version is the minimum acceptable protocl version. If zero, the
   * minimum supported version, currently SSL 3.0 and DTLS 1.0, is used */
  uint16_t min_version;

  struct ssl_cipher_preference_list_st *cipher_list;
  /* same as above but sorted for lookup */
  STACK_OF(SSL_CIPHER) *cipher_list_by_id;

  /* cipher_list_tls10 is the list of ciphers when TLS 1.0 or greater is in
   * use. This only applies to server connections as, for clients, the version
   * number is known at connect time and so the cipher list can be set then. If
   * |cipher_list_tls11| is non-NULL then this applies only to TLS 1.0
   * connections.
   *
   * TODO(agl): this exists to assist in the death of SSLv3. It can hopefully
   * be removed after that. */
  struct ssl_cipher_preference_list_st *cipher_list_tls10;

  /* cipher_list_tls11 is the list of ciphers when TLS 1.1 or greater is in
   * use. This only applies to server connections as, for clients, the version
   * number is known at connect time and so the cipher list can be set then. */
  struct ssl_cipher_preference_list_st *cipher_list_tls11;

  X509_STORE *cert_store;
  LHASH_OF(SSL_SESSION) *sessions;
  /* Most session-ids that will be cached, default is
   * SSL_SESSION_CACHE_MAX_SIZE_DEFAULT. 0 is unlimited. */
  unsigned long session_cache_size;
  SSL_SESSION *session_cache_head;
  SSL_SESSION *session_cache_tail;

  /* handshakes_since_cache_flush is the number of successful handshakes since
   * the last cache flush. */
  int handshakes_since_cache_flush;

  /* This can have one of 2 values, ored together,
   * SSL_SESS_CACHE_CLIENT,
   * SSL_SESS_CACHE_SERVER,
   * Default is SSL_SESSION_CACHE_SERVER, which means only
   * SSL_accept which cache SSL_SESSIONS. */
  int session_cache_mode;

  /* If timeout is not 0, it is the default timeout value set when SSL_new() is
   * called.  This has been put in to make life easier to set things up */
  long session_timeout;

  /* If this callback is not null, it will be called each time a session id is
   * added to the cache.  If this function returns 1, it means that the
   * callback will do a SSL_SESSION_free() when it has finished using it.
   * Otherwise, on 0, it means the callback has finished with it. If
   * remove_session_cb is not null, it will be called when a session-id is
   * removed from the cache.  After the call, OpenSSL will SSL_SESSION_free()
   * it. */
  int (*new_session_cb)(SSL *ssl, SSL_SESSION *sess);
  void (*remove_session_cb)(SSL_CTX *ctx, SSL_SESSION *sess);
  SSL_SESSION *(*get_session_cb)(SSL *ssl, uint8_t *data, int len,
                                 int *copy);

  CRYPTO_refcount_t references;

  /* if defined, these override the X509_verify_cert() calls */
  int (*app_verify_callback)(X509_STORE_CTX *store_ctx, void *arg);
  void *app_verify_arg;

  /* Default password callback. */
  pem_password_cb *default_passwd_callback;

  /* Default password callback user data. */
  void *default_passwd_callback_userdata;

  /* get client cert callback */
  int (*client_cert_cb)(SSL *ssl, X509 **x509, EVP_PKEY **pkey);

  /* get channel id callback */
  void (*channel_id_cb)(SSL *ssl, EVP_PKEY **out_pkey);

  CRYPTO_EX_DATA ex_data;

  /* custom_*_extensions stores any callback sets for custom extensions. Note
   * that these pointers will be NULL if the stack would otherwise be empty. */
  STACK_OF(SSL_CUSTOM_EXTENSION) *client_custom_extensions;
  STACK_OF(SSL_CUSTOM_EXTENSION) *server_custom_extensions;

  /* Default values used when no per-SSL value is defined follow */

  void (*info_callback)(const SSL *ssl, int type,
                        int val); /* used if SSL's info_callback is NULL */

  /* what we put in client cert requests */
  STACK_OF(X509_NAME) *client_CA;


  /* Default values to use in SSL structures follow (these are copied by
   * SSL_new) */

  uint32_t options;
  uint32_t mode;
  uint32_t max_cert_list;

  struct cert_st /* CERT */ *cert;

  /* callback that allows applications to peek at protocol messages */
  void (*msg_callback)(int write_p, int version, int content_type,
                       const void *buf, size_t len, SSL *ssl, void *arg);
  void *msg_callback_arg;

  int verify_mode;
  unsigned int sid_ctx_length;
  uint8_t sid_ctx[SSL_MAX_SID_CTX_LENGTH];
  int (*default_verify_callback)(
      int ok, X509_STORE_CTX *ctx); /* called 'verify_callback' in the SSL */

  /* Default generate session ID callback. */
  GEN_SESSION_CB generate_session_id;

  X509_VERIFY_PARAM *param;

  /* select_certificate_cb is called before most ClientHello processing and
   * before the decision whether to resume a session is made. It may return one
   * to continue the handshake or zero to cause the handshake loop to return
   * with an error and cause SSL_get_error to return
   * SSL_ERROR_PENDING_CERTIFICATE. Note: when the handshake loop is resumed, it
   * will not call the callback a second time. */
  int (*select_certificate_cb)(const struct ssl_early_callback_ctx *);

  /* dos_protection_cb is called once the resumption decision for a ClientHello
   * has been made. It returns one to continue the handshake or zero to
   * abort. */
  int (*dos_protection_cb) (const struct ssl_early_callback_ctx *);

  /* quiet_shutdown is true if the connection should not send a close_notify on
   * shutdown. */
  int quiet_shutdown;

  /* Maximum amount of data to send in one fragment. actual record size can be
   * more than this due to padding and MAC overheads. */
  uint16_t max_send_fragment;

  /* TLS extensions servername callback */
  int (*tlsext_servername_callback)(SSL *, int *, void *);
  void *tlsext_servername_arg;
  /* RFC 4507 session ticket keys */
  uint8_t tlsext_tick_key_name[SSL_TICKET_KEY_NAME_LEN];
  uint8_t tlsext_tick_hmac_key[16];
  uint8_t tlsext_tick_aes_key[16];
  /* Callback to support customisation of ticket key setting */
  int (*tlsext_ticket_key_cb)(SSL *ssl, uint8_t *name, uint8_t *iv,
                              EVP_CIPHER_CTX *ectx, HMAC_CTX *hctx, int enc);

  /* Server-only: psk_identity_hint is the default identity hint to send in
   * PSK-based key exchanges. */
  char *psk_identity_hint;

  unsigned int (*psk_client_callback)(SSL *ssl, const char *hint,
                                      char *identity,
                                      unsigned int max_identity_len,
                                      uint8_t *psk, unsigned int max_psk_len);
  unsigned int (*psk_server_callback)(SSL *ssl, const char *identity,
                                      uint8_t *psk, unsigned int max_psk_len);


  /* retain_only_sha256_of_client_certs is true if we should compute the SHA256
   * hash of the peer's certifiate and then discard it to save memory and
   * session space. Only effective on the server side. */
  char retain_only_sha256_of_client_certs;

  /* Next protocol negotiation information */
  /* (for experimental NPN extension). */

  /* For a server, this contains a callback function by which the set of
   * advertised protocols can be provided. */
  int (*next_protos_advertised_cb)(SSL *ssl, const uint8_t **out,
                                   unsigned *out_len, void *arg);
  void *next_protos_advertised_cb_arg;
  /* For a client, this contains a callback function that selects the
   * next protocol from the list provided by the server. */
  int (*next_proto_select_cb)(SSL *ssl, uint8_t **out, uint8_t *out_len,
                              const uint8_t *in, unsigned in_len, void *arg);
  void *next_proto_select_cb_arg;

  /* ALPN information
   * (we are in the process of transitioning from NPN to ALPN.) */

  /* For a server, this contains a callback function that allows the
   * server to select the protocol for the connection.
   *   out: on successful return, this must point to the raw protocol
   *        name (without the length prefix).
   *   outlen: on successful return, this contains the length of |*out|.
   *   in: points to the client's list of supported protocols in
   *       wire-format.
   *   inlen: the length of |in|. */
  int (*alpn_select_cb)(SSL *s, const uint8_t **out, uint8_t *out_len,
                        const uint8_t *in, unsigned in_len, void *arg);
  void *alpn_select_cb_arg;

  /* For a client, this contains the list of supported protocols in wire
   * format. */
  uint8_t *alpn_client_proto_list;
  unsigned alpn_client_proto_list_len;

  /* SRTP profiles we are willing to do from RFC 5764 */
  STACK_OF(SRTP_PROTECTION_PROFILE) *srtp_profiles;

  /* EC extension values inherited by SSL structure */
  size_t tlsext_ellipticcurvelist_length;
  uint16_t *tlsext_ellipticcurvelist;

  /* If true, a client will advertise the Channel ID extension and a server
   * will echo it. */
  char tlsext_channel_id_enabled;
  /* The client's Channel ID private key. */
  EVP_PKEY *tlsext_channel_id_private;

  /* If true, a client will request certificate timestamps. */
  char signed_cert_timestamps_enabled;

  /* Signed certificate timestamp list to be sent to the client, if requested */
  uint8_t *signed_cert_timestamp_list;
  size_t signed_cert_timestamp_list_length;

  /* If true, a client will request a stapled OCSP response. */
  char ocsp_stapling_enabled;

  /* OCSP response to be sent to the client, if requested. */
  uint8_t *ocsp_response;
  size_t ocsp_response_length;

  /* If not NULL, session key material will be logged to this BIO for debugging
   * purposes. The format matches NSS's and is readable by Wireshark. */
  BIO *keylog_bio;

  /* current_time_cb, if not NULL, is the function to use to get the current
   * time. It sets |*out_clock| to the current time. */
  void (*current_time_cb)(const SSL *ssl, struct timeval *out_clock);
};

struct ssl_st {
  /* version is the protocol version. */
  int version;

  /* method is the method table corresponding to the current protocol (DTLS or
   * TLS). */
  const SSL_PROTOCOL_METHOD *method;

  /* enc_method is the method table corresponding to the current protocol
   * version. */
  const SSL3_ENC_METHOD *enc_method;

  /* max_version is the maximum acceptable protocol version. If zero, the
   * maximum supported version, currently (D)TLS 1.2, is used. */
  uint16_t max_version;

  /* min_version is the minimum acceptable protocl version. If zero, the
   * minimum supported version, currently SSL 3.0 and DTLS 1.0, is used */
  uint16_t min_version;

  /* There are 2 BIO's even though they are normally both the same. This is so
   * data can be read and written to different handlers */

  BIO *rbio; /* used by SSL_read */
  BIO *wbio; /* used by SSL_write */
  BIO *bbio; /* used during session-id reuse to concatenate
              * messages */

  /* This holds a variable that indicates what we were doing when a 0 or -1 is
   * returned.  This is needed for non-blocking IO so we know what request
   * needs re-doing when in SSL_accept or SSL_connect */
  int rwstate;

  /* true when we are actually in SSL_accept() or SSL_connect() */
  int in_handshake;
  int (*handshake_func)(SSL *);

  /* Imagine that here's a boolean member "init" that is switched as soon as
   * SSL_set_{accept/connect}_state is called for the first time, so that
   * "state" and "handshake_func" are properly initialized.  But as
   * handshake_func is == 0 until then, we use this test instead of an "init"
   * member. */

  /* server is true iff the this SSL* is the server half. Note: before the SSL*
   * is initialized by either SSL_set_accept_state or SSL_set_connect_state,
   * the side is not determined. In this state, server is always false. */
  int server;

  /* quiet_shutdown is true if the connection should not send a close_notify on
   * shutdown. */
  int quiet_shutdown;

  int shutdown; /* we have shut things down, 0x01 sent, 0x02
                 * for received */
  int state;    /* where we are */

  BUF_MEM *init_buf; /* buffer used during init */
  uint8_t *init_msg; /* pointer to handshake message body, set by
                        ssl3_get_message() */
  int init_num;      /* amount read/written */
  int init_off;      /* amount read/written */

  struct ssl3_state_st *s3;  /* SSLv3 variables */
  struct dtls1_state_st *d1; /* DTLSv1 variables */

  /* callback that allows applications to peek at protocol messages */
  void (*msg_callback)(int write_p, int version, int content_type,
                       const void *buf, size_t len, SSL *ssl, void *arg);
  void *msg_callback_arg;

  int hit; /* reusing a previous session */

  X509_VERIFY_PARAM *param;

  /* crypto */
  struct ssl_cipher_preference_list_st *cipher_list;
  STACK_OF(SSL_CIPHER) *cipher_list_by_id;

  SSL_AEAD_CTX *aead_read_ctx;
  SSL_AEAD_CTX *aead_write_ctx;

  /* session info */

  /* client cert? */
  /* This is used to hold the server certificate used */
  struct cert_st /* CERT */ *cert;

  /* the session_id_context is used to ensure sessions are only reused
   * in the appropriate context */
  unsigned int sid_ctx_length;
  uint8_t sid_ctx[SSL_MAX_SID_CTX_LENGTH];

  /* This can also be in the session once a session is established */
  SSL_SESSION *session;

  /* Default generate session ID callback. */
  GEN_SESSION_CB generate_session_id;

  /* Used in SSL2 and SSL3 */
  int verify_mode; /* 0 don't care about verify failure.
                    * 1 fail if verify fails */
  int (*verify_callback)(int ok,
                         X509_STORE_CTX *ctx); /* fail if callback returns 0 */

  void (*info_callback)(const SSL *ssl, int type,
                        int val); /* optional informational callback */

  /* Server-only: psk_identity_hint is the identity hint to send in
   * PSK-based key exchanges. */
  char *psk_identity_hint;

  unsigned int (*psk_client_callback)(SSL *ssl, const char *hint,
                                      char *identity,
                                      unsigned int max_identity_len,
                                      uint8_t *psk, unsigned int max_psk_len);
  unsigned int (*psk_server_callback)(SSL *ssl, const char *identity,
                                      uint8_t *psk, unsigned int max_psk_len);

  SSL_CTX *ctx;

  /* extra application data */
  long verify_result;
  CRYPTO_EX_DATA ex_data;

  /* for server side, keep the list of CA_dn we can use */
  STACK_OF(X509_NAME) *client_CA;

  uint32_t options; /* protocol behaviour */
  uint32_t mode;    /* API behaviour */
  uint32_t max_cert_list;
  int client_version; /* what was passed, used for
                       * SSLv3/TLS rollback check */
  uint16_t max_send_fragment;
  char *tlsext_hostname;
  /* RFC4507 session ticket expected to be received or sent */
  int tlsext_ticket_expected;
  size_t tlsext_ellipticcurvelist_length;
  uint16_t *tlsext_ellipticcurvelist; /* our list */

  SSL_CTX *initial_ctx; /* initial ctx, used to store sessions */

  /* Next protocol negotiation. For the client, this is the protocol that we
   * sent in NextProtocol and is set when handling ServerHello extensions.
   *
   * For a server, this is the client's selected_protocol from NextProtocol and
   * is set when handling the NextProtocol message, before the Finished
   * message. */
  uint8_t *next_proto_negotiated;
  size_t next_proto_negotiated_len;

  /* srtp_profiles is the list of configured SRTP protection profiles for
   * DTLS-SRTP. */
  STACK_OF(SRTP_PROTECTION_PROFILE) *srtp_profiles;

  /* srtp_profile is the selected SRTP protection profile for
   * DTLS-SRTP. */
  const SRTP_PROTECTION_PROFILE *srtp_profile;

  /* Copied from the SSL_CTX. For a server, means that we'll accept Channel IDs
   * from clients. For a client, means that we'll advertise support. */
  char tlsext_channel_id_enabled;
  /* The client's Channel ID private key. */
  EVP_PKEY *tlsext_channel_id_private;

  /* Enable signed certificate time stamps. Currently client only. */
  char signed_cert_timestamps_enabled;

  /* ocsp_stapling_enabled is only used by client connections and indicates
   * whether OCSP stapling will be requested. */
  char ocsp_stapling_enabled;

  /* For a client, this contains the list of supported protocols in wire
   * format. */
  uint8_t *alpn_client_proto_list;
  unsigned alpn_client_proto_list_len;

  /* accept_peer_renegotiations, if one, accepts renegotiation attempts from the
   * peer. Otherwise, they will be rejected with a fatal error. */
  char accept_peer_renegotiations;

  /* These fields are always NULL and exist only to keep wpa_supplicant happy
   * about the change to EVP_AEAD. They are only needed for EAP-FAST, which we
   * don't support. */
  EVP_CIPHER_CTX *enc_read_ctx;
  EVP_MD_CTX *read_hash;
};


/* Android compatibility section.
 *
 * These functions are declared, temporarily, for Android because
 * wpa_supplicant will take a little time to sync with upstream. Outside of
 * Android they'll have no definition. */

#define SSL_F_SSL_SET_SESSION_TICKET_EXT doesnt_exist

OPENSSL_EXPORT int SSL_set_session_ticket_ext(SSL *s, void *ext_data,
                                              int ext_len);
OPENSSL_EXPORT int SSL_set_session_secret_cb(SSL *s, void *cb, void *arg);
OPENSSL_EXPORT int SSL_set_session_ticket_ext_cb(SSL *s, void *cb, void *arg);
OPENSSL_EXPORT int SSL_set_ssl_method(SSL *s, const SSL_METHOD *method);

#define OPENSSL_VERSION_TEXT "BoringSSL"

#define SSLEAY_VERSION 0

/* SSLeay_version is a compatibility function that returns the string
 * "BoringSSL". */
OPENSSL_EXPORT const char *SSLeay_version(int unused);


/* Preprocessor compatibility section.
 *
 * Historically, a number of APIs were implemented in OpenSSL as macros and
 * constants to 'ctrl' functions. To avoid breaking #ifdefs in consumers, this
 * section defines a number of legacy macros.
 *
 * Although using either the CTRL values or their wrapper macros in #ifdefs is
 * still supported, the CTRL values may not be passed to |SSL_ctrl| and
 * |SSL_CTX_ctrl|. Call the functions (previously wrapper macros) instead. */

#define DTLS_CTRL_GET_TIMEOUT doesnt_exist
#define DTLS_CTRL_HANDLE_TIMEOUT doesnt_exist
#define SSL_CTRL_CHAIN doesnt_exist
#define SSL_CTRL_CHAIN_CERT doesnt_exist
#define SSL_CTRL_CHANNEL_ID doesnt_exist
#define SSL_CTRL_CLEAR_EXTRA_CHAIN_CERTS doesnt_exist
#define SSL_CTRL_CLEAR_MODE doesnt_exist
#define SSL_CTRL_CLEAR_OPTIONS doesnt_exist
#define SSL_CTRL_EXTRA_CHAIN_CERT doesnt_exist
#define SSL_CTRL_GET_CHAIN_CERTS doesnt_exist
#define SSL_CTRL_GET_CHANNEL_ID doesnt_exist
#define SSL_CTRL_GET_CLIENT_CERT_TYPES doesnt_exist
#define SSL_CTRL_GET_EXTRA_CHAIN_CERTS doesnt_exist
#define SSL_CTRL_GET_MAX_CERT_LIST doesnt_exist
#define SSL_CTRL_GET_NUM_RENEGOTIATIONS doesnt_exist
#define SSL_CTRL_GET_READ_AHEAD doesnt_exist
#define SSL_CTRL_GET_RI_SUPPORT doesnt_exist
#define SSL_CTRL_GET_SESSION_REUSED doesnt_exist
#define SSL_CTRL_GET_SESS_CACHE_MODE doesnt_exist
#define SSL_CTRL_GET_SESS_CACHE_SIZE doesnt_exist
#define SSL_CTRL_GET_TLSEXT_TICKET_KEYS doesnt_exist
#define SSL_CTRL_GET_TOTAL_RENEGOTIATIONS doesnt_exist
#define SSL_CTRL_MODE doesnt_exist
#define SSL_CTRL_NEED_TMP_RSA doesnt_exist
#define SSL_CTRL_OPTIONS doesnt_exist
#define SSL_CTRL_SESS_NUMBER doesnt_exist
#define SSL_CTRL_SET_CHANNEL_ID doesnt_exist
#define SSL_CTRL_SET_CURVES doesnt_exist
#define SSL_CTRL_SET_MAX_CERT_LIST doesnt_exist
#define SSL_CTRL_SET_MAX_SEND_FRAGMENT doesnt_exist
#define SSL_CTRL_SET_MSG_CALLBACK doesnt_exist
#define SSL_CTRL_SET_MSG_CALLBACK_ARG doesnt_exist
#define SSL_CTRL_SET_MTU doesnt_exist
#define SSL_CTRL_SET_READ_AHEAD doesnt_exist
#define SSL_CTRL_SET_SESS_CACHE_MODE doesnt_exist
#define SSL_CTRL_SET_SESS_CACHE_SIZE doesnt_exist
#define SSL_CTRL_SET_TLSEXT_HOSTNAME doesnt_exist
#define SSL_CTRL_SET_TLSEXT_SERVERNAME_ARG doesnt_exist
#define SSL_CTRL_SET_TLSEXT_SERVERNAME_CB doesnt_exist
#define SSL_CTRL_SET_TLSEXT_TICKET_KEYS doesnt_exist
#define SSL_CTRL_SET_TLSEXT_TICKET_KEY_CB doesnt_exist
#define SSL_CTRL_SET_TMP_DH doesnt_exist
#define SSL_CTRL_SET_TMP_DH_CB doesnt_exist
#define SSL_CTRL_SET_TMP_ECDH doesnt_exist
#define SSL_CTRL_SET_TMP_ECDH_CB doesnt_exist
#define SSL_CTRL_SET_TMP_RSA doesnt_exist
#define SSL_CTRL_SET_TMP_RSA_CB doesnt_exist

#define DTLSv1_get_timeout DTLSv1_get_timeout
#define DTLSv1_handle_timeout DTLSv1_handle_timeout
#define SSL_CTX_add0_chain_cert SSL_CTX_add0_chain_cert
#define SSL_CTX_add1_chain_cert SSL_CTX_add1_chain_cert
#define SSL_CTX_add_extra_chain_cert SSL_CTX_add_extra_chain_cert
#define SSL_CTX_clear_extra_chain_certs SSL_CTX_clear_extra_chain_certs
#define SSL_CTX_clear_chain_certs SSL_CTX_clear_chain_certs
#define SSL_CTX_clear_mode SSL_CTX_clear_mode
#define SSL_CTX_clear_options SSL_CTX_clear_options
#define SSL_CTX_enable_tls_channel_id SSL_CTX_enable_tls_channel_id
#define SSL_CTX_get0_chain_certs SSL_CTX_get0_chain_certs
#define SSL_CTX_get_extra_chain_certs SSL_CTX_get_extra_chain_certs
#define SSL_CTX_get_max_cert_list SSL_CTX_get_max_cert_list
#define SSL_CTX_get_mode SSL_CTX_get_mode
#define SSL_CTX_get_options SSL_CTX_get_options
#define SSL_CTX_get_read_ahead SSL_CTX_get_read_ahead
#define SSL_CTX_get_session_cache_mode SSL_CTX_get_session_cache_mode
#define SSL_CTX_get_tlsext_ticket_keys SSL_CTX_get_tlsext_ticket_keys
#define SSL_CTX_need_tmp_RSA SSL_CTX_need_tmp_RSA
#define SSL_CTX_sess_get_cache_size SSL_CTX_sess_get_cache_size
#define SSL_CTX_sess_number SSL_CTX_sess_number
#define SSL_CTX_sess_set_cache_size SSL_CTX_sess_set_cache_size
#define SSL_CTX_set0_chain SSL_CTX_set0_chain
#define SSL_CTX_set1_chain SSL_CTX_set1_chain
#define SSL_CTX_set1_curves SSL_CTX_set1_curves
#define SSL_CTX_set1_tls_channel_id SSL_CTX_set1_tls_channel_id
#define SSL_CTX_set_max_cert_list SSL_CTX_set_max_cert_list
#define SSL_CTX_set_max_send_fragment SSL_CTX_set_max_send_fragment
#define SSL_CTX_set_mode SSL_CTX_set_mode
#define SSL_CTX_set_msg_callback_arg SSL_CTX_set_msg_callback_arg
#define SSL_CTX_set_options SSL_CTX_set_options
#define SSL_CTX_set_read_ahead SSL_CTX_set_read_ahead
#define SSL_CTX_set_session_cache_mode SSL_CTX_set_session_cache_mode
#define SSL_CTX_set_tlsext_servername_arg SSL_CTX_set_tlsext_servername_arg
#define SSL_CTX_set_tlsext_servername_callback \
    SSL_CTX_set_tlsext_servername_callback
#define SSL_CTX_set_tlsext_ticket_key_cb SSL_CTX_set_tlsext_ticket_key_cb
#define SSL_CTX_set_tlsext_ticket_keys SSL_CTX_set_tlsext_ticket_keys
#define SSL_CTX_set_tmp_dh SSL_CTX_set_tmp_dh
#define SSL_CTX_set_tmp_ecdh SSL_CTX_set_tmp_ecdh
#define SSL_CTX_set_tmp_rsa SSL_CTX_set_tmp_rsa
#define SSL_add0_chain_cert SSL_add0_chain_cert
#define SSL_add1_chain_cert SSL_add1_chain_cert
#define SSL_clear_chain_certs SSL_clear_chain_certs
#define SSL_clear_mode SSL_clear_mode
#define SSL_clear_options SSL_clear_options
#define SSL_enable_tls_channel_id SSL_enable_tls_channel_id
#define SSL_get0_certificate_types SSL_get0_certificate_types
#define SSL_get0_chain_certs SSL_get0_chain_certs
#define SSL_get_max_cert_list SSL_get_max_cert_list
#define SSL_get_mode SSL_get_mode
#define SSL_get_options SSL_get_options
#define SSL_get_secure_renegotiation_support \
    SSL_get_secure_renegotiation_support
#define SSL_get_tls_channel_id SSL_get_tls_channel_id
#define SSL_need_tmp_RSA SSL_need_tmp_RSA
#define SSL_num_renegotiations SSL_num_renegotiations
#define SSL_session_reused SSL_session_reused
#define SSL_set0_chain SSL_set0_chain
#define SSL_set1_chain SSL_set1_chain
#define SSL_set1_curves SSL_set1_curves
#define SSL_set1_tls_channel_id SSL_set1_tls_channel_id
#define SSL_set_max_cert_list SSL_set_max_cert_list
#define SSL_set_max_send_fragment SSL_set_max_send_fragment
#define SSL_set_mode SSL_set_mode
#define SSL_set_msg_callback_arg SSL_set_msg_callback_arg
#define SSL_set_mtu SSL_set_mtu
#define SSL_set_options SSL_set_options
#define SSL_set_tlsext_host_name SSL_set_tlsext_host_name
#define SSL_set_tmp_dh SSL_set_tmp_dh
#define SSL_set_tmp_ecdh SSL_set_tmp_ecdh
#define SSL_set_tmp_rsa SSL_set_tmp_rsa
#define SSL_total_renegotiations SSL_total_renegotiations


#if defined(__cplusplus)
} /* extern C */
#endif


/* Library consumers assume these headers are included by ssl.h, but they depend
 * on ssl.h, so include them after all declarations.
 *
 * TODO(davidben): The separation between ssl.h and these version-specific
 * headers introduces circular dependencies and is inconsistent. The function
 * declarations should move to ssl.h. Many of the constants can probably be
 * pruned or unexported. */
#include <openssl/ssl3.h>
#include <openssl/tls1.h> /* This is mostly sslv3 with a few tweaks */


/* BEGIN ERROR CODES */
/* The following lines are auto generated by the script make_errors.go. Any
 * changes made after this point may be overwritten when the script is next run.
 */
#define SSL_R_APP_DATA_IN_HANDSHAKE 100
#define SSL_R_ATTEMPT_TO_REUSE_SESSION_IN_DIFFERENT_CONTEXT 101
#define SSL_R_BAD_ALERT 102
#define SSL_R_BAD_CHANGE_CIPHER_SPEC 103
#define SSL_R_BAD_DATA_RETURNED_BY_CALLBACK 104
#define SSL_R_BAD_DH_P_LENGTH 105
#define SSL_R_BAD_DIGEST_LENGTH 106
#define SSL_R_BAD_ECC_CERT 107
#define SSL_R_BAD_ECPOINT 108
#define SSL_R_BAD_HANDSHAKE_LENGTH 109
#define SSL_R_BAD_HANDSHAKE_RECORD 110
#define SSL_R_BAD_HELLO_REQUEST 111
#define SSL_R_BAD_LENGTH 112
#define SSL_R_BAD_PACKET_LENGTH 113
#define SSL_R_BAD_RSA_ENCRYPT 114
#define SSL_R_BAD_SIGNATURE 115
#define SSL_R_BAD_SRTP_MKI_VALUE 116
#define SSL_R_BAD_SRTP_PROTECTION_PROFILE_LIST 117
#define SSL_R_BAD_SSL_FILETYPE 118
#define SSL_R_BAD_WRITE_RETRY 119
#define SSL_R_BIO_NOT_SET 120
#define SSL_R_BN_LIB 121
#define SSL_R_CANNOT_SERIALIZE_PUBLIC_KEY 122
#define SSL_R_CA_DN_LENGTH_MISMATCH 123
#define SSL_R_CA_DN_TOO_LONG 124
#define SSL_R_CCS_RECEIVED_EARLY 125
#define SSL_R_CERTIFICATE_VERIFY_FAILED 126
#define SSL_R_CERT_CB_ERROR 127
#define SSL_R_CERT_LENGTH_MISMATCH 128
#define SSL_R_CHANNEL_ID_NOT_P256 129
#define SSL_R_CHANNEL_ID_SIGNATURE_INVALID 130
#define SSL_R_CIPHER_CODE_WRONG_LENGTH 131
#define SSL_R_CIPHER_OR_HASH_UNAVAILABLE 132
#define SSL_R_CLIENTHELLO_PARSE_FAILED 133
#define SSL_R_CLIENTHELLO_TLSEXT 134
#define SSL_R_CONNECTION_REJECTED 135
#define SSL_R_CONNECTION_TYPE_NOT_SET 136
#define SSL_R_COOKIE_MISMATCH 137
#define SSL_R_D2I_ECDSA_SIG 138
#define SSL_R_DATA_BETWEEN_CCS_AND_FINISHED 139
#define SSL_R_DATA_LENGTH_TOO_LONG 140
#define SSL_R_DECODE_ERROR 141
#define SSL_R_DECRYPTION_FAILED 142
#define SSL_R_DECRYPTION_FAILED_OR_BAD_RECORD_MAC 143
#define SSL_R_DH_PUBLIC_VALUE_LENGTH_IS_WRONG 144
#define SSL_R_DIGEST_CHECK_FAILED 145
#define SSL_R_DTLS_MESSAGE_TOO_BIG 146
#define SSL_R_ECC_CERT_NOT_FOR_SIGNING 147
#define SSL_R_EMPTY_SRTP_PROTECTION_PROFILE_LIST 148
#define SSL_R_ENCRYPTED_LENGTH_TOO_LONG 149
#define SSL_R_ERROR_IN_RECEIVED_CIPHER_LIST 150
#define SSL_R_EVP_DIGESTSIGNFINAL_FAILED 151
#define SSL_R_EVP_DIGESTSIGNINIT_FAILED 152
#define SSL_R_EXCESSIVE_MESSAGE_SIZE 153
#define SSL_R_EXTRA_DATA_IN_MESSAGE 154
#define SSL_R_GOT_A_FIN_BEFORE_A_CCS 155
#define SSL_R_GOT_CHANNEL_ID_BEFORE_A_CCS 156
#define SSL_R_GOT_NEXT_PROTO_BEFORE_A_CCS 157
#define SSL_R_GOT_NEXT_PROTO_WITHOUT_EXTENSION 158
#define SSL_R_HANDSHAKE_FAILURE_ON_CLIENT_HELLO 159
#define SSL_R_HANDSHAKE_RECORD_BEFORE_CCS 160
#define SSL_R_HTTPS_PROXY_REQUEST 161
#define SSL_R_HTTP_REQUEST 162
#define SSL_R_INAPPROPRIATE_FALLBACK 163
#define SSL_R_INVALID_COMMAND 164
#define SSL_R_INVALID_MESSAGE 165
#define SSL_R_INVALID_SSL_SESSION 166
#define SSL_R_INVALID_TICKET_KEYS_LENGTH 167
#define SSL_R_LENGTH_MISMATCH 168
#define SSL_R_LIBRARY_HAS_NO_CIPHERS 169
#define SSL_R_MISSING_DH_KEY 170
#define SSL_R_MISSING_ECDSA_SIGNING_CERT 171
#define SSL_R_MISSING_RSA_CERTIFICATE 172
#define SSL_R_MISSING_RSA_ENCRYPTING_CERT 173
#define SSL_R_MISSING_RSA_SIGNING_CERT 174
#define SSL_R_MISSING_TMP_DH_KEY 175
#define SSL_R_MISSING_TMP_ECDH_KEY 176
#define SSL_R_MIXED_SPECIAL_OPERATOR_WITH_GROUPS 177
#define SSL_R_MTU_TOO_SMALL 178
#define SSL_R_NESTED_GROUP 179
#define SSL_R_NO_CERTIFICATES_RETURNED 180
#define SSL_R_NO_CERTIFICATE_ASSIGNED 181
#define SSL_R_NO_CERTIFICATE_SET 182
#define SSL_R_NO_CIPHERS_AVAILABLE 183
#define SSL_R_NO_CIPHERS_PASSED 184
#define SSL_R_NO_CIPHERS_SPECIFIED 185
#define SSL_R_NO_CIPHER_MATCH 186
#define SSL_R_NO_COMPRESSION_SPECIFIED 187
#define SSL_R_NO_METHOD_SPECIFIED 188
#define SSL_R_NO_P256_SUPPORT 189
#define SSL_R_NO_PRIVATE_KEY_ASSIGNED 190
#define SSL_R_NO_RENEGOTIATION 191
#define SSL_R_NO_REQUIRED_DIGEST 192
#define SSL_R_NO_SHARED_CIPHER 193
#define SSL_R_NO_SHARED_SIGATURE_ALGORITHMS 194
#define SSL_R_NO_SRTP_PROFILES 195
#define SSL_R_NULL_SSL_CTX 196
#define SSL_R_NULL_SSL_METHOD_PASSED 197
#define SSL_R_OLD_SESSION_CIPHER_NOT_RETURNED 198
#define SSL_R_PACKET_LENGTH_TOO_LONG 199
#define SSL_R_PARSE_TLSEXT 200
#define SSL_R_PATH_TOO_LONG 201
#define SSL_R_PEER_DID_NOT_RETURN_A_CERTIFICATE 202
#define SSL_R_PEER_ERROR_UNSUPPORTED_CERTIFICATE_TYPE 203
#define SSL_R_PROTOCOL_IS_SHUTDOWN 204
#define SSL_R_PSK_IDENTITY_NOT_FOUND 205
#define SSL_R_PSK_NO_CLIENT_CB 206
#define SSL_R_PSK_NO_SERVER_CB 207
#define SSL_R_READ_BIO_NOT_SET 208
#define SSL_R_READ_TIMEOUT_EXPIRED 209
#define SSL_R_RECORD_LENGTH_MISMATCH 210
#define SSL_R_RECORD_TOO_LARGE 211
#define SSL_R_RENEGOTIATE_EXT_TOO_LONG 212
#define SSL_R_RENEGOTIATION_ENCODING_ERR 213
#define SSL_R_RENEGOTIATION_MISMATCH 214
#define SSL_R_REQUIRED_CIPHER_MISSING 215
#define SSL_R_SCSV_RECEIVED_WHEN_RENEGOTIATING 216
#define SSL_R_SERVERHELLO_TLSEXT 217
#define SSL_R_SESSION_ID_CONTEXT_UNINITIALIZED 218
#define SSL_R_SESSION_MAY_NOT_BE_CREATED 219
#define SSL_R_SIGNATURE_ALGORITHMS_ERROR 220
#define SSL_R_SRTP_COULD_NOT_ALLOCATE_PROFILES 221
#define SSL_R_SRTP_PROTECTION_PROFILE_LIST_TOO_LONG 222
#define SSL_R_SRTP_UNKNOWN_PROTECTION_PROFILE 223
#define SSL_R_SSL3_EXT_INVALID_SERVERNAME 224
#define SSL_R_SSL3_EXT_INVALID_SERVERNAME_TYPE 225
#define SSL_R_SSL_CTX_HAS_NO_DEFAULT_SSL_VERSION 226
#define SSL_R_SSL_HANDSHAKE_FAILURE 227
#define SSL_R_SSL_SESSION_ID_CALLBACK_FAILED 228
#define SSL_R_SSL_SESSION_ID_CONFLICT 229
#define SSL_R_SSL_SESSION_ID_CONTEXT_TOO_LONG 230
#define SSL_R_SSL_SESSION_ID_HAS_BAD_LENGTH 231
#define SSL_R_TLS_CLIENT_CERT_REQ_WITH_ANON_CIPHER 232
#define SSL_R_TLS_ILLEGAL_EXPORTER_LABEL 233
#define SSL_R_TLS_INVALID_ECPOINTFORMAT_LIST 234
#define SSL_R_TLS_PEER_DID_NOT_RESPOND_WITH_CERTIFICATE_LIST 235
#define SSL_R_TLS_RSA_ENCRYPTED_VALUE_LENGTH_IS_WRONG 236
#define SSL_R_TOO_MANY_EMPTY_FRAGMENTS 237
#define SSL_R_UNABLE_TO_FIND_ECDH_PARAMETERS 238
#define SSL_R_UNABLE_TO_FIND_PUBLIC_KEY_PARAMETERS 239
#define SSL_R_UNEXPECTED_GROUP_CLOSE 240
#define SSL_R_UNEXPECTED_MESSAGE 241
#define SSL_R_UNEXPECTED_OPERATOR_IN_GROUP 242
#define SSL_R_UNEXPECTED_RECORD 243
#define SSL_R_UNINITIALIZED 244
#define SSL_R_UNKNOWN_ALERT_TYPE 245
#define SSL_R_UNKNOWN_CERTIFICATE_TYPE 246
#define SSL_R_UNKNOWN_CIPHER_RETURNED 247
#define SSL_R_UNKNOWN_CIPHER_TYPE 248
#define SSL_R_UNKNOWN_DIGEST 249
#define SSL_R_UNKNOWN_KEY_EXCHANGE_TYPE 250
#define SSL_R_UNKNOWN_PROTOCOL 251
#define SSL_R_UNKNOWN_SSL_VERSION 252
#define SSL_R_UNKNOWN_STATE 253
#define SSL_R_UNPROCESSED_HANDSHAKE_DATA 254
#define SSL_R_UNSAFE_LEGACY_RENEGOTIATION_DISABLED 255
#define SSL_R_UNSUPPORTED_CIPHER 256
#define SSL_R_UNSUPPORTED_COMPRESSION_ALGORITHM 257
#define SSL_R_UNSUPPORTED_ELLIPTIC_CURVE 258
#define SSL_R_UNSUPPORTED_PROTOCOL 259
#define SSL_R_UNSUPPORTED_SSL_VERSION 260
#define SSL_R_USE_SRTP_NOT_NEGOTIATED 261
#define SSL_R_WRONG_CERTIFICATE_TYPE 262
#define SSL_R_WRONG_CIPHER_RETURNED 263
#define SSL_R_WRONG_CURVE 264
#define SSL_R_WRONG_MESSAGE_TYPE 265
#define SSL_R_WRONG_SIGNATURE_TYPE 266
#define SSL_R_WRONG_SSL_VERSION 267
#define SSL_R_WRONG_VERSION_NUMBER 268
#define SSL_R_X509_LIB 269
#define SSL_R_X509_VERIFICATION_SETUP_PROBLEMS 270
#define SSL_R_FRAGMENT_MISMATCH 271
#define SSL_R_BUFFER_TOO_SMALL 272
#define SSL_R_OLD_SESSION_VERSION_NOT_RETURNED 273
#define SSL_R_OUTPUT_ALIASES_INPUT 274
#define SSL_R_RESUMED_EMS_SESSION_WITHOUT_EMS_EXTENSION 275
#define SSL_R_EMS_STATE_INCONSISTENT 276
#define SSL_R_RESUMED_NON_EMS_SESSION_WITH_EMS_EXTENSION 277
#define SSL_R_TOO_MANY_WARNING_ALERTS 278
#define SSL_R_UNEXPECTED_EXTENSION 279
#define SSL_R_SIGNATURE_ALGORITHMS_EXTENSION_SENT_BY_SERVER 280
#define SSL_R_ERROR_ADDING_EXTENSION 281
#define SSL_R_ERROR_PARSING_EXTENSION 282
#define SSL_R_MISSING_EXTENSION 283
#define SSL_R_CUSTOM_EXTENSION_CONTENTS_TOO_LARGE 284
#define SSL_R_CUSTOM_EXTENSION_ERROR 285
#define SSL_R_NEGOTIATED_BOTH_NPN_AND_ALPN 286
#define SSL_R_SSLV3_ALERT_CLOSE_NOTIFY 1000
#define SSL_R_SSLV3_ALERT_UNEXPECTED_MESSAGE 1010
#define SSL_R_SSLV3_ALERT_BAD_RECORD_MAC 1020
#define SSL_R_TLSV1_ALERT_DECRYPTION_FAILED 1021
#define SSL_R_TLSV1_ALERT_RECORD_OVERFLOW 1022
#define SSL_R_SSLV3_ALERT_DECOMPRESSION_FAILURE 1030
#define SSL_R_SSLV3_ALERT_HANDSHAKE_FAILURE 1040
#define SSL_R_SSLV3_ALERT_NO_CERTIFICATE 1041
#define SSL_R_SSLV3_ALERT_BAD_CERTIFICATE 1042
#define SSL_R_SSLV3_ALERT_UNSUPPORTED_CERTIFICATE 1043
#define SSL_R_SSLV3_ALERT_CERTIFICATE_REVOKED 1044
#define SSL_R_SSLV3_ALERT_CERTIFICATE_EXPIRED 1045
#define SSL_R_SSLV3_ALERT_CERTIFICATE_UNKNOWN 1046
#define SSL_R_SSLV3_ALERT_ILLEGAL_PARAMETER 1047
#define SSL_R_TLSV1_ALERT_UNKNOWN_CA 1048
#define SSL_R_TLSV1_ALERT_ACCESS_DENIED 1049
#define SSL_R_TLSV1_ALERT_DECODE_ERROR 1050
#define SSL_R_TLSV1_ALERT_DECRYPT_ERROR 1051
#define SSL_R_TLSV1_ALERT_EXPORT_RESTRICTION 1060
#define SSL_R_TLSV1_ALERT_PROTOCOL_VERSION 1070
#define SSL_R_TLSV1_ALERT_INSUFFICIENT_SECURITY 1071
#define SSL_R_TLSV1_ALERT_INTERNAL_ERROR 1080
#define SSL_R_TLSV1_ALERT_INAPPROPRIATE_FALLBACK 1086
#define SSL_R_TLSV1_ALERT_USER_CANCELLED 1090
#define SSL_R_TLSV1_ALERT_NO_RENEGOTIATION 1100
#define SSL_R_TLSV1_UNSUPPORTED_EXTENSION 1110
#define SSL_R_TLSV1_CERTIFICATE_UNOBTAINABLE 1111
#define SSL_R_TLSV1_UNRECOGNIZED_NAME 1112
#define SSL_R_TLSV1_BAD_CERTIFICATE_STATUS_RESPONSE 1113
#define SSL_R_TLSV1_BAD_CERTIFICATE_HASH_VALUE 1114

#endif /* OPENSSL_HEADER_SSL_H */
