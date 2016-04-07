/* Copyright (c) 2014, Google Inc.
 *
 * Permission to use, copy, modify, and/or distribute this software for any
 * purpose with or without fee is hereby granted, provided that the above
 * copyright notice and this permission notice appear in all copies.
 *
 * THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES
 * WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF
 * MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY
 * SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
 * WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN ACTION
 * OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF OR IN
 * CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE. */

#include <openssl/ssl.h>

#include <assert.h>
#include <string.h>

#include <openssl/bytestring.h>
#include <openssl/err.h>
#include <openssl/mem.h>
#include <openssl/stack.h>

#include "internal.h"


void SSL_CUSTOM_EXTENSION_free(SSL_CUSTOM_EXTENSION *custom_extension) {
  OPENSSL_free(custom_extension);
}

static const SSL_CUSTOM_EXTENSION *custom_ext_find(
    STACK_OF(SSL_CUSTOM_EXTENSION) *stack,
    unsigned *out_index, uint16_t value) {
  size_t i;
  for (i = 0; i < sk_SSL_CUSTOM_EXTENSION_num(stack); i++) {
    const SSL_CUSTOM_EXTENSION *ext = sk_SSL_CUSTOM_EXTENSION_value(stack, i);
    if (ext->value == value) {
      if (out_index != NULL) {
        *out_index = i;
      }
      return ext;
    }
  }

  return NULL;
}

/* default_add_callback is used as the |add_callback| when the user doesn't
 * provide one. For servers, it does nothing while, for clients, it causes an
 * empty extension to be included. */
static int default_add_callback(SSL *ssl, unsigned extension_value,
                                const uint8_t **out, size_t *out_len,
                                int *out_alert_value, void *add_arg) {
  if (ssl->server) {
    return 0;
  }
  *out_len = 0;
  return 1;
}

static int custom_ext_add_hello(SSL *ssl, CBB *extensions) {
  STACK_OF(SSL_CUSTOM_EXTENSION) *stack = ssl->ctx->client_custom_extensions;
  if (ssl->server) {
    stack = ssl->ctx->server_custom_extensions;
  }

  if (stack == NULL) {
    return 1;
  }

  size_t i;
  for (i = 0; i < sk_SSL_CUSTOM_EXTENSION_num(stack); i++) {
    const SSL_CUSTOM_EXTENSION *ext = sk_SSL_CUSTOM_EXTENSION_value(stack, i);

    if (ssl->server &&
        !(ssl->s3->tmp.custom_extensions.received & (1u << i))) {
      /* Servers cannot echo extensions that the client didn't send. */
      continue;
    }

    const uint8_t *contents;
    size_t contents_len;
    int alert = SSL_AD_DECODE_ERROR;
    CBB contents_cbb;

    switch (ext->add_callback(ssl, ext->value, &contents, &contents_len, &alert,
                              ext->add_arg)) {
      case 1:
        if (!CBB_add_u16(extensions, ext->value) ||
            !CBB_add_u16_length_prefixed(extensions, &contents_cbb) ||
            !CBB_add_bytes(&contents_cbb, contents, contents_len) ||
            !CBB_flush(extensions)) {
          OPENSSL_PUT_ERROR(SSL, ERR_R_INTERNAL_ERROR);
          ERR_add_error_dataf("extension: %u", (unsigned) ext->value);
          if (ext->free_callback && 0 < contents_len) {
            ext->free_callback(ssl, ext->value, contents, ext->add_arg);
          }
          return 0;
        }

        if (ext->free_callback && 0 < contents_len) {
          ext->free_callback(ssl, ext->value, contents, ext->add_arg);
        }

        if (!ssl->server) {
          assert((ssl->s3->tmp.custom_extensions.sent & (1u << i)) == 0);
          ssl->s3->tmp.custom_extensions.sent |= (1u << i);
        }
        break;

      case 0:
        break;

      default:
        ssl3_send_alert(ssl, SSL3_AL_FATAL, alert);
        OPENSSL_PUT_ERROR(SSL, SSL_R_CUSTOM_EXTENSION_ERROR);
        ERR_add_error_dataf("extension: %u", (unsigned) ext->value);
        return 0;
    }
  }

  return 1;
}

int custom_ext_add_clienthello(SSL *ssl, CBB *extensions) {
  return custom_ext_add_hello(ssl, extensions);
}

int custom_ext_parse_serverhello(SSL *ssl, int *out_alert, uint16_t value,
                                 const CBS *extension) {
  unsigned index;
  const SSL_CUSTOM_EXTENSION *ext =
      custom_ext_find(ssl->ctx->client_custom_extensions, &index, value);

  if (/* Unknown extensions are not allowed in a ServerHello. */
      ext == NULL ||
      /* Also, if we didn't send the extension, that's also unacceptable. */
      !(ssl->s3->tmp.custom_extensions.sent & (1u << index))) {
    OPENSSL_PUT_ERROR(SSL, SSL_R_UNEXPECTED_EXTENSION);
    ERR_add_error_dataf("extension: %u", (unsigned)value);
    *out_alert = SSL_AD_DECODE_ERROR;
    return 0;
  }

  if (ext->parse_callback != NULL &&
      !ext->parse_callback(ssl, value, CBS_data(extension), CBS_len(extension),
                           out_alert, ext->parse_arg)) {
    OPENSSL_PUT_ERROR(SSL, SSL_R_CUSTOM_EXTENSION_ERROR);
    ERR_add_error_dataf("extension: %u", (unsigned)ext->value);
    return 0;
  }

  return 1;
}

int custom_ext_parse_clienthello(SSL *ssl, int *out_alert, uint16_t value,
                                 const CBS *extension) {
  unsigned index;
  const SSL_CUSTOM_EXTENSION *ext =
      custom_ext_find(ssl->ctx->server_custom_extensions, &index, value);

  if (ext == NULL) {
    return 1;
  }

  assert((ssl->s3->tmp.custom_extensions.received & (1u << index)) == 0);
  ssl->s3->tmp.custom_extensions.received |= (1u << index);

  if (ext->parse_callback &&
      !ext->parse_callback(ssl, value, CBS_data(extension), CBS_len(extension),
                           out_alert, ext->parse_arg)) {
    OPENSSL_PUT_ERROR(SSL, SSL_R_CUSTOM_EXTENSION_ERROR);
    ERR_add_error_dataf("extension: %u", (unsigned)ext->value);
    return 0;
  }

  return 1;
}

int custom_ext_add_serverhello(SSL *ssl, CBB *extensions) {
  return custom_ext_add_hello(ssl, extensions);
}

/* MAX_NUM_CUSTOM_EXTENSIONS is the maximum number of custom extensions that
 * can be set on an |SSL_CTX|. It's determined by the size of the bitset used
 * to track when an extension has been sent. */
#define MAX_NUM_CUSTOM_EXTENSIONS \
  (sizeof(((struct ssl3_state_st *)NULL)->tmp.custom_extensions.sent) * 8)

static int custom_ext_append(STACK_OF(SSL_CUSTOM_EXTENSION) **stack,
                             unsigned extension_value,
                             SSL_custom_ext_add_cb add_cb,
                             SSL_custom_ext_free_cb free_cb, void *add_arg,
                             SSL_custom_ext_parse_cb parse_cb,
                             void *parse_arg) {
  if (add_cb == NULL ||
      0xffff < extension_value ||
      SSL_extension_supported(extension_value) ||
      /* Specifying a free callback without an add callback is nonsensical
       * and an error. */
      (*stack != NULL &&
       (MAX_NUM_CUSTOM_EXTENSIONS <= sk_SSL_CUSTOM_EXTENSION_num(*stack) ||
        custom_ext_find(*stack, NULL, extension_value) != NULL))) {
    return 0;
  }

  SSL_CUSTOM_EXTENSION *ext = OPENSSL_malloc(sizeof(SSL_CUSTOM_EXTENSION));
  if (ext == NULL) {
    return 0;
  }
  ext->add_callback = add_cb;
  ext->add_arg = add_arg;
  ext->free_callback = free_cb;
  ext->parse_callback = parse_cb;
  ext->parse_arg = parse_arg;
  ext->value = extension_value;

  if (*stack == NULL) {
    *stack = sk_SSL_CUSTOM_EXTENSION_new_null();
    if (*stack == NULL) {
      SSL_CUSTOM_EXTENSION_free(ext);
      return 0;
    }
  }

  if (!sk_SSL_CUSTOM_EXTENSION_push(*stack, ext)) {
    SSL_CUSTOM_EXTENSION_free(ext);
    if (sk_SSL_CUSTOM_EXTENSION_num(*stack) == 0) {
      sk_SSL_CUSTOM_EXTENSION_free(*stack);
      *stack = NULL;
    }
    return 0;
  }

  return 1;
}

int SSL_CTX_add_client_custom_ext(SSL_CTX *ctx, unsigned extension_value,
                                  SSL_custom_ext_add_cb add_cb,
                                  SSL_custom_ext_free_cb free_cb, void *add_arg,
                                  SSL_custom_ext_parse_cb parse_cb,
                                  void *parse_arg) {
  return custom_ext_append(&ctx->client_custom_extensions, extension_value,
                           add_cb ? add_cb : default_add_callback, free_cb,
                           add_arg, parse_cb, parse_arg);
}

int SSL_CTX_add_server_custom_ext(SSL_CTX *ctx, unsigned extension_value,
                                  SSL_custom_ext_add_cb add_cb,
                                  SSL_custom_ext_free_cb free_cb, void *add_arg,
                                  SSL_custom_ext_parse_cb parse_cb,
                                  void *parse_arg) {
  return custom_ext_append(&ctx->server_custom_extensions, extension_value,
                           add_cb ? add_cb : default_add_callback, free_cb,
                           add_arg, parse_cb, parse_arg);
}
