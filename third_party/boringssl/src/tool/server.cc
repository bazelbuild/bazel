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

#include <openssl/base.h>

#include <openssl/err.h>
#include <openssl/ssl.h>

#include "internal.h"
#include "transport_common.h"


static const struct argument kArguments[] = {
    {
     "-accept", kRequiredArgument,
     "The port of the server to bind on; eg 45102",
    },
    {
     "-cipher", kOptionalArgument,
     "An OpenSSL-style cipher suite string that configures the offered ciphers",
    },
    {
      "-key", kOptionalArgument,
      "Private-key file to use (default is server.pem)",
    },
    {
      "-ocsp-response", kOptionalArgument,
      "OCSP response file to send",
    },
    {
     "", kOptionalArgument, "",
    },
};

static bool LoadOCSPResponse(SSL_CTX *ctx, const char *filename) {
  void *data = NULL;
  bool ret = false;
  size_t bytes_read;
  long length;

  FILE *f = fopen(filename, "rb");

  if (f == NULL ||
      fseek(f, 0, SEEK_END) != 0) {
    goto out;
  }

  length = ftell(f);
  if (length < 0) {
    goto out;
  }

  data = malloc(length);
  if (data == NULL) {
    goto out;
  }
  rewind(f);

  bytes_read = fread(data, 1, length, f);
  if (ferror(f) != 0 ||
      bytes_read != (size_t)length ||
      !SSL_CTX_set_ocsp_response(ctx, (uint8_t*)data, bytes_read)) {
    goto out;
  }

  ret = true;
out:
  if (f != NULL) {
      fclose(f);
  }
  free(data);
  return ret;
}

bool Server(const std::vector<std::string> &args) {
  if (!InitSocketLibrary()) {
    return false;
  }

  std::map<std::string, std::string> args_map;

  if (!ParseKeyValueArguments(&args_map, args, kArguments)) {
    PrintUsage(kArguments);
    return false;
  }

  SSL_CTX *ctx = SSL_CTX_new(SSLv23_server_method());
  SSL_CTX_set_options(ctx, SSL_OP_NO_SSLv3);

  // Server authentication is required.
  std::string key_file = "server.pem";
  if (args_map.count("-key") != 0) {
    key_file = args_map["-key"];
  }
  if (SSL_CTX_use_PrivateKey_file(ctx, key_file.c_str(), SSL_FILETYPE_PEM) <= 0) {
    fprintf(stderr, "Failed to load private key: %s\n", key_file.c_str());
    return false;
  }
  if (SSL_CTX_use_certificate_chain_file(ctx, key_file.c_str()) != 1) {
    fprintf(stderr, "Failed to load cert chain: %s\n", key_file.c_str());
    return false;
  }

  if (args_map.count("-cipher") != 0 &&
      !SSL_CTX_set_cipher_list(ctx, args_map["-cipher"].c_str())) {
    fprintf(stderr, "Failed setting cipher list\n");
    return false;
  }

  if (args_map.count("-ocsp-response") != 0 &&
      !LoadOCSPResponse(ctx, args_map["-ocsp-response"].c_str())) {
    fprintf(stderr, "Failed to load OCSP response: %s\n", args_map["-ocsp-response"].c_str());
    return false;
  }

  int sock = -1;
  if (!Accept(&sock, args_map["-accept"])) {
    return false;
  }

  BIO *bio = BIO_new_socket(sock, BIO_CLOSE);
  SSL *ssl = SSL_new(ctx);
  SSL_set_bio(ssl, bio, bio);

  int ret = SSL_accept(ssl);
  if (ret != 1) {
    int ssl_err = SSL_get_error(ssl, ret);
    fprintf(stderr, "Error while connecting: %d\n", ssl_err);
    ERR_print_errors_cb(PrintErrorCallback, stderr);
    return false;
  }

  fprintf(stderr, "Connected.\n");
  PrintConnectionInfo(ssl);

  bool ok = TransferData(ssl, sock);

  SSL_free(ssl);
  SSL_CTX_free(ctx);
  return ok;
}
