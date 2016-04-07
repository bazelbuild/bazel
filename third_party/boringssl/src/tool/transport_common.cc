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

#include <string>
#include <vector>

#include <errno.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>

#if !defined(OPENSSL_WINDOWS)
#include <arpa/inet.h>
#include <fcntl.h>
#include <netdb.h>
#include <netinet/in.h>
#include <sys/select.h>
#include <sys/socket.h>
#include <unistd.h>
#else
#include <io.h>
#pragma warning(push, 3)
#include <winsock2.h>
#include <ws2tcpip.h>
#pragma warning(pop)

typedef int ssize_t;
#pragma comment(lib, "Ws2_32.lib")
#endif

#include <openssl/err.h>
#include <openssl/ssl.h>

#include "internal.h"


#if !defined(OPENSSL_WINDOWS)
static int closesocket(int sock) {
  return close(sock);
}
#endif

bool InitSocketLibrary() {
#if defined(OPENSSL_WINDOWS)
  WSADATA wsaData;
  int err = WSAStartup(MAKEWORD(2, 2), &wsaData);
  if (err != 0) {
    fprintf(stderr, "WSAStartup failed with error %d\n", err);
    return false;
  }
#endif
  return true;
}

// Connect sets |*out_sock| to be a socket connected to the destination given
// in |hostname_and_port|, which should be of the form "www.example.com:123".
// It returns true on success and false otherwise.
bool Connect(int *out_sock, const std::string &hostname_and_port) {
  const size_t colon_offset = hostname_and_port.find_last_of(':');
  std::string hostname, port;

  if (colon_offset == std::string::npos) {
    hostname = hostname_and_port;
    port = "443";
  } else {
    hostname = hostname_and_port.substr(0, colon_offset);
    port = hostname_and_port.substr(colon_offset + 1);
  }

  struct addrinfo hint, *result;
  memset(&hint, 0, sizeof(hint));
  hint.ai_family = AF_UNSPEC;
  hint.ai_socktype = SOCK_STREAM;

  int ret = getaddrinfo(hostname.c_str(), port.c_str(), &hint, &result);
  if (ret != 0) {
    fprintf(stderr, "getaddrinfo returned: %s\n", gai_strerror(ret));
    return false;
  }

  bool ok = false;
  char buf[256];

  *out_sock =
      socket(result->ai_family, result->ai_socktype, result->ai_protocol);
  if (*out_sock < 0) {
    perror("socket");
    goto out;
  }

  switch (result->ai_family) {
    case AF_INET: {
      struct sockaddr_in *sin =
          reinterpret_cast<struct sockaddr_in *>(result->ai_addr);
      fprintf(stderr, "Connecting to %s:%d\n",
              inet_ntop(result->ai_family, &sin->sin_addr, buf, sizeof(buf)),
              ntohs(sin->sin_port));
      break;
    }
    case AF_INET6: {
      struct sockaddr_in6 *sin6 =
          reinterpret_cast<struct sockaddr_in6 *>(result->ai_addr);
      fprintf(stderr, "Connecting to [%s]:%d\n",
              inet_ntop(result->ai_family, &sin6->sin6_addr, buf, sizeof(buf)),
              ntohs(sin6->sin6_port));
      break;
    }
  }

  if (connect(*out_sock, result->ai_addr, result->ai_addrlen) != 0) {
    perror("connect");
    goto out;
  }
  ok = true;

out:
  freeaddrinfo(result);
  return ok;
}

bool Accept(int *out_sock, const std::string &port) {
  struct sockaddr_in addr, cli_addr;
  socklen_t cli_addr_len = sizeof(cli_addr);
  memset(&addr, 0, sizeof(addr));

  addr.sin_family = AF_INET;
  addr.sin_addr.s_addr = INADDR_ANY;
  addr.sin_port = htons(atoi(port.c_str()));

  bool ok = false;
  int server_sock = -1;

  server_sock =
      socket(addr.sin_family, SOCK_STREAM, 0);
  if (server_sock < 0) {
    perror("socket");
    goto out;
  }

  if (bind(server_sock, (struct sockaddr*)&addr, sizeof(addr)) != 0) {
    perror("connect");
    goto out;
  }
  listen(server_sock, 1);
  *out_sock = accept(server_sock, (struct sockaddr*)&cli_addr, &cli_addr_len);

  ok = true;

out:
  closesocket(server_sock);
  return ok;
}

void PrintConnectionInfo(const SSL *ssl) {
  const SSL_CIPHER *cipher = SSL_get_current_cipher(ssl);

  fprintf(stderr, "  Version: %s\n", SSL_get_version(ssl));
  fprintf(stderr, "  Resumed session: %s\n",
          SSL_session_reused(ssl) ? "yes" : "no");
  fprintf(stderr, "  Cipher: %s\n", SSL_CIPHER_get_name(cipher));
  fprintf(stderr, "  Secure renegotiation: %s\n",
          SSL_get_secure_renegotiation_support(ssl) ? "yes" : "no");

  const uint8_t *next_proto;
  unsigned next_proto_len;
  SSL_get0_next_proto_negotiated(ssl, &next_proto, &next_proto_len);
  fprintf(stderr, "  Next protocol negotiated: %.*s\n", next_proto_len,
          next_proto);

  const uint8_t *alpn;
  unsigned alpn_len;
  SSL_get0_alpn_selected(ssl, &alpn, &alpn_len);
  fprintf(stderr, "  ALPN protocol: %.*s\n", alpn_len, alpn);
}

bool SocketSetNonBlocking(int sock, bool is_non_blocking) {
  bool ok;

#if defined(OPENSSL_WINDOWS)
  u_long arg = is_non_blocking;
  ok = 0 == ioctlsocket(sock, FIONBIO, &arg);
#else
  int flags = fcntl(sock, F_GETFL, 0);
  if (flags < 0) {
    return false;
  }
  if (is_non_blocking) {
    flags |= O_NONBLOCK;
  } else {
    flags &= ~O_NONBLOCK;
  }
  ok = 0 == fcntl(sock, F_SETFL, flags);
#endif
  if (!ok) {
    fprintf(stderr, "Failed to set socket non-blocking.\n");
  }
  return ok;
}

// PrintErrorCallback is a callback function from OpenSSL's
// |ERR_print_errors_cb| that writes errors to a given |FILE*|.
int PrintErrorCallback(const char *str, size_t len, void *ctx) {
  fwrite(str, len, 1, reinterpret_cast<FILE*>(ctx));
  return 1;
}

bool TransferData(SSL *ssl, int sock) {
  bool stdin_open = true;

  fd_set read_fds;
  FD_ZERO(&read_fds);

  if (!SocketSetNonBlocking(sock, true)) {
    return false;
  }

  for (;;) {
    if (stdin_open) {
      FD_SET(0, &read_fds);
    }
    FD_SET(sock, &read_fds);

    int ret = select(sock + 1, &read_fds, NULL, NULL, NULL);
    if (ret <= 0) {
      perror("select");
      return false;
    }

    if (FD_ISSET(0, &read_fds)) {
      uint8_t buffer[512];
      ssize_t n;

      do {
        n = read(0, buffer, sizeof(buffer));
      } while (n == -1 && errno == EINTR);

      if (n == 0) {
        FD_CLR(0, &read_fds);
        stdin_open = false;
#if !defined(OPENSSL_WINDOWS)
        shutdown(sock, SHUT_WR);
#else
        shutdown(sock, SD_SEND);
#endif
        continue;
      } else if (n < 0) {
        perror("read from stdin");
        return false;
      }

      if (!SocketSetNonBlocking(sock, false)) {
        return false;
      }
      int ssl_ret = SSL_write(ssl, buffer, n);
      if (!SocketSetNonBlocking(sock, true)) {
        return false;
      }

      if (ssl_ret <= 0) {
        int ssl_err = SSL_get_error(ssl, ssl_ret);
        fprintf(stderr, "Error while writing: %d\n", ssl_err);
        ERR_print_errors_cb(PrintErrorCallback, stderr);
        return false;
      } else if (ssl_ret != n) {
        fprintf(stderr, "Short write from SSL_write.\n");
        return false;
      }
    }

    if (FD_ISSET(sock, &read_fds)) {
      uint8_t buffer[512];
      int ssl_ret = SSL_read(ssl, buffer, sizeof(buffer));

      if (ssl_ret < 0) {
        int ssl_err = SSL_get_error(ssl, ssl_ret);
        if (ssl_err == SSL_ERROR_WANT_READ) {
          continue;
        }
        fprintf(stderr, "Error while reading: %d\n", ssl_err);
        ERR_print_errors_cb(PrintErrorCallback, stderr);
        return false;
      } else if (ssl_ret == 0) {
        return true;
      }

      ssize_t n;
      do {
        n = write(1, buffer, ssl_ret);
      } while (n == -1 && errno == EINTR);

      if (n != ssl_ret) {
        fprintf(stderr, "Short write to stderr.\n");
        return false;
      }
    }
  }
}
