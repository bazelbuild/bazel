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

#if !defined(_POSIX_C_SOURCE)
#define _POSIX_C_SOURCE 201410L
#endif

#include <openssl/base.h>

#if !defined(OPENSSL_WINDOWS)
#include <arpa/inet.h>
#include <fcntl.h>
#include <netinet/in.h>
#include <string.h>
#include <sys/socket.h>
#include <unistd.h>
#else
#include <io.h>
#pragma warning(push, 3)
#include <winsock2.h>
#include <ws2tcpip.h>
#pragma warning(pop)
#endif

#include <openssl/bio.h>
#include <openssl/crypto.h>
#include <openssl/err.h>
#include <openssl/mem.h>

#include <algorithm>

#include "../test/scoped_types.h"


#if !defined(OPENSSL_WINDOWS)
static int closesocket(int sock) {
  return close(sock);
}

static void PrintSocketError(const char *func) {
  perror(func);
}
#else
static void PrintSocketError(const char *func) {
  fprintf(stderr, "%s: %d\n", func, WSAGetLastError());
}
#endif

class ScopedSocket {
 public:
  ScopedSocket(int sock) : sock_(sock) {}
  ~ScopedSocket() {
    closesocket(sock_);
  }

 private:
  const int sock_;
};

static bool TestSocketConnect() {
  static const char kTestMessage[] = "test";

  int listening_sock = socket(AF_INET, SOCK_STREAM, 0);
  if (listening_sock == -1) {
    PrintSocketError("socket");
    return false;
  }
  ScopedSocket listening_sock_closer(listening_sock);

  struct sockaddr_in sin;
  memset(&sin, 0, sizeof(sin));
  sin.sin_family = AF_INET;
  if (!inet_pton(AF_INET, "127.0.0.1", &sin.sin_addr)) {
    PrintSocketError("inet_pton");
    return false;
  }
  if (bind(listening_sock, (struct sockaddr *)&sin, sizeof(sin)) != 0) {
    PrintSocketError("bind");
    return false;
  }
  if (listen(listening_sock, 1)) {
    PrintSocketError("listen");
    return false;
  }
  socklen_t sockaddr_len = sizeof(sin);
  if (getsockname(listening_sock, (struct sockaddr *)&sin, &sockaddr_len) ||
      sockaddr_len != sizeof(sin)) {
    PrintSocketError("getsockname");
    return false;
  }

  char hostname[80];
  BIO_snprintf(hostname, sizeof(hostname), "%s:%d", "127.0.0.1",
               ntohs(sin.sin_port));
  ScopedBIO bio(BIO_new_connect(hostname));
  if (!bio) {
    fprintf(stderr, "BIO_new_connect failed.\n");
    return false;
  }

  if (BIO_write(bio.get(), kTestMessage, sizeof(kTestMessage)) !=
      sizeof(kTestMessage)) {
    fprintf(stderr, "BIO_write failed.\n");
    ERR_print_errors_fp(stderr);
    return false;
  }

  int sock = accept(listening_sock, (struct sockaddr *) &sin, &sockaddr_len);
  if (sock == -1) {
    PrintSocketError("accept");
    return false;
  }
  ScopedSocket sock_closer(sock);

  char buf[5];
  if (recv(sock, buf, sizeof(buf), 0) != sizeof(kTestMessage)) {
    PrintSocketError("read");
    return false;
  }
  if (memcmp(buf, kTestMessage, sizeof(kTestMessage))) {
    return false;
  }

  return true;
}


// BioReadZeroCopyWrapper is a wrapper around the zero-copy APIs to make
// testing easier.
static size_t BioReadZeroCopyWrapper(BIO *bio, uint8_t *data, size_t len) {
  uint8_t *read_buf;
  size_t read_buf_offset;
  size_t available_bytes;
  size_t len_read = 0;

  do {
    if (!BIO_zero_copy_get_read_buf(bio, &read_buf, &read_buf_offset,
                                    &available_bytes)) {
      return 0;
    }

    available_bytes = std::min(available_bytes, len - len_read);
    memmove(data + len_read, read_buf + read_buf_offset, available_bytes);

    BIO_zero_copy_get_read_buf_done(bio, available_bytes);

    len_read += available_bytes;
  } while (len - len_read > 0 && available_bytes > 0);

  return len_read;
}

// BioWriteZeroCopyWrapper is a wrapper around the zero-copy APIs to make
// testing easier.
static size_t BioWriteZeroCopyWrapper(BIO *bio, const uint8_t *data,
                                      size_t len) {
  uint8_t *write_buf;
  size_t write_buf_offset;
  size_t available_bytes;
  size_t len_written = 0;

  do {
    if (!BIO_zero_copy_get_write_buf(bio, &write_buf, &write_buf_offset,
                                     &available_bytes)) {
      return 0;
    }

    available_bytes = std::min(available_bytes, len - len_written);
    memmove(write_buf + write_buf_offset, data + len_written, available_bytes);

    BIO_zero_copy_get_write_buf_done(bio, available_bytes);

    len_written += available_bytes;
  } while (len - len_written > 0 && available_bytes > 0);

  return len_written;
}

static bool TestZeroCopyBioPairs() {
  // Test read and write, especially triggering the ring buffer wrap-around.
  uint8_t bio1_application_send_buffer[1024];
  uint8_t bio2_application_recv_buffer[1024];

  const size_t kLengths[] = {254, 255, 256, 257, 510, 511, 512, 513};

  // These trigger ring buffer wrap around.
  const size_t kPartialLengths[] = {0, 1, 2, 3, 128, 255, 256, 257, 511, 512};

  static const size_t kBufferSize = 512;

  srand(1);
  for (size_t i = 0; i < sizeof(bio1_application_send_buffer); i++) {
    bio1_application_send_buffer[i] = rand() & 255;
  }

  // Transfer bytes from bio1_application_send_buffer to
  // bio2_application_recv_buffer in various ways.
  for (size_t i = 0; i < sizeof(kLengths) / sizeof(kLengths[0]); i++) {
    for (size_t j = 0; j < sizeof(kPartialLengths) / sizeof(kPartialLengths[0]);
         j++) {
      size_t total_write = 0;
      size_t total_read = 0;

      BIO *bio1, *bio2;
      if (!BIO_new_bio_pair(&bio1, kBufferSize, &bio2, kBufferSize)) {
        return false;
      }
      ScopedBIO bio1_scoper(bio1);
      ScopedBIO bio2_scoper(bio2);

      total_write += BioWriteZeroCopyWrapper(
          bio1, bio1_application_send_buffer, kLengths[i]);

      // This tests interleaved read/write calls. Do a read between zero copy
      // write calls.
      uint8_t *write_buf;
      size_t write_buf_offset;
      size_t available_bytes;
      if (!BIO_zero_copy_get_write_buf(bio1, &write_buf, &write_buf_offset,
                                       &available_bytes)) {
        return false;
      }

      // Free kPartialLengths[j] bytes in the beginning of bio1 write buffer.
      // This enables ring buffer wrap around for the next write.
      total_read += BIO_read(bio2, bio2_application_recv_buffer + total_read,
                             kPartialLengths[j]);

      size_t interleaved_write_len = std::min(kPartialLengths[j],
                                              available_bytes);

      // Write the data for the interleaved write call. If the buffer becomes
      // empty after a read, the write offset is normally set to 0. Check that
      // this does not happen for interleaved read/write and that
      // |write_buf_offset| is still valid.
      memcpy(write_buf + write_buf_offset,
             bio1_application_send_buffer + total_write, interleaved_write_len);
      if (BIO_zero_copy_get_write_buf_done(bio1, interleaved_write_len)) {
        total_write += interleaved_write_len;
      }

      // Do another write in case |write_buf_offset| was wrapped.
      total_write += BioWriteZeroCopyWrapper(
          bio1, bio1_application_send_buffer + total_write,
          kPartialLengths[j] - interleaved_write_len);

      // Drain the rest.
      size_t bytes_left = BIO_pending(bio2);
      total_read += BioReadZeroCopyWrapper(
          bio2, bio2_application_recv_buffer + total_read, bytes_left);

      if (total_read != total_write) {
        fprintf(stderr, "Lengths not equal in round (%u, %u)\n", (unsigned)i,
                (unsigned)j);
        return false;
      }
      if (total_read > kLengths[i] + kPartialLengths[j]) {
        fprintf(stderr, "Bad lengths in round (%u, %u)\n", (unsigned)i,
                (unsigned)j);
        return false;
      }
      if (memcmp(bio1_application_send_buffer, bio2_application_recv_buffer,
                 total_read) != 0) {
        fprintf(stderr, "Buffers not equal in round (%u, %u)\n", (unsigned)i,
                (unsigned)j);
        return false;
      }
    }
  }

  return true;
}

static bool TestPrintf() {
  // Test a short output, a very long one, and various sizes around
  // 256 (the size of the buffer) to ensure edge cases are correct.
  static const size_t kLengths[] = { 5, 250, 251, 252, 253, 254, 1023 };

  ScopedBIO bio(BIO_new(BIO_s_mem()));
  if (!bio) {
    fprintf(stderr, "BIO_new failed\n");
    return false;
  }

  for (size_t i = 0; i < sizeof(kLengths) / sizeof(kLengths[0]); i++) {
    char string[1024];
    if (kLengths[i] >= sizeof(string)) {
      fprintf(stderr, "Bad test string length\n");
      return false;
    }
    memset(string, 'a', sizeof(string));
    string[kLengths[i]] = '\0';

    int ret = BIO_printf(bio.get(), "test %s", string);
    if (ret < 0 || static_cast<size_t>(ret) != 5 + kLengths[i]) {
      fprintf(stderr, "BIO_printf failed: %d\n", ret);
      return false;
    }
    const uint8_t *contents;
    size_t len;
    if (!BIO_mem_contents(bio.get(), &contents, &len)) {
      fprintf(stderr, "BIO_mem_contents failed\n");
      return false;
    }
    if (len != 5 + kLengths[i] ||
        strncmp((const char *)contents, "test ", 5) != 0 ||
        strncmp((const char *)contents + 5, string, kLengths[i]) != 0) {
      fprintf(stderr, "Contents did not match: %.*s\n", (int)len, contents);
      return false;
    }

    if (!BIO_reset(bio.get())) {
      fprintf(stderr, "BIO_reset failed\n");
      return false;
    }
  }

  return true;
}

static bool ReadASN1(bool should_succeed, const uint8_t *data, size_t data_len,
                     size_t expected_len, size_t max_len) {
  ScopedBIO bio(BIO_new_mem_buf(const_cast<uint8_t*>(data), data_len));

  uint8_t *out;
  size_t out_len;
  int ok = BIO_read_asn1(bio.get(), &out, &out_len, max_len);
  if (!ok) {
    out = nullptr;
  }
  ScopedOpenSSLBytes out_storage(out);

  if (should_succeed != (ok == 1)) {
    return false;
  }

  if (should_succeed &&
      (out_len != expected_len || memcmp(data, out, expected_len) != 0)) {
    return false;
  }

  return true;
}

static bool TestASN1() {
  static const uint8_t kData1[] = {0x30, 2, 1, 2, 0, 0};
  static const uint8_t kData2[] = {0x30, 3, 1, 2};  /* truncated */
  static const uint8_t kData3[] = {0x30, 0x81, 1, 1};  /* should be short len */
  static const uint8_t kData4[] = {0x30, 0x82, 0, 1, 1};  /* zero padded. */

  if (!ReadASN1(true, kData1, sizeof(kData1), 4, 100) ||
      !ReadASN1(false, kData2, sizeof(kData2), 0, 100) ||
      !ReadASN1(false, kData3, sizeof(kData3), 0, 100) ||
      !ReadASN1(false, kData4, sizeof(kData4), 0, 100)) {
    return false;
  }

  static const size_t kLargePayloadLen = 8000;
  static const uint8_t kLargePrefix[] = {0x30, 0x82, kLargePayloadLen >> 8,
                                         kLargePayloadLen & 0xff};
  ScopedOpenSSLBytes large(reinterpret_cast<uint8_t *>(
      OPENSSL_malloc(sizeof(kLargePrefix) + kLargePayloadLen)));
  if (!large) {
    return false;
  }
  memset(large.get() + sizeof(kLargePrefix), 0, kLargePayloadLen);
  memcpy(large.get(), kLargePrefix, sizeof(kLargePrefix));

  if (!ReadASN1(true, large.get(), sizeof(kLargePrefix) + kLargePayloadLen,
                sizeof(kLargePrefix) + kLargePayloadLen,
                kLargePayloadLen * 2)) {
    fprintf(stderr, "Large payload test failed.\n");
    return false;
  }

  if (!ReadASN1(false, large.get(), sizeof(kLargePrefix) + kLargePayloadLen,
                sizeof(kLargePrefix) + kLargePayloadLen,
                kLargePayloadLen - 1)) {
    fprintf(stderr, "max_len test failed.\n");
    return false;
  }

  static const uint8_t kIndefPrefix[] = {0x30, 0x80};
  memcpy(large.get(), kIndefPrefix, sizeof(kIndefPrefix));
  if (!ReadASN1(true, large.get(), sizeof(kLargePrefix) + kLargePayloadLen,
                sizeof(kLargePrefix) + kLargePayloadLen,
                kLargePayloadLen*2)) {
    fprintf(stderr, "indefinite length test failed.\n");
    return false;
  }

  if (!ReadASN1(false, large.get(), sizeof(kLargePrefix) + kLargePayloadLen,
                sizeof(kLargePrefix) + kLargePayloadLen,
                kLargePayloadLen-1)) {
    fprintf(stderr, "indefinite length, max_len test failed.\n");
    return false;
  }

  return true;
}

int main(void) {
  CRYPTO_library_init();
  ERR_load_crypto_strings();

#if defined(OPENSSL_WINDOWS)
  // Initialize Winsock.
  WORD wsa_version = MAKEWORD(2, 2);
  WSADATA wsa_data;
  int wsa_err = WSAStartup(wsa_version, &wsa_data);
  if (wsa_err != 0) {
    fprintf(stderr, "WSAStartup failed: %d\n", wsa_err);
    return 1;
  }
  if (wsa_data.wVersion != wsa_version) {
    fprintf(stderr, "Didn't get expected version: %x\n", wsa_data.wVersion);
    return 1;
  }
#endif

  if (!TestSocketConnect() ||
      !TestPrintf() ||
      !TestZeroCopyBioPairs() ||
      !TestASN1()) {
    return 1;
  }

  printf("PASS\n");
  return 0;
}
