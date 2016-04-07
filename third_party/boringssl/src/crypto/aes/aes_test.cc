/* Copyright (c) 2015, Google Inc.
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

#include <stdio.h>
#include <string.h>

#include <openssl/aes.h>
#include <openssl/crypto.h>


static bool TestAES(const uint8_t *key, size_t key_len,
                    const uint8_t plaintext[AES_BLOCK_SIZE],
                    const uint8_t ciphertext[AES_BLOCK_SIZE]) {
  AES_KEY aes_key;
  if (AES_set_encrypt_key(key, key_len * 8, &aes_key) != 0) {
    fprintf(stderr, "AES_set_encrypt_key failed\n");
    return false;
  }

  // Test encryption.
  uint8_t block[AES_BLOCK_SIZE];
  AES_encrypt(plaintext, block, &aes_key);
  if (memcmp(block, ciphertext, AES_BLOCK_SIZE) != 0) {
    fprintf(stderr, "AES_encrypt gave the wrong output\n");
    return false;
  }

  // Test in-place encryption.
  memcpy(block, plaintext, AES_BLOCK_SIZE);
  AES_encrypt(block, block, &aes_key);
  if (memcmp(block, ciphertext, AES_BLOCK_SIZE) != 0) {
    fprintf(stderr, "AES_encrypt gave the wrong output\n");
    return false;
  }

  if (AES_set_decrypt_key(key, key_len * 8, &aes_key) != 0) {
    fprintf(stderr, "AES_set_decrypt_key failed\n");
    return false;
  }

  // Test decryption.
  AES_decrypt(ciphertext, block, &aes_key);
  if (memcmp(block, plaintext, AES_BLOCK_SIZE) != 0) {
    fprintf(stderr, "AES_decrypt gave the wrong output\n");
    return false;
  }

  // Test in-place decryption.
  memcpy(block, ciphertext, AES_BLOCK_SIZE);
  AES_decrypt(block, block, &aes_key);
  if (memcmp(block, plaintext, AES_BLOCK_SIZE) != 0) {
    fprintf(stderr, "AES_decrypt gave the wrong output\n");
    return false;
  }
  return true;
}

int main() {
  CRYPTO_library_init();

  // Test vectors from FIPS-197, Appendix C.
  if (!TestAES((const uint8_t *)"\x00\x01\x02\x03\x04\x05\x06\x07"
                                "\x08\x09\x0a\x0b\x0c\x0d\x0e\x0f",
               128 / 8,
               (const uint8_t *)"\x00\x11\x22\x33\x44\x55\x66\x77"
                                "\x88\x99\xaa\xbb\xcc\xdd\xee\xff",
               (const uint8_t *)"\x69\xc4\xe0\xd8\x6a\x7b\x04\x30"
                                "\xd8\xcd\xb7\x80\x70\xb4\xc5\x5a") ||
      !TestAES((const uint8_t *)"\x00\x01\x02\x03\x04\x05\x06\x07"
                                "\x08\x09\x0a\x0b\x0c\x0d\x0e\x0f"
                                "\x10\x11\x12\x13\x14\x15\x16\x17",
               192 / 8,
               (const uint8_t *)"\x00\x11\x22\x33\x44\x55\x66\x77"
                                "\x88\x99\xaa\xbb\xcc\xdd\xee\xff",
               (const uint8_t *)"\xdd\xa9\x7c\xa4\x86\x4c\xdf\xe0"
                                "\x6e\xaf\x70\xa0\xec\x0d\x71\x91") ||
      !TestAES((const uint8_t *)"\x00\x01\x02\x03\x04\x05\x06\x07"
                                "\x08\x09\x0a\x0b\x0c\x0d\x0e\x0f"
                                "\x10\x11\x12\x13\x14\x15\x16\x17"
                                "\x18\x19\x1a\x1b\x1c\x1d\x1e\x1f",
               256 / 8,
               (const uint8_t *)"\x00\x11\x22\x33\x44\x55\x66\x77"
                                "\x88\x99\xaa\xbb\xcc\xdd\xee\xff",
               (const uint8_t *)"\x8e\xa2\xb7\xca\x51\x67\x45\xbf"
                                "\xea\xfc\x49\x90\x4b\x49\x60\x89")) {
    return false;
  }

  printf("PASS\n");
  return 0;
}
