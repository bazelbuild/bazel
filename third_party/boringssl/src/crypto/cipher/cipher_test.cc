/*
 * Written by Dr Stephen N Henson (steve@openssl.org) for the OpenSSL
 * project.
 */
/* ====================================================================
 * Copyright (c) 2015 The OpenSSL Project.  All rights reserved.
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
 */

#include <stdlib.h>
#include <string.h>

#include <string>
#include <vector>

#include <openssl/cipher.h>
#include <openssl/crypto.h>
#include <openssl/err.h>

#include "../test/file_test.h"
#include "../test/scoped_types.h"
#include "../test/stl_compat.h"


static const EVP_CIPHER *GetCipher(const std::string &name) {
  if (name == "DES-CBC") {
    return EVP_des_cbc();
  } else if (name == "DES-ECB") {
    return EVP_des_ecb();
  } else if (name == "DES-EDE") {
    return EVP_des_ede();
  } else if (name == "DES-EDE-CBC") {
    return EVP_des_ede_cbc();
  } else if (name == "DES-EDE3-CBC") {
    return EVP_des_ede3_cbc();
  } else if (name == "RC4") {
    return EVP_rc4();
  } else if (name == "AES-128-ECB") {
    return EVP_aes_128_ecb();
  } else if (name == "AES-256-ECB") {
    return EVP_aes_256_ecb();
  } else if (name == "AES-128-CBC") {
    return EVP_aes_128_cbc();
  } else if (name == "AES-128-GCM") {
    return EVP_aes_128_gcm();
  } else if (name == "AES-128-OFB") {
    return EVP_aes_128_ofb();
  } else if (name == "AES-192-CBC") {
    return EVP_aes_192_cbc();
  } else if (name == "AES-192-ECB") {
    return EVP_aes_192_ecb();
  } else if (name == "AES-256-CBC") {
    return EVP_aes_256_cbc();
  } else if (name == "AES-128-CTR") {
    return EVP_aes_128_ctr();
  } else if (name == "AES-256-CTR") {
    return EVP_aes_256_ctr();
  } else if (name == "AES-256-GCM") {
    return EVP_aes_256_gcm();
  } else if (name == "AES-256-OFB") {
    return EVP_aes_256_ofb();
  }
  return nullptr;
}

static bool TestOperation(FileTest *t,
                          const EVP_CIPHER *cipher,
                          bool encrypt,
                          bool streaming,
                          const std::vector<uint8_t> &key,
                          const std::vector<uint8_t> &iv,
                          const std::vector<uint8_t> &plaintext,
                          const std::vector<uint8_t> &ciphertext,
                          const std::vector<uint8_t> &aad,
                          const std::vector<uint8_t> &tag) {
  const std::vector<uint8_t> *in, *out;
  if (encrypt) {
    in = &plaintext;
    out = &ciphertext;
  } else {
    in = &ciphertext;
    out = &plaintext;
  }

  bool is_aead = EVP_CIPHER_mode(cipher) == EVP_CIPH_GCM_MODE;

  ScopedEVP_CIPHER_CTX ctx;
  if (!EVP_CipherInit_ex(ctx.get(), cipher, nullptr, nullptr, nullptr,
                         encrypt ? 1 : 0)) {
    return false;
  }
  if (t->HasAttribute("IV")) {
    if (is_aead) {
      if (!EVP_CIPHER_CTX_ctrl(ctx.get(), EVP_CTRL_GCM_SET_IVLEN,
                               iv.size(), 0)) {
        return false;
      }
    } else if (iv.size() != (size_t)EVP_CIPHER_CTX_iv_length(ctx.get())) {
      t->PrintLine("Bad IV length.");
      return false;
    }
  }
  if (is_aead && !encrypt &&
      !EVP_CIPHER_CTX_ctrl(ctx.get(), EVP_CTRL_GCM_SET_TAG, tag.size(),
                           const_cast<uint8_t*>(bssl::vector_data(&tag)))) {
    return false;
  }
  // The ciphers are run with no padding. For each of the ciphers we test, the
  // output size matches the input size.
  std::vector<uint8_t> result(in->size());
  if (in->size() != out->size()) {
    t->PrintLine("Input/output size mismatch (%u vs %u).", (unsigned)in->size(),
                 (unsigned)out->size());
    return false;
  }
  // Note: the deprecated |EVP_CIPHER|-based AES-GCM API is sensitive to whether
  // parameters are NULL, so it is important to skip the |in| and |aad|
  // |EVP_CipherUpdate| calls when empty.
  int unused, result_len1 = 0, result_len2;
  if (!EVP_CIPHER_CTX_set_key_length(ctx.get(), key.size()) ||
      !EVP_CipherInit_ex(ctx.get(), nullptr, nullptr, bssl::vector_data(&key),
                         bssl::vector_data(&iv), -1) ||
      (!aad.empty() &&
       !EVP_CipherUpdate(ctx.get(), nullptr, &unused, bssl::vector_data(&aad),
                         aad.size())) ||
      !EVP_CIPHER_CTX_set_padding(ctx.get(), 0)) {
    t->PrintLine("Operation failed.");
    return false;
  }
  if (streaming) {
    for (size_t i = 0; i < in->size(); i++) {
      uint8_t c = (*in)[i];
      int len;
      if (!EVP_CipherUpdate(ctx.get(), bssl::vector_data(&result) + result_len1,
                            &len, &c, 1)) {
        t->PrintLine("Operation failed.");
        return false;
      }
      result_len1 += len;
    }
  } else if (!in->empty() &&
             !EVP_CipherUpdate(ctx.get(), bssl::vector_data(&result),
                               &result_len1, bssl::vector_data(in),
                               in->size())) {
    t->PrintLine("Operation failed.");
    return false;
  }
  if (!EVP_CipherFinal_ex(ctx.get(), bssl::vector_data(&result) + result_len1,
                          &result_len2)) {
    t->PrintLine("Operation failed.");
    return false;
  }
  result.resize(result_len1 + result_len2);
  if (!t->ExpectBytesEqual(bssl::vector_data(out), out->size(),
                           bssl::vector_data(&result), result.size())) {
    return false;
  }
  if (encrypt && is_aead) {
    uint8_t rtag[16];
    if (tag.size() > sizeof(rtag)) {
      t->PrintLine("Bad tag length.");
      return false;
    }
    if (!EVP_CIPHER_CTX_ctrl(ctx.get(), EVP_CTRL_GCM_GET_TAG, tag.size(),
                             rtag) ||
        !t->ExpectBytesEqual(bssl::vector_data(&tag), tag.size(), rtag,
                             tag.size())) {
      return false;
    }
  }
  return true;
}

static bool TestCipher(FileTest *t, void *arg) {
  std::string cipher_str;
  if (!t->GetAttribute(&cipher_str, "Cipher")) {
    return false;
  }
  const EVP_CIPHER *cipher = GetCipher(cipher_str);
  if (cipher == nullptr) {
    t->PrintLine("Unknown cipher: '%s'.", cipher_str.c_str());
    return false;
  }

  std::vector<uint8_t> key, iv, plaintext, ciphertext, aad, tag;
  if (!t->GetBytes(&key, "Key") ||
      !t->GetBytes(&plaintext, "Plaintext") ||
      !t->GetBytes(&ciphertext, "Ciphertext")) {
    return false;
  }
  if (EVP_CIPHER_iv_length(cipher) > 0 &&
      !t->GetBytes(&iv, "IV")) {
    return false;
  }
  if (EVP_CIPHER_mode(cipher) == EVP_CIPH_GCM_MODE) {
    if (!t->GetBytes(&aad, "AAD") ||
        !t->GetBytes(&tag, "Tag")) {
      return false;
    }
  }

  enum {
    kEncrypt,
    kDecrypt,
    kBoth,
  } operation = kBoth;
  if (t->HasAttribute("Operation")) {
    const std::string &str = t->GetAttributeOrDie("Operation");
    if (str == "ENCRYPT") {
      operation = kEncrypt;
    } else if (str == "DECRYPT") {
      operation = kDecrypt;
    } else {
      t->PrintLine("Unknown operation: '%s'.", str.c_str());
      return false;
    }
  }

  // By default, both directions are run, unless overridden by the operation.
  if (operation != kDecrypt) {
    if (!TestOperation(t, cipher, true /* encrypt */, false /* single-shot */,
                       key, iv, plaintext, ciphertext, aad, tag) ||
        !TestOperation(t, cipher, true /* encrypt */, true /* streaming */, key,
                       iv, plaintext, ciphertext, aad, tag)) {
      return false;
    }
  }
  if (operation != kEncrypt) {
    if (!TestOperation(t, cipher, false /* decrypt */, false /* single-shot */,
                       key, iv, plaintext, ciphertext, aad, tag) ||
        !TestOperation(t, cipher, false /* decrypt */, true /* streaming */,
                       key, iv, plaintext, ciphertext, aad, tag)) {
      return false;
    }
  }

  return true;
}

int main(int argc, char **argv) {
  CRYPTO_library_init();

  if (argc != 2) {
    fprintf(stderr, "%s <test file>\n", argv[0]);
    return 1;
  }

  return FileTestMain(TestCipher, nullptr, argv[1]);
}
