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

#include <stdio.h>
#include <string.h>

#include <string>
#include <vector>

#include <openssl/base64.h>
#include <openssl/bio.h>
#include <openssl/err.h>
#include <openssl/ssl.h>

#include "test/scoped_types.h"
#include "../crypto/test/test_util.h"


struct ExpectedCipher {
  unsigned long id;
  int in_group_flag;
};

struct CipherTest {
  // The rule string to apply.
  const char *rule;
  // The list of expected ciphers, in order, terminated with -1.
  const ExpectedCipher *expected;
};

// Selecting individual ciphers should work.
static const char kRule1[] =
    "ECDHE-ECDSA-CHACHA20-POLY1305:"
    "ECDHE-RSA-CHACHA20-POLY1305:"
    "ECDHE-ECDSA-AES128-GCM-SHA256:"
    "ECDHE-RSA-AES128-GCM-SHA256";

static const ExpectedCipher kExpected1[] = {
  { TLS1_CK_ECDHE_ECDSA_CHACHA20_POLY1305, 0 },
  { TLS1_CK_ECDHE_RSA_CHACHA20_POLY1305, 0 },
  { TLS1_CK_ECDHE_ECDSA_WITH_AES_128_GCM_SHA256, 0 },
  { TLS1_CK_ECDHE_RSA_WITH_AES_128_GCM_SHA256, 0 },
  { 0, 0 },
};

// + reorders selected ciphers to the end, keeping their relative
// order.
static const char kRule2[] =
    "ECDHE-ECDSA-CHACHA20-POLY1305:"
    "ECDHE-RSA-CHACHA20-POLY1305:"
    "ECDHE-ECDSA-AES128-GCM-SHA256:"
    "ECDHE-RSA-AES128-GCM-SHA256:"
    "+aRSA";

static const ExpectedCipher kExpected2[] = {
  { TLS1_CK_ECDHE_ECDSA_CHACHA20_POLY1305, 0 },
  { TLS1_CK_ECDHE_ECDSA_WITH_AES_128_GCM_SHA256, 0 },
  { TLS1_CK_ECDHE_RSA_CHACHA20_POLY1305, 0 },
  { TLS1_CK_ECDHE_RSA_WITH_AES_128_GCM_SHA256, 0 },
  { 0, 0 },
};

// ! banishes ciphers from future selections.
static const char kRule3[] =
    "!aRSA:"
    "ECDHE-ECDSA-CHACHA20-POLY1305:"
    "ECDHE-RSA-CHACHA20-POLY1305:"
    "ECDHE-ECDSA-AES128-GCM-SHA256:"
    "ECDHE-RSA-AES128-GCM-SHA256";

static const ExpectedCipher kExpected3[] = {
  { TLS1_CK_ECDHE_ECDSA_CHACHA20_POLY1305, 0 },
  { TLS1_CK_ECDHE_ECDSA_WITH_AES_128_GCM_SHA256, 0 },
  { 0, 0 },
};

// Multiple masks can be ANDed in a single rule.
static const char kRule4[] = "kRSA+AESGCM+AES128";

static const ExpectedCipher kExpected4[] = {
  { TLS1_CK_RSA_WITH_AES_128_GCM_SHA256, 0 },
  { 0, 0 },
};

// - removes selected ciphers, but preserves their order for future
// selections. Select AES_128_GCM, but order the key exchanges RSA,
// DHE_RSA, ECDHE_RSA.
static const char kRule5[] =
    "ALL:-kECDHE:-kDHE:-kRSA:-ALL:"
    "AESGCM+AES128+aRSA";

static const ExpectedCipher kExpected5[] = {
  { TLS1_CK_RSA_WITH_AES_128_GCM_SHA256, 0 },
  { TLS1_CK_DHE_RSA_WITH_AES_128_GCM_SHA256, 0 },
  { TLS1_CK_ECDHE_RSA_WITH_AES_128_GCM_SHA256, 0 },
  { 0, 0 },
};

// Unknown selectors are no-ops.
static const char kRule6[] =
    "ECDHE-ECDSA-CHACHA20-POLY1305:"
    "ECDHE-RSA-CHACHA20-POLY1305:"
    "ECDHE-ECDSA-AES128-GCM-SHA256:"
    "ECDHE-RSA-AES128-GCM-SHA256:"
    "BOGUS1:-BOGUS2:+BOGUS3:!BOGUS4";

static const ExpectedCipher kExpected6[] = {
  { TLS1_CK_ECDHE_ECDSA_CHACHA20_POLY1305, 0 },
  { TLS1_CK_ECDHE_RSA_CHACHA20_POLY1305, 0 },
  { TLS1_CK_ECDHE_ECDSA_WITH_AES_128_GCM_SHA256, 0 },
  { TLS1_CK_ECDHE_RSA_WITH_AES_128_GCM_SHA256, 0 },
  { 0, 0 },
};

// Square brackets specify equi-preference groups.
static const char kRule7[] =
    "[ECDHE-ECDSA-CHACHA20-POLY1305|ECDHE-ECDSA-AES128-GCM-SHA256]:"
    "[ECDHE-RSA-CHACHA20-POLY1305]:"
    "ECDHE-RSA-AES128-GCM-SHA256";

static const ExpectedCipher kExpected7[] = {
  { TLS1_CK_ECDHE_ECDSA_CHACHA20_POLY1305, 1 },
  { TLS1_CK_ECDHE_ECDSA_WITH_AES_128_GCM_SHA256, 0 },
  { TLS1_CK_ECDHE_RSA_CHACHA20_POLY1305, 0 },
  { TLS1_CK_ECDHE_RSA_WITH_AES_128_GCM_SHA256, 0 },
  { 0, 0 },
};

// @STRENGTH performs a stable strength-sort of the selected
// ciphers and only the selected ciphers.
static const char kRule8[] =
    // To simplify things, banish all but {ECDHE_RSA,RSA} x
    // {CHACHA20,AES_256_CBC,AES_128_CBC,RC4} x SHA1.
    "!kEDH:!AESGCM:!3DES:!SHA256:!MD5:!SHA384:"
    // Order some ciphers backwards by strength.
    "ALL:-CHACHA20:-AES256:-AES128:-RC4:-ALL:"
    // Select ECDHE ones and sort them by strength. Ties should resolve
    // based on the order above.
    "kECDHE:@STRENGTH:-ALL:"
    // Now bring back everything uses RSA. ECDHE_RSA should be first,
    // sorted by strength. Then RSA, backwards by strength.
    "aRSA";

static const ExpectedCipher kExpected8[] = {
  { TLS1_CK_ECDHE_RSA_WITH_AES_256_CBC_SHA, 0 },
  { TLS1_CK_ECDHE_RSA_CHACHA20_POLY1305, 0 },
  { TLS1_CK_ECDHE_RSA_WITH_RC4_128_SHA, 0 },
  { TLS1_CK_ECDHE_RSA_WITH_AES_128_CBC_SHA, 0 },
  { SSL3_CK_RSA_RC4_128_SHA, 0 },
  { TLS1_CK_RSA_WITH_AES_128_SHA, 0 },
  { TLS1_CK_RSA_WITH_AES_256_SHA, 0 },
  { 0, 0 },
};

// Exact ciphers may not be used in multi-part rules; they are treated
// as unknown aliases.
static const char kRule9[] =
    "ECDHE-ECDSA-CHACHA20-POLY1305:"
    "ECDHE-RSA-CHACHA20-POLY1305:"
    "!ECDHE-RSA-CHACHA20-POLY1305+RSA:"
    "!ECDSA+ECDHE-ECDSA-CHACHA20-POLY1305";

static const ExpectedCipher kExpected9[] = {
  { TLS1_CK_ECDHE_ECDSA_CHACHA20_POLY1305, 0 },
  { TLS1_CK_ECDHE_RSA_CHACHA20_POLY1305, 0 },
  { 0, 0 },
};

static CipherTest kCipherTests[] = {
  { kRule1, kExpected1 },
  { kRule2, kExpected2 },
  { kRule3, kExpected3 },
  { kRule4, kExpected4 },
  { kRule5, kExpected5 },
  { kRule6, kExpected6 },
  { kRule7, kExpected7 },
  { kRule8, kExpected8 },
  { kRule9, kExpected9 },
  { NULL, NULL },
};

static const char *kBadRules[] = {
  // Invalid brackets.
  "[ECDHE-RSA-CHACHA20-POLY1305|ECDHE-RSA-AES128-GCM-SHA256",
  "RSA]",
  "[[RSA]]",
  // Operators inside brackets.
  "[+RSA]",
  // Unknown directive.
  "@BOGUS",
  // Empty cipher lists error at SSL_CTX_set_cipher_list.
  "",
  "BOGUS",
  // COMPLEMENTOFDEFAULT is empty.
  "COMPLEMENTOFDEFAULT",
  // Invalid command.
  "?BAR",
  // Special operators are not allowed if groups are used.
  "[ECDHE-RSA-CHACHA20-POLY1305|ECDHE-RSA-AES128-GCM-SHA256]:+FOO",
  "[ECDHE-RSA-CHACHA20-POLY1305|ECDHE-RSA-AES128-GCM-SHA256]:!FOO",
  "[ECDHE-RSA-CHACHA20-POLY1305|ECDHE-RSA-AES128-GCM-SHA256]:-FOO",
  "[ECDHE-RSA-CHACHA20-POLY1305|ECDHE-RSA-AES128-GCM-SHA256]:@STRENGTH",
  NULL,
};

static const char *kMustNotIncludeNull[] = {
  "ALL",
  "DEFAULT",
  "ALL:!eNULL",
  "ALL:!NULL",
  "FIPS",
  "SHA",
  "SHA1",
  "RSA",
  "SSLv3",
  "TLSv1",
  "TLSv1.2",
  NULL
};

static void PrintCipherPreferenceList(ssl_cipher_preference_list_st *list) {
  bool in_group = false;
  for (size_t i = 0; i < sk_SSL_CIPHER_num(list->ciphers); i++) {
    const SSL_CIPHER *cipher = sk_SSL_CIPHER_value(list->ciphers, i);
    if (!in_group && list->in_group_flags[i]) {
      fprintf(stderr, "\t[\n");
      in_group = true;
    }
    fprintf(stderr, "\t");
    if (in_group) {
      fprintf(stderr, "  ");
    }
    fprintf(stderr, "%s\n", SSL_CIPHER_get_name(cipher));
    if (in_group && !list->in_group_flags[i]) {
      fprintf(stderr, "\t]\n");
      in_group = false;
    }
  }
}

static bool TestCipherRule(CipherTest *t) {
  ScopedSSL_CTX ctx(SSL_CTX_new(TLS_method()));
  if (!ctx) {
    return false;
  }

  if (!SSL_CTX_set_cipher_list(ctx.get(), t->rule)) {
    fprintf(stderr, "Error testing cipher rule '%s'\n", t->rule);
    return false;
  }

  // Compare the two lists.
  size_t i;
  for (i = 0; i < sk_SSL_CIPHER_num(ctx->cipher_list->ciphers); i++) {
    const SSL_CIPHER *cipher =
        sk_SSL_CIPHER_value(ctx->cipher_list->ciphers, i);
    if (t->expected[i].id != SSL_CIPHER_get_id(cipher) ||
        t->expected[i].in_group_flag != ctx->cipher_list->in_group_flags[i]) {
      fprintf(stderr, "Error: cipher rule '%s' evaluated to:\n", t->rule);
      PrintCipherPreferenceList(ctx->cipher_list);
      return false;
    }
  }

  if (t->expected[i].id != 0) {
    fprintf(stderr, "Error: cipher rule '%s' evaluated to:\n", t->rule);
    PrintCipherPreferenceList(ctx->cipher_list);
    return false;
  }

  return true;
}

static bool TestRuleDoesNotIncludeNull(const char *rule) {
  ScopedSSL_CTX ctx(SSL_CTX_new(SSLv23_server_method()));
  if (!ctx) {
    return false;
  }
  if (!SSL_CTX_set_cipher_list(ctx.get(), rule)) {
    fprintf(stderr, "Error: cipher rule '%s' failed\n", rule);
    return false;
  }
  for (size_t i = 0; i < sk_SSL_CIPHER_num(ctx->cipher_list->ciphers); i++) {
    if (SSL_CIPHER_is_NULL(sk_SSL_CIPHER_value(ctx->cipher_list->ciphers, i))) {
      fprintf(stderr, "Error: cipher rule '%s' includes NULL\n",rule);
      return false;
    }
  }
  return true;
}

static bool TestCipherRules() {
  for (size_t i = 0; kCipherTests[i].rule != NULL; i++) {
    if (!TestCipherRule(&kCipherTests[i])) {
      return false;
    }
  }

  for (size_t i = 0; kBadRules[i] != NULL; i++) {
    ScopedSSL_CTX ctx(SSL_CTX_new(SSLv23_server_method()));
    if (!ctx) {
      return false;
    }
    if (SSL_CTX_set_cipher_list(ctx.get(), kBadRules[i])) {
      fprintf(stderr, "Cipher rule '%s' unexpectedly succeeded\n", kBadRules[i]);
      return false;
    }
    ERR_clear_error();
  }

  for (size_t i = 0; kMustNotIncludeNull[i] != NULL; i++) {
    if (!TestRuleDoesNotIncludeNull(kMustNotIncludeNull[i])) {
      return false;
    }
  }

  return true;
}

// kOpenSSLSession is a serialized SSL_SESSION generated from openssl
// s_client -sess_out.
static const char kOpenSSLSession[] =
    "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"
    "i4gv7Y5oliyn";

// kCustomSession is a custom serialized SSL_SESSION generated by
// filling in missing fields from |kOpenSSLSession|. This includes
// providing |peer_sha256|, so |peer| is not serialized.
static const char kCustomSession[] =
    "MIIBdgIBAQICAwMEAsAvBCAG5Q1ndq4Yfmbeo1zwLkNRKmCXGdNgWvGT3cskV0yQ"
    "kAQwJlrlzkAWBOWiLj/jJ76D7l+UXoizP2KI2C7I2FccqMmIfFmmkUy32nIJ0mZH"
    "IWoJoQYCBFRDO46iBAICASykAwQBAqUDAgEUphAEDnd3dy5nb29nbGUuY29tqAcE"
    "BXdvcmxkqQUCAwGJwKqBpwSBpBwUQvoeOk0Kg36SYTcLEkXqKwOBfF9vE4KX0Nxe"
    "LwjcDTpsuh3qXEaZ992r1N38VDcyS6P7I6HBYN9BsNHM362zZnY27GpTw+Kwd751"
    "CLoXFPoaMOe57dbBpXoro6Pd3BTbf/Tzr88K06yEOTDKPNj3+inbMaVigtK4PLyP"
    "q+Topyzvx9USFgRvyuoxn0Hgb+R0A3j6SLRuyOdAi4gv7Y5oliynrSIEIAYGBgYG"
    "BgYGBgYGBgYGBgYGBgYGBgYGBgYGBgYGBgYGrgMEAQevAwQBBLADBAEF";

// kBoringSSLSession is a serialized SSL_SESSION generated from bssl client.
static const char kBoringSSLSession[] =
    "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"
    "lQI5kQsO/jgu0R0FmvZNPm8PBx2vLB6PYDni+jZTEznUXiYr2z2oFL0y6xgDKFIE"
    "ceWrMz3hOLsHNoRinHnqFjD0X8Ar6HFr5PkCAwEAAaOB8DCB7TAfBgNVHSMEGDAW"
    "gBRI5mj5K9KylddH2CMgEE8zmJCf1DAdBgNVHQ4EFgQUwHqYaI2J+6sFZAwRfap9"
    "ZbjKzE4wDwYDVR0TAQH/BAUwAwEB/zAOBgNVHQ8BAf8EBAMCAQYwOgYDVR0fBDMw"
    "MTAvoC2gK4YpaHR0cDovL2NybC5nZW90cnVzdC5jb20vY3Jscy9zZWN1cmVjYS5j"
    "cmwwTgYDVR0gBEcwRTBDBgRVHSAAMDswOQYIKwYBBQUHAgEWLWh0dHBzOi8vd3d3"
    "Lmdlb3RydXN0LmNvbS9yZXNvdXJjZXMvcmVwb3NpdG9yeTANBgkqhkiG9w0BAQUF"
    "AAOBgQB24RJuTksWEoYwBrKBCM/wCMfHcX5m7sLt1Dsf//DwyE7WQziwuTB9GNBV"
    "g6JqyzYRnOhIZqNtf7gT1Ef+i1pcc/yu2RsyGTirlzQUqpbS66McFAhJtrvlke+D"
    "NusdVm/K2rxzY5Dkf3s+Iss9B+1fOHSc4wNQTqGvmO5h8oQ/Eg==";

// kBadSessionExtraField is a custom serialized SSL_SESSION generated by replacing
// the final (optional) element of |kCustomSession| with tag number 30.
static const char kBadSessionExtraField[] =
    "MIIBdgIBAQICAwMEAsAvBCAG5Q1ndq4Yfmbeo1zwLkNRKmCXGdNgWvGT3cskV0yQ"
    "kAQwJlrlzkAWBOWiLj/jJ76D7l+UXoizP2KI2C7I2FccqMmIfFmmkUy32nIJ0mZH"
    "IWoJoQYCBFRDO46iBAICASykAwQBAqUDAgEUphAEDnd3dy5nb29nbGUuY29tqAcE"
    "BXdvcmxkqQUCAwGJwKqBpwSBpBwUQvoeOk0Kg36SYTcLEkXqKwOBfF9vE4KX0Nxe"
    "LwjcDTpsuh3qXEaZ992r1N38VDcyS6P7I6HBYN9BsNHM362zZnY27GpTw+Kwd751"
    "CLoXFPoaMOe57dbBpXoro6Pd3BTbf/Tzr88K06yEOTDKPNj3+inbMaVigtK4PLyP"
    "q+Topyzvx9USFgRvyuoxn0Hgb+R0A3j6SLRuyOdAi4gv7Y5oliynrSIEIAYGBgYG"
    "BgYGBgYGBgYGBgYGBgYGBgYGBgYGBgYGBgYGrgMEAQevAwQBBL4DBAEF";

// kBadSessionVersion is a custom serialized SSL_SESSION generated by replacing
// the version of |kCustomSession| with 2.
static const char kBadSessionVersion[] =
    "MIIBdgIBAgICAwMEAsAvBCAG5Q1ndq4Yfmbeo1zwLkNRKmCXGdNgWvGT3cskV0yQ"
    "kAQwJlrlzkAWBOWiLj/jJ76D7l+UXoizP2KI2C7I2FccqMmIfFmmkUy32nIJ0mZH"
    "IWoJoQYCBFRDO46iBAICASykAwQBAqUDAgEUphAEDnd3dy5nb29nbGUuY29tqAcE"
    "BXdvcmxkqQUCAwGJwKqBpwSBpBwUQvoeOk0Kg36SYTcLEkXqKwOBfF9vE4KX0Nxe"
    "LwjcDTpsuh3qXEaZ992r1N38VDcyS6P7I6HBYN9BsNHM362zZnY27GpTw+Kwd751"
    "CLoXFPoaMOe57dbBpXoro6Pd3BTbf/Tzr88K06yEOTDKPNj3+inbMaVigtK4PLyP"
    "q+Topyzvx9USFgRvyuoxn0Hgb+R0A3j6SLRuyOdAi4gv7Y5oliynrSIEIAYGBgYG"
    "BgYGBgYGBgYGBgYGBgYGBgYGBgYGBgYGBgYGrgMEAQevAwQBBLADBAEF";

// kBadSessionTrailingData is a custom serialized SSL_SESSION with trailing data
// appended.
static const char kBadSessionTrailingData[] =
    "MIIBdgIBAQICAwMEAsAvBCAG5Q1ndq4Yfmbeo1zwLkNRKmCXGdNgWvGT3cskV0yQ"
    "kAQwJlrlzkAWBOWiLj/jJ76D7l+UXoizP2KI2C7I2FccqMmIfFmmkUy32nIJ0mZH"
    "IWoJoQYCBFRDO46iBAICASykAwQBAqUDAgEUphAEDnd3dy5nb29nbGUuY29tqAcE"
    "BXdvcmxkqQUCAwGJwKqBpwSBpBwUQvoeOk0Kg36SYTcLEkXqKwOBfF9vE4KX0Nxe"
    "LwjcDTpsuh3qXEaZ992r1N38VDcyS6P7I6HBYN9BsNHM362zZnY27GpTw+Kwd751"
    "CLoXFPoaMOe57dbBpXoro6Pd3BTbf/Tzr88K06yEOTDKPNj3+inbMaVigtK4PLyP"
    "q+Topyzvx9USFgRvyuoxn0Hgb+R0A3j6SLRuyOdAi4gv7Y5oliynrSIEIAYGBgYG"
    "BgYGBgYGBgYGBgYGBgYGBgYGBgYGBgYGBgYGrgMEAQevAwQBBLADBAEFAAAA";

static bool DecodeBase64(std::vector<uint8_t> *out, const char *in) {
  size_t len;
  if (!EVP_DecodedLength(&len, strlen(in))) {
    fprintf(stderr, "EVP_DecodedLength failed\n");
    return false;
  }

  out->resize(len);
  if (!EVP_DecodeBase64(bssl::vector_data(out), &len, len, (const uint8_t *)in,
                        strlen(in))) {
    fprintf(stderr, "EVP_DecodeBase64 failed\n");
    return false;
  }
  out->resize(len);
  return true;
}

static bool TestSSL_SESSIONEncoding(const char *input_b64) {
  const uint8_t *cptr;
  uint8_t *ptr;

  // Decode the input.
  std::vector<uint8_t> input;
  if (!DecodeBase64(&input, input_b64)) {
    return false;
  }

  // Verify the SSL_SESSION decodes.
  ScopedSSL_SESSION session(SSL_SESSION_from_bytes(bssl::vector_data(&input),
                                                   input.size()));
  if (!session) {
    fprintf(stderr, "SSL_SESSION_from_bytes failed\n");
    return false;
  }

  // Verify the SSL_SESSION encoding round-trips.
  size_t encoded_len;
  ScopedOpenSSLBytes encoded;
  uint8_t *encoded_raw;
  if (!SSL_SESSION_to_bytes(session.get(), &encoded_raw, &encoded_len)) {
    fprintf(stderr, "SSL_SESSION_to_bytes failed\n");
    return false;
  }
  encoded.reset(encoded_raw);
  if (encoded_len != input.size() ||
      memcmp(bssl::vector_data(&input), encoded.get(), input.size()) != 0) {
    fprintf(stderr, "SSL_SESSION_to_bytes did not round-trip\n");
    hexdump(stderr, "Before: ", input.data(), input.size());
    hexdump(stderr, "After:  ", encoded_raw, encoded_len);
    return false;
  }

  // Verify the SSL_SESSION also decodes with the legacy API.
  cptr = bssl::vector_data(&input);
  session.reset(d2i_SSL_SESSION(NULL, &cptr, input.size()));
  if (!session || cptr != bssl::vector_data(&input) + input.size()) {
    fprintf(stderr, "d2i_SSL_SESSION failed\n");
    return false;
  }

  // Verify the SSL_SESSION encoding round-trips via the legacy API.
  int len = i2d_SSL_SESSION(session.get(), NULL);
  if (len < 0 || (size_t)len != input.size()) {
    fprintf(stderr, "i2d_SSL_SESSION(NULL) returned invalid length\n");
    return false;
  }

  encoded.reset((uint8_t *)OPENSSL_malloc(input.size()));
  if (!encoded) {
    fprintf(stderr, "malloc failed\n");
    return false;
  }

  ptr = encoded.get();
  len = i2d_SSL_SESSION(session.get(), &ptr);
  if (len < 0 || (size_t)len != input.size()) {
    fprintf(stderr, "i2d_SSL_SESSION returned invalid length\n");
    return false;
  }
  if (ptr != encoded.get() + input.size()) {
    fprintf(stderr, "i2d_SSL_SESSION did not advance ptr correctly\n");
    return false;
  }
  if (memcmp(bssl::vector_data(&input), encoded.get(), input.size()) != 0) {
    fprintf(stderr, "i2d_SSL_SESSION did not round-trip\n");
    return false;
  }

  return true;
}

static bool TestBadSSL_SESSIONEncoding(const char *input_b64) {
  std::vector<uint8_t> input;
  if (!DecodeBase64(&input, input_b64)) {
    return false;
  }

  // Verify that the SSL_SESSION fails to decode.
  ScopedSSL_SESSION session(SSL_SESSION_from_bytes(bssl::vector_data(&input),
                                                   input.size()));
  if (session) {
    fprintf(stderr, "SSL_SESSION_from_bytes unexpectedly succeeded\n");
    return false;
  }
  ERR_clear_error();
  return true;
}

static bool TestDefaultVersion(uint16_t version,
                               const SSL_METHOD *(*method)(void)) {
  ScopedSSL_CTX ctx(SSL_CTX_new(method()));
  if (!ctx) {
    return false;
  }
  return ctx->min_version == version && ctx->max_version == version;
}

static bool CipherGetRFCName(std::string *out, uint16_t value) {
  const SSL_CIPHER *cipher = SSL_get_cipher_by_value(value);
  if (cipher == NULL) {
    return false;
  }
  ScopedOpenSSLString rfc_name(SSL_CIPHER_get_rfc_name(cipher));
  if (!rfc_name) {
    return false;
  }
  out->assign(rfc_name.get());
  return true;
}

typedef struct {
  int id;
  const char *rfc_name;
} CIPHER_RFC_NAME_TEST;

static const CIPHER_RFC_NAME_TEST kCipherRFCNameTests[] = {
  { SSL3_CK_RSA_DES_192_CBC3_SHA, "TLS_RSA_WITH_3DES_EDE_CBC_SHA" },
  { SSL3_CK_RSA_RC4_128_MD5, "TLS_RSA_WITH_RC4_MD5" },
  { TLS1_CK_RSA_WITH_AES_128_SHA, "TLS_RSA_WITH_AES_128_CBC_SHA" },
  { TLS1_CK_DHE_RSA_WITH_AES_256_SHA, "TLS_DHE_RSA_WITH_AES_256_CBC_SHA" },
  { TLS1_CK_DHE_RSA_WITH_AES_256_SHA256,
    "TLS_DHE_RSA_WITH_AES_256_CBC_SHA256" },
  { TLS1_CK_ECDHE_RSA_WITH_AES_128_SHA256,
    "TLS_ECDHE_RSA_WITH_AES_128_CBC_SHA256" },
  { TLS1_CK_ECDHE_RSA_WITH_AES_256_SHA384,
    "TLS_ECDHE_RSA_WITH_AES_256_CBC_SHA384" },
  { TLS1_CK_ECDHE_RSA_WITH_AES_128_GCM_SHA256,
    "TLS_ECDHE_RSA_WITH_AES_128_GCM_SHA256" },
  { TLS1_CK_ECDHE_ECDSA_WITH_AES_128_GCM_SHA256,
    "TLS_ECDHE_ECDSA_WITH_AES_128_GCM_SHA256" },
  { TLS1_CK_ECDHE_ECDSA_WITH_AES_256_GCM_SHA384,
    "TLS_ECDHE_ECDSA_WITH_AES_256_GCM_SHA384" },
  { TLS1_CK_PSK_WITH_RC4_128_SHA, "TLS_PSK_WITH_RC4_SHA" },
  { TLS1_CK_ECDHE_PSK_WITH_AES_128_CBC_SHA,
    "TLS_ECDHE_PSK_WITH_AES_128_CBC_SHA" },
  // These names are non-standard:
  { TLS1_CK_ECDHE_RSA_CHACHA20_POLY1305,
    "TLS_ECDHE_RSA_WITH_CHACHA20_POLY1305_SHA256" },
  { TLS1_CK_ECDHE_ECDSA_CHACHA20_POLY1305,
    "TLS_ECDHE_ECDSA_WITH_CHACHA20_POLY1305_SHA256" },
};

static bool TestCipherGetRFCName(void) {
  for (size_t i = 0;
       i < sizeof(kCipherRFCNameTests) / sizeof(kCipherRFCNameTests[0]); i++) {
    const CIPHER_RFC_NAME_TEST *test = &kCipherRFCNameTests[i];
    std::string rfc_name;
    if (!CipherGetRFCName(&rfc_name, test->id & 0xffff)) {
      fprintf(stderr, "SSL_CIPHER_get_rfc_name failed\n");
      return false;
    }
    if (rfc_name != test->rfc_name) {
      fprintf(stderr, "SSL_CIPHER_get_rfc_name: got '%s', wanted '%s'\n",
              rfc_name.c_str(), test->rfc_name);
      return false;
    }
  }
  return true;
}

// CreateSessionWithTicket returns a sample |SSL_SESSION| with the ticket
// replaced for one of length |ticket_len| or nullptr on failure.
static ScopedSSL_SESSION CreateSessionWithTicket(size_t ticket_len) {
  std::vector<uint8_t> der;
  if (!DecodeBase64(&der, kOpenSSLSession)) {
    return nullptr;
  }
  ScopedSSL_SESSION session(SSL_SESSION_from_bytes(bssl::vector_data(&der),
                                                   der.size()));
  if (!session) {
    return nullptr;
  }

  // Swap out the ticket for a garbage one.
  OPENSSL_free(session->tlsext_tick);
  session->tlsext_tick = reinterpret_cast<uint8_t*>(OPENSSL_malloc(ticket_len));
  if (session->tlsext_tick == nullptr) {
    return nullptr;
  }
  memset(session->tlsext_tick, 'a', ticket_len);
  session->tlsext_ticklen = ticket_len;
  return session;
}

// GetClientHelloLen creates a client SSL connection with a ticket of length
// |ticket_len| and records the ClientHello. It returns the length of the
// ClientHello, not including the record header, on success and zero on error.
static size_t GetClientHelloLen(size_t ticket_len) {
  ScopedSSL_CTX ctx(SSL_CTX_new(TLS_method()));
  ScopedSSL_SESSION session = CreateSessionWithTicket(ticket_len);
  if (!ctx || !session) {
    return 0;
  }
  ScopedSSL ssl(SSL_new(ctx.get()));
  ScopedBIO bio(BIO_new(BIO_s_mem()));
  if (!ssl || !bio || !SSL_set_session(ssl.get(), session.get())) {
    return 0;
  }
  // Do not configure a reading BIO, but record what's written to a memory BIO.
  SSL_set_bio(ssl.get(), nullptr /* rbio */, BIO_up_ref(bio.get()));
  int ret = SSL_connect(ssl.get());
  if (ret > 0) {
    // SSL_connect should fail without a BIO to write to.
    return 0;
  }
  ERR_clear_error();

  const uint8_t *unused;
  size_t client_hello_len;
  if (!BIO_mem_contents(bio.get(), &unused, &client_hello_len) ||
      client_hello_len <= SSL3_RT_HEADER_LENGTH) {
    return 0;
  }
  return client_hello_len - SSL3_RT_HEADER_LENGTH;
}

struct PaddingTest {
  size_t input_len, padded_len;
};

static const PaddingTest kPaddingTests[] = {
    // ClientHellos of length below 0x100 do not require padding.
    {0xfe, 0xfe},
    {0xff, 0xff},
    // ClientHellos of length 0x100 through 0x1fb are padded up to 0x200.
    {0x100, 0x200},
    {0x123, 0x200},
    {0x1fb, 0x200},
    // ClientHellos of length 0x1fc through 0x1ff get padded beyond 0x200. The
    // padding extension takes a minimum of four bytes plus one required content
    // byte. (To work around yet more server bugs, we avoid empty final
    // extensions.)
    {0x1fc, 0x201},
    {0x1fd, 0x202},
    {0x1fe, 0x203},
    {0x1ff, 0x204},
    // Finally, larger ClientHellos need no padding.
    {0x200, 0x200},
    {0x201, 0x201},
};

static bool TestPaddingExtension() {
  // Sample a baseline length.
  size_t base_len = GetClientHelloLen(1);
  if (base_len == 0) {
    return false;
  }

  for (const PaddingTest &test : kPaddingTests) {
    if (base_len > test.input_len) {
      fprintf(stderr, "Baseline ClientHello too long.\n");
      return false;
    }

    size_t padded_len = GetClientHelloLen(1 + test.input_len - base_len);
    if (padded_len != test.padded_len) {
      fprintf(stderr, "%u-byte ClientHello padded to %u bytes, not %u.\n",
              static_cast<unsigned>(test.input_len),
              static_cast<unsigned>(padded_len),
              static_cast<unsigned>(test.padded_len));
      return false;
    }
  }
  return true;
}

// Test that |SSL_get_client_CA_list| echoes back the configured parameter even
// before configuring as a server.
static bool TestClientCAList() {
  ScopedSSL_CTX ctx(SSL_CTX_new(TLS_method()));
  if (!ctx) {
    return false;
  }
  ScopedSSL ssl(SSL_new(ctx.get()));
  if (!ssl) {
    return false;
  }

  STACK_OF(X509_NAME) *stack = sk_X509_NAME_new_null();
  if (stack == nullptr) {
    return false;
  }
  // |SSL_set_client_CA_list| takes ownership.
  SSL_set_client_CA_list(ssl.get(), stack);

  return SSL_get_client_CA_list(ssl.get()) == stack;
}

int main() {
  SSL_library_init();

  if (!TestCipherRules() ||
      !TestSSL_SESSIONEncoding(kOpenSSLSession) ||
      !TestSSL_SESSIONEncoding(kCustomSession) ||
      !TestSSL_SESSIONEncoding(kBoringSSLSession) ||
      !TestBadSSL_SESSIONEncoding(kBadSessionExtraField) ||
      !TestBadSSL_SESSIONEncoding(kBadSessionVersion) ||
      !TestBadSSL_SESSIONEncoding(kBadSessionTrailingData) ||
      !TestDefaultVersion(0, &TLS_method) ||
      !TestDefaultVersion(SSL3_VERSION, &SSLv3_method) ||
      !TestDefaultVersion(TLS1_VERSION, &TLSv1_method) ||
      !TestDefaultVersion(TLS1_1_VERSION, &TLSv1_1_method) ||
      !TestDefaultVersion(TLS1_2_VERSION, &TLSv1_2_method) ||
      !TestDefaultVersion(0, &DTLS_method) ||
      !TestDefaultVersion(DTLS1_VERSION, &DTLSv1_method) ||
      !TestDefaultVersion(DTLS1_2_VERSION, &DTLSv1_2_method) ||
      !TestCipherGetRFCName() ||
      !TestPaddingExtension() ||
      !TestClientCAList()) {
    ERR_print_errors_fp(stderr);
    return 1;
  }

  printf("PASS\n");
  return 0;
}
