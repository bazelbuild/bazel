# This file is created by generate_build_files.py. Do not edit manually.

test_support_sources = [
    "src/crypto/test/test_util.cc",
    "src/crypto/test/file_test.cc",
]

def create_tests(copts):
  test_support_sources_complete = test_support_sources + \
      native.glob(["src/crypto/test/*.h"])
  native.cc_test(
      name = "aes_test",
      size = "small",
      srcs = ["src/crypto/aes/aes_test.cc"] + test_support_sources_complete,
      copts = copts,
      deps = [":crypto"],
  )

  native.cc_test(
      name = "base64_test",
      size = "small",
      srcs = ["src/crypto/base64/base64_test.cc"] + test_support_sources_complete,
      copts = copts,
      deps = [":crypto"],
  )

  native.cc_test(
      name = "bio_test",
      size = "small",
      srcs = ["src/crypto/bio/bio_test.cc"] + test_support_sources_complete,
      copts = copts,
      deps = [":crypto"],
  )

  native.cc_test(
      name = "bn_test",
      size = "small",
      srcs = ["src/crypto/bn/bn_test.cc"] + test_support_sources_complete,
      copts = copts,
      deps = [":crypto"],
  )

  native.cc_test(
      name = "bytestring_test",
      size = "small",
      srcs = ["src/crypto/bytestring/bytestring_test.cc"] + test_support_sources_complete,
      copts = copts,
      deps = [":crypto"],
  )

  native.cc_test(
      name = "aead_test_aes_128_gcm",
      size = "small",
      srcs = ["src/crypto/cipher/aead_test.cc"] + test_support_sources_complete,
      args = [
          "aes-128-gcm",
          "$(location src/crypto/cipher/test/aes_128_gcm_tests.txt)",
      ],
      copts = copts,
      data = [
          "src/crypto/cipher/test/aes_128_gcm_tests.txt",
      ],
      deps = [":crypto"],
  )

  native.cc_test(
      name = "aead_test_aes_128_key_wrap",
      size = "small",
      srcs = ["src/crypto/cipher/aead_test.cc"] + test_support_sources_complete,
      args = [
          "aes-128-key-wrap",
          "$(location src/crypto/cipher/test/aes_128_key_wrap_tests.txt)",
      ],
      copts = copts,
      data = [
          "src/crypto/cipher/test/aes_128_key_wrap_tests.txt",
      ],
      deps = [":crypto"],
  )

  native.cc_test(
      name = "aead_test_aes_256_gcm",
      size = "small",
      srcs = ["src/crypto/cipher/aead_test.cc"] + test_support_sources_complete,
      args = [
          "aes-256-gcm",
          "$(location src/crypto/cipher/test/aes_256_gcm_tests.txt)",
      ],
      copts = copts,
      data = [
          "src/crypto/cipher/test/aes_256_gcm_tests.txt",
      ],
      deps = [":crypto"],
  )

  native.cc_test(
      name = "aead_test_aes_256_key_wrap",
      size = "small",
      srcs = ["src/crypto/cipher/aead_test.cc"] + test_support_sources_complete,
      args = [
          "aes-256-key-wrap",
          "$(location src/crypto/cipher/test/aes_256_key_wrap_tests.txt)",
      ],
      copts = copts,
      data = [
          "src/crypto/cipher/test/aes_256_key_wrap_tests.txt",
      ],
      deps = [":crypto"],
  )

  native.cc_test(
      name = "aead_test_chacha20_poly1305",
      size = "small",
      srcs = ["src/crypto/cipher/aead_test.cc"] + test_support_sources_complete,
      args = [
          "chacha20-poly1305",
          "$(location src/crypto/cipher/test/chacha20_poly1305_tests.txt)",
      ],
      copts = copts,
      data = [
          "src/crypto/cipher/test/chacha20_poly1305_tests.txt",
      ],
      deps = [":crypto"],
  )

  native.cc_test(
      name = "aead_test_rc4_md5_tls",
      size = "small",
      srcs = ["src/crypto/cipher/aead_test.cc"] + test_support_sources_complete,
      args = [
          "rc4-md5-tls",
          "$(location src/crypto/cipher/test/rc4_md5_tls_tests.txt)",
      ],
      copts = copts,
      data = [
          "src/crypto/cipher/test/rc4_md5_tls_tests.txt",
      ],
      deps = [":crypto"],
  )

  native.cc_test(
      name = "aead_test_rc4_sha1_tls",
      size = "small",
      srcs = ["src/crypto/cipher/aead_test.cc"] + test_support_sources_complete,
      args = [
          "rc4-sha1-tls",
          "$(location src/crypto/cipher/test/rc4_sha1_tls_tests.txt)",
      ],
      copts = copts,
      data = [
          "src/crypto/cipher/test/rc4_sha1_tls_tests.txt",
      ],
      deps = [":crypto"],
  )

  native.cc_test(
      name = "aead_test_aes_128_cbc_sha1_tls",
      size = "small",
      srcs = ["src/crypto/cipher/aead_test.cc"] + test_support_sources_complete,
      args = [
          "aes-128-cbc-sha1-tls",
          "$(location src/crypto/cipher/test/aes_128_cbc_sha1_tls_tests.txt)",
      ],
      copts = copts,
      data = [
          "src/crypto/cipher/test/aes_128_cbc_sha1_tls_tests.txt",
      ],
      deps = [":crypto"],
  )

  native.cc_test(
      name = "aead_test_aes_128_cbc_sha1_tls_implicit_iv",
      size = "small",
      srcs = ["src/crypto/cipher/aead_test.cc"] + test_support_sources_complete,
      args = [
          "aes-128-cbc-sha1-tls-implicit-iv",
          "$(location src/crypto/cipher/test/aes_128_cbc_sha1_tls_implicit_iv_tests.txt)",
      ],
      copts = copts,
      data = [
          "src/crypto/cipher/test/aes_128_cbc_sha1_tls_implicit_iv_tests.txt",
      ],
      deps = [":crypto"],
  )

  native.cc_test(
      name = "aead_test_aes_128_cbc_sha256_tls",
      size = "small",
      srcs = ["src/crypto/cipher/aead_test.cc"] + test_support_sources_complete,
      args = [
          "aes-128-cbc-sha256-tls",
          "$(location src/crypto/cipher/test/aes_128_cbc_sha256_tls_tests.txt)",
      ],
      copts = copts,
      data = [
          "src/crypto/cipher/test/aes_128_cbc_sha256_tls_tests.txt",
      ],
      deps = [":crypto"],
  )

  native.cc_test(
      name = "aead_test_aes_256_cbc_sha1_tls",
      size = "small",
      srcs = ["src/crypto/cipher/aead_test.cc"] + test_support_sources_complete,
      args = [
          "aes-256-cbc-sha1-tls",
          "$(location src/crypto/cipher/test/aes_256_cbc_sha1_tls_tests.txt)",
      ],
      copts = copts,
      data = [
          "src/crypto/cipher/test/aes_256_cbc_sha1_tls_tests.txt",
      ],
      deps = [":crypto"],
  )

  native.cc_test(
      name = "aead_test_aes_256_cbc_sha1_tls_implicit_iv",
      size = "small",
      srcs = ["src/crypto/cipher/aead_test.cc"] + test_support_sources_complete,
      args = [
          "aes-256-cbc-sha1-tls-implicit-iv",
          "$(location src/crypto/cipher/test/aes_256_cbc_sha1_tls_implicit_iv_tests.txt)",
      ],
      copts = copts,
      data = [
          "src/crypto/cipher/test/aes_256_cbc_sha1_tls_implicit_iv_tests.txt",
      ],
      deps = [":crypto"],
  )

  native.cc_test(
      name = "aead_test_aes_256_cbc_sha256_tls",
      size = "small",
      srcs = ["src/crypto/cipher/aead_test.cc"] + test_support_sources_complete,
      args = [
          "aes-256-cbc-sha256-tls",
          "$(location src/crypto/cipher/test/aes_256_cbc_sha256_tls_tests.txt)",
      ],
      copts = copts,
      data = [
          "src/crypto/cipher/test/aes_256_cbc_sha256_tls_tests.txt",
      ],
      deps = [":crypto"],
  )

  native.cc_test(
      name = "aead_test_aes_256_cbc_sha384_tls",
      size = "small",
      srcs = ["src/crypto/cipher/aead_test.cc"] + test_support_sources_complete,
      args = [
          "aes-256-cbc-sha384-tls",
          "$(location src/crypto/cipher/test/aes_256_cbc_sha384_tls_tests.txt)",
      ],
      copts = copts,
      data = [
          "src/crypto/cipher/test/aes_256_cbc_sha384_tls_tests.txt",
      ],
      deps = [":crypto"],
  )

  native.cc_test(
      name = "aead_test_des_ede3_cbc_sha1_tls",
      size = "small",
      srcs = ["src/crypto/cipher/aead_test.cc"] + test_support_sources_complete,
      args = [
          "des-ede3-cbc-sha1-tls",
          "$(location src/crypto/cipher/test/des_ede3_cbc_sha1_tls_tests.txt)",
      ],
      copts = copts,
      data = [
          "src/crypto/cipher/test/des_ede3_cbc_sha1_tls_tests.txt",
      ],
      deps = [":crypto"],
  )

  native.cc_test(
      name = "aead_test_des_ede3_cbc_sha1_tls_implicit_iv",
      size = "small",
      srcs = ["src/crypto/cipher/aead_test.cc"] + test_support_sources_complete,
      args = [
          "des-ede3-cbc-sha1-tls-implicit-iv",
          "$(location src/crypto/cipher/test/des_ede3_cbc_sha1_tls_implicit_iv_tests.txt)",
      ],
      copts = copts,
      data = [
          "src/crypto/cipher/test/des_ede3_cbc_sha1_tls_implicit_iv_tests.txt",
      ],
      deps = [":crypto"],
  )

  native.cc_test(
      name = "aead_test_rc4_md5_ssl3",
      size = "small",
      srcs = ["src/crypto/cipher/aead_test.cc"] + test_support_sources_complete,
      args = [
          "rc4-md5-ssl3",
          "$(location src/crypto/cipher/test/rc4_md5_ssl3_tests.txt)",
      ],
      copts = copts,
      data = [
          "src/crypto/cipher/test/rc4_md5_ssl3_tests.txt",
      ],
      deps = [":crypto"],
  )

  native.cc_test(
      name = "aead_test_rc4_sha1_ssl3",
      size = "small",
      srcs = ["src/crypto/cipher/aead_test.cc"] + test_support_sources_complete,
      args = [
          "rc4-sha1-ssl3",
          "$(location src/crypto/cipher/test/rc4_sha1_ssl3_tests.txt)",
      ],
      copts = copts,
      data = [
          "src/crypto/cipher/test/rc4_sha1_ssl3_tests.txt",
      ],
      deps = [":crypto"],
  )

  native.cc_test(
      name = "aead_test_aes_128_cbc_sha1_ssl3",
      size = "small",
      srcs = ["src/crypto/cipher/aead_test.cc"] + test_support_sources_complete,
      args = [
          "aes-128-cbc-sha1-ssl3",
          "$(location src/crypto/cipher/test/aes_128_cbc_sha1_ssl3_tests.txt)",
      ],
      copts = copts,
      data = [
          "src/crypto/cipher/test/aes_128_cbc_sha1_ssl3_tests.txt",
      ],
      deps = [":crypto"],
  )

  native.cc_test(
      name = "aead_test_aes_256_cbc_sha1_ssl3",
      size = "small",
      srcs = ["src/crypto/cipher/aead_test.cc"] + test_support_sources_complete,
      args = [
          "aes-256-cbc-sha1-ssl3",
          "$(location src/crypto/cipher/test/aes_256_cbc_sha1_ssl3_tests.txt)",
      ],
      copts = copts,
      data = [
          "src/crypto/cipher/test/aes_256_cbc_sha1_ssl3_tests.txt",
      ],
      deps = [":crypto"],
  )

  native.cc_test(
      name = "aead_test_des_ede3_cbc_sha1_ssl3",
      size = "small",
      srcs = ["src/crypto/cipher/aead_test.cc"] + test_support_sources_complete,
      args = [
          "des-ede3-cbc-sha1-ssl3",
          "$(location src/crypto/cipher/test/des_ede3_cbc_sha1_ssl3_tests.txt)",
      ],
      copts = copts,
      data = [
          "src/crypto/cipher/test/des_ede3_cbc_sha1_ssl3_tests.txt",
      ],
      deps = [":crypto"],
  )

  native.cc_test(
      name = "aead_test_aes_128_ctr_hmac_sha256",
      size = "small",
      srcs = ["src/crypto/cipher/aead_test.cc"] + test_support_sources_complete,
      args = [
          "aes-128-ctr-hmac-sha256",
          "$(location src/crypto/cipher/test/aes_128_ctr_hmac_sha256.txt)",
      ],
      copts = copts,
      data = [
          "src/crypto/cipher/test/aes_128_ctr_hmac_sha256.txt",
      ],
      deps = [":crypto"],
  )

  native.cc_test(
      name = "aead_test_aes_256_ctr_hmac_sha256",
      size = "small",
      srcs = ["src/crypto/cipher/aead_test.cc"] + test_support_sources_complete,
      args = [
          "aes-256-ctr-hmac-sha256",
          "$(location src/crypto/cipher/test/aes_256_ctr_hmac_sha256.txt)",
      ],
      copts = copts,
      data = [
          "src/crypto/cipher/test/aes_256_ctr_hmac_sha256.txt",
      ],
      deps = [":crypto"],
  )

  native.cc_test(
      name = "cipher_test",
      size = "small",
      srcs = ["src/crypto/cipher/cipher_test.cc"] + test_support_sources_complete,
      args = [
          "$(location src/crypto/cipher/test/cipher_test.txt)",
      ],
      copts = copts,
      data = [
          "src/crypto/cipher/test/cipher_test.txt",
      ],
      deps = [":crypto"],
  )

  native.cc_test(
      name = "cmac_test",
      size = "small",
      srcs = ["src/crypto/cmac/cmac_test.cc"] + test_support_sources_complete,
      copts = copts,
      deps = [":crypto"],
  )

  native.cc_test(
      name = "constant_time_test",
      size = "small",
      srcs = ["src/crypto/constant_time_test.c"] + test_support_sources_complete,
      copts = copts,
      deps = [":crypto"],
  )

  native.cc_test(
      name = "dh_test",
      size = "small",
      srcs = ["src/crypto/dh/dh_test.cc"] + test_support_sources_complete,
      copts = copts,
      deps = [":crypto"],
  )

  native.cc_test(
      name = "digest_test",
      size = "small",
      srcs = ["src/crypto/digest/digest_test.cc"] + test_support_sources_complete,
      copts = copts,
      deps = [":crypto"],
  )

  native.cc_test(
      name = "dsa_test",
      size = "small",
      srcs = ["src/crypto/dsa/dsa_test.c"] + test_support_sources_complete,
      copts = copts,
      deps = [":crypto"],
  )

  native.cc_test(
      name = "ec_test",
      size = "small",
      srcs = ["src/crypto/ec/ec_test.cc"] + test_support_sources_complete,
      copts = copts,
      deps = [":crypto"],
  )

  native.cc_test(
      name = "example_mul",
      size = "small",
      srcs = ["src/crypto/ec/example_mul.c"] + test_support_sources_complete,
      copts = copts,
      deps = [":crypto"],
  )

  native.cc_test(
      name = "ecdsa_test",
      size = "small",
      srcs = ["src/crypto/ecdsa/ecdsa_test.cc"] + test_support_sources_complete,
      copts = copts,
      deps = [":crypto"],
  )

  native.cc_test(
      name = "err_test",
      size = "small",
      srcs = ["src/crypto/err/err_test.cc"] + test_support_sources_complete,
      copts = copts,
      deps = [":crypto"],
  )

  native.cc_test(
      name = "evp_extra_test",
      size = "small",
      srcs = ["src/crypto/evp/evp_extra_test.cc"] + test_support_sources_complete,
      copts = copts,
      deps = [":crypto"],
  )

  native.cc_test(
      name = "evp_test",
      size = "small",
      srcs = ["src/crypto/evp/evp_test.cc"] + test_support_sources_complete,
      args = [
          "$(location src/crypto/evp/evp_tests.txt)",
      ],
      copts = copts,
      data = [
          "src/crypto/evp/evp_tests.txt",
      ],
      deps = [":crypto"],
  )

  native.cc_test(
      name = "pbkdf_test",
      size = "small",
      srcs = ["src/crypto/evp/pbkdf_test.cc"] + test_support_sources_complete,
      copts = copts,
      deps = [":crypto"],
  )

  native.cc_test(
      name = "hkdf_test",
      size = "small",
      srcs = ["src/crypto/hkdf/hkdf_test.c"] + test_support_sources_complete,
      copts = copts,
      deps = [":crypto"],
  )

  native.cc_test(
      name = "hmac_test",
      size = "small",
      srcs = ["src/crypto/hmac/hmac_test.cc"] + test_support_sources_complete,
      args = [
          "$(location src/crypto/hmac/hmac_tests.txt)",
      ],
      copts = copts,
      data = [
          "src/crypto/hmac/hmac_tests.txt",
      ],
      deps = [":crypto"],
  )

  native.cc_test(
      name = "lhash_test",
      size = "small",
      srcs = ["src/crypto/lhash/lhash_test.c"] + test_support_sources_complete,
      copts = copts,
      deps = [":crypto"],
  )

  native.cc_test(
      name = "gcm_test",
      size = "small",
      srcs = ["src/crypto/modes/gcm_test.c"] + test_support_sources_complete,
      copts = copts,
      deps = [":crypto"],
  )

  native.cc_test(
      name = "pkcs8_test",
      size = "small",
      srcs = ["src/crypto/pkcs8/pkcs8_test.cc"] + test_support_sources_complete,
      copts = copts,
      deps = [":crypto"],
  )

  native.cc_test(
      name = "pkcs12_test",
      size = "small",
      srcs = ["src/crypto/pkcs8/pkcs12_test.cc"] + test_support_sources_complete,
      copts = copts,
      deps = [":crypto"],
  )

  native.cc_test(
      name = "poly1305_test",
      size = "small",
      srcs = ["src/crypto/poly1305/poly1305_test.cc"] + test_support_sources_complete,
      args = [
          "$(location src/crypto/poly1305/poly1305_test.txt)",
      ],
      copts = copts,
      data = [
          "src/crypto/poly1305/poly1305_test.txt",
      ],
      deps = [":crypto"],
  )

  native.cc_test(
      name = "refcount_test",
      size = "small",
      srcs = ["src/crypto/refcount_test.c"] + test_support_sources_complete,
      copts = copts,
      deps = [":crypto"],
  )

  native.cc_test(
      name = "rsa_test",
      size = "small",
      srcs = ["src/crypto/rsa/rsa_test.cc"] + test_support_sources_complete,
      copts = copts,
      deps = [":crypto"],
  )

  native.cc_test(
      name = "thread_test",
      size = "small",
      srcs = ["src/crypto/thread_test.c"] + test_support_sources_complete,
      copts = copts,
      deps = [":crypto"],
  )

  native.cc_test(
      name = "pkcs7_test",
      size = "small",
      srcs = ["src/crypto/x509/pkcs7_test.c"] + test_support_sources_complete,
      copts = copts,
      deps = [":crypto"],
  )

  native.cc_test(
      name = "tab_test",
      size = "small",
      srcs = ["src/crypto/x509v3/tab_test.c"] + test_support_sources_complete,
      copts = copts,
      deps = [":crypto"],
  )

  native.cc_test(
      name = "v3name_test",
      size = "small",
      srcs = ["src/crypto/x509v3/v3name_test.c"] + test_support_sources_complete,
      copts = copts,
      deps = [":crypto"],
  )

  native.cc_test(
      name = "pqueue_test",
      size = "small",
      srcs = ["src/ssl/pqueue/pqueue_test.c"] + test_support_sources_complete,
      copts = copts,
      deps = [
          ":crypto",
          ":ssl",
      ],
  )

  native.cc_test(
      name = "ssl_test",
      size = "small",
      srcs = ["src/ssl/ssl_test.cc"] + test_support_sources_complete,
      copts = copts,
      deps = [
          ":crypto",
          ":ssl",
      ],
  )
