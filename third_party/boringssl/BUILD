licenses(["unencumbered"])

crypto_headers = glob(["src/include/openssl/*.h"])

crypto_internal_headers = glob(["src/crypto/**/*.h"])

crypto_sources = ["err_data.c"] + glob(
    ["src/crypto/**/*.c"],
    exclude = [
        "src/crypto/**/*_test.c",
        "src/crypto/**/example_*.c",
    ],
)

ssl_headers = [
    "src/include/openssl/dtls1.h",
    "src/include/openssl/ssl.h",
    "src/include/openssl/ssl3.h",
    "src/include/openssl/tls1.h",
]

ssl_internal_headers = glob(["src/ssl/**/*.h"])

ssl_sources = glob(["src/ssl/**/*.c"])

cc_library(
    name = "crypto",
    srcs = crypto_internal_headers + crypto_sources,
    hdrs = crypto_headers,
    copts = ["-DOPENSSL_NO_ASM"],
    includes = ["src/include"],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "ssl",
    srcs = ssl_internal_headers + ssl_sources,
    hdrs = ssl_headers,
    includes = ["src/include"],
    visibility = ["//visibility:public"],
    deps = [
        ":crypto",
    ],
)
