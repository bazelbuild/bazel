licenses(["unencumbered"])

crypto_sources_aarch64 = glob(["linux-aarch64/crypto/**/*.S"])

crypto_sources_arm = glob(["linux-arm/crypto/**/*.S"]) + [
    "src/crypto/chacha/chacha_vec_arm.S",
    "src/crypto/cpu-arm-asm.S",
    "src/crypto/poly1305/poly1305_arm_asm.S",
]

crypto_sources_x86 = glob(["linux-x86/crypto/**/*.S"])

crypto_sources_x86_64 = glob(["linux-x86_64/crypto/**/*.S"])

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

config_setting(
    name = "linux_aarch64",
    values = {"cpu": "aarch64"},
)

config_setting(
    name = "linux_arm",
    values = {"cpu": "arm"},
)

config_setting(
    name = "linux_x86",
    values = {"cpu": "piii"},
)

config_setting(
    name = "linux_x86_64",
    values = {"cpu": "k8"},
)

cc_library(
    name = "crypto",
    srcs = crypto_internal_headers + crypto_sources + select({
        ":linux_aarch64": crypto_sources_aarch64,
        ":linux_arm": crypto_sources_arm,
        ":linux_x86": crypto_sources_x86,
        ":linux_x86_64": crypto_sources_x86_64,
        "//conditions:default": crypto_sources_x86_64,
    }),
    hdrs = crypto_headers,
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
