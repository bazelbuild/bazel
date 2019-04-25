# Description: brotli research tools.

package(default_visibility = ["//visibility:public"])

licenses(["notice"])  # MIT

cc_library(
    name = "dm",
    srcs = ["deorummolae.cc"],
    hdrs = [
        "deorummolae.h",
        "esaxx/sais.hxx",
    ],
)

cc_library(
    name = "durchschlag",
    srcs = ["durchschlag.cc"],
    hdrs = ["durchschlag.h"],
    deps = ["@divsufsort//:libdivsufsort"],
)

cc_library(
    name = "sieve",
    srcs = ["sieve.cc"],
    hdrs = ["sieve.h"],
)

cc_binary(
    name = "dictionary_generator",
    srcs = ["dictionary_generator.cc"],
    deps = [
        ":dm",
        ":durchschlag",
        ":sieve",
    ],
)

cc_binary(
    name = "brotli_decoder",
    srcs = ["brotli_decoder.c"],
    linkstatic = 1,
    deps = ["@org_brotli//:brotlidec"],
)
