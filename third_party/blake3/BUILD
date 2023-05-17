load("@rules_license//rules:license.bzl", "license")
load("//tools/distributions:distribution_rules.bzl", "distrib_cc_binary", "distrib_cc_library")

package(default_applicable_licenses = [":license"])

licenses(["notice"])  #  BSD/MIT-like license

license(
    name = "license",
    package_name = "blake3",
    license_kinds = [
        "@rules_license//licenses/spdx:Apache-2.0",
    ],
    license_text = "LICENSE",
    package_version = "1.3.3",
)

filegroup(
    name = "srcs",
    srcs = glob(["src/**"]),
    visibility = ["//third_party:__pkg__"],
)

distrib_cc_library(
    name = "blake3",
    srcs = [
        "src/c/blake3.c",
        "src/c/blake3_avx2_x86-64_unix.S",
        "src/c/blake3_avx512_x86-64_unix.S",
        "src/c/blake3_dispatch.c",
        "src/c/blake3_portable.c",
        "src/c/blake3_sse2_x86-64_unix.S",
        "src/c/blake3_sse41_x86-64_unix.S",
    ],
    hdrs = [
        "src/c/blake3.h",
        "src/c/blake3_impl.h",
    ],
    copts = [
        "-w",
        "-Dverbose=-1",
    ],
    enable_distributions = ["debian"],
    includes = ["."],
    visibility = ["//visibility:public"],
)

distrib_cc_binary(
    name = "blake3_example",
    srcs = [
        "src/c/example.c",
    ],
    copts = [
        "-w",
        "-Dverbose=-1",
    ],
    enable_distributions = ["debian"],
    includes = ["."],
    visibility = ["//visibility:public"],
    deps = [
        ":blake3",
    ],
)
