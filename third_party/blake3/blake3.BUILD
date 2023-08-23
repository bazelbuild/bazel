load("@rules_license//rules:license.bzl", "license")
load("@rules_cc//cc:defs.bzl", "cc_binary", "cc_library")

licenses(["notice"])  #  BSD/MIT-like license

exports_files(["LICENSE"])

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
    srcs = glob(["**"]),
    visibility = ["//third_party:__pkg__"],
)

cc_library(
    name = "blake3",
    srcs = [
        "c/blake3.c",
        "c/blake3_dispatch.c",
        "c/blake3_portable.c",
    ] + select({
        "@bazel_tools//src/conditions:linux_x86_64": [
            "c/blake3_avx2_x86-64_unix.S",
            # Disable to appease bazel-ci which uses ubuntu-18 (EOL) and GCC 7
            # lacking the headers to compile AVX512.
            # "c/blake3_avx512_x86-64_unix.S",
            "c/blake3_sse2_x86-64_unix.S",
            "c/blake3_sse41_x86-64_unix.S",
        ],
        "@bazel_tools//src/conditions:windows_x64": [
            "c/blake3_avx2_x86-64_windows_msvc.asm",
            "c/blake3_avx512_x86-64_windows_msvc.asm",
            "c/blake3_sse2_x86-64_windows_msvc.asm",
            "c/blake3_sse41_x86-64_windows_msvc.asm",
        ],
        "@bazel_tools//src/conditions:darwin_arm64": [
            "c/blake3_neon.c",
        ],
        "//conditions:default": [],
    }),
    hdrs = [
        "c/blake3.h",
        "c/blake3_impl.h",
    ],
    copts = select({
        "@bazel_tools//src/conditions:linux_x86_64": [
	    # Disable to appease bazel-ci which uses ubuntu-18 (EOL) and GCC 7
            # lacking the headers to compile AVX512.
	    "-DBLAKE3_NO_AVX512",
	],
        "@bazel_tools//src/conditions:windows_x64": [],
        "@bazel_tools//src/conditions:windows_arm64": [
            "-DBLAKE3_USE_NEON=0",
        ],
        "@bazel_tools//src/conditions:darwin_arm64": [
            "-DBLAKE3_USE_NEON=1",
        ],
        "//conditions:default": [
            "-DBLAKE3_NO_AVX2",
            "-DBLAKE3_NO_AVX512",
            "-DBLAKE3_NO_NEON",
            "-DBLAKE3_NO_SSE2",
            "-DBLAKE3_NO_SSE41",
        ],
    }),
    includes = ["."],
    visibility = ["//visibility:public"],
)

cc_binary(
    name = "example",
    srcs = [
        "c/example.c",
    ],
    copts = [
        "-w",
        "-O3",
    ],
    includes = ["."],
    visibility = ["//visibility:public"],
    deps = [
        ":blake3",
    ],
)
