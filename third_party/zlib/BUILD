load("@rules_license//rules:license.bzl", "license")
load("//tools/distributions:distribution_rules.bzl", "distrib_cc_library")

package(default_applicable_licenses = [":license"])

licenses(["notice"])  #  BSD/MIT-like license

license(
    name = "license",
    package_name = "zlib",
    copyright_notice = "(C) 1995-2022 Jean-loup Gailly and Mark Adler",
    license_kinds = [
        "@rules_license//licenses/spdx:0BSD",
    ],
    license_text = "LICENSE",
    package_version = "1.3.1",
)

filegroup(
    name = "srcs",
    srcs = glob(["**"]),
    visibility = ["//third_party:__pkg__"],
)

filegroup(
    name = "embedded_tools",
    srcs = glob(["*.c"]) + glob(["*.h"]) + [":embedded_build_file"] + ["LICENSE"],
    visibility = ["//visibility:public"],
)

genrule(
    name = "embedded_build_file",
    srcs = ["BUILD.tools"],
    # Rename BUILD.bazel instead of BUILD to not be conflict with the BUILD file in source.
    outs = ["BUILD.bazel"],
    cmd = "cp $< $@",
)

distrib_cc_library(
    name = "zlib",
    srcs = [
        "adler32.c",
        "compress.c",
        "crc32.c",
        "deflate.c",
        "gzclose.c",
        "gzlib.c",
        "gzread.c",
        "gzwrite.c",
        "infback.c",
        "inffast.c",
        "inflate.c",
        "inftrees.c",
        "trees.c",
        "uncompr.c",
        "zutil.c",
    ],
    hdrs = [
        "crc32.h",
        "deflate.h",
        "gzguts.h",
        "inffast.h",
        "inffixed.h",
        "inflate.h",
        "inftrees.h",
        "trees.h",
        "zconf.h",
        "zlib.h",
        "zutil.h",
    ],
    copts = select({
        "@platforms//os:windows": [],
        "//conditions:default": [
            "-Wno-deprecated-non-prototype",
            "-Wno-unused-variable",
            "-Wno-implicit-function-declaration",
        ],
    }),
    enable_distributions = ["debian"],
    includes = ["."],
    visibility = ["//visibility:public"],
)
