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
    package_version = "1.2.12",
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
    srcs = glob(["*.c"]),
    hdrs = glob(["*.h"]),
    # Use -Dverbose=-1 to turn off zlib's trace logging. (#3280)
    copts = [
        "-w",
        "-Dverbose=-1",
    ],
    enable_distributions = ["debian"],
    includes = ["."],
    visibility = ["//visibility:public"],
)
