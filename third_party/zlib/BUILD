load("//tools/distributions:distribution_rules.bzl", "distrib_cc_library")

licenses(["notice"])  #  BSD/MIT-like license

filegroup(
    name = "srcs",
    srcs = glob(["**"]),
    visibility = ["//third_party:__pkg__"],
)

filegroup(
    name = "embedded_tools",
    srcs = glob(["*.c"]) + glob(["*.h"]) + [":embedded_build_file"] + ["LICENSE.txt"],
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
    includes = ["."],
    visibility = ["//visibility:public"],
    enable_distributions = ["debian"],
)
