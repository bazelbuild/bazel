licenses(["notice"])  #  BSD/MIT-like license

load("//tools/build_defs/pkg:pkg.bzl", "pkg_tar")

filegroup(
    name = "srcs",
    srcs = glob(["**"]),
    visibility = ["//third_party:__pkg__"],
)

filegroup(
    name = "embedded_tools",
    srcs = glob(["*.c"]) + glob(["*.h"]) + ["BUILD"] + ["LICENSE.txt"],
    visibility = ["//visibility:public"],
)

cc_library(
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
)

# Part of the Java tools remote archive. Not embedded or used in Bazel.
pkg_tar(
    name = "zlib_pkg",
    srcs = [":embedded_tools"],
    extension = "tar.gz",
    mode = "755",
    # Create a new pkg_tar to preserve the directory structure (zlib/).
    package_dir = "zlib",
    visibility = ["//third_party:__pkg__"],
)

genrule(
    name = "zlib_zip",
    srcs = [":embedded_tools"],
    outs = ["zlib.zip"],
    cmd = "$(location //third_party:zip_java_tools.sh) zlib $(OUTS) $(SRCS)",
    tools = ["//third_party:zip_java_tools.sh"],
    visibility = ["//third_party:__pkg__"],
)
