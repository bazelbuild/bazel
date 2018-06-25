licenses(["notice"])  # Apache 2.0

exports_files(["action_names.bzl"])

filegroup(
    name = "srcs",
    srcs = glob(["**"]),
    visibility = [
        "//:__pkg__",
        "//tools:__pkg__",
    ],
)

# bazel_osx_p4deps specifies the set of dependencies needed to run Bazel on OS X.
# Consumed by Google internal tests.
filegroup(
    name = "bazel_osx_p4deps",
    testonly = 1,
    srcs = glob(["**"]),
    visibility = ["//tools/osx:__pkg__"],
)
