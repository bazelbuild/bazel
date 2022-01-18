licenses(["notice"])  # Apache 2.0

exports_files([
    "action_names.bzl",
    "cc_import.bzl",
])

filegroup(
    name = "srcs",
    srcs = glob(["**"]),
    visibility = [
        "//devtools/blaze/integration:__pkg__",
        "//:__pkg__",
        "//tools/build_defs:__pkg__",
    ],
)

filegroup(
    name = "action_names",
    srcs = [
        "action_names.bzl",
    ],
    visibility = ["//visibility:public"],
)

filegroup(
    name = "action_names_test_files",
    testonly = True,
    srcs = [
        "BUILD",
        "action_names.bzl",
    ],
    visibility = ["//visibility:public"],
)

# bazel_osx_p4deps specifies the set of dependencies needed to run Bazel on OS X.
# Consumed by Google internal tests.
filegroup(
    name = "bazel_osx_p4deps",
    testonly = 1,
    srcs = glob(["**"]),
    visibility = ["//tools/osx:__pkg__"],
)
