load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
load("@bazel_tools//tools/build_defs/repo:utils.bzl", "maybe")
load("@bazel_tools//tools/cpp:cc_configure.bzl", "cc_configure")

# rules_cc is used in @bazel_tools//tools/cpp, so must be loaded here.
maybe(
    http_archive,
    "rules_cc",
    sha256 = "d0c573b94a6ef20ef6ff20154a23d0efcb409fb0e1ff0979cec318dfe42f0cdd",
    strip_prefix = "rules_cc-b1c40e1de81913a3c40e5948f78719c28152486d",
    urls = [
        "https://mirror.bazel.build/github.com/bazelbuild/rules_cc/archive/b1c40e1de81913a3c40e5948f78719c28152486d.zip",
        "https://github.com/bazelbuild/rules_cc/archive/b1c40e1de81913a3c40e5948f78719c28152486d.zip",
    ],
)

cc_configure()
