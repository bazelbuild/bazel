# Tools for gathering OSS licenses

load("@rules_license//rules:compliance.bzl", "check_license")

licenses(["notice"])  # Apache 2.0

filegroup(
    name = "srcs",
    srcs = glob(["**"]),
    visibility = [
        "//tools:__subpackages__",
        "@bazel_tools//tools:__subpackages__",
    ],
)

check_license(
    name = "check_bazel",
    check_conditions = False,
    copyright_notices = "bazel_copyrights.txt",
    license_texts = "bazel_licenses.txt",
    report = "bazel_report",
    deps = [
        "//src:bazel-bin_nojdk",
    ],
)

check_license(
    name = "check_srcs",
    check_conditions = False,
    copyright_notices = "srcs_copyrights.txt",
    license_texts = "srcs_licenses.txt",
    report = "srcs_report",
    deps = [
        "//:bazel-srcs",
    ],
)
