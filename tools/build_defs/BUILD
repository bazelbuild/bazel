filegroup(
    name = "srcs",
    srcs = glob(["**"]) + [
        "//tools/build_defs/cc:srcs",
        "//tools/build_defs/cc/whitelists/parse_headers_and_layering_check:srcs",
        "//tools/build_defs/cc/whitelists/starlark_hdrs_check:srcs",
        "//tools/build_defs/hash:srcs",
        "//tools/build_defs/pkg:srcs",
        "//tools/build_defs/repo:srcs",
        "//tools/build_defs/cc/tests:cc_import_tests_files",
    ],
    visibility = ["//tools:__pkg__"],
)

filegroup(
    name = "embedded_tools",
    srcs = glob(["**"]) + [
        "//tools/build_defs/cc:srcs",
        "//tools/build_defs/cc/whitelists/parse_headers_and_layering_check:srcs",
        "//tools/build_defs/cc/whitelists/starlark_hdrs_check:srcs",
        "//tools/build_defs/hash:srcs",
        "//tools/build_defs/pkg:srcs",
        "//tools/build_defs/repo:embedded_tools",
    ],
    visibility = ["//tools:__pkg__"],
)

test_suite(
    name = "all_windows_tests",
    tests = [
        "//tools/build_defs/hash:all_windows_tests",
        "//tools/build_defs/pkg:all_windows_tests",
    ],
    visibility = ["//tools:__pkg__"],
)

filegroup(
    name = "bzl_srcs",
    srcs = [
        "//tools/build_defs/cc:action_names.bzl",
        "//tools/build_defs/cc:cc_import.bzl",
        "//tools/build_defs/hash:hash.bzl",
        "//tools/build_defs/pkg:bzl_srcs",
        "//tools/build_defs/repo:bzl_srcs",
    ],
    visibility = ["//tools:__pkg__"],
)
