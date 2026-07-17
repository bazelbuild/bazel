licenses(["notice"])  # Apache 2.0

# TODO(gnish): Re-enable tests once bazel picks up changes.
# cc_import_test_suite(
#    name = "cc_import_tests",
# )

filegroup(
    name = "cc_import_tests_files",
    srcs = glob(["**"]),
    visibility = [
        "//src/test/shell/bazel:__pkg__",
        "//tools/build_defs:__pkg__",
        "//tools/build_defs/cc:__pkg__",
    ],
)
