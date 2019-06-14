package(default_visibility = ["//visibility:public"])

# Files which shouldn't be publicly visible and dependencies of all objc_* or ios_* rules should be excluded.
exports_files(glob(
    ["**"],
    exclude = ["wrapped_clang_test.sh"],
))

filegroup(
    name = "srcs",
    srcs = glob(["**"]),
)

cc_binary(
    name = "wrapped_clang",
    testonly = True,
    srcs = [
        "wrapped_clang.cc",
    ],
)

sh_test(
    name = "wrapped_clang_test",
    size = "small",
    srcs = [
        "wrapped_clang_test.sh",
    ],
    data = [
        ":wrapped_clang",
        "//src/test/shell:bashunit",
        "@bazel_tools//tools/bash/runfiles",
    ],
)
