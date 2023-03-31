package(default_visibility = ["//visibility:private"])

filegroup(
    name = "srcs",
    srcs = glob(["**"]),
    visibility = ["//src/test/shell/bazel:__pkg__"],
)

sh_library(
    name = "apple_common",
    testonly = 1,
    srcs = ["apple_common.sh"],
)

filegroup(
    name = "objc-deps",
    testonly = 1,
    srcs = select({
        "//src/conditions:darwin": [
            "//tools/osx:xcode-locator",
        ],
        "//conditions:default": [],
    }),
)

sh_test(
    name = "bazel_apple_test",
    size = "large",
    srcs = ["bazel_apple_test.sh"],
    data = [
        ":apple_common",
        ":objc-deps",
        "//:workspace-file",
        "//src/test/shell/bazel:test-deps",
    ],
    tags = ["no_windows"],
)

sh_test(
    name = "bazel_objc_test",
    size = "large",
    srcs = ["bazel_objc_test.sh"],
    data = [
        ":apple_common",
        ":objc-deps",
        "//src/test/shell/bazel:test-deps",
    ],
    tags = ["no_windows"],
)
