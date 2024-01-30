package(default_visibility = ["//visibility:private"])

filegroup(
    name = "srcs",
    srcs = glob(["**"]),
    visibility = ["//src/test/shell/bazel:__pkg__"],
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
        ":objc-deps",
        "//src/test/shell/bazel:test-deps",
    ],
    tags = ["no_windows"],
)
