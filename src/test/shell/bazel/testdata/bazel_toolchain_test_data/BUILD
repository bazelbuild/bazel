filegroup(
    name = "srcs",
    srcs = glob(["**"]) + [
        "//src/test/shell/bazel/testdata/bazel_toolchain_test_data/tools/arm_compiler:srcs",
    ],
    visibility = ["//src/test/shell/bazel/testdata:__pkg__"],
)

cc_binary(
    name = "hello",
    srcs = ["hello.cc"],
)
