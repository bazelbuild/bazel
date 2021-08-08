filegroup(
    name = "srcs",
    srcs = glob(["**"]),
    visibility = [
        "//src/test/shell/bazel:__pkg__",
    ],
)

sh_test(
    name = "proto_toolchain_test",
    size = "large",  # Downloads and compiles protobuf for *every* *test* *case*
    srcs = [
        "proto_toolchain_test.sh",
    ],
    data = [
        "//src/test/shell/bazel:test-deps",
    ],
)
