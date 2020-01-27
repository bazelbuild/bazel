local_repository(
    name = "bazel_tools",
    path = __embedded_dir__ + "/embedded_tools",
)

bind(
    name = "cc_toolchain",
    actual = "@bazel_tools//tools/cpp:default-toolchain",
)
