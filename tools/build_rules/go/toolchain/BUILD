# TODO(bazel-team): support darwin.

filegroup(
    name = "toolchain",
    srcs = select({
        "//src:darwin": ["@golang-darwin-amd64//:toolchain"],
        "//conditions:default": ["@golang-linux-amd64//:toolchain"],
    }),
    visibility = ["//visibility:public"],
)

filegroup(
    name = "go_tool",
    srcs = select({
        "//src:darwin": ["@golang-darwin-amd64//:go_tool"],
        "//conditions:default": ["@golang-linux-amd64//:go_tool"],
    }),
    visibility = ["//visibility:public"],
)
