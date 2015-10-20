package(
    default_visibility = ["//src:__subpackages__"],
)

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

filegroup(
    name = "srcs",
    srcs = [
        "BUILD",
        "BUILD.go-toolchain",
        "WORKSPACE.go-toolchain",
        ":go_tool",
        ":toolchain",
    ],
)
