load("@rules_go//go:def.bzl", "go_binary")

go_binary(
    name = "runfile",
    srcs = ["runfile.go"],
    data = ["//examples:runfile.txt"],
    deps = ["@rules_go//go/runfiles:go_default_library"],
)

filegroup(
    name = "srcs",
    srcs = glob(["**"]),
    visibility = ["//examples:__pkg__"],
)
