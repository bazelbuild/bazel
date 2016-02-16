package(default_visibility = ["//scripts/release:__pkg__"])

filegroup(
    name = "git",
    srcs = glob([".git/**"]),
)

filegroup(
    name = "dummy",
    visibility = ["//visibility:public"],
)

load("//tools/build_rules/go:def.bzl", "go_prefix")

go_prefix("github.com/bazelbuild/bazel")
