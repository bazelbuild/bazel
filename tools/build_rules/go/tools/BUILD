package(default_visibility = ["//visibility:public"])

load("//tools/build_rules/go:def.bzl", "go_binary")

# This binary is used implicitly by go_test().
go_binary(
    name = "generate_test_main",
    srcs = ["generate_test_main.go"],
)

filegroup(
    name = "srcs",
    srcs = [
        "BUILD",
        "generate_test_main.go",
    ],
)
