package(default_visibility = ["//visibility:public"])

licenses(["notice"])  # MIT

load("@io_bazel_rules_go//go:def.bzl", "go_library", "go_test")

go_library(
    name = "cbrotli",
    srcs = [
        "reader.go",
        "writer.go",
    ],
    cdeps = [
        "@org_brotli//:brotlidec",
        "@org_brotli//:brotlienc",
    ],
    cgo = True,
    importpath = "github.com/google/brotli/go/cbrotli",
)

go_test(
    name = "cbrotli_test",
    size = "small",
    srcs = ["cbrotli_test.go"],
    embed = [":cbrotli"],
)
