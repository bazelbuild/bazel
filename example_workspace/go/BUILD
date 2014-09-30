package(default_visibility = ["//visibility:public"])

load("rules/go_rules", "go_library", "go_binary")

go_binary(
    name = "fib",
    srcs = glob(
        ["*.go"],
        exclude = ["*_test.go"],
    ),
    deps = ["//go/lib1"],
)
