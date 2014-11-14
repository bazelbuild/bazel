package(default_visibility = ["//visibility:public"])

load("tools/build_rules/go_rules", "go_library", "go_binary")

go_binary(
    name = "fib",
    srcs = glob(
        ["*.go"],
        exclude = ["*_test.go"],
    ),
    deps = ["//examples/go/lib1"],
)
