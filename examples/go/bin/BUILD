package(
    default_visibility = ["//visibility:public"],
)

load("/tools/build_rules/go/def", "go_binary")

go_binary(
    name = "bin",
    srcs = ["bin.go"],
    deps = [
        "//examples/go/lib:go_default_library",
        "//examples/go/vendor/github_com/user/vendored:go_default_library",
    ],
)
