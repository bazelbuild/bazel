package(default_visibility = ["//visibility:public"])

load("/tools/build_rules/go_rules", "go_library", "go_binary", "go_test")

go_library(
    name = "lib1",
    srcs = glob(
        ["*.go"],
        exclude = ["*_test.go"],
    ),
)

go_test(
    name = "lib1_test",
    srcs = [
        "lib1_test.go",
    ],
    library = ":lib1",
)

go_test(
    name = "fail_test",
    srcs = ["fail_test.go"],
)

filegroup(
    name = "srcs",
    srcs = ["BUILD"] + glob(["**/*.go"]),
)
