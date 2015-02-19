package(default_visibility = ["//visibility:public"])

load("tools/build_rules/go_rules", "go_binary", "go_library")

# This should be a symlink to a GOROOT holding bin/ and pkg/
exports_files(srcs = ["go_root"])

go_binary(
    name = "generate_test_main",
    srcs = ["generate_test_main.go"],
)
