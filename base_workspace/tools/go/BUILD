package(default_visibility = ["//visibility:public"])

load("rules/go_rules", "go_binary", "go_library")

# This should be a symlink to the go tool.
exports_files(srcs = ["go"])

go_binary(
    name = "generate_test_main",
    srcs = ["generate_test_main.go"],
)
