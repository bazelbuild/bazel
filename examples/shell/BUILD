package(default_visibility = ["//visibility:public"])

sh_binary(
    name = "bin",
    srcs = ["bin.sh"],
    deps = [":lib"],
)

sh_library(
    name = "lib",
    data = [
        "data/file.txt",
        "lib.sh",
    ],
)

sh_test(
    name = "test",
    size = "small",
    srcs = ["test.sh"],
    data = ["data/test_file.txt"],
    deps = [":lib"],
)

filegroup(
    name = "srcs",
    srcs = ["BUILD"] + glob(["**/*.sh"]) + glob(["**/*.txt"]),
)
