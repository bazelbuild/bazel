filegroup(
    name = "srcs",
    srcs = glob(["**"]) + [
        "//src/test/shell/bazel/testdata/bazel_toolchain_test_data:srcs",
    ],
    visibility = ["//src/test/shell/bazel:__pkg__"],
)

filegroup(
    name = "git-repos",
    testonly = 1,
    srcs = [
        "outer-planets-repo.tar.gz",
        "pluto-repo.tar.gz",
        "refetch-repo.tar.gz",
    ],
    visibility = ["//src/test/shell/bazel:__pkg__"],
)

filegroup(
    name = "bazel_toolchain_test_project_pkg",
    testonly = 1,
    srcs = [
        "//src/test/shell/bazel/testdata/bazel_toolchain_test_data:srcs",
    ],
    visibility = ["//src/test/shell/bazel:__pkg__"],
)
