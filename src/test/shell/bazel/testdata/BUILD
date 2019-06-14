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

filegroup(
    name = "embedded_tools_deps_test_data",
    testonly = 1,
    srcs = [
        "//src/test/shell/bazel/testdata:embedded_tools_srcs_deps",
    ],
    visibility = ["//src/test/shell/bazel:__pkg__"],
)

filegroup(
    name = "jdk_http_archives_filegroup",
    testonly = 1,
    srcs = [
        "//src/test/shell/bazel/testdata:jdk_http_archives",
    ],
    visibility = ["//src/test/shell/bazel:__pkg__"],
)
