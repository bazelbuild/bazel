load("//src/test/tools:wasm.bzl", "wat_binary")

package(default_testonly = True)

filegroup(
    name = "srcs",
    testonly = False,
    srcs = glob(["**"]),
    visibility = ["//src/test/shell/bazel:__pkg__"],
)

filegroup(
    name = "git-repos",
    srcs = [
        "outer-planets-repo.tar.gz",
        "pluto-repo.tar.gz",
        "refetch-repo.tar.gz",
        "strip-prefix-repo.tar.gz",
    ],
    visibility = ["//src/test/shell/bazel:__pkg__"],
)

filegroup(
    name = "embedded_tools_deps_test_data",
    srcs = [
        "//src/test/shell/bazel/testdata:embedded_tools_srcs_deps",
    ],
    visibility = ["//src/test/shell/bazel:__pkg__"],
)

exports_files(["zstd_test_archive.tar.zst"])

wat_binary(
    name = "exec_wasm",
    src = "exec_wasm.wat",
    visibility = ["//src/test/shell/bazel:__pkg__"],
)
