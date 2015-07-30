# Scripts for building Bazel releases
package(default_visibility = ["//visibility:private"])

sh_library(
    name = "relnotes",
    srcs = ["relnotes.sh"],
)

sh_test(
    name = "relnotes_test",
    srcs = ["relnotes_test.sh"],
    data = [
        "testenv.sh",
        "//:git",
        "//src/test/shell:bashunit",
    ],
    shard_count = 2,
    tags = ["need_git"],
    deps = [
        ":relnotes",
    ],
)

sh_library(
    name = "release",
    srcs = [
        "common.sh",
        "release.sh",
    ],
    deps = [":relnotes"],
)

sh_test(
    name = "release_test",
    srcs = ["release_test.sh"],
    data = [
        "testenv.sh",
        "//:git",
        "//src/test/shell:bashunit",
    ],
    tags = ["need_git"],
    deps = [":release"],
)
