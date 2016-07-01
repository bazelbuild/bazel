package(default_visibility = ["//visibility:private"])

filegroup(
    name = "srcs",
    srcs = glob(["**"]) + [
        "//src/test/shell/bazel:srcs",
        "//src/test/shell/integration:srcs",
    ],
    visibility = ["//src:__pkg__"],
)

sh_library(
    name = "bashunit",
    srcs = ["unittest.bash"],
    data = [
        "testenv.sh",
        "//tools:test_sharding_compliant",
    ],
    visibility = ["//visibility:public"],
)

sh_test(
    name = "bashunit_test",
    srcs = ["unittest_test.sh"],
    data = [":bashunit"],
)
