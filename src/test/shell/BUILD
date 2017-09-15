package(default_visibility = ["//visibility:private"])

exports_files([
    "bin/bazel",
    "testenv.sh",
    "integration_test_setup.sh",
    "unittest.bash",
])

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

sh_library(
    name = "shell_utils",
    srcs = ["shell_utils.sh"],
    visibility = ["//src/test/shell/integration:__pkg__"],
)

sh_test(
    name = "shell_utils_test",
    srcs = ["shell_utils_test.sh"],
    data = [
        "testenv.sh",
        ":bashunit",
        ":shell_utils",
    ],
)
