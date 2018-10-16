package(default_visibility = ["//visibility:private"])

exports_files([
    "bin/bazel",
    "testenv.sh",
    "integration_test_setup.sh",
    "sandboxing_test_utils.sh",
    "unittest.bash",
    "shell_utils.sh",
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
    data = [
        ":bashunit",
        "@bazel_tools//tools/bash/runfiles",
    ],
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
        "@bazel_tools//tools/bash/runfiles",
    ],
)

sh_test(
    name = "shell_utils_symlinks_test",
    srcs = ["shell_utils_symlinks_test.sh"],
    data = [
        "testenv.sh",
        ":bashunit",
        ":shell_utils",
        "@bazel_tools//tools/bash/runfiles",
    ],
    tags = [
        "no_windows",  # test asserts symlink behavior
    ],
)

test_suite(
    name = "windows_tests",
    tags = [
        "-no_windows",
        "-slow",
    ],
    visibility = ["//visibility:private"],
)

test_suite(
    name = "all_windows_tests",
    tests = [
        ":windows_tests",
        "//src/test/shell/bazel:all_windows_tests",
        "//src/test/shell/integration:all_windows_tests",
    ],
    visibility = ["//src:__pkg__"],
)
