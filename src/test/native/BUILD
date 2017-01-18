# Description:
#   C++ utility tests for Bazel
package(default_visibility = ["//visibility:public"])

filegroup(
    name = "srcs",
    srcs = glob(["**"]),
    visibility = ["//src:__pkg__"],
)

cc_test(
    name = "windows_util_test",
    srcs = ["windows_util_test.cc"],
    deps = [
        "//src/main/native:windows_jni_utils",
        "//third_party:gtest",
    ],
)

test_suite(name = "all_tests")
