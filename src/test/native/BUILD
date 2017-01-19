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
    srcs = select({
        "//src:windows": ["windows_util_test.cc"],
        "//src:windows_msvc": ["windows_util_test.cc"],
        "//conditions:default": ["dummy_test.cc"],
    }),
    deps = select({
        "//src:windows": [
            "//src/main/native:windows_jni_lib",
            "//third_party:gtest",
        ],
        "//src:windows_msvc": [
            "//src/main/native:windows_jni_lib",
            "//third_party:gtest",
        ],
        "//conditions:default": [],
    }),
)

test_suite(name = "all_tests")
