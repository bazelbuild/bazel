# Description:
#   C++ utility tests for Bazel
package(default_visibility = ["//visibility:public"])

filegroup(
    name = "srcs",
    srcs = glob(["**"]),
    visibility = ["//src:__pkg__"],
)

cc_test(
    name = "windows_jni_test",
    size = "small",
    srcs = select({
        "//src:windows": [
            "windows_util_test.cc",
            "windows_file_operations_test.cc",
        ],
        "//src:windows_msys": [
            "windows_util_test.cc",
            "windows_file_operations_test.cc",
        ],
        "//src:windows_msvc": [
            "windows_util_test.cc",
            "windows_file_operations_test.cc",
        ],
        "//conditions:default": ["dummy_test.cc"],
    }),
    deps = select({
        "//src:windows": [
            "//src/main/native:windows_jni_lib",
            "//src/test/cpp/util:windows_test_util",
            "//third_party:gtest",
        ],
        "//src:windows_msys": [
            "//src/main/native:windows_jni_lib",
            "//src/test/cpp/util:windows_test_util",
            "//third_party:gtest",
        ],
        "//src:windows_msvc": [
            "//src/main/native:windows_jni_lib",
            "//src/test/cpp/util:windows_test_util",
            "//third_party:gtest",
        ],
        "//conditions:default": [],
    }),
)

test_suite(name = "all_tests")
