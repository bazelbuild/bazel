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
            "windows/util_test.cc",
            "windows/file_test.cc",
        ],
        "//src:windows_msys": [
            "windows/util_test.cc",
            "windows/file_test.cc",
        ],
        "//src:windows_msvc": [
            "windows/util_test.cc",
            "windows/file_test.cc",
        ],
        "//conditions:default": ["dummy_test.cc"],
    }),
    deps = select({
        "//src:windows": [
            "//src/main/native/windows:lib-file",
            "//src/test/cpp/util:windows_test_util",
            "//third_party:gtest",
        ],
        "//src:windows_msys": [
            "//src/main/native/windows:lib-file",
            "//src/test/cpp/util:windows_test_util",
            "//third_party:gtest",
        ],
        "//src:windows_msvc": [
            "//src/main/native/windows:lib-file",
            "//src/test/cpp/util:windows_test_util",
            "//third_party:gtest",
        ],
        "//conditions:default": [],
    }),
)

test_suite(name = "all_tests")
