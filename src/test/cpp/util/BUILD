# Description:
#   C++ utility tests for Bazel
package(default_visibility = ["//visibility:public"])

filegroup(
    name = "srcs",
    srcs = glob(["**"]),
    visibility = ["//src/test/cpp:__pkg__"],
)

cc_test(
    name = "md5_test",
    srcs = ["md5_test.cc"],
    deps = [
        "//src/main/cpp/util",
        "//src/main/cpp/util:md5",
        "//third_party:gtest",
    ],
)

cc_test(
    name = "file_test",
    size = "small",
    srcs = ["file_test.cc"] + select({
        "//src:windows": [
            "file_windows_test.cc",
        ],
        "//src:windows_msys": [
            "file_windows_test.cc",
        ],
        "//src:windows_msvc": [
            "file_windows_test.cc",
        ],
        "//conditions:default": [
            "file_posix_test.cc",
        ],
    }),
    deps = [
        ":test_util",
        "//src/main/cpp/util:file",
        "//third_party:gtest",
    ] + select({
        "//src:windows": [
            ":windows_test_util",
            "//src/main/native/windows:lib-file",
        ],
        "//src:windows_msys": [
            ":windows_test_util",
            "//src/main/native/windows:lib-file",
        ],
        "//src:windows_msvc": [
            ":windows_test_util",
            "//src/main/native/windows:lib-file",
        ],
        "//conditions:default": [],
    }),
)

cc_test(
    name = "logging_test",
    srcs = ["logging_test.cc"],
    deps = [
        "//src/main/cpp:blaze_util",
        "//src/main/cpp/util:bazel_log_handler",
        "//src/main/cpp/util:logging",
        "//third_party:gtest",
    ],
)

cc_test(
    name = "numbers_test",
    srcs = ["numbers_test.cc"],
    deps = [
        "//src/main/cpp/util",
        "//third_party:gtest",
    ],
)

cc_test(
    name = "strings_test",
    srcs = ["strings_test.cc"],
    shard_count = 2,
    deps = [
        "//src/main/cpp/util:strings",
        "//third_party:gtest",
    ],
)

cc_library(
    name = "windows_test_util",
    testonly = 1,
    srcs = select({
        "//src:windows": ["windows_test_util.cc"],
        "//src:windows_msys": ["windows_test_util.cc"],
        "//src:windows_msvc": ["windows_test_util.cc"],
        "//conditions:default": [],
    }),
    hdrs = select({
        "//src:windows": ["windows_test_util.h"],
        "//src:windows_msys": ["windows_test_util.h"],
        "//src:windows_msvc": ["windows_test_util.h"],
        "//conditions:default": [],
    }),
    visibility = [
        "//src/test/cpp:__subpackages__",
        "//src/test/native:__subpackages__",
    ],
)

cc_test(
    name = "windows_test_util_test",
    size = "small",
    srcs = select({
        "//src:windows": ["windows_test_util_test.cc"],
        "//src:windows_msys": ["windows_test_util_test.cc"],
        "//src:windows_msvc": ["windows_test_util_test.cc"],
        "//conditions:default": ["dummy_test.cc"],
    }),
    deps = select({
        "//src:windows": [
            ":windows_test_util",
            "//third_party:gtest",
        ],
        "//src:windows_msys": [
            ":windows_test_util",
            "//third_party:gtest",
        ],
        "//src:windows_msvc": [
            ":windows_test_util",
            "//third_party:gtest",
        ],
        "//conditions:default": [],
    }),
)

cc_library(
    name = "test_util",
    testonly = 1,
    hdrs = ["test_util.h"],
    visibility = ["//visibility:private"],
)

test_suite(name = "all_tests")
