# Description:
#   C++ utility tests for Bazel

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
    srcs = ["file_test.cc"],
    deps = [
        "//src/main/cpp/util",
        "//third_party:gtest",
    ],
)

cc_test(
    name = "strings_test",
    srcs = ["strings_test.cc"],
    deps = [
        "//src/main/cpp/util:strings",
        "//third_party:gtest",
    ],
)

test_suite(
    name = "all_tests",
)
