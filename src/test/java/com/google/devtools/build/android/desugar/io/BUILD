# Description:
#   Tests for the Java 8 desugaring tool for Android.
package(
    default_testonly = 1,
)

licenses(["notice"])  # Apache 2.0

filegroup(
    name = "srcs",
    testonly = 0,
    srcs = glob(["**"]),
    visibility = ["//src/test/java/com/google/devtools/build/android/desugar:__pkg__"],
)

java_test(
    name = "FieldInfoTest",
    size = "small",
    srcs = ["FieldInfoTest.java"],
    deps = [
        "//src/tools/android/java/com/google/devtools/build/android/desugar/io",
        "//third_party:junit4",
        "//third_party:truth",
    ],
)

java_test(
    name = "IndexedInputsTest",
    size = "small",
    srcs = ["IndexedInputsTest.java"],
    deps = [
        "//src/tools/android/java/com/google/devtools/build/android/desugar/io",
        "//third_party:guava",
        "//third_party:junit4",
        "//third_party:truth",
    ],
)

test_suite(
    name = "windows_tests",
    tags = [
        "-no_windows",
        "-slow",
    ],
)

test_suite(
    name = "all_windows_tests",
    tests = [
        ":windows_tests",
    ],
    visibility = ["//src/test/java/com/google/devtools/build/android/desugar:__pkg__"],
)
