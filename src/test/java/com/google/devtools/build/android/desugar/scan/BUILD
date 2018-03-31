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

sh_test(
    name = "test_keep_scanner",
    srcs = ["test_keep_scanner.sh"],
    args = [
        "$(location //src/tools/android/java/com/google/devtools/build/android/desugar/scan:KeepScanner)",
        "$(location :testdata)",
        "$(location testdata_golden.txt)",
    ],
    data = [
        "testdata_golden.txt",
        ":testdata",
        "//src/tools/android/java/com/google/devtools/build/android/desugar/scan:KeepScanner",
    ],
)

java_library(
    name = "testdata",
    srcs = glob(["testdata/**/*.java"]),
)
