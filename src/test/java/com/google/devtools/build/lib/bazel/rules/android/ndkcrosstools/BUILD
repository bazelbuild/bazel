load("@rules_java//java:defs.bzl", "java_library", "java_test")

package(
    default_testonly = 1,
    default_visibility = ["//src:__subpackages__"],
)

filegroup(
    name = "srcs",
    testonly = 0,
    srcs = glob(["*"]),
    visibility = ["//src:__subpackages__"],
)

java_library(
    name = "NdkCrosstoolsTest_lib",
    srcs = glob(["*.java"]),
    resources = glob(["*.txt"]),
    deps = [
        "//src/main/java/com/google/devtools/build/lib/bazel/rules/android/ndkcrosstools",
        "//src/main/java/com/google/devtools/build/lib/bazel/rules/android/ndkcrosstools:android_ndk_crosstools",
        "//src/main/java/com/google/devtools/build/lib/bazel/rules/android/ndkcrosstools/r10e",
        "//src/main/java/com/google/devtools/build/lib/bazel/rules/android/ndkcrosstools/r12",
        "//src/main/java/com/google/devtools/build/lib/events",
        "//src/main/java/com/google/devtools/build/lib/util",
        "//src/main/protobuf:crosstool_config_java_proto",
        "//third_party:guava",
        "//third_party:junit4",
        "//third_party:truth",
    ],
)

java_test(
    name = "NdkCrosstoolsTest",
    test_class = "com.google.devtools.build.lib.AllTests",
    runtime_deps = [
        ":NdkCrosstoolsTest_lib",
        "//src/test/java/com/google/devtools/build/lib:test_runner",
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
)
