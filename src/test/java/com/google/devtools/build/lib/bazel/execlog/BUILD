load("@rules_java//java:defs.bzl", "java_test")

package(
    default_testonly = 1,
    default_visibility = ["//src:__subpackages__"],
)

filegroup(
    name = "srcs",
    testonly = 0,
    srcs = glob(["**"]),
    visibility = ["//src:__subpackages__"],
)

java_library(
    name = "StableSortTest_lib",
    srcs = ["StableSortTest.java"],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/bazel/execlog:stable_sort",
        "//src/main/java/com/google/devtools/build/lib/util/io",
        "//src/main/protobuf:spawn_java_proto",
        "//third_party:guava",
        "//third_party:junit4",
        "//third_party:truth",
        "//third_party/protobuf:protobuf_java",
    ],
)

java_test(
    name = "StableSortTest",
    size = "small",
    test_class = "com.google.devtools.build.lib.bazel.execlog.StableSortTest",
    runtime_deps = [
        ":StableSortTest_lib",
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
