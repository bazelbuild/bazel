load("@rules_java//java:defs.bzl", "java_test")

package(
    default_visibility = ["//src:__subpackages__"],
)

filegroup(
    name = "srcs",
    testonly = 0,
    srcs = glob(["**"]),
    visibility = ["//src:__subpackages__"],
)

java_test(
    name = "OptionProcessorTest",
    srcs = ["OptionProcessorTest.java"],
    resources = [":OptionTestFiles"],
    deps = [
        "//src/main/java/com/google/devtools/common/options",  
        "//src/main/java/com/google/devtools/common/options/processor:options_preprocessor",
        "//third_party:compile_testing",
        "//third_party:guava",
        "//third_party:junit4",
        "//third_party:truth",
    ],
)

filegroup(
    name = "OptionTestFiles",
    srcs = glob(["optiontestsources/*.java"]),
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
