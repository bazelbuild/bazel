load("@rules_java//java:defs.bzl", "java_test")

package(
    default_applicable_licenses = ["//:license"],
    default_visibility = ["//src:__subpackages__"],
)

filegroup(
    name = "srcs",
    testonly = 0,
    srcs = glob(["**"]),
)

java_test(
    name = "OptionProcessorTest",
    srcs = ["OptionProcessorTest.java"],
    resources = [":OptionTestFiles"],
    deps = [
        "//src/main/java/com/google/devtools/common/options",
        "//src/main/java/com/google/devtools/common/options/processor:options_preprocessor_lib",
        "//third_party/java/compile_testing",
        "//third_party/java/guava:io",
        "//third_party/java/junit",
        "//third_party/java/truth",
    ],
)

java_test(
    name = "OptionsClassProcessorTest",
    srcs = ["OptionsClassProcessorTest.java"],
    resources = [":OptionTestFiles"],
    deps = [
        "//src/main/java/com/google/devtools/common/options",
        "//src/main/java/com/google/devtools/common/options/processor:options_preprocessor_lib",
        "//third_party/java/compile_testing",
        "//third_party/java/guava:io",
        "//third_party/java/junit",
        "//third_party/java/truth",
    ],
)

filegroup(
    name = "OptionTestFiles",
    srcs = glob(["optiontestsources/*.java"]),
)
