load("@rules_java//java:defs.bzl", "java_test")

package(
    default_applicable_licenses = ["//:license"],
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
        "//src/main/java/com/google/devtools/common/options/processor:options_preprocessor_lib",
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
