load("@rules_java//java:defs.bzl", "java_library", "java_test")

# Description:
#   Tests of the testing tools for the devtools-common options parser.
package(
    default_applicable_licenses = ["//:license"],
    default_testonly = 1,
    default_visibility = ["//src:__subpackages__"],
)

filegroup(
    name = "srcs",
    testonly = 0,
    srcs = glob(
        ["**"],
    ),
    visibility = ["//src:__subpackages__"],
)

java_library(
    name = "OptionsTesterTest_lib",
    srcs = glob(["*.java"]),
    javacopts = [
        "-XepOpt:CheckReturnValue:CheckAllConstructors=false",  # b/226969262
    ],
    deps = [
        "//src/main/java/com/google/devtools/common/options",
        "//src/main/java/com/google/devtools/common/options/testing",
        "//src/test/java/com/google/devtools/build/lib/testutil:TestSuite",
        "//third_party:guava",
        "//third_party:junit4",
        "//third_party:truth",
    ],
)

java_test(
    name = "OptionsTesterTest",
    test_class = "com.google.devtools.common.options.testing.AllTests",
    runtime_deps = [":OptionsTesterTest_lib"],
)
