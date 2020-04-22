load("@rules_java//java:defs.bzl", "java_library", "java_test")

java_library(
    name = "deps",
    testonly = 1,
    exports = [
        "//src/test/java/com/google/devtools/build/lib/blackbox:base_deps",
        "//src/test/java/com/google/devtools/build/lib/blackbox:test_deps",
    ],
)

java_library(
    name = "framework",
    testonly = 1,
    srcs = [
        "BlackBoxTestContext.java",
        "BlackBoxTestEnvironment.java",
        "BuilderRunner.java",
        "PathUtils.java",
        "ProcessParameters.java",
        "ProcessResult.java",
        "ProcessRunner.java",
        "ToolsSetup.java",
    ],
    visibility = ["//src/test/java/com/google/devtools/build/lib/blackbox:__subpackages__"],
    deps = [":deps"],
)

java_test(
    name = "framework_tests",
    testonly = 1,
    srcs = [
        "PathUtilsTest.java",
        "ProcessRunnerTest.java",
    ],
    tags = ["black_box_test"],
    test_class = "com.google.devtools.build.lib.AllTests",
    visibility = ["//src/test/java/com/google/devtools/build/lib/blackbox/tests:__pkg__"],
    deps = [
        ":deps",
        ":framework",
    ],
)

filegroup(
    name = "srcs",
    testonly = 0,
    srcs = glob(["**"]),
    visibility = ["//src/test/java/com/google/devtools/build/lib/blackbox:__pkg__"],
)
