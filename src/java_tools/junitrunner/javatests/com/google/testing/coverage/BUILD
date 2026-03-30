load("@rules_java//java:defs.bzl", "java_library", "java_test")
load("@rules_shell//shell:sh_test.bzl", "sh_test")

package(default_visibility = ["//src:__subpackages__"])

licenses(["notice"])

filegroup(
    name = "srcs",
    srcs = glob(["*"]),
    visibility = ["//src:__subpackages__"],
)

java_test(
    name = "JacocoLCOVFormatterUninstrumentedTest",
    size = "small",
    srcs = [
        "JacocoLCOVFormatterUninstrumentedTest.java",
    ],
    deps = [
        "//src/java_tools/junitrunner/java/com/google/testing/coverage:JacocoCoverageLib",
        "//third_party:guava",
        "//third_party:junit4",
        "//third_party:mockito",
        "//third_party:truth",
        "//third_party/java/jacoco:core",
        "//third_party/java/jacoco:report",
    ],
)

java_library(
    name = "jar+with+plus",
    srcs = ["TestClass.java"],
)

sh_test(
    name = "jacoco_runner_integration_test",
    size = "medium",
    srcs = ["jacoco_runner_integration_tests.sh"],
    args = [
        "$(location //src/test/shell:unittest.bash)",
        "$(location //src/java_tools/junitrunner/java/com/google/testing/coverage:JacocoCoverage)",
        "$(location :jar+with+plus)",
    ],
    data = [
        ":jar+with+plus",
        "//src/java_tools/junitrunner/java/com/google/testing/coverage:JacocoCoverage",
        "//src/test/shell:bashunit",
        "//src/test/shell:unittest.bash",
    ],
)
