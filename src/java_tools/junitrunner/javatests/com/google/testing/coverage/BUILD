load("@rules_java//java:defs.bzl", "java_test")

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
        "//third_party/java/jacoco:core-0.8.3",
        "//third_party/java/jacoco:report-0.8.3",
    ],
)
