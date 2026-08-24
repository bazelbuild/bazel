load("@rules_java//java:defs.bzl", "java_library")

package(
    default_applicable_licenses = ["//:license"],
    default_visibility = [
        "//src/java_tools/junitrunner:junitrunner_package_group",
    ],
)

java_library(
    name = "util",
    srcs = glob(["*.java"]),
    visibility = ["//visibility:public"],
    deps = [
        "//third_party/java/error_prone:annotations",
        "//third_party/java/jsr305_annotations",
        "//third_party/java/junit:not-testonly",
    ],
)

java_library(
    name = "util-android",
    srcs = [
        "TestPropertyExporter.java",
        "TestPropertyRunnerIntegration.java",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//third_party/java/jsr305_annotations",
    ],
)

java_library(
    name = "current_test",
    srcs = [
        "CurrentRunningTest.java",
        "TestNameProvider.java",
    ],
    deps = [
        "//third_party/java/junit:not-testonly",
    ],
)

java_library(
    name = "current_test-android",
    srcs = [
        "CurrentRunningTest.java",
        "TestNameProvider.java",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//third_party/java/junit:not-testonly-android",
    ],
)

filegroup(
    name = "srcs",
    srcs = glob(["**/*.java"]) + ["BUILD"],
    visibility = ["//src:__subpackages__"],
)
