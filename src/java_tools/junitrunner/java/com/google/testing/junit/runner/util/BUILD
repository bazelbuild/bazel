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
        "//third_party:error_prone_annotations",
        "//third_party:jsr305",
        "//third_party:junit4",
    ],
)

java_library(
    name = "current_test",
    srcs = [
        "CurrentRunningTest.java",
        "TestNameProvider.java",
    ],
    deps = [
        "//third_party:junit4",
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
        "//third_party:jsr305",
    ],
)

filegroup(
    name = "srcs",
    srcs = glob(["**/*.java"]) + ["BUILD"],
    visibility = ["//src:__subpackages__"],
)
