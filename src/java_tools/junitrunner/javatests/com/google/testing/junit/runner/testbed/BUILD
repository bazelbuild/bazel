load("@rules_java//java:defs.bzl", "java_library")

package(
    default_applicable_licenses = ["//:license"],
    default_visibility = [
        "//src/java_tools/junitrunner:junitrunner_package_group",
    ],
)

java_library(
    name = "testbed",
    testonly = 1,
    srcs = glob(["*.java"]),
    deps = [
        "//third_party/java/guava:base",
        "//third_party/java/guava:io",
        "//third_party/java/junit",
        "//third_party/java/mockito",
        "//third_party/java/truth",
    ],
)

exports_files([
    "XmlOutputExercises.ant.xml",
])

filegroup(
    name = "srcs",
    srcs = glob(["*.java"]) + [
        "BUILD",
        "XmlOutputExercises.ant.xml",
    ],
    visibility = ["//src:__subpackages__"],
)
