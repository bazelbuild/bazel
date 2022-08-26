load("@rules_java//java:defs.bzl", "java_library")

package(default_visibility = [
    "//src/java_tools/junitrunner:junitrunner_package_group",
])

java_library(
    name = "testbed",
    testonly = 1,
    srcs = glob(["*.java"]),
    deps = [
        "//third_party:guava",
        "//third_party:junit4",
        "//third_party:truth",
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
