load("@rules_java//java:defs.bzl", "java_library")

package(default_visibility = ["//src:__subpackages__"])

filegroup(
    name = "srcs",
    srcs = glob(["**"]) + [
        "//src/main/java/com/google/devtools/build/lib/analysis/skylark/annotations/processor:srcs",
    ],
    visibility = ["//src:__subpackages__"],
)

java_library(
    name = "annotations",
    exported_plugins = ["//src/main/java/com/google/devtools/build/lib/analysis/skylark/annotations/processor:annotation_preprocessor"],
    exports = [":annotations_internal"],
)

java_library(
    name = "annotations_internal",
    srcs = glob(["*.java"]),
)
