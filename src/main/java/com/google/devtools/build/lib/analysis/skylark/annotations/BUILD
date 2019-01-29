licenses(["notice"])  # Apache 2.0

package(
    default_visibility = ["//src:__subpackages__"],
)

filegroup(
    name = "srcs",
    srcs = glob(["**"]),
    visibility = ["//src/main/java/com/google/devtools/build/lib:__pkg__"],
)

java_library(
    name = "annotations",
    exported_plugins = ["//src/main/java/com/google/devtools/build/lib/analysis/skylark/annotations/processor:annotation_preprocessor"],
    exports = [":annotations_internal"],
)

java_library(
    name = "annotations_internal",
    srcs = glob(["*.java"]),
    visibility = ["//src/main/java/com/google/devtools/build/lib/analysis/skylark/annotations/processor:__pkg__"],
)
