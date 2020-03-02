load("@rules_java//java:defs.bzl", "java_library")

package(default_visibility = ["//src:__subpackages__"])

licenses(["notice"])  # Apache 2.0

filegroup(
    name = "srcs",
    srcs = glob(["**"]),
)

# Annotations to make Java classes and methods accessible from Starlark.
# Depending on this library adds annotations processing to the build.
java_library(
    name = "skylarkinterface",
    exported_plugins = ["//src/main/java/com/google/devtools/build/lib/skylarkinterface/processor:annotation_preprocessor"],
    visibility = ["//visibility:public"],
    exports = [":annotations"],
)

# Annotations without processor, to avoid a dependency cycle in the processor itself.
java_library(
    name = "annotations",
    srcs = glob(["*.java"]),
    visibility = [
        "//src/main/java/com/google/devtools/build/lib/skylarkinterface/processor:__pkg__",
    ],
    deps = ["//third_party:jsr305"],
)
