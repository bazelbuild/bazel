load("@rules_java//java:defs.bzl", "java_plugin")

# Description:
#   A preprocessor for Starlark annotations.
package(default_visibility = ["//src:__subpackages__"])

filegroup(
    name = "srcs",
    srcs = glob(["**"]),
    visibility = ["//src:__subpackages__"],
)

java_plugin(
    name = "annotation_preprocessor",
    srcs = glob(["*.java"]),
    processor_class = "com.google.devtools.build.lib.analysis.skylark.annotations.processor.SkylarkConfigurationFieldProcessor",
    deps = [
        "//src/main/java/com/google/devtools/build/lib/analysis/skylark/annotations:annotations_internal",
        "//src/main/java/com/google/devtools/build/lib/skylarkinterface",
        "//third_party:guava",
    ],
)
