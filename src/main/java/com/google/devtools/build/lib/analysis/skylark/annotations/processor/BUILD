# Description:
#   A preprocessor for skylark annotations.
package(default_visibility = ["//src:__subpackages__"])

licenses(["notice"])  # Apache 2.0

filegroup(
    name = "srcs",
    srcs = glob(
        ["**"],
    ),
)

java_plugin(
    name = "annotation_preprocessor",
    srcs = glob(["*.java"]),
    processor_class = "com.google.devtools.build.lib.analysis.skylark.annotations.processor.SkylarkConfigurationFieldProcessor",
    deps = [
        "//src/main/java/com/google/devtools/build/lib:skylarkinterface",
        "//src/main/java/com/google/devtools/build/lib/analysis/skylark/annotations:annotations_internal",
    ],
)
