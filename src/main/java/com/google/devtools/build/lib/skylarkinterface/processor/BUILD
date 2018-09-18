# Description:
#   A preprocessor for skylarkinterface annotations.
package(default_visibility = [
    "//src/main/java/com/google/devtools/build/lib:__pkg__",
    "//src/test/java/com/google/devtools/build/lib/skylarkinterface/processor:__pkg__",
])

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
    processor_class = "com.google.devtools.build.lib.skylarkinterface.processor.SkylarkCallableProcessor",
    deps = [
        "//src/main/java/com/google/devtools/build/lib:skylark_semantics",
        "//src/main/java/com/google/devtools/build/lib:skylarkinterface_internal",
        "//third_party:guava",
    ],
)
