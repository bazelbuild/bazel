load("@rules_java//java:defs.bzl", "java_plugin")

package(default_visibility = [
    "//src/main/java/com/google/devtools/build/lib:__pkg__",
    "//src/main/java/com/google/devtools/build/lib/skylarkinterface:__pkg__",
    "//src/test/java/com/google/devtools/build/lib/skylarkinterface/processor:__pkg__",
])

licenses(["notice"])  # Apache 2.0

filegroup(
    name = "srcs",
    srcs = glob(
        ["**"],
    ),
)

# A processor for Starlark annotations.
java_plugin(
    name = "annotation_preprocessor",
    srcs = glob(["*.java"]),
    processor_class = "com.google.devtools.build.lib.skylarkinterface.processor.SkylarkCallableProcessor",
    deps = [
        "//src/main/java/com/google/devtools/build/lib/skylarkinterface:annotations",
        "//third_party:guava",
    ],
)
