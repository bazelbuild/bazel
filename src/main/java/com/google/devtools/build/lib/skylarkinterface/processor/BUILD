load("@rules_java//java:defs.bzl", "java_plugin")

package(default_visibility = ["//src:__subpackages__"])

filegroup(
    name = "srcs",
    srcs = glob(["**"]),
    visibility = ["//src:__subpackages__"],
)

# A processor for Starlark annotations.
java_plugin(
    name = "annotation_preprocessor",
    srcs = glob(["*.java"]),
    processor_class = "com.google.devtools.build.lib.skylarkinterface.processor.SkylarkCallableProcessor",
    deps = [
        "//src/main/java/com/google/devtools/build/lib/skylarkinterface:annotations",
        "//third_party:error_prone_annotations",
        "//third_party:guava",
    ],
)
