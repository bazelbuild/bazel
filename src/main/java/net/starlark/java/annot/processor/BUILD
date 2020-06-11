load("@rules_java//java:defs.bzl", "java_plugin")

package(default_visibility = ["//src:__subpackages__"])

filegroup(
    name = "srcs",
    srcs = glob(["**"]),
    visibility = ["//src:__subpackages__"],
)

# A processor for Starlark annotations.
java_plugin(
    name = "processor",
    srcs = glob(["*.java"]),
    processor_class = "net.starlark.java.annot.processor.StarlarkMethodProcessor",
    deps = [
        "//src/main/java/net/starlark/java/annot:annot_sans_processor",
        "//third_party:error_prone_annotations",
        "//third_party:guava",
    ],
)
