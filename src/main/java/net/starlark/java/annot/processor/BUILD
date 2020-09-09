load("@rules_java//java:defs.bzl", "java_plugin")

package(default_visibility = ["//src:__subpackages__"])

filegroup(
    name = "srcs",
    srcs = glob(["**"]),
    visibility = ["//src:__subpackages__"],
)

# A processor for Starlark annotations.
# (Legacy target for clients still depending on
# devtools/build/lib/syntax evaluator.)
java_plugin(
    name = "processor",
    srcs = ["StarlarkMethodProcessor.java"],
    processor_class = "net.starlark.java.annot.processor.StarlarkMethodProcessor",
    deps = [
        "//src/main/java/net/starlark/java/annot:annot_sans_processor",
        "//third_party:error_prone_annotations",
        "//third_party:guava",
    ],
)

# Processor for future use.
java_plugin(
    name = "processor_new",
    srcs = ["StarlarkMethodProcessor.java"],
    processor_class = "net.starlark.java.annot.processor.StarlarkMethodProcessor$New",
    deps = [
        "//src/main/java/net/starlark/java/annot:annot_sans_processor",
        "//third_party:error_prone_annotations",
        "//third_party:guava",
    ],
)
