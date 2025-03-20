load("@rules_java//java:defs.bzl", "java_library", "java_plugin")

package(
    default_applicable_licenses = ["//:license"],
    default_visibility = ["//src:__subpackages__"],
)

filegroup(
    name = "srcs",
    srcs = glob(["**"]),
    visibility = ["//src:__subpackages__"],
)

# A processor for Starlark annotations.
java_plugin(
    name = "processor",
    processor_class = "net.starlark.java.annot.processor.StarlarkMethodProcessor",
    visibility = ["//src/main/java/net/starlark/java:starlark"],
    deps = [":processor_lib"],
)

java_library(
    name = "processor_lib",
    srcs = ["StarlarkMethodProcessor.java"],
    visibility = ["//src/main/java/net/starlark/java:starlark"],
    deps = [
        "//src/main/java/net/starlark/java/annot:annot_sans_processor",
        "//third_party:error_prone_annotations",
        "//third_party:guava",
        "//third_party:jsr305",
    ],
)
