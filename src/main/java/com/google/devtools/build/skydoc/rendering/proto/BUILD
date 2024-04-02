# TODO(arostovtsev): move this package to docgen or rules/starlarkdocextract when we remove legacy Stardoc.

load("@rules_java//java:defs.bzl", "java_proto_library")
load("@rules_proto//proto:defs.bzl", "proto_library")
load("//tools/build_rules:utilities.bzl", "java_library_srcs")

package(
    default_visibility = [
        "//src/main/java/com/google/devtools/build/docgen:__subpackages__",
        "//src/main/java/com/google/devtools/build/lib/rules/starlarkdocextract:__subpackages__",
        "//src/main/java/com/google/devtools/build/skydoc:__subpackages__",
        "//src/test/java/com/google/devtools/build/docgen:__subpackages__",
        "//src/test/java/com/google/devtools/build/lib/rules/starlarkdocextract:__subpackages__",
        "//src/test/java/com/google/devtools/build/skydoc:__subpackages__",
    ],
)

licenses(["notice"])  # Apache 2.0

filegroup(
    name = "srcs",
    srcs = glob(["**"]),
)

proto_library(
    name = "stardoc_output_proto",
    srcs = ["stardoc_output.proto"],
)

java_proto_library(
    name = "stardoc_output_java_proto",
    deps = [":stardoc_output_proto"],
)

java_library_srcs(
    name = "dist_jars",
    visibility = ["//src:__pkg__"],
    deps = [":stardoc_output_java_proto"],
)
