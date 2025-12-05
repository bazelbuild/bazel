load("@rules_java//java:defs.bzl", "java_library")

package(
    default_visibility = ["//src:__subpackages__"],
)

filegroup(
    name = "srcs",
    srcs = glob(["**"]),
    visibility = ["//src:__subpackages__"],
)

java_library(
    name = "rendering",
    srcs = glob(["*.java"]),
    deps = [
        "//src/main/java/build/stack/starlark/v1beta1:starlark_java_grpc",
        "//src/main/java/build/stack/starlark/v1beta1:starlark_java_proto",
        "//src/main/java/com/google/devtools/build/lib/events",
        "//src/main/java/com/google/devtools/build/lib/starlarkdocextract:extractionexception",
        "//src/main/java/com/google/devtools/build/lib/starlarkdocextract:labelrenderer",
        "//src/main/java/com/google/devtools/build/lib/starlarkdocextract:starlarkfunctioninfoextractor",
        "//src/main/protobuf:stardoc_output_java_proto",
        "//src/main/java/net/starlark/java/eval",
        "//src/main/java/net/starlark/java/syntax",
        "//src/tools/starlark/java/com/google/devtools/starlark/common",
        "//third_party:apache_velocity",
        "//third_party:guava",
        "//third_party:jsr305",
    ],
)
