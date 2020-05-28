load("@rules_java//java:defs.bzl", "java_library")

package(
    default_visibility = [
        "//src/test/java/com/google/devtools/build/android/desugar:__subpackages__",
        "//src/tools/android/java/com/google/devtools/build/android/desugar:__subpackages__",
    ],
)

java_library(
    name = "langmodel",
    srcs = glob(["*.java"]),
    deps = [
        ":desugar_class_attr_java_proto",
        "//third_party:asm",
        "//third_party:asm-commons",
        "//third_party:asm-tree",
        "//third_party:auto_value",
        "//third_party:guava",
        "//third_party:jsr305",
        "//third_party/protobuf:protobuf_java",
    ],
)

proto_library(
    name = "desugar_class_attr_proto",
    srcs = [
        "desugar_method_attr.proto",
    ],
)

java_proto_library(
    name = "desugar_class_attr_java_proto",
    deps = [":desugar_class_attr_proto"],
)

filegroup(
    name = "srcs",
    srcs = glob(["**"]),
    visibility = ["//src/tools/android/java/com/google/devtools/build/android/desugar:__pkg__"],
)
