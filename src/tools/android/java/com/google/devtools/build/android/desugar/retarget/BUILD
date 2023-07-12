package(
    default_visibility = [
        "//src/test/java/com/google/devtools/build/android/desugar:__subpackages__",
        "//src/tools/android/java/com/google/devtools/build/android/desugar:__subpackages__",
    ],
)

proto_library(
    name = "retarget_proto",
    srcs = ["retarget.proto"],
    deps = ["//src/tools/android/java/com/google/devtools/build/android/desugar/langmodel:desugar_class_attr_proto"],
)

java_proto_library(
    name = "retarget_java_proto",
    deps = [":retarget_proto"],
)

filegroup(
    name = "retarget_config_textproto",
    srcs = ["retarget_config.textproto"],
)

java_library(
    name = "retarget",
    srcs = glob(["*.java"]),
    resources = [
        ":retarget_config_textproto",
    ],
    deps = [
        ":retarget_java_proto",
        "//src/tools/android/java/com/google/devtools/build/android/desugar/langmodel",
        "//third_party:asm",
        "//third_party:asm-commons",
        "//third_party:asm-tree",
        "//third_party:auto_value",
        "//third_party:flogger",
        "//third_party:guava",
        "//third_party:jsr305",
        "//third_party/protobuf:protobuf_java",
    ],
)

filegroup(
    name = "srcs",
    srcs = glob(["**"]),
    visibility = ["//src/tools/android/java/com/google/devtools/build/android/desugar:__pkg__"],
)
