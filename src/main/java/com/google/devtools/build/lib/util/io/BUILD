load("@rules_java//java:defs.bzl", "java_library")

package(
    default_visibility = ["//src:__subpackages__"],
)

filegroup(
    name = "srcs",
    srcs = glob(["**"]),
    visibility = ["//src:__subpackages__"],
)

# I/O utilities.

OUT_ERR_SRCS = [
    "OutErr.java",
    "LinePrefixingOutputStream.java",
    "LineFlushingOutputStream.java",
]

java_library(
    name = "io",
    srcs = glob(
        ["*.java"],
        exclude = OUT_ERR_SRCS,
    ),
    deps = [
        ":out-err",
        "//src/main/java/com/google/devtools/build/lib/clock",
        "//src/main/java/com/google/devtools/build/lib/concurrent",
        "//src/main/java/com/google/devtools/build/lib/profiler",
        "//src/main/java/com/google/devtools/build/lib/vfs",
        "//src/main/java/com/google/devtools/build/lib/vfs:pathfragment",
        "//third_party:flogger",
        "//third_party:guava",
        "//third_party/protobuf:protobuf_java",
        "//third_party/protobuf:protobuf_java_util",
    ],
)

java_library(
    name = "out-err",
    srcs = OUT_ERR_SRCS,
)
