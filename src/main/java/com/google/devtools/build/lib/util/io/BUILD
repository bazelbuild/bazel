load("@rules_java//java:defs.bzl", "java_library")

package(default_visibility = ["//src:__subpackages__"])

filegroup(
    name = "srcs",
    srcs = glob(["**"]),
    visibility = ["//src/main/java/com/google/devtools/build/lib:__pkg__"],
)

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
        "//third_party:guava",
        "@com_google_protobuf//:protobuf_java",
        "@com_google_protobuf//:protobuf_java_util",
    ],
)

java_library(
    name = "out-err",
    srcs = OUT_ERR_SRCS,
)
