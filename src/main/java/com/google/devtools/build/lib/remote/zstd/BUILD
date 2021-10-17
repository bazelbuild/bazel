package(
    default_visibility = ["//src:__subpackages__",]
)

filegroup(
    name = "srcs",
    srcs = glob(["*"]),
)

java_library(
    name = "zstd",
    srcs = glob(["*.java"]),
    deps = [
        "//third_party:guava",
        "//third_party:apache_commons_compress",
        "//third_party/protobuf:protobuf_java",
        "@zstd-jni//:zstd-jni",
    ],
)
