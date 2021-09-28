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
        "//third_party:zstd-jni",
    ],
)
