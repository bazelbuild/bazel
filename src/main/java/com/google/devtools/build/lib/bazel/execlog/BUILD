filegroup(
    name = "srcs",
    srcs = glob(["**"]),
    visibility = ["//src/main/java/com/google/devtools/build/lib:__pkg__"],
)

java_library(
    name = "stable_sort",
    srcs = glob(["*.java"]),
    visibility = [
        "//src/main/java/com/google/devtools/build/lib:__pkg__",
        "//src/test/java/com/google/devtools/build/lib/bazel/execlog:__pkg__",
    ],
    deps = [
        "//src/main/java/com/google/devtools/build/lib:io",
        "//src/main/java/com/google/devtools/common/options",
        "//src/main/protobuf:spawn_java_proto",
        "//third_party:guava",
        "@com_google_protobuf//:protobuf_java_util",
    ],
)
