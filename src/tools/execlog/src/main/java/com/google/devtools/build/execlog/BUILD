filegroup(
    name = "srcs",
    srcs = glob(["**"]),
    visibility = ["//src/tools/execlog:__pkg__"],
)

java_library(
    name = "parser",
    srcs = glob(["*.java"]),
    visibility = ["//src/tools/execlog:__subpackages__"],
    deps = [
        "//src/main/java/com/google/devtools/common/options",
        "//src/main/protobuf:spawn_java_proto",
        "//third_party:guava",
    ],
)
