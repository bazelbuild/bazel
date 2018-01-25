package(
    default_visibility = ["//src:__subpackages__"],
)

filegroup(
    name = "srcs",
    srcs = glob(["**"]),
)

java_library(
    name = "serializers",
    srcs = glob(["*.java"]),
    deps = [
        "//src/main/java/com/google/devtools/build/lib/skyframe/serialization:kryo",
        "//third_party:guava",
        "//third_party/protobuf:protobuf_java",
    ],
)
