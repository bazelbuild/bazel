package(default_visibility = ["//src/tools/benchmark:__subpackages__"])

load("//tools/build_rules:genproto.bzl", "java_proto_library")

java_proto_library(
    name = "build_data_proto",
    src = "build_data.proto",
)

java_binary(
    name = "benchmark",
    srcs = glob(["*.java"]),
    main_class = "com.google.devtools.build.benchmark.Main",
    deps = [
        ":build_data_proto",
        "//src/main/java/com/google/devtools/build/lib:shell",
        "//src/main/java/com/google/devtools/build/lib:vfs",
        "//src/main/java/com/google/devtools/common/options",
        "//src/tools/benchmark/java/com/google/devtools/build/benchmark/codegenerator:codegenerator_lib",
        "//third_party:guava",
        "//third_party/protobuf:protobuf-util",
        "//third_party/protobuf:protobuf_java",
    ],
)

java_library(
    name = "benchmark_lib",
    testonly = 1,
    srcs = glob(["*.java"]),
    deps = [
        ":build_data_proto",
        "//src/main/java/com/google/devtools/build/lib:shell",
        "//src/main/java/com/google/devtools/build/lib:vfs",
        "//src/main/java/com/google/devtools/common/options",
        "//src/tools/benchmark/java/com/google/devtools/build/benchmark/codegenerator:codegenerator_lib",
        "//third_party:guava",
        "//third_party/protobuf:protobuf-util",
        "//third_party/protobuf:protobuf_java",
    ],
)

filegroup(
    name = "srcs",
    srcs = glob(["**"]),
)
