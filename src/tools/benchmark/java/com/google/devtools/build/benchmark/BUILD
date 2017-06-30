package(default_visibility = ["//src/tools/benchmark:__subpackages__"])

java_proto_library(
    name = "build_data_java_proto",
    deps = [":build_data_proto"],
)

proto_library(
    name = "build_data_proto",
    srcs = ["build_data.proto"],
)

java_binary(
    name = "benchmark",
    srcs = glob(["*.java"]),
    main_class = "com.google.devtools.build.benchmark.Main",
    deps = [
        ":build_data_java_proto",
        "//src/main/java/com/google/devtools/build/lib:shell",
        "//src/main/java/com/google/devtools/build/lib:vfs",
        "//src/main/java/com/google/devtools/common/options",
        "//src/main/protobuf:option_filters_java_proto",
        "//src/tools/benchmark/java/com/google/devtools/build/benchmark/codegenerator:codegenerator_lib",
        "//third_party:auto_value",
        "//third_party:guava",
        "//third_party/protobuf:protobuf_java",
        "//third_party/protobuf:protobuf_java_util",
    ],
)

java_library(
    name = "benchmark_lib",
    testonly = 1,
    srcs = glob(["*.java"]),
    deps = [
        ":build_data_java_proto",
        "//src/main/java/com/google/devtools/build/lib:shell",
        "//src/main/java/com/google/devtools/build/lib:vfs",
        "//src/main/java/com/google/devtools/common/options",
        "//src/main/protobuf:option_filters_java_proto",
        "//src/tools/benchmark/java/com/google/devtools/build/benchmark/codegenerator:codegenerator_lib",
        "//third_party:auto_value",
        "//third_party:guava",
        "//third_party/protobuf:protobuf_java",
        "//third_party/protobuf:protobuf_java_util",
    ],
)

filegroup(
    name = "srcs",
    srcs = glob(["**"]),
)
