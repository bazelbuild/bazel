load("@rules_java//java:defs.bzl", "java_test")

filegroup(
    name = "srcs",
    srcs = glob(["**"]) + ["//src/test/java/build/stack/devtools/build/constellate/testdata:srcs"],
)

java_test(
    name = "ConstellateTest",
    size = "small",
    srcs = ["ConstellateTest.java"],
    data = ["//src/test/java/build/stack/devtools/build/constellate/testdata"],
    deps = [
        "//src/main/java/build/stack/devtools/build/constellate:constellate_lib",
        "//src/main/java/build/stack/devtools/build/constellate/rendering",
        "//src/main/java/build/stack/starlark/v1beta1:starlark_java_proto",
        "//src/main/java/com/google/devtools/build/lib/cmdline",
        "//src/main/java/com/google/devtools/build/lib/packages/semantics",
        "//src/main/java/com/google/devtools/common/options",
        "//src/main/java/net/starlark/java/eval",
        "//src/main/java/net/starlark/java/syntax",
        "//src/main/protobuf:stardoc_output_java_proto",
        "//third_party:guava",
        "//third_party:junit4",
        "//third_party:truth",
        "@maven//:junit_junit",
    ],
)

java_test(
    name = "DebugTest",
    size = "small",
    srcs = ["DebugTest.java"],
    data = ["//src/test/java/build/stack/devtools/build/constellate/testdata"],
    deps = [
        "//src/main/java/build/stack/devtools/build/constellate:constellate_lib",
        "//src/main/java/build/stack/starlark/v1beta1:starlark_java_proto",
        "//src/main/java/com/google/devtools/build/lib/cmdline",
        "//src/main/java/com/google/devtools/build/lib/packages/semantics",
        "//src/main/java/com/google/devtools/common/options",
        "//src/main/java/net/starlark/java/eval",
        "//src/main/java/net/starlark/java/syntax",
        "//src/main/protobuf:stardoc_output_java_proto",
        "//third_party:guava",
        "//third_party:junit4",
        "@maven//:junit_junit",
    ],
)

java_test(
    name = "GrpcIntegrationTest",
    size = "small",
    srcs = ["GrpcIntegrationTest.java"],
    data = ["//src/test/java/build/stack/devtools/build/constellate/testdata"],
    deps = [
        "//src/main/java/build/stack/devtools/build/constellate:server_lib",
        "//src/main/java/build/stack/starlark/v1beta1:starlark_java_grpc",
        "//src/main/java/build/stack/starlark/v1beta1:starlark_java_proto",
        "//src/main/java/com/google/devtools/build/lib/packages/semantics",
        "//src/main/java/com/google/devtools/common/options",
        "//src/main/java/net/starlark/java/eval",
        "//src/main/protobuf:stardoc_output_java_proto",
        "//third_party:guava",
        "//third_party:junit4",
        "//third_party/grpc-java:grpc-jar",
        "@grpc-java//api",
        "@grpc-java//core:inprocess",
        "@grpc-java//stub",
        "@maven//:junit_junit",
    ],
)

java_test(
    name = "GrpcWrapperTest",
    size = "small",
    srcs = ["GrpcWrapperTest.java"],
    data = ["//src/test/java/build/stack/devtools/build/constellate/testdata"],
    deps = [
        "//src/main/java/build/stack/devtools/build/constellate:server_lib",
        "//src/main/java/build/stack/starlark/v1beta1:starlark_java_grpc",
        "//src/main/java/build/stack/starlark/v1beta1:starlark_java_proto",
        "//src/main/java/com/google/devtools/build/lib/packages/semantics",
        "//src/main/java/com/google/devtools/common/options",
        "//src/main/java/net/starlark/java/eval",
        "//src/main/protobuf:stardoc_output_java_proto",
        "//third_party:guava",
        "//third_party:junit4",
        "//third_party/grpc-java:grpc-jar",
        "@grpc-java//api",
        "@grpc-java//core:inprocess",
        "@grpc-java//stub",
        "@maven//:junit_junit",
    ],
)
