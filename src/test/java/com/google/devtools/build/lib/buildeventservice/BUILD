load("@rules_java//java:defs.bzl", "java_test")

package(
    default_testonly = 1,
    default_visibility = ["//src:__subpackages__"],
)

licenses(["notice"])

filegroup(
    name = "srcs",
    testonly = 0,
    srcs = glob(["**"]),
    visibility = ["//src:__subpackages__"],
)

java_test(
    name = "BuildEventTransportTest",
    srcs = [
        "BuildEventServiceGrpcClientTest.java",
        "BuildEventServiceProtoUtilTest.java",
    ],
    test_class = "com.google.devtools.build.lib.AllTests",
    runtime_deps = ["//src/test/java/com/google/devtools/build/lib:test_runner"],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/buildeventservice",
        "//src/main/java/com/google/devtools/build/lib/buildeventservice/client",
        "//src/test/java/com/google/devtools/build/lib/testutil",
        "//third_party:guava",
        "//third_party:junit4",
        "//third_party:truth",
        "//third_party/grpc-java:grpc-jar",
        "//third_party/protobuf:protobuf_java",
        "//third_party/protobuf:protobuf_java_util",
        "@googleapis//:google_devtools_build_v1_build_events_java_proto",
        "@googleapis//:google_devtools_build_v1_build_status_java_proto",
        "@googleapis//:google_devtools_build_v1_publish_build_event_java_grpc",
        "@googleapis//:google_devtools_build_v1_publish_build_event_java_proto",
    ],
)

java_test(
    name = "BazelBuildEventServiceModuleTest",
    srcs = ["BazelBuildEventServiceModuleTest.java"],
    shard_count = 10,
    tags = [
        "no_windows",
    ],
    deps = [
        "//src/main/java/com/google/devtools/build/lib:runtime",
        "//src/main/java/com/google/devtools/build/lib/actions:action_lookup_data",
        "//src/main/java/com/google/devtools/build/lib/authandtls",
        "//src/main/java/com/google/devtools/build/lib/bugreport",
        "//src/main/java/com/google/devtools/build/lib/buildeventservice",
        "//src/main/java/com/google/devtools/build/lib/buildeventservice:buildeventservice-options",
        "//src/main/java/com/google/devtools/build/lib/buildeventstream",
        "//src/main/java/com/google/devtools/build/lib/buildeventstream/proto:build_event_stream_java_proto",
        "//src/main/java/com/google/devtools/build/lib/buildeventstream/transports",
        "//src/main/java/com/google/devtools/build/lib/network:connectivity_status",
        "//src/main/java/com/google/devtools/build/lib/network:noop_connectivity",
        "//src/main/java/com/google/devtools/build/lib/util:abrupt_exit_exception",
        "//src/main/java/com/google/devtools/build/lib/util:exit_code",
        "//src/test/java/com/google/devtools/build/lib/analysis/util",
        "//src/test/java/com/google/devtools/build/lib/buildtool/util",
        "//src/test/java/com/google/devtools/build/lib/testutil:TestConstants",
        "//src/test/java/com/google/devtools/build/skyframe:testutil",
        "//third_party:guava",
        "//third_party:jsr305",
        "//third_party:junit4",
        "//third_party:truth",
        "//third_party/grpc-java:grpc-jar",
        "//third_party/protobuf:protobuf_java",
        "@googleapis//:google_devtools_build_v1_build_events_java_proto",
        "@googleapis//:google_devtools_build_v1_publish_build_event_java_grpc",
        "@googleapis//:google_devtools_build_v1_publish_build_event_java_proto",
    ],
)
