package(default_visibility = ["//src:__subpackages__"])

filegroup(
    name = "srcs",
    srcs = glob(["**"]),
    visibility = ["//src/main/java/com/google/devtools/build/lib:__pkg__"],
)

java_library(
    name = "buildeventservice-options",
    srcs = ["BuildEventServiceOptions.java"],
    deps = [
        "//src/main/java/com/google/devtools/common/options",
    ],
)

java_library(
    name = "buildeventservice",
    srcs = glob(
        ["*.java"],
        exclude = ["BuildEventServiceOptions.java"],
    ),
    visibility = [
        "//visibility:public",
    ],
    deps = [
        ":buildeventservice-options",
        "//src/main/java/com/google/devtools/build/lib:events",
        "//src/main/java/com/google/devtools/build/lib:io",
        "//src/main/java/com/google/devtools/build/lib:runtime",
        "//src/main/java/com/google/devtools/build/lib:util",
        "//src/main/java/com/google/devtools/build/lib/authandtls",
        "//src/main/java/com/google/devtools/build/lib/buildeventservice/client",
        "//src/main/java/com/google/devtools/build/lib/buildeventstream",
        "//src/main/java/com/google/devtools/build/lib/buildeventstream/proto:build_event_stream_java_proto",
        "//src/main/java/com/google/devtools/build/lib/buildeventstream/transports",
        "//src/main/java/com/google/devtools/build/lib/vfs",
        "//src/main/java/com/google/devtools/common/options",
        "//third_party:auto_value",
        "//third_party:guava",
        "//third_party:jsr305",
        "//third_party/grpc:grpc-jar",
        "@com_google_protobuf//:any_proto",
        "@com_google_protobuf//:protobuf_java",
        "@com_google_protobuf//:protobuf_java_util",
        "@googleapis//:google_devtools_build_v1_build_events_java_proto",
        "@googleapis//:google_devtools_build_v1_build_status_java_proto",
        "@googleapis//:google_devtools_build_v1_publish_build_event_java_proto",
    ],
)
