package(default_visibility = ["//src:__subpackages__"])

filegroup(
    name = "srcs",
    srcs = glob(["**"]),
    visibility = ["//src/main/java/com/google/devtools/build/lib:__pkg__"],
)

java_library(
    name = "client",
    srcs = glob(["*.java"]),
    runtime_deps = [
        # This is required for client TLS.
        "//third_party:netty_tcnative",
    ],
    deps = [
        "//third_party:guava",
        "//third_party:jsr305",
        "//third_party/grpc:grpc-jar",
        "@googleapis//:google_devtools_build_v1_publish_build_event_java_grpc",
        "@googleapis//:google_devtools_build_v1_publish_build_event_java_proto",
    ],
)
