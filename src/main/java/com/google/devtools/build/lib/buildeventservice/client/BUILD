filegroup(
    name = "srcs",
    srcs = glob(["**"]),
    visibility = ["//src/main/java/com/google/devtools/build/lib:__pkg__"],
)

java_library(
    name = "client",
    srcs = glob(["*.java"]),
    visibility = [
        "//src/main/java/com/google/devtools/build/lib:__pkg__",
        "//src/test/java/com/google/devtools/build/lib/buildeventservice:__pkg__",
    ],
    runtime_deps = [
        # This is required for client TLS.
        "//third_party:netty_tcnative",
    ],
    deps = [
        "//src/main/java/com/google/devtools/build/lib:util",
        "//third_party:apache_httpclient",
        "//third_party:apache_httpcore",
        "//third_party:auth",
        "//third_party:gson",
        "//third_party:guava",
        "//third_party:joda_time",
        "//third_party:jsr305",
        "//third_party:netty",
        "//third_party/grpc:grpc-jar",
        "@googleapis//:google_devtools_build_v1_publish_build_event_java_grpc",
        "@googleapis//:google_devtools_build_v1_publish_build_event_java_proto",
    ],
)
