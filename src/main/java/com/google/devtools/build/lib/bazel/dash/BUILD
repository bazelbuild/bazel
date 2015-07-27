filegroup(
    name = "srcs",
    srcs = glob(["**"]),
    visibility = ["//src/main/java:__pkg__"],
)

java_library(
    name = "dash",
    srcs = glob(["*.java"]),
    visibility = [
        "//src/main/java:__pkg__",
    ],
    deps = [
        "//src/main/java:analysis-exec-rules-skyframe",
        "//src/main/java:events",
        "//src/main/java:options",
        "//src/main/java:packages",
        "//src/main/java:runtime",
        "//src/main/java:vfs",
        "//src/main/protobuf:proto_dash",
        "//third_party:apache_httpclient",
        "//third_party:apache_httpcore",
        "//third_party:guava",
        "//third_party:protobuf",
    ],
)
