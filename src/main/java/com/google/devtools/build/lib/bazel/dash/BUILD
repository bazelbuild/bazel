filegroup(
    name = "srcs",
    srcs = glob(["**"]),
    visibility = ["//src/main/java/com/google/devtools/build/lib:__pkg__"],
)

java_library(
    name = "dash",
    srcs = glob(["*.java"]),
    visibility = [
        "//src/main/java/com/google/devtools/build/lib:__pkg__",
    ],
    deps = [
        "//src/main/java/com/google/devtools/build/lib:analysis-exec-rules-skyframe",
        "//src/main/java/com/google/devtools/build/lib:events",
        "//src/main/java/com/google/devtools/build/lib:packages",
        "//src/main/java/com/google/devtools/build/lib:runtime",
        "//src/main/java/com/google/devtools/build/lib:vfs",
        "//src/main/java/com/google/devtools/common/options",
        "//src/main/protobuf:dash_proto",
        "//third_party:apache_httpclient",
        "//third_party:apache_httpcore",
        "//third_party:guava",
        "//third_party:protobuf",
    ],
)
