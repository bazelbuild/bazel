package(default_visibility = ["//src:__subpackages__"])

java_library(
    name = "dash",
    srcs = glob(["*.java"]),
    deps = [
        "//src/main/java/com/google/devtools/build/lib:build-base",
        "//src/main/java/com/google/devtools/build/lib:events",
        "//src/main/java/com/google/devtools/build/lib:packages-internal",
        "//src/main/java/com/google/devtools/build/lib:runtime",
        "//src/main/java/com/google/devtools/common/options",
        "//src/main/protobuf:dash_java_proto",
        "//third_party:apache_httpclient",
        "//third_party:apache_httpcore",
        "//third_party:guava",
        "//third_party:protobuf",
    ],
)

filegroup(
    name = "srcs",
    srcs = glob(["**"]),
)
