package(
    default_visibility = ["//src:__subpackages__"],
)

java_library(
    name = "worker",
    srcs = glob(["*.java"]),
    deps = [
        "//src/main/java/com/google/devtools/build/lib:build-base",
        "//src/main/java/com/google/devtools/build/lib:concurrent",
        "//src/main/java/com/google/devtools/build/lib:events",
        "//src/main/java/com/google/devtools/build/lib:io",
        "//src/main/java/com/google/devtools/build/lib:packages-internal",
        "//src/main/java/com/google/devtools/build/lib:runtime",
        "//src/main/java/com/google/devtools/build/lib:util",
        "//src/main/java/com/google/devtools/build/lib:vfs",
        "//src/main/java/com/google/devtools/build/lib/actions",
        "//src/main/java/com/google/devtools/build/lib/standalone",
        "//src/main/java/com/google/devtools/common/options",
        "//src/main/protobuf:worker_protocol_java_proto",
        "//third_party:apache_commons_pool2",
        "//third_party:guava",
        "//third_party:jsr305",
        "//third_party:protobuf",
    ],
)

filegroup(
    name = "srcs",
    srcs = glob(["**"]),
)
