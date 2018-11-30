package(default_visibility = ["//src:__subpackages__"])

filegroup(
    name = "srcs",
    srcs = glob(["**"]),
    visibility = ["//src/main/java/com/google/devtools/build/lib:__pkg__"],
)

java_library(
    name = "authandtls",
    srcs = glob(["*.java"]),
    deps = [
        "//src/main/java/com/google/devtools/common/options",
        "//third_party:auth",
        "//third_party:guava",
        "//third_party:jsr305",
        "//third_party:netty",
        "//third_party/aws-sdk-auth-lite",
        "//third_party/grpc:grpc-jar",
    ],
)
