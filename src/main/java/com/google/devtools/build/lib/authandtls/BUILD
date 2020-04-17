load("@rules_java//java:defs.bzl", "java_library")

package(default_visibility = ["//src:__subpackages__"])

filegroup(
    name = "srcs",
    srcs = glob(["**"]),
    visibility = ["//src:__subpackages__"],
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
        "//third_party/grpc:grpc-jar",
    ],
)
