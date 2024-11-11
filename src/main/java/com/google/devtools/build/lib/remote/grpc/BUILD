load("@rules_java//java:defs.bzl", "java_library")

package(
    default_applicable_licenses = ["//:license"],
    default_visibility = ["//src:__subpackages__"],
)

filegroup(
    name = "srcs",
    srcs = glob(["*"]),
    visibility = ["//src:__subpackages__"],
)

java_library(
    name = "grpc",
    srcs = glob(["*.java"]),
    deps = [
        "//src/main/java/com/google/devtools/build/lib/concurrent",
        "//third_party:guava",
        "//third_party:jsr305",
        "//third_party:netty",
        "//third_party:rxjava3",
        "//third_party/grpc-java:grpc-jar",
    ],
)
