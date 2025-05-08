load("@rules_java//java:defs.bzl", "java_library")

package(
    default_applicable_licenses = ["//:license"],
    default_visibility = ["//src:__subpackages__"],
)

filegroup(
    name = "srcs",
    srcs = glob(["**"]),
    visibility = ["//src:__subpackages__"],
)

java_library(
    name = "client",
    srcs = glob(["*.java"]),
    runtime_deps = [
        # This is required for client TLS.
        "//third_party:netty_tcnative",
    ],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/remote/util",
        "//third_party:guava",
        "//third_party:jsr305",
        "//third_party/grpc-java:grpc-jar",
        "@googleapis//google/devtools/build/v1:build_java_grpc",
        "@googleapis//google/devtools/build/v1:build_java_proto",
    ],
)
