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
    name = "chunking",
    srcs = [
        "ChunkingConfig.java",
        "FastCdcChunker.java",
    ],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/remote/util:digest_utils",
        "//third_party:guava",
        "//third_party:jsr305",
        "@remoteapis//:build_bazel_remote_execution_v2_remote_execution_java_proto",
    ],
)
