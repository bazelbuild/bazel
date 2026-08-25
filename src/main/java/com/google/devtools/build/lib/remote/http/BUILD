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
    name = "http",
    srcs = glob(["*.java"]),
    runtime_deps = [
        # This is required for client TLS.
        "//third_party/java/netty_tcnative",
    ],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/analysis:blaze_version_info",
        "//src/main/java/com/google/devtools/build/lib/authandtls",
        "//src/main/java/com/google/devtools/build/lib/remote:Retrier",
        "//src/main/java/com/google/devtools/build/lib/remote/common",
        "//src/main/java/com/google/devtools/build/lib/remote/common:action_key",
        "//src/main/java/com/google/devtools/build/lib/remote/common:cache_not_found_exception",
        "//src/main/java/com/google/devtools/build/lib/remote/util:digest_output_stream",
        "//src/main/java/com/google/devtools/build/lib/remote/util:digest_util",
        "//src/main/java/com/google/devtools/build/lib/remote/util:utils",
        "//third_party/java/auth_kit",
        "//third_party/java/flogger",
        "//third_party/java/guava:base",
        "//third_party/java/guava:collect",
        "//third_party/java/guava:concurrent",
        "//third_party/java/guava:io",
        "//third_party/java/jsr305_annotations",
        "//third_party/java/netty4:buffer",
        "//third_party/java/netty4:codec",
        "//third_party/java/netty4:codec_http",
        "//third_party/java/netty4:common",
        "//third_party/java/netty4:handler",
        "//third_party/java/netty4:transport",
        "//third_party/java/netty4:transport_native_epoll",
        "//third_party/java/netty4:transport_native_kqueue",
        "//third_party/java/netty4:transport_native_unix_common",
        "//third_party/java/protobuf:protobuf_lite",
        "@remoteapis//:build_bazel_remote_execution_v2_remote_execution_java_proto",
    ],
)
