load("@rules_java//java:defs.bzl", "java_test")

package(
    default_applicable_licenses = ["//:license"],
    default_testonly = 1,
    default_visibility = ["//src:__subpackages__"],
)

filegroup(
    name = "srcs",
    testonly = 0,
    srcs = glob(["**"]),
)

java_test(
    name = "http",
    srcs = glob(["*.java"]),
    jvm_flags = [
        # Required for Mockito.spy.
        "--add-opens=java.base/java.io=ALL-UNNAMED",
    ],
    tags = [
        "requires-network",
    ],
    test_class = "com.google.devtools.build.lib.AllTests",
    deps = [
        "//src/main/java/com/google/devtools/build/lib/actions",
        "//src/main/java/com/google/devtools/build/lib/authandtls",
        "//src/main/java/com/google/devtools/build/lib/exec:spawn_runner",
        "//src/main/java/com/google/devtools/build/lib/remote:Retrier",
        "//src/main/java/com/google/devtools/build/lib/remote:combined_cache_client_factory",
        "//src/main/java/com/google/devtools/build/lib/remote/common",
        "//src/main/java/com/google/devtools/build/lib/remote/common:action_key",
        "//src/main/java/com/google/devtools/build/lib/remote/http",
        "//src/main/java/com/google/devtools/build/lib/remote/options",
        "//src/main/java/com/google/devtools/build/lib/remote/util:digest_util",
        "//src/main/java/com/google/devtools/build/lib/remote/util:futures",
        "//src/main/java/com/google/devtools/build/lib/remote/util:tracing_metadata_utils",
        "//src/main/java/com/google/devtools/build/lib/vfs",
        "//src/main/java/com/google/devtools/common/options",
        "//src/test/java/com/google/devtools/build/lib:test_runner",  # unuseddeps: keep buildcleaner: keep
        "//src/tools/remote/src/main/java/com/google/devtools/build/remote/worker/http",
        "//third_party/java/auth_kit:credentials",
        "//third_party/java/guava:base",
        "//third_party/java/guava:collect",
        "//third_party/java/guava:concurrent",
        "//third_party/java/guava:net",
        "//third_party/java/jsr305_annotations",
        "//third_party/java/junit",
        "//third_party/java/mockito",
        "//third_party/java/netty4:buffer",
        "//third_party/java/netty4:codec",
        "//third_party/java/netty4:codec_http",
        "//third_party/java/netty4:transport",
        "//third_party/java/netty4:transport_native_epoll",
        "//third_party/java/netty4:transport_native_kqueue",
        "//third_party/java/netty4:transport_native_unix_common",
        "//third_party/java/protobuf:protobuf_lite",
        "//third_party/java/truth",
        "@remoteapis//:build_bazel_remote_execution_v2_remote_execution_java_proto",
    ],
)
