load("@rules_java//java:defs.bzl", "java_test")

package(
    default_testonly = 1,
    default_visibility = ["//src:__subpackages__"],
)

filegroup(
    name = "srcs",
    testonly = 0,
    srcs = glob(["**"]),
    visibility = ["//src/test/java/com/google/devtools/build/lib/remote:__pkg__"],
)

java_test(
    name = "http",
    srcs = glob(["*.java"]),
    tags = [
        "requires-network",
    ],
    test_class = "com.google.devtools.build.lib.AllTests",
    deps = [
        "//src/main/java/com/google/devtools/build/lib/remote/http",
        "//src/main/java/com/google/devtools/build/lib/remote/util",
        "//src/main/java/com/google/devtools/build/lib/vfs",
        "//src/test/java/com/google/devtools/build/lib:test_runner",
        "//src/test/java/com/google/devtools/build/lib:testutil",
        "//src/tools/remote/src/main/java/com/google/devtools/build/remote/worker/http",
        "//third_party:api_client",
        "//third_party:auth",
        "//third_party:guava",
        "//third_party:mockito",
        "//third_party:netty",
        "//third_party:truth",
        "//third_party/protobuf:protobuf_java",
        "@bazel_remote_apis//build/bazel/remote/execution/v2:remote_execution_java_proto",
    ],
)
