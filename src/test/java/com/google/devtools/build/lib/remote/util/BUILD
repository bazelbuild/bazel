load("@rules_java//java:defs.bzl", "java_library")

package(
    default_applicable_licenses = ["//:license"],
    default_testonly = 1,
    default_visibility = ["//src:__subpackages__"],
)

filegroup(
    name = "srcs",
    testonly = 0,
    srcs = glob(["**"]),
    visibility = ["//src:__subpackages__"],
)

INTEGRATION_TEST_UTILS_SRCS = ["IntegrationTestUtils.java"]

FREE_PORT_FINDER_SRCS = ["FreePortFinder.java"]

java_library(
    name = "util",
    srcs = glob(
        ["*.java"],
        exclude = INTEGRATION_TEST_UTILS_SRCS + FREE_PORT_FINDER_SRCS,
    ),
    visibility = ["//visibility:public"],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/actions",
        "//src/main/java/com/google/devtools/build/lib/actions:action_input",
        "//src/main/java/com/google/devtools/build/lib/actions:artifacts",
        "//src/main/java/com/google/devtools/build/lib/exec:spawn_input_expander",
        "//src/main/java/com/google/devtools/build/lib/exec:spawn_runner",
        "//src/main/java/com/google/devtools/build/lib/remote:Retrier",
        "//src/main/java/com/google/devtools/build/lib/remote:remote_action_file_system",
        "//src/main/java/com/google/devtools/build/lib/remote/common",
        "//src/main/java/com/google/devtools/build/lib/remote/common:action_key",
        "//src/main/java/com/google/devtools/build/lib/remote/common:bulk_transfer_exception",
        "//src/main/java/com/google/devtools/build/lib/remote/common:cache_not_found_exception",
        "//src/main/java/com/google/devtools/build/lib/remote/util:async_task_cache",
        "//src/main/java/com/google/devtools/build/lib/remote/util:rx_futures",
        "//src/main/java/com/google/devtools/build/lib/remote/util:rx_utils",
        "//src/main/java/com/google/devtools/build/lib/util/io",
        "//src/main/java/com/google/devtools/build/lib/vfs",
        "//src/main/java/com/google/devtools/build/lib/vfs:pathfragment",
        "//src/main/protobuf:spawn_java_proto",
        "//third_party/google/bytestream:bytestream_grpc",
        "//third_party/google/bytestream:bytestream_java_proto",
        "//third_party/java/grpc:core",
        "//third_party/java/grpc:stub",
        "//third_party/java/guava:base",
        "//third_party/java/guava:collect",
        "//third_party/java/guava:concurrent",
        "//third_party/java/jsr305_annotations",
        "//third_party/java/junit",
        "//third_party/java/protobuf:protobuf_lite",
        "//third_party/java/rxjava3",
        "//third_party/java/truth",
        "@remoteapis//:build_bazel_remote_execution_v2_remote_execution_java_proto",
    ],
)

java_library(
    name = "free_port_finder",
    srcs = FREE_PORT_FINDER_SRCS,
)

java_library(
    name = "integration_test_utils",
    srcs = INTEGRATION_TEST_UTILS_SRCS,
    data = [
        "//src/tools/remote:worker",
        "//src/tools/remote:worker_deploy.jar",
    ],
    deps = [
        ":free_port_finder",
        "//src/main/java/com/google/devtools/build/lib/shell",
        "//src/main/java/com/google/devtools/build/lib/util:os",
        "//src/main/java/com/google/devtools/build/lib/vfs:pathfragment",
        "//third_party/java/guava:base",
        "//third_party/java/guava:collect",
        "//third_party/java/guava:hash",
        "//third_party/java/jsr305_annotations",
        "//third_party/java/junit",
        "@remoteapis//:build_bazel_remote_execution_v2_remote_execution_java_proto",
        "@rules_java//java/runfiles",
    ],
)
