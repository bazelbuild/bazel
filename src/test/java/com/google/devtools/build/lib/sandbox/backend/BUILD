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
    visibility = ["//src/test/java/com/google/devtools/build/lib/sandbox:__pkg__"],
)

COMMON_DEPS = [
    "//src/main/java/com/google/devtools/build/lib/remote/util:digest_util",
    "//src/main/java/com/google/devtools/build/lib/sandbox/backend",
    "//src/main/java/com/google/devtools/build/lib/vfs",
    "//third_party/java/guava:collect",
    "//third_party/java/junit",
    "//third_party/java/protobuf:protobuf_lite",
    "//third_party/java/truth",
]

java_test(
    name = "FileUriBlobsTest",
    srcs = ["FileUriBlobsTest.java"],
    test_class = "com.google.devtools.build.lib.sandbox.backend.FileUriBlobsTest",
    deps = COMMON_DEPS + [
        "@remoteapis//:build_bazel_remote_execution_v2_remote_execution_java_proto",
    ],
)

java_test(
    name = "BackendConfigTest",
    srcs = ["BackendConfigTest.java"],
    test_class = "com.google.devtools.build.lib.sandbox.backend.BackendConfigTest",
    deps = COMMON_DEPS,
)

java_test(
    name = "SandboxBackendSpawnRunnerTest",
    srcs = [
        "FakeBackend.java",
        "SandboxBackendSpawnRunnerTest.java",
    ],
    test_class = "com.google.devtools.build.lib.sandbox.backend.SandboxBackendSpawnRunnerTest",
    deps = COMMON_DEPS + [
        "//src/main/java/com/google/devtools/build/lib/actions",
        "//src/main/java/com/google/devtools/build/lib/actions:action_input",
        "//src/main/java/com/google/devtools/build/lib/actions:artifacts",
        "//src/main/java/com/google/devtools/build/lib/actions:file_metadata",
        "//src/main/java/com/google/devtools/build/lib/exec:spawn_runner",
        "//src/main/java/com/google/devtools/build/lib/util/io",
        "//src/main/java/com/google/devtools/build/lib/vfs:pathfragment",
        "//src/test/java/com/google/devtools/build/lib/actions/util",
        "//src/test/java/com/google/devtools/build/lib/exec/util",
        "//src/test/java/com/google/devtools/build/lib/remote/util",
        "//src/test/java/com/google/devtools/build/lib/vfs/util",
        "//third_party/google/bytestream:bytestream_grpc",
        "//third_party/google/bytestream:bytestream_java_proto",
        "//third_party/google/longrunning:operations_java_proto",
        "//third_party/google/protobuf:any_java_proto",
        "//third_party/google/rpc:error_details_java_proto",
        "//third_party/google/rpc:status_java_proto",
        "//third_party/java/grpc:core",
        "//third_party/java/grpc:inprocess",
        "//third_party/java/grpc:stub",
        "//third_party/java/guava:concurrent",
        "@remoteapis//:build_bazel_remote_execution_v2_remote_execution_java_grpc",
        "@remoteapis//:build_bazel_remote_execution_v2_remote_execution_java_proto",
    ],
)

java_test(
    name = "SandboxBackendSessionTest",
    srcs = ["SandboxBackendSessionTest.java"],
    test_class = "com.google.devtools.build.lib.sandbox.backend.SandboxBackendSessionTest",
    deps = COMMON_DEPS,
)
