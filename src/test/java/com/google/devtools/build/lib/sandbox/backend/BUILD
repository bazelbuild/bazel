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
    "@remoteapis//:build_bazel_semver_semver_java_proto",
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
    deps = COMMON_DEPS + [
        "//src/main/java/com/google/devtools/build/lib/util:os",
    ],
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
        "//src/main/java/com/google/devtools/build/lib/remote/common",
        "//src/main/protobuf:spawn_java_proto",
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
        "//third_party/java/grpc:netty",
        "//third_party/java/grpc:stub",
        "//third_party/java/guava:concurrent",
        "//third_party/java/mockito",
        "//third_party/java/netty4:transport",
        "//third_party/java/netty4:transport_native_epoll",
        "//third_party/java/netty4:transport_native_kqueue",
        "//third_party/java/netty4:transport_native_unix_common",
        "@remoteapis//:build_bazel_remote_execution_v2_remote_execution_java_grpc",
        "@remoteapis//:build_bazel_remote_execution_v2_remote_execution_java_proto",
    ],
)

java_test(
    name = "SandboxBackendCapabilitiesTest",
    srcs = [
        "FakeBackend.java",
        "SandboxBackendCapabilitiesTest.java",
    ],
    test_class = "com.google.devtools.build.lib.sandbox.backend.SandboxBackendCapabilitiesTest",
    deps = COMMON_DEPS + [
        "//third_party/google/bytestream:bytestream_grpc",
        "//third_party/google/bytestream:bytestream_java_proto",
        "//third_party/google/longrunning:operations_java_proto",
        "//third_party/google/protobuf:any_java_proto",
        "//third_party/google/rpc:error_details_java_proto",
        "//third_party/google/rpc:status_java_proto",
        "//third_party/java/grpc:core",
        "//third_party/java/grpc:inprocess",
        "//third_party/java/grpc:netty",
        "//third_party/java/grpc:stub",
        "//third_party/java/netty4:transport",
        "//third_party/java/netty4:transport_native_epoll",
        "//third_party/java/netty4:transport_native_kqueue",
        "//third_party/java/netty4:transport_native_unix_common",
        "@remoteapis//:build_bazel_remote_execution_v2_remote_execution_java_grpc",
        "@remoteapis//:build_bazel_remote_execution_v2_remote_execution_java_proto",
    ],
)

java_test(
    name = "SandboxBackendTransportTest",
    srcs = ["SandboxBackendTransportTest.java"],
    test_class = "com.google.devtools.build.lib.sandbox.backend.SandboxBackendTransportTest",
    deps = COMMON_DEPS + [
        "//src/main/java/com/google/devtools/build/lib/util:os",
    ],
)

java_test(
    name = "SandboxBackendsTest",
    srcs = [
        "FakeBackend.java",
        "SandboxBackendsTest.java",
    ],
    test_class = "com.google.devtools.build.lib.sandbox.backend.SandboxBackendsTest",
    deps = COMMON_DEPS + [
        "//src/main/java/com/google/devtools/build/lib/actions",
        "//src/main/java/com/google/devtools/build/lib/exec:execution_options",
        "//src/main/java/com/google/devtools/build/lib/exec:spawn_runner",
        "//src/main/java/com/google/devtools/build/lib/exec:spawn_strategy_registry",
        "//src/main/java/com/google/devtools/build/lib/runtime:blaze_command_cluster",
        "//src/main/java/com/google/devtools/common/options",
        "//src/test/java/com/google/devtools/build/lib/exec/util",
        "//src/test/java/com/google/devtools/build/lib/vfs/util",
        "//third_party/java/mockito",
        "//third_party/google/bytestream:bytestream_grpc",
        "//third_party/google/bytestream:bytestream_java_proto",
        "//third_party/google/longrunning:operations_java_proto",
        "//third_party/google/protobuf:any_java_proto",
        "//third_party/google/rpc:error_details_java_proto",
        "//third_party/google/rpc:status_java_proto",
        "//third_party/java/grpc:core",
        "//third_party/java/grpc:inprocess",
        "//third_party/java/grpc:netty",
        "//third_party/java/grpc:stub",
        "//third_party/java/netty4:transport",
        "//third_party/java/netty4:transport_native_epoll",
        "//third_party/java/netty4:transport_native_kqueue",
        "//third_party/java/netty4:transport_native_unix_common",
        "@remoteapis//:build_bazel_remote_execution_v2_remote_execution_java_grpc",
        "@remoteapis//:build_bazel_remote_execution_v2_remote_execution_java_proto",
    ],
)

java_test(
    name = "SandboxBackendSessionTest",
    srcs = [
        "FakeBackend.java",
        "SandboxBackendSessionTest.java",
    ],
    test_class = "com.google.devtools.build.lib.sandbox.backend.SandboxBackendSessionTest",
    deps = COMMON_DEPS + [
        "//third_party/google/bytestream:bytestream_grpc",
        "//third_party/google/bytestream:bytestream_java_proto",
        "//third_party/google/longrunning:operations_java_proto",
        "//third_party/google/protobuf:any_java_proto",
        "//third_party/google/rpc:error_details_java_proto",
        "//third_party/google/rpc:status_java_proto",
        "//third_party/java/grpc:core",
        "//third_party/java/grpc:inprocess",
        "//third_party/java/grpc:netty",
        "//third_party/java/grpc:stub",
        "//third_party/java/netty4:transport",
        "//third_party/java/netty4:transport_native_epoll",
        "//third_party/java/netty4:transport_native_kqueue",
        "//third_party/java/netty4:transport_native_unix_common",
        "@remoteapis//:build_bazel_remote_execution_v2_remote_execution_java_grpc",
        "@remoteapis//:build_bazel_remote_execution_v2_remote_execution_java_proto",
    ],
)


java_test(
    name = "SandboxBackendSocketTest",
    srcs = [
        "FakeBackend.java",
        "SandboxBackendSocketTest.java",
    ],
    test_class = "com.google.devtools.build.lib.sandbox.backend.SandboxBackendSocketTest",
    deps = COMMON_DEPS + [
        "//src/main/java/com/google/devtools/build/lib/actions",
        "//src/main/java/com/google/devtools/build/lib/actions:action_input",
        "//src/main/java/com/google/devtools/build/lib/actions:artifacts",
        "//src/main/java/com/google/devtools/build/lib/actions:file_metadata",
        "//src/main/java/com/google/devtools/build/lib/exec:spawn_runner",
        "//src/main/java/com/google/devtools/build/lib/remote/common",
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
        "//third_party/java/grpc:netty",
        "//third_party/java/grpc:stub",
        "//third_party/java/guava:concurrent",
        "//third_party/java/netty4:transport",
        "//third_party/java/netty4:transport_native_epoll",
        "//third_party/java/netty4:transport_native_kqueue",
        "//third_party/java/netty4:transport_native_unix_common",
        "@remoteapis//:build_bazel_remote_execution_v2_remote_execution_java_grpc",
        "@remoteapis//:build_bazel_remote_execution_v2_remote_execution_java_proto",
    ],
)

java_test(
    name = "SandboxBackendDynamicExecutionIntegrationTest",
    size = "medium",
    srcs = [
        "FakeBackend.java",
        "SandboxBackendDynamicExecutionIntegrationTest.java",
    ],
    jvm_flags = [
        # Required for compatibility with records.
        "-Djol.magicFieldOffset=true",
    ],
    test_class = "com.google.devtools.build.lib.sandbox.backend.SandboxBackendDynamicExecutionIntegrationTest",
    runtime_deps = [
        "//third_party/java/grpc:util",
    ],
    deps = COMMON_DEPS + [
        "//src/main/java/com/google/devtools/build/lib/actions",
        "//src/main/java/com/google/devtools/build/lib/actions:artifacts",
        "//src/main/java/com/google/devtools/build/lib/dynamic",
        "//src/main/java/com/google/devtools/build/lib/runtime:blaze_command_cluster",
        "//src/test/java/com/google/devtools/build/lib/buildtool/util",
        "//third_party/google/bytestream:bytestream_grpc",
        "//third_party/google/bytestream:bytestream_java_proto",
        "//third_party/google/longrunning:operations_java_proto",
        "//third_party/google/protobuf:any_java_proto",
        "//third_party/google/rpc:error_details_java_proto",
        "//third_party/google/rpc:status_java_proto",
        "//third_party/java/grpc:core",
        "//third_party/java/grpc:inprocess",
        "//third_party/java/grpc:netty",
        "//third_party/java/grpc:stub",
        "//third_party/java/netty4:transport",
        "//third_party/java/netty4:transport_native_epoll",
        "//third_party/java/netty4:transport_native_kqueue",
        "//third_party/java/netty4:transport_native_unix_common",
        "@remoteapis//:build_bazel_remote_execution_v2_remote_execution_java_grpc",
        "@remoteapis//:build_bazel_remote_execution_v2_remote_execution_java_proto",
    ],
)

java_test(
    name = "SandboxBackendDiskCacheIntegrationTest",
    size = "medium",
    srcs = [
        "FakeBackend.java",
        "SandboxBackendDiskCacheIntegrationTest.java",
    ],
    jvm_flags = [
        # Required for compatibility with records.
        "-Djol.magicFieldOffset=true",
    ],
    test_class = "com.google.devtools.build.lib.sandbox.backend.SandboxBackendDiskCacheIntegrationTest",
    runtime_deps = [
        "//third_party/java/grpc:util",
    ],
    deps = COMMON_DEPS + [
        "//src/main/java/com/google/devtools/build/lib/actions",
        "//src/main/java/com/google/devtools/build/lib/authandtls/credentialhelper:credential_module",
        "//src/main/java/com/google/devtools/build/lib/remote:remote_module",
        "//src/main/java/com/google/devtools/build/lib/remote/options",
        "//src/main/java/com/google/devtools/build/lib/runtime:blaze_command_cluster",
        "//src/main/java/com/google/devtools/build/lib/runtime:block_waiting_module",
        "//src/main/java/com/google/devtools/build/lib/runtime:build_summary_stats_module",
        "//src/main/java/com/google/devtools/common/options:options_internal",
        "//src/test/java/com/google/devtools/build/lib/buildtool/util",
        "//third_party/google/bytestream:bytestream_grpc",
        "//third_party/google/bytestream:bytestream_java_proto",
        "//third_party/google/longrunning:operations_java_proto",
        "//third_party/google/protobuf:any_java_proto",
        "//third_party/google/rpc:error_details_java_proto",
        "//third_party/google/rpc:status_java_proto",
        "//third_party/java/grpc:core",
        "//third_party/java/grpc:inprocess",
        "//third_party/java/grpc:netty",
        "//third_party/java/grpc:stub",
        "//third_party/java/netty4:transport",
        "//third_party/java/netty4:transport_native_epoll",
        "//third_party/java/netty4:transport_native_kqueue",
        "//third_party/java/netty4:transport_native_unix_common",
        "@remoteapis//:build_bazel_remote_execution_v2_remote_execution_java_grpc",
        "@remoteapis//:build_bazel_remote_execution_v2_remote_execution_java_proto",
    ],
)
