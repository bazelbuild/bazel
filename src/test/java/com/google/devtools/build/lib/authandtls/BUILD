load("@rules_java//java:defs.bzl", "java_library", "java_test")

package(
    default_applicable_licenses = ["//:license"],
    default_testonly = 1,
    default_visibility = ["//src:__subpackages__"],
)

filegroup(
    name = "srcs",
    testonly = 0,
    srcs = glob(["**"]) + [
        "//src/test/java/com/google/devtools/build/lib/authandtls/credentialhelper:srcs",
    ],
    visibility = ["//src/test/java/com/google/devtools/build/lib:__pkg__"],
)

java_library(
    name = "AuthAndTlsTests_lib",
    srcs = glob(
        [
            "*.java",
        ],
    ),
    deps = [
        "//src/main/java/com/google/devtools/build/lib/authandtls",
        "//src/main/java/com/google/devtools/build/lib/authandtls:grpc",
        "//src/main/java/com/google/devtools/build/lib/authandtls/credentialhelper",
        "//src/main/java/com/google/devtools/build/lib/events",
        "//src/main/java/com/google/devtools/build/lib/events:event_bus_event_handler",
        "//src/main/java/com/google/devtools/build/lib/runtime:command_line_path_factory",
        "//src/main/java/com/google/devtools/build/lib/util:string_encoding",
        "//src/main/java/com/google/devtools/build/lib/vfs",
        "//src/main/java/com/google/devtools/build/lib/vfs/inmemoryfs",
        "//src/main/java/com/google/devtools/common/options",
        "//src/test/java/com/google/devtools/build/lib/testutil",
        "//third_party/java/auth_kit:credentials",
        "//third_party/java/grpc:core",
        "//third_party/java/guava:base",
        "//third_party/java/guava:collect",
        "//third_party/java/junit",
        "//third_party/java/netty4:transport_native_epoll",
        "//third_party/java/netty4:transport_native_kqueue",
        "//third_party/java/testparameterinjector",
        "//third_party/java/truth",
    ],
)

java_test(
    name = "AuthAndTlsTests",
    test_class = "com.google.devtools.build.lib.AllTests",
    runtime_deps = [
        ":AuthAndTlsTests_lib",
        "//src/test/java/com/google/devtools/build/lib:test_runner",
    ],
)
