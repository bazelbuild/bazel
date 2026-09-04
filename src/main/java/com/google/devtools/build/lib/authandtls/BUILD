load("@rules_java//java:defs.bzl", "java_library")

package(
    default_applicable_licenses = ["//:license"],
    default_visibility = ["//src:__subpackages__"],
)

filegroup(
    name = "srcs",
    srcs = glob(["**"]) + [
        "//src/main/java/com/google/devtools/build/lib/authandtls/credentialhelper:srcs",
    ],
    visibility = ["//src/main/java/com/google/devtools/build/lib:__pkg__"],
)

java_library(
    name = "authandtls",
    srcs = glob(
        ["*.java"],
        exclude = [
            "CallCredentialsProvider.java",
            "GoogleAuthCallCredentialsProvider.java",
            "GoogleAuthUtils.java",
        ],
    ),
    deps = [
        "//src/main/java/com/google/devtools/build/lib/unsafe:string",
        "//src/main/java/com/google/devtools/common/options",
        "//third_party/java/auth_kit",
        "//third_party/java/auto:auto_value",
        "//third_party/java/guava:base",
        "//third_party/java/guava:collect",
        "//third_party/java/jsr305_annotations",
    ],
)

java_library(
    name = "grpc",
    srcs = [
        "CallCredentialsProvider.java",
        "GoogleAuthCallCredentialsProvider.java",
        "GoogleAuthUtils.java",
    ],
    deps = [
        ":authandtls",
        "//src/main/java/com/google/devtools/build/lib/authandtls/credentialhelper",
        "//src/main/java/com/google/devtools/build/lib/concurrent:thread_safety",
        "//src/main/java/com/google/devtools/build/lib/events",
        "//src/main/java/com/google/devtools/build/lib/runtime:command_line_path_factory",
        "//src/main/java/com/google/devtools/build/lib/vfs",
        "//third_party/java/auth_kit",
        "//third_party/java/caffeine",
        "//third_party/java/grpc:auth",
        "//third_party/java/grpc:core",
        "//third_party/java/grpc:netty",
        "//third_party/java/guava:annotations",
        "//third_party/java/guava:base",
        "//third_party/java/jsr305_annotations",
        "//third_party/java/netty4:handler",
        "//third_party/java/netty4:transport",
        "//third_party/java/netty4:transport_native_epoll",
        "//third_party/java/netty4:transport_native_kqueue",
        "//third_party/java/netty4:transport_native_unix_common",
    ],
)
