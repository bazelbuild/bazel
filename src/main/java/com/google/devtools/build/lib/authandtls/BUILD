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
    visibility = ["//src:__subpackages__"],
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
        "//third_party:auth",
        "//third_party:auto_value",
        "//third_party:guava",
        "//third_party:jsr305",
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
        "//third_party:auth",
        "//third_party:caffeine",
        "//third_party:guava",
        "//third_party:jsr305",
        "//third_party:netty",
        "//third_party/grpc-java:grpc-jar",
    ],
)
