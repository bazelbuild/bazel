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
    name = "disk",
    srcs = glob(["*.java"]),
    deps = [
        "//src/main/java/com/google/devtools/build/lib/concurrent",
        "//src/main/java/com/google/devtools/build/lib/exec:spawn_runner",
        "//src/main/java/com/google/devtools/build/lib/jni",
        "//src/main/java/com/google/devtools/build/lib/profiler",
        "//src/main/java/com/google/devtools/build/lib/remote:store",
        "//src/main/java/com/google/devtools/build/lib/remote/common",
        "//src/main/java/com/google/devtools/build/lib/remote/common:cache_not_found_exception",
        "//src/main/java/com/google/devtools/build/lib/remote/util",
        "//src/main/java/com/google/devtools/build/lib/vfs",
        "//third_party:guava",
        "//third_party:jsr305",
        "//third_party/protobuf:protobuf_java",
        "@remoteapis//:build_bazel_remote_execution_v2_remote_execution_java_proto",
    ],
)
