load("@rules_java//java:defs.bzl", "java_library")

package(
    default_applicable_licenses = ["//:license"],
    default_visibility = ["//src:__subpackages__"],
)

filegroup(
    name = "srcs",
    srcs = glob(["*"]),
    visibility = ["//src/main/java/com/google/devtools/build/lib/remote:__pkg__"],
)

java_library(
    name = "zstd",
    srcs = glob(["*.java"]),
    deps = [
        "//third_party/java/guava:base",
        "//third_party/java/zstd_jni",
    ],
)
