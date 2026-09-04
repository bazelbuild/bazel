load("@rules_java//java:defs.bzl", "java_library")

package(
    default_applicable_licenses = ["//:license"],
    default_visibility = ["//src:__subpackages__"],
)

filegroup(
    name = "srcs",
    srcs = glob(["*"]),
    visibility = ["//src/main/java/com/google/devtools/build/lib/bazel/repository:__pkg__"],
)

java_library(
    name = "decompressor",
    srcs = glob(["*.java"]),
    runtime_deps = [
        "//third_party/brotli/java/org/brotli/dec",
    ],
    deps = [
        "//src/java_tools/singlejar/java/com/google/devtools/build/zip",
        "//src/main/java/com/google/devtools/build/lib/bazel/repository:exception",
        "//src/main/java/com/google/devtools/build/lib/concurrent:thread_safety",
        "//src/main/java/com/google/devtools/build/lib/util:pair",
        "//src/main/java/com/google/devtools/build/lib/util:string_encoding",
        "//src/main/java/com/google/devtools/build/lib/vfs",
        "//src/main/java/com/google/devtools/build/lib/vfs:pathfragment",
        "//src/main/java/com/google/devtools/build/skyframe:skyframe-objects",
        "//src/main/java/net/starlark/java/eval",
        "//third_party/java/apache_commons_compress:ar",
        "//third_party/java/apache_commons_compress:brotli",
        "//third_party/java/apache_commons_compress:bzip2",
        "//third_party/java/apache_commons_compress:gzip",
        "//third_party/java/apache_commons_compress:sevenz",
        "//third_party/java/apache_commons_compress:tar",
        "//third_party/java/apache_commons_compress:xz",
        "//third_party/java/auto:auto_service",
        "//third_party/java/auto:auto_value",
        "//third_party/java/guava:annotations",
        "//third_party/java/guava:base",
        "//third_party/java/guava:collect",
        "//third_party/java/guava:io",
        "//third_party/java/java_diff_utils",
        "//third_party/java/jsr305_annotations",
        "//third_party/java/xz",
        "//third_party/java/zstd_jni",
    ],
)
