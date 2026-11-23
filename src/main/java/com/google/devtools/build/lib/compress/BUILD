load("@rules_java//java:defs.bzl", "java_library")

package(
    default_applicable_licenses = ["//:license"],
    default_visibility = ["//src/main/java/com/google/devtools/build/lib:build_lib"],
)

filegroup(
    name = "srcs",
    srcs = glob(["**"]),
    visibility = ["//src/main/java/com/google/devtools/build/lib:__pkg__"],
)

java_library(
    name = "compression_service",
    srcs = ["CompressionService.java"],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/runtime:blaze_service",
    ],
)

java_library(
    name = "compression_service_impl",
    srcs = ["CompressionServiceImpl.java"],
    deps = [
        ":compression_service",
        "//third_party/java/zstd_jni",
    ],
)
