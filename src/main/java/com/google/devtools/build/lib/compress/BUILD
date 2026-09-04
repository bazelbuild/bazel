load("@rules_java//java:defs.bzl", "java_library")

package(
    default_applicable_licenses = ["//:license"],
    default_visibility = ["//src:__subpackages__"],
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
        "//src/main/java/com/google/devtools/build/lib/skybridge:skybridge_interface",
    ],
)

java_library(
    name = "compression_service_impl",
    srcs = ["CompressionServiceImpl.java"],
    deps = [
        ":compression_service",
        "//src/main/java/com/google/devtools/build/lib/skybridge:sc_only",
        "//third_party/java/zstd_jni",
    ],
)
