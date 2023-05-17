load("@rules_java//java:defs.bzl", "java_library")

package(
    default_applicable_licenses = ["//:license"],
    default_visibility = ["//src:__subpackages__"],
)

filegroup(
    name = "srcs",
    testonly = 0,
    srcs = glob(["**"]),
    visibility = ["//src:__subpackages__"],
)

java_library(
    name = "blake3_jni",
    srcs = ["Blake3JNI.java"],
    deps = ["//src/main/java/com/google/devtools/build/lib/jni"],
)

java_library(
    name = "blake3_hasher",
    srcs = ["Blake3Hasher.java"],
    deps = [
        ":blake3_jni",
    ],
)
