load("@rules_java//java:defs.bzl", "java_library")

package(
    default_applicable_licenses = ["//:license"],
    default_visibility = ["//src:__subpackages__"],
)

filegroup(
    name = "srcs",
    srcs = glob(["**"]),
    visibility = ["//src:__subpackages__"],
)

# Library of collection utilities.
java_library(
    name = "compacthashset",
    srcs = ["CompactHashSet.java"],
    deps = [
        "//third_party/java/guava:base",
        "//third_party/java/guava:primitives",
        "//third_party/java/jsr305_annotations",
    ],
)

java_library(
    name = "compacthashset_codec",
    srcs = ["CompactHashSetCodec.java"],
    deps = [
        ":compacthashset",
        "//src/main/java/com/google/devtools/build/lib/skyframe/serialization",
        "//third_party/java/protobuf:protobuf_lite",
    ],
)
