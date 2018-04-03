package(default_visibility = ["//src:__subpackages__"])

filegroup(
    name = "srcs",
    srcs = glob(["**"]),
    visibility = ["//src/main/java/com/google/devtools/build/lib:__pkg__"],
)

# Library of collection utilities.
java_library(
    name = "nestedset",
    srcs = [
        "NestedSet.java",
        "NestedSetBuilder.java",
        "NestedSetCodec.java",
        "NestedSetView.java",
        "NestedSetVisitor.java",
        "Order.java",
    ],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/collect/compacthashset",
        "//src/main/java/com/google/devtools/build/lib/skyframe/serialization",
        "//third_party:guava",
        "//third_party:jsr305",
        "//third_party/protobuf:protobuf_java",
    ],
)

java_library(
    name = "fingerprint_cache",
    srcs = [
        "DigestMap.java",
        "NestedSetFingerprintCache.java",
    ],
    deps = [
        ":nestedset",
        "//src/main/java/com/google/devtools/build/lib:util",
        "//src/main/java/com/google/devtools/build/lib/actions:commandline_item",
        "//third_party:guava",
    ],
)
