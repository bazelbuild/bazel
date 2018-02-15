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
        "NestedSetView.java",
        "NestedSetVisitor.java",
        "Order.java",
    ],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/collect/compacthashset",
        "//src/main/java/com/google/devtools/build/lib/concurrent",
        "//third_party:guava",
        "//third_party:jsr305",
    ],
)

java_library(
    name = "serialization",
    srcs = [
        "NestedSetCodec.java",
        "NestedSetSerializer.java",
    ],
    deps = [
        ":nestedset",
        "//src/main/java/com/google/devtools/build/lib/skyframe/serialization",
        "//src/main/java/com/google/devtools/build/lib/skyframe/serialization:kryo",
        "//third_party:guava",
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
