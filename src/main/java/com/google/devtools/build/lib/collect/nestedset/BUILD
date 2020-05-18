load("@rules_java//java:defs.bzl", "java_library")

package(default_visibility = ["//src:__subpackages__"])

filegroup(
    name = "srcs",
    srcs = glob(["*"]),
    visibility = ["//src:__subpackages__"],
)

# Library of collection utilities.

java_library(
    name = "nestedset",
    srcs = [
        "Depset.java",
        "NestedSet.java",
        "NestedSetBuilder.java",
        "NestedSetCodecWithStore.java",
        "NestedSetExpander.java",
        "NestedSetStore.java",
        "NestedSetView.java",
        "NestedSetVisitor.java",
        "Order.java",
    ],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/bugreport",
        "//src/main/java/com/google/devtools/build/lib/collect/compacthashset",
        "//src/main/java/com/google/devtools/build/lib/concurrent",
        "//src/main/java/com/google/devtools/build/lib/skyframe/serialization",
        "//src/main/java/com/google/devtools/build/lib/skyframe/serialization:constants",
        "//src/main/java/com/google/devtools/build/lib/skyframe/serialization/autocodec",
        "//src/main/java/com/google/devtools/build/lib/syntax:evaluator",
        "//src/main/java/com/google/devtools/build/lib/util:exit_code",
        "//src/main/java/net/starlark/java/annot",
        "//third_party:auto_value",
        "//third_party:error_prone_annotations",
        "//third_party:flogger",
        "//third_party:guava",
        "//third_party:jsr305",
        "//third_party/protobuf:protobuf_java",
    ],
)

java_library(
    name = "options",
    srcs = ["NestedSetOptionsModule.java"],
    deps = [
        ":nestedset",
        "//src/main/java/com/google/devtools/build/lib:runtime",
        "//src/main/java/com/google/devtools/common/options",
        "//third_party:guava",
    ],
)

java_library(
    name = "testutils",
    testonly = 1,
    srcs = ["NestedSetCodecTestUtils.java"],
    deps = [
        ":nestedset",
        "//src/main/java/com/google/devtools/build/lib/skyframe/serialization",
        "//src/main/java/com/google/devtools/build/lib/skyframe/serialization/autocodec",
        "//src/main/java/com/google/devtools/build/lib/skyframe/serialization/testutils",
        "//third_party:guava",
        "//third_party:truth",
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
        "//src/main/java/com/google/devtools/build/lib/actions:commandline_item",
        "//src/main/java/com/google/devtools/build/lib/util",
        "//src/main/java/com/google/devtools/build/lib/vfs",
        "//third_party:guava",
    ],
)
