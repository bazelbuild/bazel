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

# Library of collection utilities.

java_library(
    name = "nestedset",
    srcs = [
        "DeferredNestedSetCodec.java",
        "Depset.java",
        "NestedArrayCodec.java",
        "NestedSet.java",
        "NestedSetBuilder.java",
        "NestedSetCodecWithStore.java",
        "NestedSetInterner.java",
        "NestedSetSerializationCache.java",
        "NestedSetStore.java",
        "NestedSetVisitor.java",
        "Order.java",
    ],
    visibility = [
        "//src/main/java/com/google/devtools/build/lib:build_lib",
        "//src/main/java/com/google/devtools/build/skyframe:__subpackages__",
        "//src/test/java/com/google/devtools/build/skyframe:__subpackages__",
    ],
    deps = [
        ":nested_sets_should_be_interned_by_equality",
        "//src/main/java/com/google/devtools/build/docgen/annot",
        "//src/main/java/com/google/devtools/build/lib/bugreport",
        "//src/main/java/com/google/devtools/build/lib/collect:concurrent_identity_set",
        "//src/main/java/com/google/devtools/build/lib/collect/compacthashset",
        "//src/main/java/com/google/devtools/build/lib/concurrent",
        "//src/main/java/com/google/devtools/build/lib/concurrent:thread_safety",
        "//src/main/java/com/google/devtools/build/lib/packages/semantics",
        "//src/main/java/com/google/devtools/build/lib/runtime:memory_optimizations",
        "//src/main/java/com/google/devtools/build/lib/skyframe/serialization",
        "//src/main/java/com/google/devtools/build/lib/skyframe/serialization:constants",
        "//src/main/java/com/google/devtools/build/lib/skyframe/serialization:fingerprint_value_store",
        "//src/main/java/com/google/devtools/build/lib/skyframe/serialization:visible-for-serialization",
        "//src/main/java/com/google/devtools/build/lib/skyframe/serialization:write_status",
        "//src/main/java/com/google/devtools/build/lib/skyframe/serialization:write_statuses",
        "//src/main/java/com/google/devtools/build/lib/skyframe/serialization/autocodec",
        "//src/main/java/com/google/devtools/build/lib/skyframe/serialization/autocodec:serialization-constant",
        "//src/main/java/com/google/devtools/build/lib/util:detailed_exit_code",
        "//src/main/java/net/starlark/java/annot",
        "//src/main/java/net/starlark/java/eval",
        "//src/main/protobuf:failure_details_java_proto",
        "//third_party/java/caffeine",
        "//third_party/java/flogger",
        "//third_party/java/guava:annotations",
        "//third_party/java/guava:base",
        "//third_party/java/guava:collect",
        "//third_party/java/guava:concurrent",
        "//third_party/java/guava:hash",
        "//third_party/java/jsr305_annotations",
        "//third_party/java/protobuf:protobuf_lite",
    ],
)

java_library(
    name = "nested_sets_should_be_interned_by_equality",
    srcs = ["NestedSetsShouldBeInternedByEquality.java"],
    visibility = ["//src/main/java/com/google/devtools/build/lib:build_lib"],
)

java_library(
    name = "artifact_nested_set_key",
    srcs = [
        "ArtifactNestedSetKey.java",
    ],
    deps = [
        ":nestedset",
        "//src/main/java/com/google/devtools/build/lib/actions:action_input",
        "//src/main/java/com/google/devtools/build/lib/actions:artifacts",
        "//src/main/java/com/google/devtools/build/lib/skyframe:sky_functions",
        "//src/main/java/com/google/devtools/build/skyframe",
        "//src/main/java/com/google/devtools/build/skyframe:skyframe-objects",
        "//third_party/java/guava:base",
        "//third_party/java/guava:collect",
        "//third_party/java/guava:graph",
    ],
)

java_library(
    name = "testutils",
    testonly = 1,
    srcs = ["NestedSetCodecTestUtils.java"],
    deps = [
        ":nestedset",
        "//src/main/java/com/google/devtools/build/lib/skyframe/serialization",
        "//src/main/java/com/google/devtools/build/lib/skyframe/serialization/testutils",
        "//third_party/java/guava:base",
        "//third_party/java/guava:concurrent",
        "//third_party/java/truth",
    ],
)

java_library(
    name = "fingerprint_cache",
    srcs = [
        "DigestDeduper.java",
        "DigestMap.java",
        "NestedSetFingerprintCache.java",
    ],
    deps = [
        ":nestedset",
        "//src/main/java/com/google/devtools/build/lib/actions:commandline_item",
        "//src/main/java/com/google/devtools/build/lib/util",
        "//src/main/java/com/google/devtools/build/lib/vfs",
        "//third_party/java/guava:annotations",
        "//third_party/java/guava:base",
        "//third_party/java/guava:collect",
        "//third_party/java/jsr305_annotations",
    ],
)
