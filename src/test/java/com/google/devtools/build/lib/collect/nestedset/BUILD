load("@rules_java//java:defs.bzl", "java_library", "java_test")

package(
    default_applicable_licenses = ["//:license"],
    default_testonly = 1,
    default_visibility = ["//src:__subpackages__"],
)

filegroup(
    name = "srcs",
    testonly = 0,
    srcs = glob(["**"]),
    visibility = ["//src:__subpackages__"],
)

java_test(
    name = "CompileOrderExpanderTest",
    srcs = ["CompileOrderExpanderTest.java"],
    deps = [
        ":ExpanderTestBase",
        "//src/main/java/com/google/devtools/build/lib/collect/nestedset",
        "//third_party:guava",
        "//third_party:junit4",
    ],
)

java_test(
    name = "DeferredNestedSetCodecTest",
    srcs = ["DeferredNestedSetCodecTest.java"],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/collect/nestedset",
        "//src/main/java/com/google/devtools/build/lib/skyframe/serialization",
        "//src/main/java/com/google/devtools/build/lib/skyframe/serialization/testutils",
        "//src/main/java/com/google/devtools/build/lib/skyframe/serialization/testutils:dumper",
        "//third_party:guava",
        "//third_party:junit4",
        "//third_party:truth",
        "@com_google_protobuf//:protobuf_java",
    ],
)

java_test(
    name = "DepsetTest",
    srcs = ["DepsetTest.java"],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/collect/nestedset",
        "//src/main/java/net/starlark/java/eval",
        "//src/test/java/com/google/devtools/build/lib/starlark/util",
        "//third_party:guava",
        "//third_party:junit4",
        "//third_party:truth",
    ],
)

# Keep this as a java_library so that other tests with different hash functions can depend on it.
java_library(
    name = "DigestMapTest_lib",
    srcs = ["DigestMapTest.java"],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/collect/nestedset:fingerprint_cache",
        "//src/main/java/com/google/devtools/build/lib/util",
        "//src/main/java/com/google/devtools/build/lib/vfs",
        "//third_party:guava",
        "//third_party:junit4",
        "//third_party:truth",
    ],
)

java_test(
    name = "DigestMapTest",
    test_class = "com.google.devtools.build.lib.collect.nestedset.DigestMapTest",
    runtime_deps = [
        ":DigestMapTest_lib",
    ],
)

java_library(
    name = "ExpanderTestBase",
    srcs = ["ExpanderTestBase.java"],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/collect/nestedset",
        "//third_party:guava",
        "//third_party:junit4",
        "//third_party:truth",
    ],
)

java_test(
    name = "LinkOrderExpanderTest",
    srcs = ["LinkOrderExpanderTest.java"],
    deps = [
        ":ExpanderTestBase",
        "//src/main/java/com/google/devtools/build/lib/collect/nestedset",
        "//third_party:guava",
        "//third_party:junit4",
    ],
)

java_test(
    name = "NaiveLinkOrderExpanderTest",
    srcs = ["NaiveLinkOrderExpanderTest.java"],
    deps = [
        ":ExpanderTestBase",
        "//src/main/java/com/google/devtools/build/lib/collect/nestedset",
        "//third_party:guava",
        "//third_party:junit4",
    ],
)

java_test(
    name = "NestedSetCodecTest",
    srcs = ["NestedSetCodecTest.java"],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/bugreport",
        "//src/main/java/com/google/devtools/build/lib/collect/nestedset",
        "//src/main/java/com/google/devtools/build/lib/collect/nestedset:testutils",
        "//src/main/java/com/google/devtools/build/lib/skyframe/serialization",
        "//src/main/java/com/google/devtools/build/lib/util/io:out-err",
        "//third_party:guava",
        "//third_party:guava-testlib",
        "//third_party:junit4",
        "//third_party:mockito",
        "//third_party:truth",
        "@com_google_protobuf//:protobuf_java",
    ],
)

java_test(
    name = "NestedSetFingerprintCacheTest",
    srcs = ["NestedSetFingerprintCacheTest.java"],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/actions:commandline_item",
        "//src/main/java/com/google/devtools/build/lib/collect/nestedset",
        "//src/main/java/com/google/devtools/build/lib/collect/nestedset:fingerprint_cache",
        "//src/main/java/com/google/devtools/build/lib/util",
        "//third_party:guava",
        "//third_party:junit4",
        "//third_party:truth",
    ],
)

java_test(
    name = "NestedSetSerializationCacheTest",
    srcs = ["NestedSetSerializationCacheTest.java"],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/bugreport",
        "//src/main/java/com/google/devtools/build/lib/collect/nestedset",
        "//src/main/java/com/google/devtools/build/lib/skyframe/serialization",
        "//third_party:guava",
        "//third_party:guava-testlib",
        "//third_party:junit4",
        "//third_party:mockito",
        "//third_party:truth",
    ],
)

java_test(
    name = "NestedSetTest",
    srcs = ["NestedSetTest.java"],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/collect/nestedset",
        "//src/main/java/com/google/devtools/build/lib/skyframe/serialization",
        "//src/test/java/com/google/devtools/build/lib/testutil:TestThread",
        "//src/test/java/com/google/devtools/build/lib/testutil:TestUtils",
        "//third_party:guava",
        "//third_party:guava-testlib",
        "//third_party:junit4",
        "//third_party:truth",
    ],
)

java_test(
    name = "NestedSetTopologyTest",
    srcs = ["NestedSetTopologyTest.java"],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/collect/nestedset",
        "//third_party:guava",
        "//third_party:junit4",
        "//third_party:truth",
    ],
)

java_test(
    name = "NestedSetVisitorTest",
    srcs = ["NestedSetVisitorTest.java"],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/collect/nestedset",
        "//third_party:junit4",
        "//third_party:truth",
    ],
)

java_test(
    name = "OrderTest",
    srcs = ["OrderTest.java"],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/collect/nestedset",
        "//third_party:junit4",
        "//third_party:truth",
    ],
)

test_suite(
    name = "NestedSetTests",
)
