load("@rules_java//java:defs.bzl", "java_test")

package(
    default_applicable_licenses = ["//:license"],
    default_testonly = 1,
    default_visibility = ["//src:__subpackages__"],
)

filegroup(
    name = "srcs",
    testonly = 0,
    srcs = glob(["**"]) + [
        "//src/test/java/com/google/devtools/build/lib/collect/compacthashset:srcs",
        "//src/test/java/com/google/devtools/build/lib/collect/nestedset:srcs",
    ],
    visibility = ["//src:__subpackages__"],
)

java_test(
    name = "CollectionUtilsTest",
    srcs = ["CollectionUtilsTest.java"],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/collect:collection_utils",
        "//third_party/java/guava:collect",
        "//third_party/java/junit",
        "//third_party/java/truth",
    ],
)

java_test(
    name = "ConcurrentIdentitySetTest",
    srcs = ["ConcurrentIdentitySetTest.java"],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/collect:concurrent_identity_set",
        "//src/test/java/com/google/devtools/build/lib/testutil:TestUtils",
        "//third_party/java/guava:collect",
        "//third_party/java/junit",
        "//third_party/java/truth",
    ],
)

java_test(
    name = "ExtremaTest",
    srcs = ["ExtremaTest.java"],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/collect:extrema",
        "//third_party/java/guava:collect",
        "//third_party/java/junit",
        "//third_party/java/truth",
    ],
)

java_test(
    name = "ImmutableSharedKeyMapTest",
    srcs = ["ImmutableSharedKeyMapTest.java"],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/collect:immutable_shared_key_map",
        "//third_party/java/guava:collect",
        "//third_party/java/guava:testing",
        "//third_party/java/junit",
        "//third_party/java/testparameterinjector",
        "//third_party/java/truth",
    ],
)

java_test(
    name = "ImmutableSortedKeyListMultimapTest",
    srcs = ["ImmutableSortedKeyListMultimapTest.java"],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/collect:immutable_sorted_key_list_multimap",
        "//third_party/java/guava:collect",
        "//third_party/java/guava:collect-testing",
        "//third_party/java/guava:testing",
        "//third_party/java/junit",
        "//third_party/java/truth",
    ],
)

java_test(
    name = "PathFragmentPrefixTrieTest",
    srcs = ["PathFragmentPrefixTrieTest.java"],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/collect:path_fragment_prefix_trie",
        "//src/main/java/com/google/devtools/build/lib/vfs:pathfragment",
        "//third_party/java/junit",
        "//third_party/java/truth",
    ],
)

java_test(
    name = "SimpleTargetPatternMatcherTest",
    srcs = ["SimpleTargetPatternMatcherTest.java"],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/cmdline",
        "//src/main/java/com/google/devtools/build/lib/collect:simple_target_pattern_matcher",
        "//third_party/java/guava:base",
        "//third_party/java/guava:collect",
        "//third_party/java/junit",
        "//third_party/java/testparameterinjector",
        "//third_party/java/truth",
    ],
)

test_suite(name = "all_tests")

test_suite(
    name = "CollectTests",
    tests = [
        ":all_tests",
        "//src/test/java/com/google/devtools/build/lib/collect/nestedset:NestedSetTests",
    ],
)
