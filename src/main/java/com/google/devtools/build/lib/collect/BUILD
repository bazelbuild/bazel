load("@rules_java//java:defs.bzl", "java_library")

package(
    default_applicable_licenses = ["//:license"],
    default_visibility = ["//src:__subpackages__"],
)

filegroup(
    name = "srcs",
    srcs = glob(["**"]) + [
        "//src/main/java/com/google/devtools/build/lib/collect/compacthashmap:srcs",
        "//src/main/java/com/google/devtools/build/lib/collect/compacthashset:srcs",
    ],
    visibility = ["//src:__subpackages__"],
)

java_library(
    name = "collection_utils",
    srcs = ["CollectionUtils.java"],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/collect/compacthashset",
        "//third_party:guava",
        "//third_party:jsr305",
    ],
)

java_library(
    name = "compact_immutable_map",
    srcs = ["CompactImmutableMap.java"],
)

java_library(
    name = "concurrent_identity_set",
    srcs = ["ConcurrentIdentitySet.java"],
    deps = ["//third_party:guava"],
)

java_library(
    name = "equivalence_relation",
    srcs = ["EquivalenceRelation.java"],
)

java_library(
    name = "extrema",
    srcs = ["Extrema.java"],
    deps = ["//third_party:guava"],
)

java_library(
    name = "immutable_shared_key_map",
    srcs = ["ImmutableSharedKeyMap.java"],
    deps = [
        ":compact_immutable_map",
        "//src/main/java/com/google/devtools/build/lib/concurrent",
        "//src/main/java/com/google/devtools/build/lib/skyframe/serialization:visible-for-serialization",
        "//third_party:error_prone_annotations",
        "//third_party:guava",
        "//third_party:jsr305",
    ],
)

java_library(
    name = "immutable_sorted_key_list_multimap",
    srcs = ["ImmutableSortedKeyListMultimap.java"],
    deps = [
        "//third_party:error_prone_annotations",
        "//third_party:guava",
        "//third_party:jsr305",
    ],
)

java_library(
    name = "iterable_codecs",
    srcs = ["IterableCodecs.java"],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/skyframe/serialization",
        "//third_party:guava",
        "@com_google_protobuf//:protobuf_java",
    ],
)

java_library(
    name = "path_fragment_prefix_trie",
    srcs = ["PathFragmentPrefixTrie.java"],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/concurrent:thread_safety",
        "//src/main/java/com/google/devtools/build/lib/vfs:pathfragment",
        "//third_party:guava",
    ],
)

java_library(
    name = "simple_target_pattern_matcher",
    srcs = ["SimpleTargetPatternMatcher.java"],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/cmdline",
        "//src/main/java/com/google/devtools/build/lib/vfs:pathfragment",
        "//third_party:guava",
    ],
)
