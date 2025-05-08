load("@rules_java//java:defs.bzl", "java_library", "java_test")

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

java_test(
    name = "DocumentationTests",
    size = "medium",
    srcs = ["DocumentationTests.java"],
    shard_count = 1,
    tags = ["docgen"],
    runtime_deps = [
        ":documentation-tests",
    ],
    deps = [
        "//src/test/java/com/google/devtools/build/lib/testutil:TestSuite",
        "//third_party:junit4",
    ],
)

java_library(
    name = "documentation-tests",
    testonly = 1,
    srcs = [
        "BuildDocCollectorTest.java",
        "DocCheckerUtilsTest.java",
        "DocLinkMapTest.java",
        "RuleDocumentationAttributeTest.java",
        "RuleDocumentationTest.java",
        "RuleLinkExpanderTest.java",
        "SourceUrlMapperTest.java",
        "StarlarkDocExpanderTest.java",
        "StarlarkDocumentationTest.java",
    ],
    deps = [
        ":testutil",
        "//src/main/java/com/google/devtools/build/docgen:docgen_javalib",
        "//src/main/java/com/google/devtools/build/docgen/annot",
        "//src/main/java/com/google/devtools/build/lib/analysis:analysis_cluster",
        "//src/main/java/com/google/devtools/build/lib/collect/nestedset",
        "//src/main/java/com/google/devtools/build/lib/packages",
        "//src/main/java/com/google/devtools/build/lib/rules/cpp",
        "//src/main/java/net/starlark/java/annot",
        "//src/main/java/net/starlark/java/eval",
        "//src/main/protobuf:stardoc_output_java_proto",
        "//third_party:guava",
        "//third_party:jsr305",
        "//third_party:junit4",
        "//third_party:truth",
    ],
)

java_library(
    name = "testutil",
    testonly = 1,
    srcs = glob(["testutil/*.java"]),
    deps = [
        "//src/main/java/com/google/devtools/build/lib/actions",
        "//src/main/java/com/google/devtools/build/lib/analysis:analysis_cluster",
        "//src/main/java/com/google/devtools/build/lib/analysis:configured_target",
        "//src/main/java/com/google/devtools/build/lib/analysis:rule_definition_environment",
        "//src/main/java/com/google/devtools/build/lib/packages",
    ],
)
