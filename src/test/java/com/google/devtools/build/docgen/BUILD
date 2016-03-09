package(
    default_visibility = ["//visibility:public"],
)

licenses(["notice"])  # Apache 2.0

test_suite(
    name = "all_tests",
    tags = ["docgen"],
)

java_test(
    name = "DocumentationTests",
    size = "medium",
    srcs = ["DocumentationTests.java"],
    shard_count = 1,
    tags = ["docgen"],
    deps = [
        ":documentation-tests",
        "//src/test/java/com/google/devtools/build/lib:testutil",
        "//third_party:junit4",
    ],
)

java_library(
    name = "documentation-tests",
    testonly = 1,
    srcs = [
        "DocCheckerUtilsTest.java",
        "RuleDocumentationAttributeTest.java",
        "RuleDocumentationTest.java",
        "RuleLinkExpanderTest.java",
        "SkylarkDocumentationTest.java",
    ],
    deps = [
        ":testutil",
        "//src/main/java/com/google/devtools/build/docgen:docgen_javalib",
        "//src/main/java/com/google/devtools/build/lib:build-base",
        "//src/main/java/com/google/devtools/build/lib:packages-internal",
        "//src/main/java/com/google/devtools/build/lib:skylarkinterface",
        "//src/main/java/com/google/devtools/build/lib:syntax",
        "//src/main/java/com/google/devtools/build/lib/rules/cpp",
        "//src/test/java/com/google/devtools/build/lib:skylark_testutil",
        "//src/test/java/com/google/devtools/build/lib:syntax_testutil",
        "//src/test/java/com/google/devtools/build/lib:testutil",
        "//third_party:guava",
        "//third_party:junit4",
        "//third_party:truth",
    ],
)

java_library(
    name = "testutil",
    testonly = 1,
    srcs = glob(["testutil/*.java"]),
    deps = [
        "//src/main/java/com/google/devtools/build/lib:build-base",
        "//src/main/java/com/google/devtools/build/lib:packages-internal",
    ],
)
