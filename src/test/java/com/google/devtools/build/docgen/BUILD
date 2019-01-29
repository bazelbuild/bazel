package(
    default_testonly = 1,
    default_visibility = ["//src:__subpackages__"],
)

filegroup(
    name = "srcs",
    testonly = 0,
    srcs = glob(["**"]),
    visibility = ["//src:__pkg__"],
)

java_test(
    name = "DocumentationFailuresTests",
    size = "medium",
    shard_count = 1,
    tags = ["docgen"],
    test_class = "com.google.devtools.build.lib.AllTests",
    visibility = ["//devtools/blaze/main:__pkg__"],
    runtime_deps = [
        ":documentation-failures-tests",
        "//src/test/java/com/google/devtools/build/lib:test_runner",
        "//src/test/java/com/google/devtools/build/lib:testutil",
        "//third_party:junit4",
    ],
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
    name = "documentation-failures-tests",
    testonly = 1,
    srcs = [
        "SkylarkDocumentationFailuresTest.java",
    ],
    deps = [
        "//src/main/java/com/google/devtools/build/docgen:docgen_javalib",
        "//src/main/java/com/google/devtools/build/lib:build-base",
        "//src/main/java/com/google/devtools/build/lib:skylark_semantics",
        "//src/main/java/com/google/devtools/build/lib:skylarkinterface",
        "//src/main/java/com/google/devtools/build/lib:syntax",
        "//src/test/java/com/google/devtools/build/lib:syntax_testutil",
        "//src/test/java/com/google/devtools/build/lib:testutil",
        "//src/test/java/com/google/devtools/build/lib/skylark:testutil",
        "//third_party:guava",
        "//third_party:junit4",
        "//third_party:truth",
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
        "//src/main/java/com/google/devtools/build/lib:skylark_semantics",
        "//src/main/java/com/google/devtools/build/lib:skylarkinterface",
        "//src/main/java/com/google/devtools/build/lib:syntax",
        "//src/main/java/com/google/devtools/build/lib/collect/nestedset",
        "//src/main/java/com/google/devtools/build/lib/rules/cpp",
        "//src/test/java/com/google/devtools/build/lib:syntax_testutil",
        "//src/test/java/com/google/devtools/build/lib:testutil",
        "//src/test/java/com/google/devtools/build/lib/skylark:testutil",
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
        "//src/main/java/com/google/devtools/build/lib/actions",
    ],
)

test_suite(
    name = "all_tests",
    tags = ["docgen"],
)

test_suite(
    name = "windows_tests",
    tags = [
        "-no_windows",
        "-slow",
    ],
    visibility = ["//visibility:private"],
)

test_suite(
    name = "all_windows_tests",
    tests = [
        ":windows_tests",
    ],
    visibility = ["//src:__pkg__"],
)
