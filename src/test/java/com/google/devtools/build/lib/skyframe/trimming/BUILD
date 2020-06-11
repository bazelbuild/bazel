# Tests for the trimming support classes.

load("@rules_java//java:defs.bzl", "java_library", "java_test")

package(
    default_testonly = 1,
)

filegroup(
    name = "srcs",
    testonly = 0,
    srcs = glob(["**"]),
    visibility = ["//src:__subpackages__"],
)

java_library(
    name = "trimmable_test_fragments",
    srcs = ["TrimmableTestConfigurationFragments.java"],
    visibility = ["//src/test:__subpackages__"],
    deps = [
        "//src/main/java/com/google/devtools/build/lib:syntax",
        "//src/main/java/com/google/devtools/build/lib/actions",
        "//src/main/java/com/google/devtools/build/lib/analysis:analysis_cluster",
        "//src/main/java/com/google/devtools/build/lib/analysis:config/build_options",
        "//src/main/java/com/google/devtools/build/lib/analysis:config/core_options",
        "//src/main/java/com/google/devtools/build/lib/analysis:config/fragment",
        "//src/main/java/com/google/devtools/build/lib/analysis:config/fragment_options",
        "//src/main/java/com/google/devtools/build/lib/analysis:config/fragment_provider",
        "//src/main/java/com/google/devtools/build/lib/analysis:config/transitions/patch_transition",
        "//src/main/java/com/google/devtools/build/lib/analysis:config/transitions/transition_factory",
        "//src/main/java/com/google/devtools/build/lib/analysis:configured_target",
        "//src/main/java/com/google/devtools/build/lib/analysis:file_provider",
        "//src/main/java/com/google/devtools/build/lib/analysis:platform_options",
        "//src/main/java/com/google/devtools/build/lib/analysis:resolved_toolchain_context",
        "//src/main/java/com/google/devtools/build/lib/analysis:test/test_configuration",
        "//src/main/java/com/google/devtools/build/lib/analysis:transition_mode",
        "//src/main/java/com/google/devtools/build/lib/analysis/platform",
        "//src/main/java/com/google/devtools/build/lib/cmdline",
        "//src/main/java/com/google/devtools/build/lib/collect/nestedset",
        "//src/main/java/com/google/devtools/build/lib/events",
        "//src/main/java/com/google/devtools/build/lib/packages",
        "//src/main/java/com/google/devtools/build/lib/packages:build_type",
        "//src/main/java/com/google/devtools/build/lib/packages:type",
        "//src/main/java/com/google/devtools/build/lib/rules:core_rules",
        "//src/main/java/com/google/devtools/build/lib/rules:repository/bind_rule",
        "//src/main/java/com/google/devtools/build/lib/rules:repository/workspace_base_rule",
        "//src/main/java/com/google/devtools/build/lib/rules:toolchain_type",
        "//src/main/java/com/google/devtools/build/lib/util:filetype",
        "//src/main/java/com/google/devtools/common/options",
        "//src/main/java/net/starlark/java/annot",
        "//src/test/java/com/google/devtools/build/lib/analysis/util",
        "//src/test/java/com/google/devtools/build/lib/testutil",
        "//third_party:guava",
        "//third_party:jsr305",
    ],
)

java_library(
    name = "test_key",
    srcs = ["TestKey.java"],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/skyframe/trimming:trimmed_configuration_cache",
        "//third_party:auto_value",
        "//third_party:guava",
    ],
)

java_test(
    name = "TrimmedConfigurationCacheTests",
    size = "small",
    srcs = [
        "TestKeyTest.java",
        "TrimmedConfigurationCacheTest.java",
    ],
    test_class = "com.google.devtools.build.lib.AllTests",
    runtime_deps = ["//src/test/java/com/google/devtools/build/lib:test_runner"],
    deps = [
        ":test_key",
        "//src/main/java/com/google/devtools/build/lib/skyframe/trimming:trimmed_configuration_cache",
        "//third_party:guava",
        "//third_party:guava-testlib",
        "//third_party:junit4",
        "//third_party:truth",
    ],
)
