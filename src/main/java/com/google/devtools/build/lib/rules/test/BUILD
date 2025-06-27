load("@rules_java//java:java_library.bzl", "java_library")

package(
    default_applicable_licenses = ["//:license"],
    default_visibility = ["//src:__subpackages__"],
)

filegroup(
    name = "srcs",
    srcs = glob(["*"]),
    visibility = ["//src:__subpackages__"],
)

java_library(
    name = "starlark_testing_module",
    srcs = ["StarlarkTestingModule.java"],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/analysis:analysis_cluster",
        "//src/main/java/com/google/devtools/build/lib/analysis:rule_definition_environment",
        "//src/main/java/com/google/devtools/build/lib/analysis:run_environment_info",
        "//src/main/java/com/google/devtools/build/lib/analysis:test/execution_info",
        "//src/main/java/com/google/devtools/build/lib/cmdline",
        "//src/main/java/com/google/devtools/build/lib/events",
        "//src/main/java/com/google/devtools/build/lib/packages",
        "//src/main/java/com/google/devtools/build/lib/starlarkbuildapi/test",
        "//src/main/java/com/google/devtools/build/lib/util",
        "//src/main/java/net/starlark/java/eval",
        "//src/main/java/net/starlark/java/syntax",
    ],
)

java_library(
    name = "testing_support_rules",
    srcs = ["TestingSupportRules.java"],
    deps = [
        ":starlark_testing_module",
        "//src/main/java/com/google/devtools/build/lib/analysis:analysis_cluster",
        "//src/main/java/com/google/devtools/build/lib/analysis:test/analysis_failure_info",
        "//src/main/java/com/google/devtools/build/lib/analysis:test/analysis_test_result_info",
        "//src/main/java/com/google/devtools/build/lib/analysis:test/instrumented_files_info",
        "//src/main/java/com/google/devtools/build/lib/rules/core",
        "//src/main/java/com/google/devtools/build/lib/starlarkbuildapi/test",
        "//third_party:guava",
    ],
)

java_library(
    name = "exclusive_test_strategy",
    srcs = ["ExclusiveTestStrategy.java"],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/actions",
        "//src/main/java/com/google/devtools/build/lib/analysis:analysis_cluster",
        "//src/main/java/com/google/devtools/build/lib/vfs",
        "//src/main/protobuf:test_status_java_proto",
        "//third_party:guava",
    ],
)

java_library(
    name = "transitive_tests_provider",
    srcs = ["TransitiveTestsProvider.java"],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/analysis:transitive_info_provider",
        "//src/main/java/com/google/devtools/build/lib/concurrent:thread_safety",
    ],
)

java_library(
    name = "test_suite_rule",
    srcs = ["TestSuiteRule.java"],
    deps = [
        ":test_suite",
        "//src/main/java/com/google/devtools/build/lib/analysis:analysis_cluster",
        "//src/main/java/com/google/devtools/build/lib/analysis:rule_definition_environment",
        "//src/main/java/com/google/devtools/build/lib/analysis:test/test_configuration",
        "//src/main/java/com/google/devtools/build/lib/packages",
    ],
)

java_library(
    name = "test_suite",
    srcs = ["TestSuite.java"],
    deps = [
        ":transitive_tests_provider",
        "//src/main/java/com/google/devtools/build/lib/actions",
        "//src/main/java/com/google/devtools/build/lib/analysis:analysis_cluster",
        "//src/main/java/com/google/devtools/build/lib/analysis:configured_target",
        "//src/main/java/com/google/devtools/build/lib/analysis:transitive_info_collection",
        "//src/main/java/com/google/devtools/build/lib/packages",
        "//src/main/java/com/google/devtools/build/lib/util:pair",
        "//third_party:guava",
        "//third_party:jsr305",
    ],
)

filegroup(
    name = "docgen_src",
    srcs = glob(["*.java"]),
)
