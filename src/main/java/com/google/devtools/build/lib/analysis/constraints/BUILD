load("@rules_java//java:defs.bzl", "java_library")

package(
    default_applicable_licenses = ["//:license"],
    default_visibility = ["//src/main/java/com/google/devtools/build/lib:build_lib"],
)

filegroup(
    name = "srcs",
    srcs = glob(["**"]),
    visibility = ["//src/main/java/com/google/devtools/build/lib/analysis:__pkg__"],
)

java_library(
    name = "constraint_constants",
    srcs = ["ConstraintConstants.java"],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/analysis/platform",
        "//src/main/java/com/google/devtools/build/lib/cmdline",
        "//src/main/java/com/google/devtools/build/lib/util:os",
        "//third_party/java/guava:annotations",
        "//third_party/java/guava:collect",
        "//third_party/java/jsr305_annotations",
    ],
)

java_library(
    name = "constraint_semantics",
    srcs = ["ConstraintSemantics.java"],
    deps = [
        ":constraint_constants",
        ":environment_collection",
        ":supported_environments_provider",
        "//src/main/java/com/google/devtools/build/lib/cmdline",
        "//src/main/java/com/google/devtools/build/lib/packages",
        "//src/main/java/com/google/devtools/build/lib/skyframe:detailed_exceptions",
        "//src/main/java/com/google/devtools/build/lib/util:detailed_exit_code",
        "//src/main/protobuf:failure_details_java_proto",
        "//third_party/java/jsr305_annotations",
    ],
)

java_library(
    name = "environment",
    srcs = ["Environment.java"],
    deps = [
        ":constraint_semantics",
        ":environment_collection",
        ":supported_environments",
        ":supported_environments_provider",
        "//src/main/java/com/google/devtools/build/lib/actions",
        "//src/main/java/com/google/devtools/build/lib/analysis:analysis_cluster",
        "//src/main/java/com/google/devtools/build/lib/analysis:configured_target",
        "//src/main/java/com/google/devtools/build/lib/analysis:file_provider",
        "//src/main/java/com/google/devtools/build/lib/cmdline",
        "//src/main/java/com/google/devtools/build/lib/packages",
        "//third_party/java/guava:collect",
        "//third_party/java/jsr305_annotations",
    ],
)

java_library(
    name = "environment_collection",
    srcs = ["EnvironmentCollection.java"],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/cmdline",
        "//src/main/java/com/google/devtools/build/lib/concurrent",
        "//src/main/java/com/google/devtools/build/lib/concurrent:thread_safety",
        "//src/main/java/com/google/devtools/build/lib/packages",
        "//src/main/java/com/google/devtools/build/lib/skyframe/serialization/autocodec",
        "//src/main/java/com/google/devtools/build/lib/skyframe/serialization/autocodec:serialization-constant",
        "//third_party/java/auto:auto_value",
        "//third_party/java/guava:base",
        "//third_party/java/guava:collect",
    ],
)

java_library(
    name = "environment_rule",
    srcs = ["EnvironmentRule.java"],
    deps = [
        ":constraint_constants",
        ":environment",
        "//src/main/java/com/google/devtools/build/lib/analysis:analysis_cluster",
        "//src/main/java/com/google/devtools/build/lib/analysis:rule_definition_environment",
        "//src/main/java/com/google/devtools/build/lib/analysis/config/transitions:no_config_transition",
        "//src/main/java/com/google/devtools/build/lib/packages",
        "//src/main/java/com/google/devtools/build/lib/util:filetype",
        "//third_party/java/guava:collect",
    ],
)

java_library(
    name = "incompatible_target_checker",
    srcs = ["IncompatibleTargetChecker.java"],
    deps = [
        ":rule_context_constraint_semantics",
        ":supported_environments",
        "//src/main/java/com/google/devtools/build/lib/analysis:analysis_cluster",
        "//src/main/java/com/google/devtools/build/lib/analysis:configured_target",
        "//src/main/java/com/google/devtools/build/lib/analysis:configured_target_value",
        "//src/main/java/com/google/devtools/build/lib/analysis:dependency_kind",
        "//src/main/java/com/google/devtools/build/lib/analysis:file_provider",
        "//src/main/java/com/google/devtools/build/lib/analysis:incompatible_platform_provider",
        "//src/main/java/com/google/devtools/build/lib/analysis:target_and_configuration",
        "//src/main/java/com/google/devtools/build/lib/analysis:test/test_configuration",
        "//src/main/java/com/google/devtools/build/lib/analysis:transitive_dependency_state",
        "//src/main/java/com/google/devtools/build/lib/analysis:transitive_info_provider_map_builder",
        "//src/main/java/com/google/devtools/build/lib/analysis:visibility_provider",
        "//src/main/java/com/google/devtools/build/lib/analysis/config:build_configuration",
        "//src/main/java/com/google/devtools/build/lib/analysis/config:config_conditions",
        "//src/main/java/com/google/devtools/build/lib/analysis/config:config_matching_provider",
        "//src/main/java/com/google/devtools/build/lib/analysis/platform",
        "//src/main/java/com/google/devtools/build/lib/analysis/platform:utils",
        "//src/main/java/com/google/devtools/build/lib/cmdline",
        "//src/main/java/com/google/devtools/build/lib/collect/nestedset",
        "//src/main/java/com/google/devtools/build/lib/packages",
        "//src/main/java/com/google/devtools/build/lib/packages:configured_attribute_mapper",
        "//src/main/java/com/google/devtools/build/lib/packages:package_specification",
        "//src/main/java/com/google/devtools/build/lib/skyframe:configured_target_and_data",
        "//src/main/java/com/google/devtools/build/lib/skyframe:configured_target_key",
        "//src/main/java/com/google/devtools/build/lib/skyframe:rule_configured_target_value",
        "//src/main/java/com/google/devtools/build/lib/util",
        "//src/main/java/com/google/devtools/build/skyframe",
        "//src/main/java/com/google/devtools/build/skyframe:skyframe-objects",
        "//third_party/java/guava:collect",
        "//third_party/java/jsr305_annotations",
    ],
)

java_library(
    name = "platform_restrictions_result",
    srcs = ["PlatformRestrictionsResult.java"],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/analysis:configured_target",
        "//third_party/java/auto:auto_value",
        "//third_party/java/guava:collect",
    ],
)

java_library(
    name = "rule_context_constraint_semantics",
    srcs = ["RuleContextConstraintSemantics.java"],
    deps = [
        ":constraint_semantics",
        ":environment_collection",
        ":supported_environments_provider",
        "//src/main/java/com/google/devtools/build/lib/analysis:analysis_cluster",
        "//src/main/java/com/google/devtools/build/lib/analysis:configured_target",
        "//src/main/java/com/google/devtools/build/lib/analysis:incompatible_platform_provider",
        "//src/main/java/com/google/devtools/build/lib/analysis:label_and_location",
        "//src/main/java/com/google/devtools/build/lib/analysis:transitive_info_collection",
        "//src/main/java/com/google/devtools/build/lib/cmdline",
        "//src/main/java/com/google/devtools/build/lib/packages",
        "//third_party/java/guava:base",
        "//third_party/java/guava:collect",
        "//third_party/java/jsr305_annotations",
    ],
)

java_library(
    name = "supported_environments",
    srcs = ["SupportedEnvironments.java"],
    deps = [
        ":environment_collection",
        ":supported_environments_provider",
        "//src/main/java/com/google/devtools/build/lib/cmdline",
        "//third_party/java/guava:collect",
    ],
)

java_library(
    name = "supported_environments_provider",
    srcs = ["SupportedEnvironmentsProvider.java"],
    deps = [
        ":environment_collection",
        "//src/main/java/com/google/devtools/build/lib/analysis:label_and_location",
        "//src/main/java/com/google/devtools/build/lib/analysis:transitive_info_provider",
        "//src/main/java/com/google/devtools/build/lib/cmdline",
        "//src/main/java/com/google/devtools/build/lib/skyframe/serialization/autocodec",
    ],
)

java_library(
    name = "top_level_constraint_semantics",
    srcs = [
        "TopLevelConstraintSemantics.java",
    ],
    exports = [
        ":platform_restrictions_result",
        ":rule_context_constraint_semantics",
    ],
    deps = [
        ":constraint_semantics",
        ":environment_collection",
        ":platform_restrictions_result",
        ":rule_context_constraint_semantics",
        ":supported_environments_provider",
        "//src/main/java/com/google/devtools/build/lib/analysis:analysis_cluster",
        "//src/main/java/com/google/devtools/build/lib/analysis:configured_target",
        "//src/main/java/com/google/devtools/build/lib/analysis:incompatible_platform_provider",
        "//src/main/java/com/google/devtools/build/lib/analysis:transitive_info_collection",
        "//src/main/java/com/google/devtools/build/lib/analysis:view_creation_failed_exception",
        "//src/main/java/com/google/devtools/build/lib/analysis/config:build_configuration",
        "//src/main/java/com/google/devtools/build/lib/analysis/platform",
        "//src/main/java/com/google/devtools/build/lib/cmdline",
        "//src/main/java/com/google/devtools/build/lib/events",
        "//src/main/java/com/google/devtools/build/lib/packages",
        "//src/main/java/com/google/devtools/build/lib/pkgcache",
        "//src/main/java/com/google/devtools/build/lib/skyframe:package_value",
        "//src/main/java/com/google/devtools/build/lib/skyframe:sane_analysis_exception",
        "//src/main/java/com/google/devtools/build/lib/skyframe/config",
        "//src/main/java/com/google/devtools/build/lib/util:detailed_exit_code",
        "//src/main/java/com/google/devtools/build/skyframe",
        "//src/main/protobuf:failure_details_java_proto",
        "//third_party/java/guava:base",
        "//third_party/java/guava:collect",
        "//third_party/java/jsr305_annotations",
    ],
)
