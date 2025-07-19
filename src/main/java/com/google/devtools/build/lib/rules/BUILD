load("@rules_java//java:defs.bzl", "java_library")

package(
    default_applicable_licenses = ["//:license"],
    default_visibility = ["//src:__subpackages__"],
)

filegroup(
    name = "srcs",
    srcs = glob(["**"]) + [
        "//src/main/java/com/google/devtools/build/lib/rules/android:srcs",
        "//src/main/java/com/google/devtools/build/lib/rules/apple:srcs",
        "//src/main/java/com/google/devtools/build/lib/rules/config:srcs",
        "//src/main/java/com/google/devtools/build/lib/rules/core:srcs",
        "//src/main/java/com/google/devtools/build/lib/rules/cpp:srcs",
        "//src/main/java/com/google/devtools/build/lib/rules/genquery:srcs",
        "//src/main/java/com/google/devtools/build/lib/rules/genrule:srcs",
        "//src/main/java/com/google/devtools/build/lib/rules/java:srcs",
        "//src/main/java/com/google/devtools/build/lib/rules/objc:srcs",
        "//src/main/java/com/google/devtools/build/lib/rules/platform:srcs",
        "//src/main/java/com/google/devtools/build/lib/rules/proto:srcs",
        "//src/main/java/com/google/devtools/build/lib/rules/python:srcs",
        "//src/main/java/com/google/devtools/build/lib/rules/starlarkdocextract:srcs",
        "//src/main/java/com/google/devtools/build/lib/rules/test:srcs",
    ],
    visibility = ["//src:__subpackages__"],
)

java_library(
    name = "legacy_exports",
    exports = [
        ":alias",
        ":extra/action_listener",
        ":extra/action_listener_rule",
        ":extra/extra_action_factory",
        ":extra/extra_action_rule",
        ":filegroup/filegroup",
        ":label_build_settings",
        ":repository/repository_directory_value",
        ":toolchain_type",
    ],
)

java_library(
    name = "alias",
    srcs = [
        "Alias.java",
        "AliasConfiguredTarget.java",
        "LateBoundAlias.java",
    ],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/actions",
        "//src/main/java/com/google/devtools/build/lib/actions:action_lookup_key",
        "//src/main/java/com/google/devtools/build/lib/actions:artifacts",
        "//src/main/java/com/google/devtools/build/lib/analysis:analysis_cluster",
        "//src/main/java/com/google/devtools/build/lib/analysis:build_setting_provider",
        "//src/main/java/com/google/devtools/build/lib/analysis:configured_target",
        "//src/main/java/com/google/devtools/build/lib/analysis:file_provider",
        "//src/main/java/com/google/devtools/build/lib/analysis:required_config_fragments_provider",
        "//src/main/java/com/google/devtools/build/lib/analysis:rule_definition_environment",
        "//src/main/java/com/google/devtools/build/lib/analysis:transitive_info_provider",
        "//src/main/java/com/google/devtools/build/lib/analysis:visibility_provider",
        "//src/main/java/com/google/devtools/build/lib/analysis:visibility_provider_impl",
        "//src/main/java/com/google/devtools/build/lib/analysis/config:config_matching_provider",
        "//src/main/java/com/google/devtools/build/lib/analysis/config:fragment",
        "//src/main/java/com/google/devtools/build/lib/cmdline",
        "//src/main/java/com/google/devtools/build/lib/collect/nestedset",
        "//src/main/java/com/google/devtools/build/lib/concurrent:thread_safety",
        "//src/main/java/com/google/devtools/build/lib/packages",
        "//src/main/java/com/google/devtools/build/lib/packages:package_specification",
        "//src/main/java/com/google/devtools/build/lib/skyframe/serialization:visible-for-serialization",
        "//src/main/java/com/google/devtools/build/lib/skyframe/serialization/autocodec",
        "//src/main/java/com/google/devtools/build/lib/util:filetype",
        "//src/main/java/net/starlark/java/eval",
        "//third_party:guava",
        "//third_party:jsr305",
    ],
)

java_library(
    name = "extra/action_listener",
    srcs = ["extra/ActionListener.java"],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/actions",
        "//src/main/java/com/google/devtools/build/lib/analysis:analysis_cluster",
        "//src/main/java/com/google/devtools/build/lib/analysis:configured_target",
        "//src/main/java/com/google/devtools/build/lib/analysis:transitive_info_collection",
        "//src/main/java/com/google/devtools/build/lib/collect:immutable_sorted_key_list_multimap",
        "//src/main/java/com/google/devtools/build/lib/packages",
        "//third_party:guava",
        "//third_party:jsr305",
    ],
)

java_library(
    name = "extra/action_listener_rule",
    srcs = ["extra/ActionListenerRule.java"],
    deps = [
        ":extra/action_listener",
        "//src/main/java/com/google/devtools/build/lib/analysis:analysis_cluster",
        "//src/main/java/com/google/devtools/build/lib/analysis:rule_definition_environment",
        "//src/main/java/com/google/devtools/build/lib/packages",
    ],
)

java_library(
    name = "extra/extra_action_factory",
    srcs = ["extra/ExtraActionFactory.java"],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/actions",
        "//src/main/java/com/google/devtools/build/lib/actions:artifacts",
        "//src/main/java/com/google/devtools/build/lib/analysis:analysis_cluster",
        "//src/main/java/com/google/devtools/build/lib/analysis:configured_target",
        "//src/main/java/com/google/devtools/build/lib/packages",
        "//third_party:guava",
        "//third_party:jsr305",
    ],
)

java_library(
    name = "extra/extra_action_rule",
    srcs = ["extra/ExtraActionRule.java"],
    deps = [
        ":extra/extra_action_factory",
        "//src/main/java/com/google/devtools/build/lib/analysis:analysis_cluster",
        "//src/main/java/com/google/devtools/build/lib/analysis:rule_definition_environment",
        "//src/main/java/com/google/devtools/build/lib/analysis/config:execution_transition_factory",
        "//src/main/java/com/google/devtools/build/lib/packages",
    ],
)

java_library(
    name = "filegroup/filegroup",
    srcs = [
        "filegroup/Filegroup.java",
        "filegroup/FilegroupRule.java",
    ],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/actions",
        "//src/main/java/com/google/devtools/build/lib/actions:artifacts",
        "//src/main/java/com/google/devtools/build/lib/analysis:analysis_cluster",
        "//src/main/java/com/google/devtools/build/lib/analysis:configured_target",
        "//src/main/java/com/google/devtools/build/lib/analysis:rule_definition_environment",
        "//src/main/java/com/google/devtools/build/lib/analysis:test/instrumented_files_info",
        "//src/main/java/com/google/devtools/build/lib/analysis:transitive_info_collection",
        "//src/main/java/com/google/devtools/build/lib/analysis/config:core_options",
        "//src/main/java/com/google/devtools/build/lib/collect/nestedset",
        "//src/main/java/com/google/devtools/build/lib/packages",
        "//src/main/java/com/google/devtools/build/lib/util:filetype",
        "//third_party:jsr305",
    ],
)

java_library(
    name = "label_build_settings",
    srcs = ["LabelBuildSettings.java"],
    deps = [
        ":alias",
        "//src/main/java/com/google/devtools/build/lib/analysis:rule_definition_environment",
        "//src/main/java/com/google/devtools/build/lib/analysis/config:build_configuration",
        "//src/main/java/com/google/devtools/build/lib/cmdline",
        "//src/main/java/com/google/devtools/build/lib/packages",
        "//src/main/java/com/google/devtools/build/lib/skyframe/serialization:visible-for-serialization",
        "//src/main/java/com/google/devtools/build/lib/skyframe/serialization/autocodec:serialization-constant",
        "//src/main/java/net/starlark/java/eval",
        "//third_party:guava",
    ],
)

java_library(
    name = "repository/repo_recorded_input",
    srcs = ["repository/RepoRecordedInput.java"],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/actions:file_metadata",
        "//src/main/java/com/google/devtools/build/lib/analysis:blaze_directories",
        "//src/main/java/com/google/devtools/build/lib/cmdline",
        "//src/main/java/com/google/devtools/build/lib/cmdline:LabelValidator",
        "//src/main/java/com/google/devtools/build/lib/skyframe:action_environment_function",
        "//src/main/java/com/google/devtools/build/lib/skyframe:client_environment_value",
        "//src/main/java/com/google/devtools/build/lib/skyframe:directory_listing_value",
        "//src/main/java/com/google/devtools/build/lib/skyframe:directory_tree_digest_value",
        "//src/main/java/com/google/devtools/build/lib/skyframe:filesystem_keys",
        "//src/main/java/com/google/devtools/build/lib/skyframe:precomputed_value",
        "//src/main/java/com/google/devtools/build/lib/skyframe:repository_mapping_value",
        "//src/main/java/com/google/devtools/build/lib/skyframe/serialization/autocodec",
        "//src/main/java/com/google/devtools/build/lib/util",
        "//src/main/java/com/google/devtools/build/lib/vfs",
        "//src/main/java/com/google/devtools/build/lib/vfs:pathfragment",
        "//src/main/java/com/google/devtools/build/skyframe:skyframe-objects",
        "//third_party:guava",
        "//third_party:jsr305",
    ],
)

java_library(
    name = "repository/repository_directory_value",
    srcs = ["repository/RepositoryDirectoryValue.java"],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/cmdline",
        "//src/main/java/com/google/devtools/build/lib/skyframe:precomputed_value",
        "//src/main/java/com/google/devtools/build/lib/skyframe:sky_functions",
        "//src/main/java/com/google/devtools/build/lib/skyframe/serialization:visible-for-serialization",
        "//src/main/java/com/google/devtools/build/lib/skyframe/serialization/autocodec",
        "//src/main/java/com/google/devtools/build/lib/vfs",
        "//src/main/java/com/google/devtools/build/skyframe:skyframe-objects",
    ],
)

java_library(
    name = "repository/repository_directory_dirtiness_checker",
    srcs = ["repository/RepositoryDirectoryDirtinessChecker.java"],
    deps = [
        ":repository/repository_directory_value",
        "//src/main/java/com/google/devtools/build/lib/skyframe:sky_functions",
        "//src/main/java/com/google/devtools/build/lib/skyframe:sky_value_dirtiness_checker",
        "//src/main/java/com/google/devtools/build/lib/util/io",
        "//src/main/java/com/google/devtools/build/lib/vfs",
        "//src/main/java/com/google/devtools/build/skyframe:skyframe-objects",
        "//third_party:guava",
        "//third_party:jsr305",
    ],
)

java_library(
    name = "repository/workspace_attribute_mapper",
    srcs = ["repository/WorkspaceAttributeMapper.java"],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/packages",
        "//src/main/java/net/starlark/java/eval",
        "//third_party:guava",
        "//third_party:jsr305",
    ],
)

java_library(
    name = "toolchain_type",
    srcs = ["ToolchainType.java"],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/actions",
        "//src/main/java/com/google/devtools/build/lib/analysis:analysis_cluster",
        "//src/main/java/com/google/devtools/build/lib/analysis:configured_target",
        "//src/main/java/com/google/devtools/build/lib/analysis:rule_definition_environment",
        "//src/main/java/com/google/devtools/build/lib/analysis/platform",
        "//src/main/java/com/google/devtools/build/lib/packages",
        "//third_party:jsr305",
    ],
)

filegroup(
    name = "bazel-docgen-src",
    srcs = glob(["**/*.java"]),
)

filegroup(
    name = "docgen_src",
    srcs = glob(
        [
            "*.java",
            "filegroup/*.java",
            "extra/*.java",
        ],
    ) + [
        "//src/main/java/com/google/devtools/build/lib/rules/test:docgen_src",
    ],
)
