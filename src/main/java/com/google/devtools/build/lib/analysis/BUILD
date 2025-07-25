load("@rules_java//java:defs.bzl", "java_library")

package(
    default_applicable_licenses = ["//:license"],
    default_visibility = ["//src:__subpackages__"],
)

filegroup(
    name = "srcs",
    srcs = glob(["**"]) + [
        "//src/main/java/com/google/devtools/build/lib/analysis/config:srcs",
        "//src/main/java/com/google/devtools/build/lib/analysis/platform:srcs",
        "//src/main/java/com/google/devtools/build/lib/analysis/producers:srcs",
        "//src/main/java/com/google/devtools/build/lib/analysis/starlark/annotations:srcs",
        "//src/main/java/com/google/devtools/build/lib/analysis/stringtemplate:srcs",
    ],
    visibility = ["//src:__subpackages__"],
)

# TODO(b/144899336): Split this further.
java_library(
    name = "analysis_cluster",
    srcs = [
        "AliasProvider.java",
        "Allowlist.java",
        "AnalysisAndExecutionResult.java",
        "AnalysisEnvironment.java",
        "AnalysisFailureEvent.java",
        "AnalysisIssues.java",
        "AnalysisResult.java",
        "AnalysisRootCauseEvent.java",
        "AnalysisUtils.java",
        "AspectBaseTargetResolvedToolchainContext.java",
        "AspectCompleteEvent.java",
        "AspectContext.java",
        "AspectResolutionHelpers.java",
        "AspectValue.java",
        "BaseRuleClasses.java",
        "BashCommandConstructor.java",
        "BazelRuleAnalysisThreadContext.java",
        "CachingAnalysisEnvironment.java",
        "CommandConstructor.java",
        "CommandHelper.java",
        "ConfigurationMakeVariableContext.java",
        "ConfiguredAspect.java",
        "ConfiguredAspectFactory.java",
        "ConfiguredRuleClassProvider.java",
        "ConfiguredTargetFactory.java",
        "DefaultInfo.java",
        "DependencyResolutionHelpers.java",
        "EventHandlingErrorReporter.java",
        "Expander.java",
        "ExtraActionUtils.java",
        "ExtraActionsVisitor.java",
        "FilesToRunProvider.java",
        "FragmentCollection.java",
        "LocationExpander.java",
        "LocationTemplateContext.java",
        "OutputGroupInfo.java",
        "PackageSpecificationProvider.java",
        "PrerequisiteArtifacts.java",
        "PrerequisitesCollection.java",
        "PseudoAction.java",
        "RequiredFragmentsUtil.java",
        "RuleConfiguredTargetBuilder.java",
        "RuleConfiguredTargetFactory.java",
        "RuleContext.java",
        "RuleDefinition.java",
        "Runfiles.java",
        "RunfilesProvider.java",
        "RunfilesSupport.java",
        "ShToolchain.java",
        "ShellConfiguration.java",
        "SourceManifestAction.java",
        "StarlarkProviderValidationUtil.java",
        "TargetCompleteEvent.java",
        "TargetContext.java",
        "TopLevelArtifactHelper.java",
        "Util.java",
        "WindowsBatchCommandConstructor.java",
        "WindowsPowershellCommandConstructor.java",
        "actions/ActionConstructionContext.java",
        "actions/FileWriteAction.java",
        "actions/ShellCommand.java",
        "actions/SpawnAction.java",
        "actions/StarlarkAction.java",
        "actions/SymlinkTreeAction.java",
        "actions/SymlinkTreeActionContext.java",
        "configuredtargets/AbstractConfiguredTarget.java",
        "configuredtargets/EnvironmentGroupConfiguredTarget.java",
        "configuredtargets/FileConfiguredTarget.java",
        "configuredtargets/InputFileConfiguredTarget.java",
        "configuredtargets/MergedConfiguredTarget.java",
        "configuredtargets/OutputFileConfiguredTarget.java",
        "configuredtargets/PackageGroupConfiguredTarget.java",
        "configuredtargets/RuleConfiguredTarget.java",
        "constraints/RuleContextConstraintSemantics.java",
        "extra/ExtraAction.java",
        "extra/ExtraActionMapProvider.java",
        "extra/ExtraActionSpec.java",
        "starlark/StarlarkActionFactory.java",
        "starlark/StarlarkAspectPropagationContext.java",
        "starlark/StarlarkAttrModule.java",
        "starlark/StarlarkAttributeTransitionProvider.java",
        "starlark/StarlarkAttributesCollection.java",
        "starlark/StarlarkErrorReporter.java",
        "starlark/StarlarkGlobalsImpl.java",
        "starlark/StarlarkRuleClassFunctions.java",
        "starlark/StarlarkRuleConfiguredTargetUtil.java",
        "starlark/StarlarkRuleContext.java",
        "starlark/StarlarkRuleTransitionProvider.java",
        "starlark/StarlarkSubrule.java",
        "starlark/UnresolvedSymlinkAction.java",
        "test/AnalysisTestActionBuilder.java",
        "test/BaselineCoverageAction.java",
        "test/CoverageCommon.java",
        "test/InstrumentedFileManifestAction.java",
        "test/InstrumentedFilesCollector.java",
        "test/TestActionBuilder.java",
        "test/TestActionContext.java",
        "test/TestAttempt.java",
        "test/TestProvider.java",
        "test/TestResult.java",
        "test/TestRunnerAction.java",
        "test/TestStrategy.java",
        "test/TestTagsProvider.java",
        "test/TestTargetExecutionSettings.java",
        "test/TestTargetProperties.java",
    ],
    deps = [
        ":actions/abstract_file_write_action",
        ":actions/buildinfo_file_write_action",
        ":actions/compression",
        ":actions/custom_command_line",
        ":actions/lazy_write_nested_set_of_tuple_action",
        ":actions/parameter_file_write_action",
        ":actions/path_mappers",
        ":actions/substitution",
        ":actions/symlink_action",
        ":actions/template_expansion_action",
        ":actions_provider",
        ":aspect_aware_attribute_mapper",
        ":aspect_collection",
        ":blaze_version_info",
        ":build_setting_provider",
        ":configured_target",
        ":constraints/constraint_constants",
        ":constraints/constraint_semantics",
        ":constraints/environment_collection",
        ":constraints/supported_environments",
        ":constraints/supported_environments_provider",
        ":dependency_kind",
        ":dormant_dependency",
        ":exec_group_collection",
        ":extra/extra_action_info_file_write_action",
        ":extra_action_artifacts_provider",
        ":file_provider",
        ":incompatible_platform_provider",
        ":inconsistent_aspect_order_exception",
        ":label_and_location",
        ":label_expander",
        ":make_variable_supplier",
        ":options_diff_predicate",
        ":platform_configuration",
        ":provider_collection",
        ":repo_mapping_manifest_action",
        ":required_config_fragments_provider",
        ":resolved_toolchain_context",
        ":resolved_toolchain_data",
        ":resolved_toolchain_data_interface",
        ":rule_configured_object_value",
        ":rule_definition_environment",
        ":rule_error_consumer",
        ":run_environment_info",
        ":starlark/args",
        ":starlark/bazel_build_api_globals",
        ":starlark/function_transition_util",
        ":starlark/starlark_api_provider",
        ":starlark/starlark_exec_group_collection",
        ":starlark/starlark_late_bound_default",
        ":starlark/starlark_toolchain_context",
        ":starlark/starlark_transition",
        ":starlark/template_dict",
        ":symlink_entry",
        ":target_and_configuration",
        ":template_variable_info",
        ":test/analysis_failure",
        ":test/analysis_failure_info",
        ":test/analysis_failure_propagation_exception",
        ":test/analysis_test_result_info",
        ":test/coverage_configuration",
        ":test/execution_info",
        ":test/instrumented_files_info",
        ":test/test_configuration",
        ":test/test_sharding_strategy",
        ":toolchain_collection",
        ":toolchain_context",
        ":top_level_artifact_context",
        ":transitive_info_collection",
        ":transitive_info_provider",
        ":transitive_info_provider_effective_class_helper",
        ":transitive_info_provider_map",
        ":transitive_info_provider_map_builder",
        ":transitive_info_provider_map_impl",
        ":visibility_provider",
        "//src/main/java/com/google/devtools/build/docgen/annot",
        "//src/main/java/com/google/devtools/build/lib:runtime/build_event_streamer_utils",
        "//src/main/java/com/google/devtools/build/lib:runtime/test_summary_options",
        "//src/main/java/com/google/devtools/build/lib/actions",
        "//src/main/java/com/google/devtools/build/lib/actions:action_input_helper",
        "//src/main/java/com/google/devtools/build/lib/actions:action_lookup_key",
        "//src/main/java/com/google/devtools/build/lib/actions:artifacts",
        "//src/main/java/com/google/devtools/build/lib/actions:commandline_item",
        "//src/main/java/com/google/devtools/build/lib/actions:execution_requirements",
        "//src/main/java/com/google/devtools/build/lib/actions:file_metadata",
        "//src/main/java/com/google/devtools/build/lib/actions:fileset_output_tree",
        "//src/main/java/com/google/devtools/build/lib/actions:localhost_capacity",
        "//src/main/java/com/google/devtools/build/lib/actions:package_roots",
        "//src/main/java/com/google/devtools/build/lib/actions:runfiles_tree",
        "//src/main/java/com/google/devtools/build/lib/analysis/config:build_configuration",
        "//src/main/java/com/google/devtools/build/lib/analysis/config:build_option_details",
        "//src/main/java/com/google/devtools/build/lib/analysis/config:build_options",
        "//src/main/java/com/google/devtools/build/lib/analysis/config:common_options",
        "//src/main/java/com/google/devtools/build/lib/analysis/config:config_conditions",
        "//src/main/java/com/google/devtools/build/lib/analysis/config:config_matching_provider",
        "//src/main/java/com/google/devtools/build/lib/analysis/config:core_options",
        "//src/main/java/com/google/devtools/build/lib/analysis/config:execution_transition_factory",
        "//src/main/java/com/google/devtools/build/lib/analysis/config:feature_set",
        "//src/main/java/com/google/devtools/build/lib/analysis/config:fragment",
        "//src/main/java/com/google/devtools/build/lib/analysis/config:fragment_class_set",
        "//src/main/java/com/google/devtools/build/lib/analysis/config:fragment_options",
        "//src/main/java/com/google/devtools/build/lib/analysis/config:fragment_registry",
        "//src/main/java/com/google/devtools/build/lib/analysis/config:per_label_options",
        "//src/main/java/com/google/devtools/build/lib/analysis/config:run_under",
        "//src/main/java/com/google/devtools/build/lib/analysis/config:starlark_defined_config_transition",
        "//src/main/java/com/google/devtools/build/lib/analysis/config:symlink_definition",
        "//src/main/java/com/google/devtools/build/lib/analysis/config:toolchain_type_requirement",
        "//src/main/java/com/google/devtools/build/lib/analysis/config/transitions:composing_transition_factory",
        "//src/main/java/com/google/devtools/build/lib/analysis/config/transitions:configuration_transition",
        "//src/main/java/com/google/devtools/build/lib/analysis/config/transitions:no_config_transition",
        "//src/main/java/com/google/devtools/build/lib/analysis/config/transitions:no_transition",
        "//src/main/java/com/google/devtools/build/lib/analysis/config/transitions:patch_transition",
        "//src/main/java/com/google/devtools/build/lib/analysis/config/transitions:split_transition",
        "//src/main/java/com/google/devtools/build/lib/analysis/config/transitions:starlark_exposed_rule_transition_factory",
        "//src/main/java/com/google/devtools/build/lib/analysis/config/transitions:transition_factory",
        "//src/main/java/com/google/devtools/build/lib/analysis/platform",
        "//src/main/java/com/google/devtools/build/lib/analysis/stringtemplate",
        "//src/main/java/com/google/devtools/build/lib/bazel/bzlmod:resolution",
        "//src/main/java/com/google/devtools/build/lib/bugreport",
        "//src/main/java/com/google/devtools/build/lib/buildeventstream",
        "//src/main/java/com/google/devtools/build/lib/buildeventstream/proto:build_event_stream_java_proto",
        "//src/main/java/com/google/devtools/build/lib/causes",
        "//src/main/java/com/google/devtools/build/lib/cmdline",
        "//src/main/java/com/google/devtools/build/lib/collect:immutable_shared_key_map",
        "//src/main/java/com/google/devtools/build/lib/collect:immutable_sorted_key_list_multimap",
        "//src/main/java/com/google/devtools/build/lib/collect/compacthashset",
        "//src/main/java/com/google/devtools/build/lib/collect/nestedset",
        "//src/main/java/com/google/devtools/build/lib/concurrent",
        "//src/main/java/com/google/devtools/build/lib/concurrent:thread_safety",
        "//src/main/java/com/google/devtools/build/lib/events",
        "//src/main/java/com/google/devtools/build/lib/exec:execution_options",
        "//src/main/java/com/google/devtools/build/lib/exec:spawn_exec_exception",
        "//src/main/java/com/google/devtools/build/lib/exec:spawn_strategy_resolver",
        "//src/main/java/com/google/devtools/build/lib/exec:streamed_test_output",
        "//src/main/java/com/google/devtools/build/lib/exec:test_log_helper",
        "//src/main/java/com/google/devtools/build/lib/exec:test_xml_output_parser",
        "//src/main/java/com/google/devtools/build/lib/exec:test_xml_output_parser_exception",
        "//src/main/java/com/google/devtools/build/lib/graph",
        "//src/main/java/com/google/devtools/build/lib/packages",
        "//src/main/java/com/google/devtools/build/lib/packages:configured_attribute_mapper",
        "//src/main/java/com/google/devtools/build/lib/packages:declared_exec_group",
        "//src/main/java/com/google/devtools/build/lib/packages:package_specification",
        "//src/main/java/com/google/devtools/build/lib/packages/semantics",
        "//src/main/java/com/google/devtools/build/lib/profiler",
        "//src/main/java/com/google/devtools/build/lib/profiler:google-auto-profiler-utils",
        "//src/main/java/com/google/devtools/build/lib/profiler/memory:current_rule_tracker",
        "//src/main/java/com/google/devtools/build/lib/query2/common:cquery-node",
        "//src/main/java/com/google/devtools/build/lib/shell",
        "//src/main/java/com/google/devtools/build/lib/skyframe:aspect_key_creator",
        "//src/main/java/com/google/devtools/build/lib/skyframe:bzl_load_value",
        "//src/main/java/com/google/devtools/build/lib/skyframe:configured_target_and_data",
        "//src/main/java/com/google/devtools/build/lib/skyframe:configured_target_key",
        "//src/main/java/com/google/devtools/build/lib/skyframe:incremental_artifact_conflict_finder",
        "//src/main/java/com/google/devtools/build/lib/skyframe:repository_mapping_value",
        "//src/main/java/com/google/devtools/build/lib/skyframe:starlark_builtins_value",
        "//src/main/java/com/google/devtools/build/lib/skyframe:workspace_status_value",
        "//src/main/java/com/google/devtools/build/lib/skyframe/config",
        "//src/main/java/com/google/devtools/build/lib/skyframe/serialization:abstract-exported-starlark-symbol-codec",
        "//src/main/java/com/google/devtools/build/lib/skyframe/serialization:visible-for-serialization",
        "//src/main/java/com/google/devtools/build/lib/skyframe/serialization/autocodec",
        "//src/main/java/com/google/devtools/build/lib/skyframe/serialization/autocodec:serialization-constant",
        "//src/main/java/com/google/devtools/build/lib/skyframe/toolchains:toolchain_context_key",
        "//src/main/java/com/google/devtools/build/lib/skyframe/toolchains:unloaded_toolchain_context",
        "//src/main/java/com/google/devtools/build/lib/starlarkbuildapi",
        "//src/main/java/com/google/devtools/build/lib/starlarkbuildapi/config:configuration_transition_api",
        "//src/main/java/com/google/devtools/build/lib/starlarkbuildapi/core",
        "//src/main/java/com/google/devtools/build/lib/starlarkbuildapi/platform",
        "//src/main/java/com/google/devtools/build/lib/starlarkbuildapi/test",
        "//src/main/java/com/google/devtools/build/lib/supplier",
        "//src/main/java/com/google/devtools/build/lib/unsafe:string",
        "//src/main/java/com/google/devtools/build/lib/util",
        "//src/main/java/com/google/devtools/build/lib/util:detailed_exit_code",
        "//src/main/java/com/google/devtools/build/lib/util:filetype",
        "//src/main/java/com/google/devtools/build/lib/util:os",
        "//src/main/java/com/google/devtools/build/lib/util:shell_escaper",
        "//src/main/java/com/google/devtools/build/lib/util:string",
        "//src/main/java/com/google/devtools/build/lib/util:string_encoding",
        "//src/main/java/com/google/devtools/build/lib/util/io:out-err",
        "//src/main/java/com/google/devtools/build/lib/vfs",
        "//src/main/java/com/google/devtools/build/lib/vfs:pathfragment",
        "//src/main/java/com/google/devtools/build/lib/vfs/inmemoryfs",
        "//src/main/java/com/google/devtools/build/skyframe:skyframe-objects",
        "//src/main/java/com/google/devtools/common/options",
        "//src/main/java/net/starlark/java/annot",
        "//src/main/java/net/starlark/java/eval",
        "//src/main/java/net/starlark/java/lib/json",
        "//src/main/java/net/starlark/java/syntax",
        "//src/main/protobuf:extra_actions_base_java_proto",
        "//src/main/protobuf:failure_details_java_proto",
        "//src/main/protobuf:test_status_java_proto",
        "//third_party:auto_value",
        "//third_party:caffeine",
        "//third_party:error_prone_annotations",
        "//third_party:flogger",
        "//third_party:guava",
        "//third_party:jsr305",
        "@com_google_protobuf//:protobuf_java",
        "@com_google_protobuf//:protobuf_java_util",
    ],
)

java_library(
    name = "actions_provider",
    srcs = ["ActionsProvider.java"],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/actions",
        "//src/main/java/com/google/devtools/build/lib/actions:artifacts",
        "//src/main/java/com/google/devtools/build/lib/packages",
        "//src/main/java/com/google/devtools/build/lib/starlarkbuildapi",
        "//src/main/java/net/starlark/java/eval",
        "//src/main/java/net/starlark/java/syntax",
        "//third_party:guava",
    ],
)

java_library(
    name = "analysis_options",
    srcs = ["AnalysisOptions.java"],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/util",
        "//src/main/java/com/google/devtools/build/lib/util:cpu_resource_converter",
        "//src/main/java/com/google/devtools/build/lib/util:resource_converter",
        "//src/main/java/com/google/devtools/common/options",
    ],
)

java_library(
    name = "analysis_operation_watcher",
    srcs = ["AnalysisOperationWatcher.java"],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/skyframe:top_level_status_events",
        "//src/main/java/com/google/devtools/build/skyframe:skyframe-objects",
        "//third_party:guava",
    ],
)

java_library(
    name = "analysis_phase_complete_event",
    srcs = ["AnalysisPhaseCompleteEvent.java"],
    deps = [
        ":configured_target",
        "//src/main/java/com/google/devtools/build/lib/actions:total_and_configured_target_only_metric",
        "//src/main/java/com/google/devtools/build/lib/pkgcache",
        "//third_party:guava",
    ],
)

java_library(
    name = "analysis_phase_started_event",
    srcs = ["AnalysisPhaseStartedEvent.java"],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/cmdline",
        "//src/main/java/com/google/devtools/build/lib/packages",
        "//third_party:guava",
    ],
)

java_library(
    name = "aspect_aware_attribute_mapper",
    srcs = ["AspectAwareAttributeMapper.java"],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/cmdline",
        "//src/main/java/com/google/devtools/build/lib/packages",
        "//third_party:guava",
        "//third_party:jsr305",
    ],
)

java_library(
    name = "aspect_collection",
    srcs = ["AspectCollection.java"],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/concurrent",
        "//src/main/java/com/google/devtools/build/lib/concurrent:thread_safety",
        "//src/main/java/com/google/devtools/build/lib/packages",
        "//src/main/java/com/google/devtools/build/lib/skyframe:aspect_key_creator",
        "//src/main/java/com/google/devtools/build/lib/skyframe:configured_target_key",
        "//third_party:auto_value",
        "//third_party:error_prone_annotations",
        "//third_party:guava",
    ],
)

java_library(
    name = "aspect_configured_event",
    srcs = ["AspectConfiguredEvent.java"],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/analysis/config:build_configuration",
        "//src/main/java/com/google/devtools/build/lib/buildeventstream",
        "//src/main/java/com/google/devtools/build/lib/buildeventstream/proto:build_event_stream_java_proto",
        "//src/main/java/com/google/devtools/build/lib/cmdline",
        "//third_party:guava",
        "//third_party:jsr305",
    ],
)

java_library(
    name = "blaze_directories",
    srcs = ["BlazeDirectories.java"],
    deps = [
        ":server_directories",
        "//src/main/java/com/google/devtools/build/lib/actions:artifacts",
        "//src/main/java/com/google/devtools/build/lib/concurrent",
        "//src/main/java/com/google/devtools/build/lib/concurrent:thread_safety",
        "//src/main/java/com/google/devtools/build/lib/vfs",
        "//third_party:guava",
        "//third_party:jsr305",
    ],
)

java_library(
    name = "blaze_version_info",
    srcs = ["BlazeVersionInfo.java"],
    deps = [
        "//third_party:flogger",
        "//third_party:guava",
        "//third_party:jsr305",
    ],
)

java_library(
    name = "build_info",
    srcs = ["BuildInfo.java"],
)

java_library(
    name = "build_info_event",
    srcs = ["BuildInfoEvent.java"],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/buildeventstream",
        "//src/main/java/com/google/devtools/build/lib/buildeventstream/proto:build_event_stream_java_proto",
        "//src/main/java/com/google/devtools/build/lib/events",
        "//third_party:guava",
    ],
)

java_library(
    name = "build_setting_provider",
    srcs = ["BuildSettingProvider.java"],
    deps = [
        ":transitive_info_provider",
        "//src/main/java/com/google/devtools/build/lib/cmdline",
        "//src/main/java/com/google/devtools/build/lib/packages",
        "//third_party:guava",
    ],
)

java_library(
    name = "projects",
    srcs = ["Project.java"],
    deps = [
        ":project_resolution_exception",
        "//src/main/java/com/google/devtools/build/lib:runtime/config_flag_definitions",
        "//src/main/java/com/google/devtools/build/lib/analysis/config:build_options",
        "//src/main/java/com/google/devtools/build/lib/analysis/config:invalid_configuration_exception",
        "//src/main/java/com/google/devtools/build/lib/cmdline",
        "//src/main/java/com/google/devtools/build/lib/events",
        "//src/main/java/com/google/devtools/build/lib/skyframe:project_files_lookup_value",
        "//src/main/java/com/google/devtools/build/lib/skyframe:project_value",
        "//src/main/java/com/google/devtools/build/lib/skyframe:skyframe_cluster",
        "//src/main/java/com/google/devtools/build/lib/skyframe/config",
        "//src/main/java/com/google/devtools/build/skyframe",
        "//src/main/java/com/google/devtools/build/skyframe:skyframe-objects",
        "//src/main/protobuf:failure_details_java_proto",
        "//third_party:guava",
        "//third_party:jsr305",
    ],
)

java_library(
    name = "project_resolution_exception",
    srcs = ["ProjectResolutionException.java"],
)

java_library(
    name = "build_view",
    srcs = ["BuildView.java"],
    deps = [
        ":analysis_cluster",
        ":analysis_options",
        ":analysis_phase_started_event",
        ":blaze_directories",
        ":configured_target",
        ":constraints/platform_restrictions_result",
        ":constraints/top_level_constraint_semantics",
        ":extra_action_artifacts_provider",
        ":make_environment_event",
        ":test/coverage_report_action_factory",
        ":test/instrumented_files_info",
        ":test/test_trimming_transition_factory",
        ":top_level_artifact_context",
        ":view_creation_failed_exception",
        "//src/main/java/com/google/devtools/build/lib/actions",
        "//src/main/java/com/google/devtools/build/lib/actions:action_lookup_data",
        "//src/main/java/com/google/devtools/build/lib/actions:action_lookup_key",
        "//src/main/java/com/google/devtools/build/lib/actions:artifacts",
        "//src/main/java/com/google/devtools/build/lib/actions:package_roots",
        "//src/main/java/com/google/devtools/build/lib/actions:resource_manager",
        "//src/main/java/com/google/devtools/build/lib/analysis/config:additional_configuration_change_event",
        "//src/main/java/com/google/devtools/build/lib/analysis/config:build_configuration",
        "//src/main/java/com/google/devtools/build/lib/analysis/config:build_options",
        "//src/main/java/com/google/devtools/build/lib/analysis/config:invalid_configuration_exception",
        "//src/main/java/com/google/devtools/build/lib/bugreport",
        "//src/main/java/com/google/devtools/build/lib/cmdline",
        "//src/main/java/com/google/devtools/build/lib/collect/nestedset",
        "//src/main/java/com/google/devtools/build/lib/concurrent",
        "//src/main/java/com/google/devtools/build/lib/concurrent:thread_safety",
        "//src/main/java/com/google/devtools/build/lib/events",
        "//src/main/java/com/google/devtools/build/lib/packages",
        "//src/main/java/com/google/devtools/build/lib/pkgcache",
        "//src/main/java/com/google/devtools/build/lib/profiler",
        "//src/main/java/com/google/devtools/build/lib/skyframe:aspect_key_creator",
        "//src/main/java/com/google/devtools/build/lib/skyframe:build_result_listener",
        "//src/main/java/com/google/devtools/build/lib/skyframe:bzl_load_value",
        "//src/main/java/com/google/devtools/build/lib/skyframe:configured_target_key",
        "//src/main/java/com/google/devtools/build/lib/skyframe:coverage_report_value",
        "//src/main/java/com/google/devtools/build/lib/skyframe:repository_mapping_value",
        "//src/main/java/com/google/devtools/build/lib/skyframe:skyfocus_state",
        "//src/main/java/com/google/devtools/build/lib/skyframe:skyframe_cluster",
        "//src/main/java/com/google/devtools/build/lib/skyframe:target_pattern_phase_value",
        "//src/main/java/com/google/devtools/build/lib/skyframe/serialization/analysis:dependencies_provider",
        "//src/main/java/com/google/devtools/build/lib/skyframe/serialization/analysis:options",
        "//src/main/java/com/google/devtools/build/lib/util",
        "//src/main/java/com/google/devtools/build/lib/util:abrupt_exit_exception",
        "//src/main/java/com/google/devtools/build/lib/util:detailed_exit_code",
        "//src/main/java/com/google/devtools/build/skyframe",
        "//src/main/protobuf:failure_details_java_proto",
        "//third_party:flogger",
        "//third_party:guava",
        "//third_party:jsr305",
    ],
)

java_library(
    name = "common_prerequisite_validator",
    srcs = ["CommonPrerequisiteValidator.java"],
    deps = [
        ":analysis_cluster",
        ":rule_error_consumer",
        ":visibility_provider",
        "//src/main/java/com/google/devtools/build/lib/cmdline",
        "//src/main/java/com/google/devtools/build/lib/packages",
        "//src/main/java/com/google/devtools/build/lib/packages:package_specification",
        "//src/main/java/com/google/devtools/build/lib/skyframe:configured_target_and_data",
        "//third_party:guava",
        "//third_party:jsr305",
    ],
)

java_library(
    name = "configured_object_value",
    srcs = ["ConfiguredObjectValue.java"],
    deps = [
        ":provider_collection",
        "//src/main/java/com/google/devtools/build/lib/collect/nestedset",
        "//src/main/java/com/google/devtools/build/lib/packages",
        "//src/main/java/com/google/devtools/build/skyframe:skyframe-objects",
    ],
)

java_library(
    name = "rule_configured_object_value",
    srcs = ["RuleConfiguredObjectValue.java"],
    deps = [
        ":configured_object_value",
        "//src/main/java/com/google/devtools/build/lib/actions",
    ],
)

java_library(
    name = "configured_target",
    srcs = ["ConfiguredTarget.java"],
    deps = [
        ":transitive_info_collection",
        "//src/main/java/com/google/devtools/build/lib/actions:action_lookup_key",
        "//src/main/java/com/google/devtools/build/lib/analysis/config:config_matching_provider",
        "//src/main/java/com/google/devtools/build/lib/cmdline",
        "//src/main/java/com/google/devtools/build/lib/query2/common:cquery-node",
        "//src/main/java/com/google/devtools/build/lib/skyframe/config",
        "//src/main/java/net/starlark/java/eval",
        "//third_party:guava",
        "//third_party:jsr305",
    ],
)

java_library(
    name = "configured_target_value",
    srcs = ["ConfiguredTargetValue.java"],
    deps = [
        ":configured_object_value",
        ":configured_target",
        "//src/main/java/com/google/devtools/build/lib/packages",
        "//third_party:jsr305",
    ],
)

java_library(
    name = "dependency_kind",
    srcs = ["DependencyKind.java"],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/cmdline",
        "//src/main/java/com/google/devtools/build/lib/packages",
        "//src/main/java/com/google/devtools/build/lib/packages:declared_exec_group",
        "//third_party:auto_value",
        "//third_party:guava",
        "//third_party:jsr305",
    ],
)

java_library(
    name = "exec_group_collection",
    srcs = ["ExecGroupCollection.java"],
    deps = [
        ":resolved_toolchain_context",
        ":toolchain_collection",
        "//src/main/java/com/google/devtools/build/lib/analysis/platform",
        "//src/main/java/com/google/devtools/build/lib/packages:declared_exec_group",
        "//src/main/java/com/google/devtools/build/lib/skyframe:sane_analysis_exception",
        "//src/main/java/com/google/devtools/build/lib/util:detailed_exit_code",
        "//src/main/java/net/starlark/java/spelling",
        "//src/main/protobuf:failure_details_java_proto",
        "//third_party:auto_value",
        "//third_party:guava",
        "//third_party:jsr305",
    ],
)

java_library(
    name = "extra_action_artifacts_provider",
    srcs = ["ExtraActionArtifactsProvider.java"],
    deps = [
        ":transitive_info_provider",
        "//src/main/java/com/google/devtools/build/lib/actions:artifacts",
        "//src/main/java/com/google/devtools/build/lib/collect/nestedset",
        "//src/main/java/com/google/devtools/build/lib/concurrent",
        "//src/main/java/com/google/devtools/build/lib/concurrent:thread_safety",
        "//src/main/java/com/google/devtools/build/lib/skyframe/serialization/autocodec",
        "//src/main/java/com/google/devtools/build/lib/skyframe/serialization/autocodec:serialization-constant",
    ],
)

java_library(
    name = "file_provider",
    srcs = ["FileProvider.java"],
    deps = [
        ":transitive_info_provider",
        "//src/main/java/com/google/devtools/build/lib/actions:artifacts",
        "//src/main/java/com/google/devtools/build/lib/collect/nestedset",
        "//src/main/java/com/google/devtools/build/lib/concurrent",
        "//src/main/java/com/google/devtools/build/lib/concurrent:thread_safety",
        "//src/main/java/com/google/devtools/build/lib/starlarkbuildapi",
        "//src/main/java/net/starlark/java/eval",
    ],
)

java_library(
    name = "incompatible_platform_provider",
    srcs = ["IncompatiblePlatformProvider.java"],
    deps = [
        ":configured_target",
        "//src/main/java/com/google/devtools/build/lib/analysis/platform",
        "//src/main/java/com/google/devtools/build/lib/cmdline",
        "//src/main/java/com/google/devtools/build/lib/concurrent",
        "//src/main/java/com/google/devtools/build/lib/concurrent:thread_safety",
        "//src/main/java/com/google/devtools/build/lib/packages",
        "//src/main/java/com/google/devtools/build/lib/skyframe/serialization/autocodec",
        "//src/main/java/com/google/devtools/build/lib/starlarkbuildapi/platform",
        "//third_party:auto_value",
        "//third_party:guava",
        "//third_party:jsr305",
    ],
)

java_library(
    name = "inconsistent_aspect_order_exception",
    srcs = ["InconsistentAspectOrderException.java"],
    deps = [
        ":aspect_collection",
        "//src/main/java/com/google/devtools/build/lib/cmdline",
        "//src/main/java/net/starlark/java/syntax",
    ],
)

java_library(
    name = "inconsistent_null_config_exception",
    srcs = ["InconsistentNullConfigException.java"],
)

java_library(
    name = "invalid_visibility_dependency_exception",
    srcs = ["InvalidVisibilityDependencyException.java"],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/cmdline",
        "//src/main/java/com/google/devtools/build/lib/packages",
    ],
)

java_library(
    name = "label_and_location",
    srcs = ["LabelAndLocation.java"],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/cmdline",
        "//src/main/java/com/google/devtools/build/lib/packages",
        "//src/main/java/com/google/devtools/build/lib/skyframe/serialization/autocodec",
        "//src/main/java/net/starlark/java/syntax",
        "//third_party:auto_value",
        "//third_party:error_prone_annotations",
    ],
)

java_library(
    name = "label_expander",
    srcs = ["LabelExpander.java"],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/actions:artifacts",
        "//src/main/java/com/google/devtools/build/lib/cmdline",
        "//src/main/java/com/google/devtools/build/lib/vfs:pathfragment",
        "//third_party:guava",
        "//third_party:jsr305",
    ],
)

java_library(
    name = "make_environment_event",
    srcs = ["MakeEnvironmentEvent.java"],
    deps = ["//third_party:guava"],
)

java_library(
    name = "make_variable_supplier",
    srcs = ["MakeVariableSupplier.java"],
    deps = [
        ":template_variable_info",
        "//src/main/java/com/google/devtools/build/lib/analysis/stringtemplate",
        "//third_party:guava",
        "//third_party:jsr305",
    ],
)

java_library(
    name = "no_build_event",
    srcs = ["NoBuildEvent.java"],
    deps = [
        ":blaze_version_info",
        "//src/main/java/com/google/devtools/build/lib/buildeventstream",
        "//src/main/java/com/google/devtools/build/lib/buildeventstream/proto:build_event_stream_java_proto",
        "//third_party:guava",
        "@com_google_protobuf//java/util",
    ],
)

java_library(
    name = "no_build_request_finished_event",
    srcs = ["NoBuildRequestFinishedEvent.java"],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/buildeventstream",
        "//src/main/java/com/google/devtools/build/lib/util:exit_code",
    ],
)

java_library(
    name = "options_diff_predicate",
    srcs = ["OptionsDiffPredicate.java"],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/analysis/config:build_options",
        "//src/main/java/com/google/devtools/common/options",
    ],
)

java_library(
    name = "platform_configuration",
    srcs = ["PlatformConfiguration.java"],
    deps = [
        ":platform_options",
        "//src/main/java/com/google/devtools/build/lib/analysis/config:build_options",
        "//src/main/java/com/google/devtools/build/lib/analysis/config:fragment",
        "//src/main/java/com/google/devtools/build/lib/analysis/config:fragment_options",
        "//src/main/java/com/google/devtools/build/lib/cmdline",
        "//src/main/java/com/google/devtools/build/lib/concurrent",
        "//src/main/java/com/google/devtools/build/lib/concurrent:thread_safety",
        "//src/main/java/com/google/devtools/build/lib/events",
        "//src/main/java/com/google/devtools/build/lib/starlarkbuildapi/platform",
        "//src/main/java/com/google/devtools/build/lib/util",
        "//third_party:guava",
    ],
)

java_library(
    name = "platform_options",
    srcs = ["PlatformOptions.java"],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/analysis/config:core_option_converters",
        "//src/main/java/com/google/devtools/build/lib/analysis/config:fragment_options",
        "//src/main/java/com/google/devtools/build/lib/cmdline",
        "//src/main/java/com/google/devtools/build/lib/skyframe/config:platform_mapping_key",
        "//src/main/java/com/google/devtools/build/lib/util",
        "//src/main/java/com/google/devtools/build/lib/vfs:pathfragment",
        "//src/main/java/com/google/devtools/common/options",
        "//third_party:guava",
        "//third_party:jsr305",
    ],
)

java_library(
    name = "print_action_visitor",
    srcs = ["PrintActionVisitor.java"],
    deps = [
        ":configured_target",
        "//src/main/java/com/google/devtools/build/lib/actions",
        "//third_party:guava",
    ],
)

java_library(
    name = "provider_collection",
    srcs = [
        "ProviderCollection.java",
    ],
    deps = [
        ":transitive_info_provider",
        "//src/main/java/com/google/devtools/build/lib/packages",
        "//third_party:jsr305",
    ],
)

java_library(
    name = "actions/buildinfo_file_write_action",
    srcs = ["actions/BuildInfoFileWriteAction.java"],
    deps = [
        ":actions/substitution",
        ":actions/template",
        ":actions/template_expansion_action",
        ":workspace_status_action",
        "//src/main/java/com/google/devtools/build/lib/actions",
        "//src/main/java/com/google/devtools/build/lib/actions:artifacts",
        "//src/main/java/com/google/devtools/build/lib/cmdline",
        "//src/main/java/com/google/devtools/build/lib/collect/nestedset",
        "//src/main/java/com/google/devtools/build/lib/skyframe:workspace_status_value",
        "//src/main/java/com/google/devtools/build/lib/util",
        "//src/main/java/com/google/devtools/build/lib/util:detailed_exit_code",
        "//src/main/java/net/starlark/java/eval",
        "//src/main/protobuf:failure_details_java_proto",
        "//third_party:guava",
        "//third_party:jsr305",
    ],
)

java_library(
    name = "repo_mapping_manifest_action",
    srcs = ["RepoMappingManifestAction.java"],
    deps = [
        ":actions/abstract_file_write_action",
        ":symlink_entry",
        "//src/main/java/com/google/devtools/build/lib/actions",
        "//src/main/java/com/google/devtools/build/lib/actions:artifacts",
        "//src/main/java/com/google/devtools/build/lib/actions:commandline_item",
        "//src/main/java/com/google/devtools/build/lib/cmdline",
        "//src/main/java/com/google/devtools/build/lib/collect/nestedset",
        "//src/main/java/com/google/devtools/build/lib/events",
        "//src/main/java/com/google/devtools/build/lib/packages",
        "//src/main/java/com/google/devtools/build/lib/util",
        "//src/main/java/net/starlark/java/eval",
        "//third_party:caffeine",
        "//third_party:guava",
        "//third_party:jsr305",
    ],
)

java_library(
    name = "required_config_fragments_provider",
    srcs = ["RequiredConfigFragmentsProvider.java"],
    deps = [
        ":transitive_info_provider",
        "//src/main/java/com/google/devtools/build/lib/analysis/config:fragment",
        "//src/main/java/com/google/devtools/build/lib/analysis/config:fragment_options",
        "//src/main/java/com/google/devtools/build/lib/cmdline",
        "//src/main/java/com/google/devtools/build/lib/concurrent",
        "//src/main/java/com/google/devtools/build/lib/concurrent:thread_safety",
        "//src/main/java/com/google/devtools/build/lib/skyframe/serialization/autocodec",
        "//src/main/java/com/google/devtools/build/lib/skyframe/serialization/autocodec:serialization-constant",
        "//src/main/java/com/google/devtools/build/lib/util",
        "//third_party:auto_value",
        "//third_party:error_prone_annotations",
        "//third_party:guava",
    ],
)

java_library(
    name = "resolved_toolchain_context",
    srcs = ["ResolvedToolchainContext.java"],
    deps = [
        ":resolved_toolchain_data_interface",
        ":template_variable_info",
        "//src/main/java/com/google/devtools/build/lib/analysis/config:toolchain_type_requirement",
        "//src/main/java/com/google/devtools/build/lib/analysis/platform",
        "//src/main/java/com/google/devtools/build/lib/analysis/platform:utils",
        "//src/main/java/com/google/devtools/build/lib/cmdline",
        "//src/main/java/com/google/devtools/build/lib/concurrent",
        "//src/main/java/com/google/devtools/build/lib/concurrent:thread_safety",
        "//src/main/java/com/google/devtools/build/lib/skyframe:configured_target_and_data",
        "//src/main/java/com/google/devtools/build/lib/skyframe/toolchains:toolchain_context_key",
        "//src/main/java/com/google/devtools/build/lib/skyframe/toolchains:toolchain_exception",
        "//src/main/java/com/google/devtools/build/lib/skyframe/toolchains:unloaded_toolchain_context",
        "//src/main/protobuf:failure_details_java_proto",
        "//third_party:auto_value",
        "//third_party:guava",
        "//third_party:jsr305",
    ],
)

java_library(
    name = "resolved_toolchain_data_interface",
    srcs = ["ResolvedToolchainsDataInterface.java"],
    deps = [
        ":resolved_toolchain_data",
        ":toolchain_context",
        "//src/main/java/com/google/devtools/build/lib/analysis/platform",
        "//src/main/java/com/google/devtools/build/lib/cmdline",
        "//third_party:guava",
        "//third_party:jsr305",
    ],
)

java_library(
    name = "resolved_toolchain_data",
    srcs = ["ResolvedToolchainData.java"],
    deps = ["//src/main/java/net/starlark/java/eval"],
)

java_library(
    name = "run_environment_info",
    srcs = ["RunEnvironmentInfo.java"],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/concurrent",
        "//src/main/java/com/google/devtools/build/lib/concurrent:thread_safety",
        "//src/main/java/com/google/devtools/build/lib/packages",
        "//src/main/java/com/google/devtools/build/lib/starlarkbuildapi",
        "//src/main/java/net/starlark/java/eval",
        "//third_party:guava",
    ],
)

java_library(
    name = "rule_definition_environment",
    srcs = ["RuleDefinitionEnvironment.java"],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/cmdline",
    ],
)

java_library(
    name = "rule_error_consumer",
    srcs = ["RuleErrorConsumer.java"],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/packages",
    ],
)

java_library(
    name = "server_directories",
    srcs = ["ServerDirectories.java"],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/concurrent",
        "//src/main/java/com/google/devtools/build/lib/concurrent:thread_safety",
        "//src/main/java/com/google/devtools/build/lib/vfs",
        "//third_party:guava",
        "//third_party:jsr305",
    ],
)

java_library(
    name = "symlink_entry",
    srcs = ["SymlinkEntry.java"],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/actions:artifacts",
        "//src/main/java/com/google/devtools/build/lib/starlarkbuildapi",
        "//src/main/java/com/google/devtools/build/lib/vfs:pathfragment",
        "//src/main/java/net/starlark/java/eval",
        "//third_party:guava",
    ],
)

java_library(
    name = "target_and_configuration",
    srcs = ["TargetAndConfiguration.java"],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/analysis/config:build_configuration",
        "//src/main/java/com/google/devtools/build/lib/cmdline",
        "//src/main/java/com/google/devtools/build/lib/concurrent",
        "//src/main/java/com/google/devtools/build/lib/concurrent:thread_safety",
        "//src/main/java/com/google/devtools/build/lib/packages",
        "//third_party:guava",
        "//third_party:jsr305",
    ],
)

java_library(
    name = "target_configured_event",
    srcs = ["TargetConfiguredEvent.java"],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/analysis/config:build_configuration",
        "//src/main/java/com/google/devtools/build/lib/buildeventstream",
        "//src/main/java/com/google/devtools/build/lib/buildeventstream/proto:build_event_stream_java_proto",
        "//src/main/java/com/google/devtools/build/lib/packages",
        "//third_party:flogger",
        "//third_party:guava",
        "//third_party:jsr305",
    ],
)

java_library(
    name = "template_variable_info",
    srcs = ["TemplateVariableInfo.java"],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/concurrent",
        "//src/main/java/com/google/devtools/build/lib/concurrent:thread_safety",
        "//src/main/java/com/google/devtools/build/lib/packages",
        "//src/main/java/com/google/devtools/build/lib/skyframe/serialization/autocodec",
        "//src/main/java/com/google/devtools/build/lib/skyframe/serialization/autocodec:serialization-constant",
        "//src/main/java/com/google/devtools/build/lib/starlarkbuildapi",
        "//src/main/java/net/starlark/java/eval",
        "//src/main/java/net/starlark/java/syntax",
        "//third_party:guava",
    ],
)

java_library(
    name = "toolchain_collection",
    srcs = ["ToolchainCollection.java"],
    deps = [
        ":toolchain_context",
        "//src/main/java/com/google/devtools/build/lib/analysis/platform",
        "//src/main/java/com/google/devtools/build/lib/cmdline",
        "//src/main/java/com/google/devtools/build/lib/packages:declared_exec_group",
        "//src/main/java/com/google/devtools/build/lib/skyframe/serialization/autocodec",
        "//third_party:auto_value",
        "//third_party:error_prone_annotations",
        "//third_party:guava",
    ],
)

java_library(
    name = "toolchain_context",
    srcs = ["ToolchainContext.java"],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/analysis/config:toolchain_type_requirement",
        "//src/main/java/com/google/devtools/build/lib/analysis/platform",
        "//src/main/java/com/google/devtools/build/lib/cmdline",
        "//src/main/java/com/google/devtools/build/lib/skyframe/toolchains:toolchain_context_key",
        "//third_party:guava",
    ],
)

java_library(
    name = "top_level_artifact_context",
    srcs = ["TopLevelArtifactContext.java"],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/concurrent",
        "//src/main/java/com/google/devtools/build/lib/concurrent:thread_safety",
        "//third_party:guava",
    ],
)

java_library(
    name = "transitive_info_collection",
    srcs = ["TransitiveInfoCollection.java"],
    deps = [
        ":provider_collection",
        ":transitive_info_provider",
        "//src/main/java/com/google/devtools/build/lib/cmdline",
        "//src/main/java/com/google/devtools/build/lib/packages",
        "//src/main/java/com/google/devtools/build/lib/starlarkbuildapi/core",
        "//src/main/java/net/starlark/java/eval",
    ],
)

java_library(
    name = "transitive_info_provider",
    srcs = ["TransitiveInfoProvider.java"],
)

java_library(
    name = "transitive_info_provider_effective_class_helper",
    srcs = ["TransitiveInfoProviderEffectiveClassHelper.java"],
    deps = [
        ":transitive_info_provider",
        "//third_party:caffeine",
        "//third_party:guava",
    ],
)

java_library(
    name = "transitive_info_provider_map",
    srcs = ["TransitiveInfoProviderMap.java"],
    deps = [
        ":provider_collection",
        "//third_party:jsr305",
    ],
)

java_library(
    name = "transitive_info_provider_map_builder",
    srcs = ["TransitiveInfoProviderMapBuilder.java"],
    deps = [
        ":transitive_info_provider",
        ":transitive_info_provider_effective_class_helper",
        ":transitive_info_provider_map",
        ":transitive_info_provider_map_impl",
        "//src/main/java/com/google/devtools/build/lib/packages",
        "//third_party:guava",
        "//third_party:jsr305",
    ],
)

java_library(
    name = "transitive_info_provider_map_impl",
    srcs = ["TransitiveInfoProviderMapImpl.java"],
    deps = [
        ":transitive_info_provider",
        ":transitive_info_provider_effective_class_helper",
        ":transitive_info_provider_map",
        "//src/main/java/com/google/devtools/build/lib/collect:immutable_shared_key_map",
        "//src/main/java/com/google/devtools/build/lib/packages",
        "//src/main/java/com/google/devtools/build/lib/skyframe/serialization:visible-for-serialization",
        "//src/main/java/com/google/devtools/build/lib/skyframe/serialization/autocodec",
        "//src/main/java/com/google/devtools/build/lib/skyframe/serialization/autocodec:serialization-constant",
        "//third_party:error_prone_annotations",
        "//third_party:guava",
        "//third_party:jsr305",
        "@com_google_protobuf//:protobuf_java",
    ],
)

java_library(
    name = "view_creation_failed_exception",
    srcs = ["ViewCreationFailedException.java"],
    deps = [
        "//src/main/protobuf:failure_details_java_proto",
        "//third_party:guava",
    ],
)

java_library(
    name = "visibility_provider",
    srcs = ["VisibilityProvider.java"],
    deps = [
        ":transitive_info_provider",
        "//src/main/java/com/google/devtools/build/lib/collect/nestedset",
        "//src/main/java/com/google/devtools/build/lib/packages:package_specification",
        "//third_party:guava",
    ],
)

java_library(
    name = "visibility_provider_impl",
    srcs = ["VisibilityProviderImpl.java"],
    deps = [
        ":visibility_provider",
        "//src/main/java/com/google/devtools/build/lib/collect/nestedset",
        "//src/main/java/com/google/devtools/build/lib/concurrent",
        "//src/main/java/com/google/devtools/build/lib/concurrent:thread_safety",
        "//src/main/java/com/google/devtools/build/lib/packages:package_specification",
    ],
)

java_library(
    name = "workspace_status_action",
    srcs = ["WorkspaceStatusAction.java"],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/actions",
        "//src/main/java/com/google/devtools/build/lib/actions:artifacts",
        "//src/main/java/com/google/devtools/build/lib/collect/nestedset",
        "//src/main/java/com/google/devtools/build/lib/shell",
        "//src/main/java/com/google/devtools/build/lib/skyframe:workspace_info",
        "//src/main/java/com/google/devtools/build/lib/util",
        "//src/main/java/com/google/devtools/build/lib/util:detailed_exit_code",
        "//src/main/java/com/google/devtools/build/lib/vfs",
        "//src/main/java/com/google/devtools/build/lib/vfs:pathfragment",
        "//src/main/java/com/google/devtools/common/options",
        "//src/main/protobuf:failure_details_java_proto",
        "//third_party:guava",
        "//third_party:jsr305",
    ],
)

java_library(
    name = "starlark/starlark_build_settings_details_value",
    srcs = [
        "starlark/StarlarkBuildSettingsDetailsValue.java",
    ],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/cmdline",
        "//src/main/java/com/google/devtools/build/lib/concurrent:thread_safety",
        "//src/main/java/com/google/devtools/build/lib/packages",
        "//src/main/java/com/google/devtools/build/lib/skyframe:sky_functions",
        "//src/main/java/com/google/devtools/build/lib/skyframe/serialization/autocodec",
        "//src/main/java/com/google/devtools/build/skyframe:skyframe-objects",
        "//third_party:auto_value",
        "//third_party:error_prone_annotations",
        "//third_party:guava",
    ],
)

# TODO(b/144899336): This should be analysis/actions/BUILD
java_library(
    name = "actions/abstract_file_write_action",
    srcs = ["actions/AbstractFileWriteAction.java"],
    deps = [
        ":actions/file_write_action_context",
        "//src/main/java/com/google/devtools/build/lib/actions",
        "//src/main/java/com/google/devtools/build/lib/actions:artifacts",
        "//src/main/java/com/google/devtools/build/lib/collect/nestedset",
        "//src/main/java/com/google/devtools/build/lib/events",
        "//src/main/java/com/google/devtools/build/lib/profiler",
        "//src/main/java/com/google/devtools/build/lib/util:deterministic_writer",
        "//third_party:guava",
        "//third_party:jsr305",
    ],
)

java_library(
    name = "actions/binary_file_write_action",
    srcs = ["actions/BinaryFileWriteAction.java"],
    deps = [
        ":actions/abstract_file_write_action",
        "//src/main/java/com/google/devtools/build/lib/actions",
        "//src/main/java/com/google/devtools/build/lib/actions:artifacts",
        "//src/main/java/com/google/devtools/build/lib/collect/nestedset",
        "//src/main/java/com/google/devtools/build/lib/concurrent",
        "//src/main/java/com/google/devtools/build/lib/concurrent:thread_safety",
        "//src/main/java/com/google/devtools/build/lib/util",
        "//third_party:guava",
        "//third_party:jsr305",
    ],
)

java_library(
    name = "actions/compression",
    srcs = ["actions/Compression.java"],
)

java_library(
    name = "actions/custom_command_line",
    srcs = ["actions/CustomCommandLine.java"],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/actions",
        "//src/main/java/com/google/devtools/build/lib/actions:artifacts",
        "//src/main/java/com/google/devtools/build/lib/actions:commandline_item",
        "//src/main/java/com/google/devtools/build/lib/analysis/config:core_options",
        "//src/main/java/com/google/devtools/build/lib/cmdline",
        "//src/main/java/com/google/devtools/build/lib/collect/nestedset",
        "//src/main/java/com/google/devtools/build/lib/concurrent",
        "//src/main/java/com/google/devtools/build/lib/concurrent:thread_safety",
        "//src/main/java/com/google/devtools/build/lib/skyframe:tree_artifact_value",
        "//src/main/java/com/google/devtools/build/lib/skyframe/serialization:visible-for-serialization",
        "//src/main/java/com/google/devtools/build/lib/skyframe/serialization/autocodec",
        "//src/main/java/com/google/devtools/build/lib/skyframe/serialization/autocodec:serialization-constant",
        "//src/main/java/com/google/devtools/build/lib/util",
        "//src/main/java/com/google/devtools/build/lib/vfs:pathfragment",
        "//third_party:auto_value",
        "//third_party:guava",
        "//third_party:jsr305",
    ],
)

java_library(
    name = "actions/file_write_action_context",
    srcs = ["actions/FileWriteActionContext.java"],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/actions",
        "//src/main/java/com/google/devtools/build/lib/actions:artifacts",
        "//src/main/java/com/google/devtools/build/lib/util:deterministic_writer",
        "//third_party:guava",
    ],
)

java_library(
    name = "actions/lazy_write_nested_set_of_tuple_action",
    srcs = ["actions/LazyWriteNestedSetOfTupleAction.java"],
    deps = [
        ":actions/abstract_file_write_action",
        "//src/main/java/com/google/devtools/build/lib/actions",
        "//src/main/java/com/google/devtools/build/lib/actions:artifacts",
        "//src/main/java/com/google/devtools/build/lib/actions:commandline_item",
        "//src/main/java/com/google/devtools/build/lib/collect/nestedset",
        "//src/main/java/com/google/devtools/build/lib/unsafe:string",
        "//src/main/java/com/google/devtools/build/lib/util",
        "//src/main/java/com/google/devtools/build/lib/util:deterministic_writer",
        "//src/main/java/net/starlark/java/eval",
        "//third_party:jsr305",
    ],
)

java_library(
    name = "actions/lazy_write_paths_file_action",
    srcs = ["actions/LazyWritePathsFileAction.java"],
    deps = [
        ":actions/abstract_file_write_action",
        "//src/main/java/com/google/devtools/build/lib/actions",
        "//src/main/java/com/google/devtools/build/lib/actions:artifacts",
        "//src/main/java/com/google/devtools/build/lib/collect/nestedset",
        "//src/main/java/com/google/devtools/build/lib/unsafe:string",
        "//src/main/java/com/google/devtools/build/lib/util",
        "//src/main/java/com/google/devtools/build/lib/util:deterministic_writer",
        "//third_party:guava",
        "//third_party:jsr305",
    ],
)

java_library(
    name = "actions/parameter_file_write_action",
    srcs = ["actions/ParameterFileWriteAction.java"],
    deps = [
        ":actions/abstract_file_write_action",
        ":actions/path_mappers",
        "//src/main/java/com/google/devtools/build/lib/actions",
        "//src/main/java/com/google/devtools/build/lib/actions:artifacts",
        "//src/main/java/com/google/devtools/build/lib/actions:commandline_item",
        "//src/main/java/com/google/devtools/build/lib/actions:execution_requirements",
        "//src/main/java/com/google/devtools/build/lib/analysis/config:build_configuration",
        "//src/main/java/com/google/devtools/build/lib/analysis/config:core_options",
        "//src/main/java/com/google/devtools/build/lib/collect/nestedset",
        "//src/main/java/com/google/devtools/build/lib/concurrent:thread_safety",
        "//src/main/java/com/google/devtools/build/lib/util",
        "//src/main/java/com/google/devtools/build/lib/util:deterministic_writer",
        "//src/main/java/net/starlark/java/eval",
        "//src/main/protobuf:failure_details_java_proto",
        "//third_party:guava",
        "//third_party:jsr305",
    ],
)

java_library(
    name = "actions/path_mappers",
    srcs = [
        "actions/PathMappers.java",
        "actions/StrippingPathMapper.java",
    ],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/actions",
        "//src/main/java/com/google/devtools/build/lib/actions:action_input_helper",
        "//src/main/java/com/google/devtools/build/lib/actions:artifacts",
        "//src/main/java/com/google/devtools/build/lib/actions:commandline_item",
        "//src/main/java/com/google/devtools/build/lib/actions:execution_requirements",
        "//src/main/java/com/google/devtools/build/lib/analysis/config:build_configuration",
        "//src/main/java/com/google/devtools/build/lib/analysis/config:core_options",
        "//src/main/java/com/google/devtools/build/lib/collect/nestedset",
        "//src/main/java/com/google/devtools/build/lib/starlarkbuildapi",
        "//src/main/java/com/google/devtools/build/lib/util",
        "//src/main/java/com/google/devtools/build/lib/vfs:pathfragment",
        "//third_party:guava",
        "//third_party:jsr305",
    ],
)

java_library(
    name = "actions/spawn_action_template",
    srcs = ["actions/SpawnActionTemplate.java"],
    deps = [
        ":actions/custom_command_line",
        ":analysis_cluster",
        "//src/main/java/com/google/devtools/build/lib/actions",
        "//src/main/java/com/google/devtools/build/lib/actions:action_lookup_key",
        "//src/main/java/com/google/devtools/build/lib/actions:artifacts",
        "//src/main/java/com/google/devtools/build/lib/actions:commandline_item",
        "//src/main/java/com/google/devtools/build/lib/actions:localhost_capacity",
        "//src/main/java/com/google/devtools/build/lib/analysis/config:build_configuration",
        "//src/main/java/com/google/devtools/build/lib/analysis/config:core_options",
        "//src/main/java/com/google/devtools/build/lib/collect/nestedset",
        "//src/main/java/com/google/devtools/build/lib/skyframe:action_template_expansion_value",
        "//src/main/java/com/google/devtools/build/lib/util",
        "//src/main/java/com/google/devtools/build/lib/vfs:pathfragment",
        "//third_party:error_prone_annotations",
        "//third_party:guava",
        "//third_party:jsr305",
    ],
)

java_library(
    name = "actions/substitution",
    srcs = ["actions/Substitution.java"],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/concurrent",
        "//src/main/java/com/google/devtools/build/lib/concurrent:thread_safety",
        "//src/main/java/net/starlark/java/eval",
        "//third_party:guava",
        "//third_party:jsr305",
    ],
)

java_library(
    name = "actions/symlink_action",
    srcs = ["actions/SymlinkAction.java"],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/actions",
        "//src/main/java/com/google/devtools/build/lib/actions:artifacts",
        "//src/main/java/com/google/devtools/build/lib/actions:file_metadata",
        "//src/main/java/com/google/devtools/build/lib/actions:fileset_output_tree",
        "//src/main/java/com/google/devtools/build/lib/analysis/platform",
        "//src/main/java/com/google/devtools/build/lib/collect/nestedset",
        "//src/main/java/com/google/devtools/build/lib/exec:spawn_log_context",
        "//src/main/java/com/google/devtools/build/lib/unix",
        "//src/main/java/com/google/devtools/build/lib/util",
        "//src/main/java/com/google/devtools/build/lib/util:detailed_exit_code",
        "//src/main/java/com/google/devtools/build/lib/vfs",
        "//src/main/java/com/google/devtools/build/lib/vfs:pathfragment",
        "//src/main/protobuf:failure_details_java_proto",
        "//third_party:guava",
        "//third_party:jsr305",
    ],
)

java_library(
    name = "actions/template",
    srcs = ["actions/Template.java"],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/actions:artifacts",
        "//src/main/java/com/google/devtools/build/lib/concurrent",
        "//src/main/java/com/google/devtools/build/lib/concurrent:thread_safety",
        "//src/main/java/com/google/devtools/build/lib/util",
        "//src/main/java/com/google/devtools/build/lib/vfs",
        "//third_party:guava",
        "//third_party:jsr305",
    ],
)

java_library(
    name = "template_expansion_exception",
    srcs = ["actions/TemplateExpansionException.java"],
    deps = [
        "//src/main/java/net/starlark/java/eval",
    ],
)

java_library(
    name = "actions/template_expansion_action",
    srcs = [
        "actions/LocalTemplateExpansionStrategy.java",
        "actions/TemplateExpansionAction.java",
        "actions/TemplateExpansionContext.java",
        "actions/TemplateExpansionException.java",
    ],
    deps = [
        ":actions/file_write_action_context",
        ":actions/substitution",
        ":actions/template",
        "//src/main/java/com/google/devtools/build/lib/actions",
        "//src/main/java/com/google/devtools/build/lib/actions:artifacts",
        "//src/main/java/com/google/devtools/build/lib/collect/nestedset",
        "//src/main/java/com/google/devtools/build/lib/concurrent",
        "//src/main/java/com/google/devtools/build/lib/concurrent:thread_safety",
        "//src/main/java/com/google/devtools/build/lib/util",
        "//src/main/java/com/google/devtools/build/lib/util:detailed_exit_code",
        "//src/main/java/com/google/devtools/build/lib/util:deterministic_writer",
        "//src/main/java/com/google/devtools/build/lib/util:string",
        "//src/main/java/net/starlark/java/eval",
        "//src/main/protobuf:failure_details_java_proto",
        "//third_party:auto_value",
        "//third_party:guava",
        "//third_party:jsr305",
    ],
)

# TODO(b/144899336): This should be analysis/constraints/BUILD
java_library(
    name = "constraints/constraint_constants",
    srcs = ["constraints/ConstraintConstants.java"],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/analysis/platform",
        "//src/main/java/com/google/devtools/build/lib/cmdline",
        "//src/main/java/com/google/devtools/build/lib/util:os",
        "//third_party:guava",
    ],
)

java_library(
    name = "constraints/constraint_semantics",
    srcs = ["constraints/ConstraintSemantics.java"],
    deps = [
        ":constraints/constraint_constants",
        ":constraints/environment_collection",
        ":constraints/supported_environments_provider",
        "//src/main/java/com/google/devtools/build/lib/cmdline",
        "//src/main/java/com/google/devtools/build/lib/packages",
        "//src/main/java/com/google/devtools/build/lib/skyframe:detailed_exceptions",
        "//src/main/java/com/google/devtools/build/lib/util:detailed_exit_code",
        "//src/main/protobuf:failure_details_java_proto",
        "//third_party:jsr305",
    ],
)

java_library(
    name = "constraints/environment",
    srcs = ["constraints/Environment.java"],
    deps = [
        ":analysis_cluster",
        ":configured_target",
        ":constraints/constraint_semantics",
        ":constraints/environment_collection",
        ":constraints/supported_environments",
        ":constraints/supported_environments_provider",
        ":file_provider",
        "//src/main/java/com/google/devtools/build/lib/actions",
        "//src/main/java/com/google/devtools/build/lib/cmdline",
        "//src/main/java/com/google/devtools/build/lib/packages",
        "//third_party:guava",
        "//third_party:jsr305",
    ],
)

java_library(
    name = "constraints/environment_collection",
    srcs = ["constraints/EnvironmentCollection.java"],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/cmdline",
        "//src/main/java/com/google/devtools/build/lib/concurrent",
        "//src/main/java/com/google/devtools/build/lib/concurrent:thread_safety",
        "//src/main/java/com/google/devtools/build/lib/packages",
        "//src/main/java/com/google/devtools/build/lib/skyframe/serialization/autocodec",
        "//src/main/java/com/google/devtools/build/lib/skyframe/serialization/autocodec:serialization-constant",
        "//third_party:auto_value",
        "//third_party:guava",
    ],
)

java_library(
    name = "constraints/environment_rule",
    srcs = ["constraints/EnvironmentRule.java"],
    deps = [
        ":analysis_cluster",
        ":constraints/constraint_constants",
        ":constraints/environment",
        ":rule_definition_environment",
        "//src/main/java/com/google/devtools/build/lib/analysis/config/transitions:no_config_transition",
        "//src/main/java/com/google/devtools/build/lib/packages",
        "//src/main/java/com/google/devtools/build/lib/util:filetype",
        "//third_party:guava",
    ],
)

java_library(
    name = "constraints/supported_environments",
    srcs = ["constraints/SupportedEnvironments.java"],
    deps = [
        ":constraints/environment_collection",
        ":constraints/supported_environments_provider",
        "//src/main/java/com/google/devtools/build/lib/cmdline",
        "//third_party:guava",
    ],
)

java_library(
    name = "constraints/supported_environments_provider",
    srcs = ["constraints/SupportedEnvironmentsProvider.java"],
    deps = [
        ":constraints/environment_collection",
        ":label_and_location",
        ":transitive_info_provider",
        "//src/main/java/com/google/devtools/build/lib/cmdline",
        "//src/main/java/com/google/devtools/build/lib/skyframe/serialization/autocodec",
        "//third_party:auto_value",
    ],
)

java_library(
    name = "constraints/platform_restrictions_result",
    srcs = ["constraints/PlatformRestrictionsResult.java"],
    deps = [
        ":configured_target",
        "//third_party:auto_value",
        "//third_party:guava",
    ],
)

java_library(
    name = "constraints/top_level_constraint_semantics",
    srcs = [
        "constraints/PlatformRestrictionsResult.java",
        "constraints/TopLevelConstraintSemantics.java",
    ],
    deps = [
        ":analysis_cluster",
        ":configured_target",
        ":constraints/constraint_semantics",
        ":constraints/environment_collection",
        ":constraints/supported_environments_provider",
        ":incompatible_platform_provider",
        ":transitive_info_collection",
        ":view_creation_failed_exception",
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
        "//third_party:auto_value",
        "//third_party:guava",
        "//third_party:jsr305",
    ],
)

# TODO(b/144899336): This should be analysis/extra/BUILD
java_library(
    name = "extra/extra_action_info_file_write_action",
    srcs = ["extra/ExtraActionInfoFileWriteAction.java"],
    deps = [
        ":actions/abstract_file_write_action",
        "//src/main/java/com/google/devtools/build/lib/actions",
        "//src/main/java/com/google/devtools/build/lib/actions:artifacts",
        "//src/main/java/com/google/devtools/build/lib/actions:commandline_item",
        "//src/main/java/com/google/devtools/build/lib/collect/nestedset",
        "//src/main/java/com/google/devtools/build/lib/concurrent",
        "//src/main/java/com/google/devtools/build/lib/concurrent:thread_safety",
        "//src/main/java/com/google/devtools/build/lib/util",
        "//src/main/java/com/google/devtools/build/lib/util:deterministic_writer",
        "//src/main/java/com/google/devtools/build/lib/util:proto_deterministic_writer",
        "//src/main/protobuf:failure_details_java_proto",
        "//third_party:guava",
        "//third_party:jsr305",
    ],
)

java_library(
    name = "constraints/incompatible_target_checker",
    srcs = ["constraints/IncompatibleTargetChecker.java"],
    deps = [
        ":analysis_cluster",
        ":configured_target",
        ":configured_target_value",
        ":constraints/supported_environments",
        ":dependency_kind",
        ":file_provider",
        ":incompatible_platform_provider",
        ":target_and_configuration",
        ":test/test_configuration",
        ":transitive_dependency_state",
        ":transitive_info_provider_map_builder",
        ":visibility_provider",
        "//src/main/java/com/google/devtools/build/lib/analysis/config:build_configuration",
        "//src/main/java/com/google/devtools/build/lib/analysis/config:config_conditions",
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
        "//third_party:guava",
        "//third_party:jsr305",
    ],
)

# TODO(b/144899336): This should be analysis/starlark/BUILD
java_library(
    name = "starlark/args",
    srcs = ["starlark/Args.java"],
    deps = [
        ":starlark/starlark_custom_command_line",
        "//src/main/java/com/google/devtools/build/lib/actions",
        "//src/main/java/com/google/devtools/build/lib/actions:artifacts",
        "//src/main/java/com/google/devtools/build/lib/actions:commandline_item",
        "//src/main/java/com/google/devtools/build/lib/cmdline",
        "//src/main/java/com/google/devtools/build/lib/collect/nestedset",
        "//src/main/java/com/google/devtools/build/lib/concurrent:thread_safety",
        "//src/main/java/com/google/devtools/build/lib/starlarkbuildapi",
        "//src/main/java/com/google/devtools/build/lib/supplier",
        "//src/main/java/net/starlark/java/eval",
        "//src/main/java/net/starlark/java/syntax",
        "//third_party:error_prone_annotations",
        "//third_party:guava",
        "//third_party:jsr305",
    ],
)

java_library(
    name = "starlark/template_dict",
    srcs = ["starlark/TemplateDict.java"],
    deps = [
        ":actions/substitution",
        "//src/main/java/com/google/devtools/build/lib/collect/nestedset",
        "//src/main/java/com/google/devtools/build/lib/starlarkbuildapi",
        "//src/main/java/net/starlark/java/eval",
        "//third_party:guava",
        "//third_party:jsr305",
    ],
)

java_library(
    name = "starlark/bazel_build_api_globals",
    srcs = ["starlark/BazelBuildApiGlobals.java"],
    deps = [
        ":starlark/starlark_late_bound_default",
        "//src/main/java/com/google/devtools/build/lib/cmdline",
        "//src/main/java/com/google/devtools/build/lib/packages",
        "//src/main/java/com/google/devtools/build/lib/packages:bzl_visibility",
        "//src/main/java/com/google/devtools/build/lib/packages:package_specification",
        "//src/main/java/com/google/devtools/build/lib/packages/semantics",
        "//src/main/java/com/google/devtools/build/lib/starlarkbuildapi",
        "//src/main/java/net/starlark/java/eval",
        "//third_party:guava",
    ],
)

java_library(
    name = "starlark/function_transition_util",
    srcs = ["starlark/FunctionTransitionUtil.java"],
    deps = [
        ":test/test_configuration",
        "//src/main/java/com/google/devtools/build/lib/analysis/config:build_options",
        "//src/main/java/com/google/devtools/build/lib/analysis/config:core_option_converters",
        "//src/main/java/com/google/devtools/build/lib/analysis/config:core_options",
        "//src/main/java/com/google/devtools/build/lib/analysis/config:fragment_options",
        "//src/main/java/com/google/devtools/build/lib/analysis/config:optioninfo",
        "//src/main/java/com/google/devtools/build/lib/analysis/config:options_diff",
        "//src/main/java/com/google/devtools/build/lib/analysis/config:starlark_defined_config_transition",
        "//src/main/java/com/google/devtools/build/lib/analysis/config/transitions:configuration_transition",
        "//src/main/java/com/google/devtools/build/lib/cmdline",
        "//src/main/java/com/google/devtools/build/lib/events",
        "//src/main/java/com/google/devtools/build/lib/packages",
        "//src/main/java/com/google/devtools/common/options",
        "//src/main/java/net/starlark/java/eval",
        "//third_party:guava",
        "//third_party:jsr305",
    ],
)

java_library(
    name = "starlark/starlark_api_provider",
    srcs = ["starlark/StarlarkApiProvider.java"],
    deps = [
        ":provider_collection",
        "//third_party:guava",
    ],
)

java_library(
    name = "starlark/starlark_custom_command_line",
    srcs = ["starlark/StarlarkCustomCommandLine.java"],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/actions",
        "//src/main/java/com/google/devtools/build/lib/actions:artifacts",
        "//src/main/java/com/google/devtools/build/lib/actions:commandline_item",
        "//src/main/java/com/google/devtools/build/lib/actions:fileset_output_symlink",
        "//src/main/java/com/google/devtools/build/lib/actions:fileset_output_tree",
        "//src/main/java/com/google/devtools/build/lib/analysis/config:core_options",
        "//src/main/java/com/google/devtools/build/lib/cmdline",
        "//src/main/java/com/google/devtools/build/lib/collect/nestedset",
        "//src/main/java/com/google/devtools/build/lib/concurrent",
        "//src/main/java/com/google/devtools/build/lib/skyframe:tree_artifact_value",
        "//src/main/java/com/google/devtools/build/lib/skyframe/serialization:visible-for-serialization",
        "//src/main/java/com/google/devtools/build/lib/skyframe/serialization/autocodec",
        "//src/main/java/com/google/devtools/build/lib/skyframe/serialization/autocodec:serialization-constant",
        "//src/main/java/com/google/devtools/build/lib/starlarkbuildapi",
        "//src/main/java/com/google/devtools/build/lib/util",
        "//src/main/java/com/google/devtools/build/lib/util:hash_codes",
        "//src/main/java/com/google/devtools/build/lib/vfs:pathfragment",
        "//src/main/java/net/starlark/java/eval",
        "//src/main/java/net/starlark/java/syntax",
        "//third_party:error_prone_annotations",
        "//third_party:guava",
        "//third_party:jsr305",
    ],
)

java_library(
    name = "starlark/starlark_exec_group_collection",
    srcs = ["starlark/StarlarkExecGroupCollection.java"],
    deps = [
        ":resolved_toolchain_context",
        ":resolved_toolchain_data_interface",
        ":starlark/starlark_toolchain_context",
        ":toolchain_collection",
        "//src/main/java/com/google/devtools/build/lib/packages:declared_exec_group",
        "//src/main/java/com/google/devtools/build/lib/starlarkbuildapi/platform",
        "//src/main/java/net/starlark/java/eval",
        "//src/main/java/net/starlark/java/syntax",
        "//third_party:auto_value",
        "//third_party:guava",
    ],
)

java_library(
    name = "starlark/starlark_error_reporter",
    srcs = ["starlark/StarlarkErrorReporter.java"],
    deps = [
        ":rule_error_consumer",
        "//src/main/java/com/google/devtools/build/lib/packages",
        "//src/main/java/net/starlark/java/eval",
    ],
)

java_library(
    name = "dormant_dependency",
    srcs = ["DormantDependency.java"],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/cmdline",
        "//src/main/java/com/google/devtools/build/lib/skyframe/serialization",
        "//src/main/java/net/starlark/java/annot",
        "//src/main/java/net/starlark/java/eval",
        "//third_party:error_prone_annotations",
        "@com_google_protobuf//:protobuf_java",
    ],
)

java_library(
    name = "starlark/starlark_late_bound_default",
    srcs = ["starlark/StarlarkLateBoundDefault.java"],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/analysis/starlark/annotations",
        "//src/main/java/com/google/devtools/build/lib/cmdline",
        "//src/main/java/com/google/devtools/build/lib/packages",
        "//src/main/java/com/google/devtools/build/lib/starlarkbuildapi",
        "//src/main/java/net/starlark/java/annot",
        "//src/main/java/net/starlark/java/eval",
        "//third_party:caffeine",
        "//third_party:guava",
        "//third_party:jsr305",
    ],
)

java_library(
    name = "starlark/starlark_config",
    srcs = ["starlark/StarlarkConfig.java"],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/analysis/config:execution_transition_factory",
        "//src/main/java/com/google/devtools/build/lib/analysis/config/transitions:no_config_transition",
        "//src/main/java/com/google/devtools/build/lib/analysis/config/transitions:no_transition",
        "//src/main/java/com/google/devtools/build/lib/packages",
        "//src/main/java/com/google/devtools/build/lib/starlarkbuildapi/config",
        "//src/main/java/com/google/devtools/build/lib/starlarkbuildapi/config:configuration_transition_api",
        "//src/main/java/net/starlark/java/eval",
    ],
)

java_library(
    name = "starlark/starlark_toolchain_context",
    srcs = ["starlark/StarlarkToolchainContext.java"],
    deps = [
        ":resolved_toolchain_data",
        "//src/main/java/com/google/devtools/build/lib/analysis/platform",
        "//src/main/java/com/google/devtools/build/lib/cmdline",
        "//src/main/java/com/google/devtools/build/lib/packages",
        "//src/main/java/com/google/devtools/build/lib/starlarkbuildapi/platform",
        "//src/main/java/net/starlark/java/eval",
        "//third_party:auto_value",
        "//third_party:guava",
    ],
)

java_library(
    name = "starlark/starlark_transition",
    srcs = ["starlark/StarlarkTransition.java"],
    deps = [
        ":required_config_fragments_provider",
        ":starlark/starlark_build_settings_details_value",
        "//src/main/java/com/google/devtools/build/lib/analysis/config:build_option_details",
        "//src/main/java/com/google/devtools/build/lib/analysis/config:build_options",
        "//src/main/java/com/google/devtools/build/lib/analysis/config:fragment_options",
        "//src/main/java/com/google/devtools/build/lib/analysis/config:starlark_defined_config_transition",
        "//src/main/java/com/google/devtools/build/lib/analysis/config/transitions:configuration_transition",
        "//src/main/java/com/google/devtools/build/lib/cmdline",
        "//src/main/java/com/google/devtools/build/lib/packages",
        "//third_party:guava",
    ],
)

# TODO(b/144899336): This should be lib/analysis/test/BUILD
java_library(
    name = "test/analysis_failure",
    srcs = ["test/AnalysisFailure.java"],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/cmdline",
        "//src/main/java/com/google/devtools/build/lib/starlarkbuildapi/test",
        "//src/main/java/net/starlark/java/annot",
        "//src/main/java/net/starlark/java/eval",
        "//third_party:auto_value",
    ],
)

java_library(
    name = "test/analysis_failure_info",
    srcs = ["test/AnalysisFailureInfo.java"],
    deps = [
        ":test/analysis_failure",
        "//src/main/java/com/google/devtools/build/lib/collect/nestedset",
        "//src/main/java/com/google/devtools/build/lib/packages",
        "//src/main/java/com/google/devtools/build/lib/starlarkbuildapi/test",
    ],
)

java_library(
    name = "test/analysis_failure_propagation_exception",
    srcs = ["test/AnalysisFailurePropagationException.java"],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/cmdline",
        "//src/main/java/com/google/devtools/build/lib/skyframe:sane_analysis_exception",
        "//src/main/java/com/google/devtools/build/lib/util:detailed_exit_code",
        "//src/main/protobuf:failure_details_java_proto",
        "//third_party:guava",
    ],
)

java_library(
    name = "test/analysis_test_result_info",
    srcs = ["test/AnalysisTestResultInfo.java"],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/packages",
        "//src/main/java/com/google/devtools/build/lib/starlarkbuildapi/test",
    ],
)

java_library(
    name = "test/coverage_configuration",
    srcs = ["test/CoverageConfiguration.java"],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/analysis/config:build_options",
        "//src/main/java/com/google/devtools/build/lib/analysis/config:core_option_converters",
        "//src/main/java/com/google/devtools/build/lib/analysis/config:core_options",
        "//src/main/java/com/google/devtools/build/lib/analysis/config:fragment",
        "//src/main/java/com/google/devtools/build/lib/analysis/config:fragment_options",
        "//src/main/java/com/google/devtools/build/lib/analysis/starlark/annotations",
        "//src/main/java/com/google/devtools/build/lib/cmdline",
        "//src/main/java/com/google/devtools/build/lib/concurrent",
        "//src/main/java/com/google/devtools/build/lib/concurrent:thread_safety",
        "//src/main/java/com/google/devtools/build/lib/starlarkbuildapi/test",
        "//src/main/java/com/google/devtools/common/options",
        "//third_party:jsr305",
    ],
)

java_library(
    name = "test/coverage_report_action_factory",
    srcs = ["test/CoverageReportActionFactory.java"],
    deps = [
        ":blaze_directories",
        ":configured_target",
        "//src/main/java/com/google/devtools/build/lib/actions",
        "//src/main/java/com/google/devtools/build/lib/actions:action_lookup_key",
        "//src/main/java/com/google/devtools/build/lib/actions:artifacts",
        "//src/main/java/com/google/devtools/build/lib/collect/nestedset",
        "//src/main/java/com/google/devtools/build/lib/events",
        "//src/main/java/com/google/devtools/build/lib/skyframe:coverage_report_value",
        "//third_party:guava",
        "//third_party:jsr305",
    ],
)

java_library(
    name = "test/coverage_artifacts_known_event",
    srcs = ["test/CoverageArtifactsKnownEvent.java"],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/actions:artifacts",
        "//src/main/java/com/google/devtools/build/lib/events",
        "//third_party:auto_value",
        "//third_party:guava",
    ],
)

java_library(
    name = "test/execution_info",
    srcs = ["test/ExecutionInfo.java"],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/concurrent",
        "//src/main/java/com/google/devtools/build/lib/concurrent:thread_safety",
        "//src/main/java/com/google/devtools/build/lib/packages",
        "//src/main/java/com/google/devtools/build/lib/starlarkbuildapi/test",
        "//src/main/java/net/starlark/java/eval",
        "//third_party:guava",
    ],
)

java_library(
    name = "test/instrumented_files_info",
    srcs = ["test/InstrumentedFilesInfo.java"],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/actions:artifacts",
        "//src/main/java/com/google/devtools/build/lib/collect/nestedset",
        "//src/main/java/com/google/devtools/build/lib/packages",
        "//src/main/java/com/google/devtools/build/lib/starlarkbuildapi/test",
        "//src/main/java/net/starlark/java/eval",
        "//third_party:guava",
    ],
)

java_library(
    name = "test/test_configuration",
    srcs = [
        "test/TestConfiguration.java",
        "test/TestResourcesConverter.java",
    ],
    deps = [
        ":options_diff_predicate",
        ":test/coverage_configuration",
        ":test/test_sharding_strategy",
        "//src/main/java/com/google/devtools/build/lib/analysis/config:build_options",
        "//src/main/java/com/google/devtools/build/lib/analysis/config:core_option_converters",
        "//src/main/java/com/google/devtools/build/lib/analysis/config:core_options",
        "//src/main/java/com/google/devtools/build/lib/analysis/config:fragment",
        "//src/main/java/com/google/devtools/build/lib/analysis/config:fragment_options",
        "//src/main/java/com/google/devtools/build/lib/analysis/config:per_label_options",
        "//src/main/java/com/google/devtools/build/lib/analysis/config:run_under",
        "//src/main/java/com/google/devtools/build/lib/cmdline",
        "//src/main/java/com/google/devtools/build/lib/packages",
        "//src/main/java/com/google/devtools/build/lib/util",
        "//src/main/java/com/google/devtools/build/lib/util:resource_converter",
        "//src/main/java/com/google/devtools/common/options",
        "//third_party:guava",
    ],
)

java_library(
    name = "test/test_sharding_strategy",
    srcs = [
        "test/TestShardingStrategy.java",
        "test/TestShardingStrategyForced.java",
        "test/TestShardingStrategyNotForced.java",
    ],
    deps = [
        "//src/main/java/com/google/devtools/common/options",
        "//third_party:guava",
    ],
)

java_library(
    name = "test/test_trimming_transition_factory",
    srcs = ["test/TestTrimmingTransitionFactory.java"],
    deps = [
        ":analysis_cluster",
        ":test/test_configuration",
        ":test/test_trimming_logic",
        "//src/main/java/com/google/devtools/build/lib/analysis/config:build_options",
        "//src/main/java/com/google/devtools/build/lib/analysis/config:fragment_options",
        "//src/main/java/com/google/devtools/build/lib/analysis/config/transitions:no_transition",
        "//src/main/java/com/google/devtools/build/lib/analysis/config/transitions:patch_transition",
        "//src/main/java/com/google/devtools/build/lib/analysis/config/transitions:transition_factory",
        "//src/main/java/com/google/devtools/build/lib/cmdline",
        "//src/main/java/com/google/devtools/build/lib/events",
        "//src/main/java/com/google/devtools/build/lib/packages",
        "//src/main/java/com/google/devtools/common/options",
        "//third_party:guava",
    ],
)

java_library(
    name = "test/test_trimming_logic",
    srcs = ["test/TestTrimmingLogic.java"],
    deps = [
        ":test/test_configuration",
        "//src/main/java/com/google/devtools/build/lib/analysis/config:build_options",
        "//src/main/java/com/google/devtools/build/lib/analysis/config:build_options_cache",
        "//src/main/java/com/google/devtools/build/lib/analysis/config:core_options",
        "//src/main/java/com/google/devtools/build/lib/analysis/config:fragment_options",
        "//src/main/java/com/google/devtools/build/lib/analysis/config:run_under",
        "//third_party:guava",
    ],
)

java_library(
    name = "test/test_progress",
    srcs = ["test/TestProgress.java"],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/buildeventstream",
        "//src/main/java/com/google/devtools/build/lib/buildeventstream/proto:build_event_stream_java_proto",
        "//src/main/java/com/google/devtools/build/lib/concurrent",
        "//src/main/java/com/google/devtools/build/lib/concurrent:thread_safety",
        "//third_party:guava",
        "//third_party:jsr305",
    ],
)

java_library(
    name = "transitive_dependency_state",
    srcs = ["TransitiveDependencyState.java"],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/causes",
        "//src/main/java/com/google/devtools/build/lib/cmdline",
        "//src/main/java/com/google/devtools/build/lib/collect/nestedset",
        "//src/main/java/com/google/devtools/build/lib/packages",
        "//src/main/java/com/google/devtools/build/lib/skyframe:aspect_key_creator",
        "//src/main/java/com/google/devtools/build/lib/skyframe:configured_target_and_data",
        "//src/main/java/com/google/devtools/build/lib/skyframe:configured_target_key",
        "//src/main/java/com/google/devtools/build/lib/skyframe:prerequisite_package_function",
        "//src/main/java/com/google/devtools/build/lib/skyframe/config",
        "//third_party:guava",
        "//third_party:jsr305",
    ],
)
