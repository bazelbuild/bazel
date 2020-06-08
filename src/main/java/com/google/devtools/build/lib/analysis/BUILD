load("@rules_java//java:defs.bzl", "java_library")

package(default_visibility = ["//src:__subpackages__"])

filegroup(
    name = "srcs",
    srcs = glob(["**"]) + [
        "//src/main/java/com/google/devtools/build/lib/analysis/platform:srcs",
        "//src/main/java/com/google/devtools/build/lib/analysis/stringtemplate:srcs",
        "//src/main/java/com/google/devtools/build/lib/analysis/skylark/annotations:srcs",
    ],
    visibility = ["//src:__subpackages__"],
)

# TODO(b/144899336): Remove this when all dependers are updated.
java_library(
    name = "legacy_exports",
    exports = [
        ":actions/abstract_file_write_action",
        ":actions/binary_file_write_action",
        ":actions/byte_string_deterministic_writer",
        ":actions/compression",
        ":actions/custom_command_line",
        ":actions/deterministic_writer",
        ":actions/file_write_action_context",
        ":actions/launcher_file_write_action",
        ":actions/lazy_write_nested_set_of_pair_action",
        ":actions/lazy_write_paths_file_action",
        ":actions/parameter_file_write_action",
        ":actions/proto_deterministic_writer",
        ":actions/spawn_action_template",
        ":actions/substitution",
        ":actions/symlink_action",
        ":actions/template",
        ":actions/template_expansion_action",
        ":actions_provider",
        ":analysis_cluster",
        ":analysis_options",
        ":analysis_phase_complete_event",
        ":analysis_phase_started_event",
        ":artifacts_to_owner_labels",
        ":aspect_aware_attribute_mapper",
        ":aspect_collection",
        ":aspect_configured_event",
        ":blaze_directories",
        ":blaze_version_info",
        ":build_info",
        ":build_info_event",
        ":build_setting_provider",
        ":build_view",
        ":buildinfo/build_info_collection",
        ":buildinfo/build_info_key",
        ":common_prerequisite_validator",
        ":compilation_helper",
        ":config/auto_cpu_converter",
        ":config/build_configuration",
        ":config/build_configuration_option_details",
        ":config/build_options",
        ":config/build_options_cache",
        ":config/compilation_mode",
        ":config/config_aware_aspect_builder",
        ":config/config_matching_provider",
        ":config/core_option_converters",
        ":config/core_options",
        ":config/execution_info_modifier",
        ":config/execution_transition_factory",
        ":config/fragment",
        ":config/fragment_options",
        ":config/fragment_provider",
        ":config/host_transition",
        ":config/invalid_configuration_exception",
        ":config/per_label_options",
        ":config/run_under",
        ":config/run_under_converter",
        ":config/starlark_defined_config_transition",
        ":config/transition_factories",
        ":config/transitions/composing_transition",
        ":config/transitions/composing_transition_factory",
        ":config/transitions/configuration_transition",
        ":config/transitions/no_transition",
        ":config/transitions/null_transition",
        ":config/transitions/patch_transition",
        ":config/transitions/split_transition",
        ":config/transitions/transition_factory",
        ":config/transitive_option_details",
        ":configurations_collector",
        ":configured_object_value",
        ":configured_target",
        ":constraints/constraint_constants",
        ":constraints/constraint_semantics",
        ":constraints/environment",
        ":constraints/environment_collection",
        ":constraints/environment_rule",
        ":constraints/supported_environments",
        ":constraints/supported_environments_provider",
        ":constraints/top_level_constraint_semantics",
        ":dependency",
        ":dependency_kind",
        ":duplicate_exception",
        ":extra/extra_action_info_file_write_action",
        ":extra_action_artifacts_provider",
        ":file_provider",
        ":inconsistent_aspect_order_exception",
        ":label_and_location",
        ":label_expander",
        ":licenses_provider",
        ":make_environment_event",
        ":make_variable_supplier",
        ":middleman_provider",
        ":no_build_event",
        ":no_build_request_finished_event",
        ":options_diff_predicate",
        ":package_specification_provider",
        ":platform_configuration",
        ":platform_configuration_loader",
        ":platform_options",
        ":print_action_visitor",
        ":provider_collection",
        ":required_config_fragments_provider",
        ":rule_definition_context",
        ":rule_definition_environment",
        ":server_directories",
        ":skylark/args",
        ":skylark/bazel_build_api_globals",
        ":skylark/function_transition_util",
        ":skylark/starlark_api_provider",
        ":skylark/starlark_command_line",
        ":skylark/starlark_config",
        ":skylark/starlark_custom_command_line",
        ":skylark/starlark_error_reporter",
        ":skylark/starlark_late_bound_default",
        ":statically_linked_marker_provider",
        ":target_configured_event",
        ":template_variable_info",
        ":test/analysis_failure",
        ":test/analysis_failure_info",
        ":test/analysis_test_result_info",
        ":test/baseline_coverage_result",
        ":test/coverage_report_action_factory",
        ":test/execution_info",
        ":test/instrumented_files_info",
        ":test/test_configuration",
        ":test/test_environment_info",
        ":test/test_sharding_strategy",
        ":test/test_trimming_transition_factory",
        ":toolchain_collection",
        ":toolchain_context",
        ":top_level_artifact_context",
        ":transition_mode",
        ":transitive_info_collection",
        ":transitive_info_provider",
        ":transitive_info_provider_effective_class_helper",
        ":transitive_info_provider_map",
        ":transitive_info_provider_map_builder",
        ":transitive_info_provider_map_impl",
        ":view_creation_failed_exception",
        ":visibility_provider",
        ":visibility_provider_impl",
        ":workspace_status_action",
        ":xcode_config_event",
    ],
)

# TODO(b/144899336): Split this further.
java_library(
    name = "analysis_cluster",
    srcs = [
        "AliasProvider.java",
        "AnalysisEnvironment.java",
        "AnalysisFailureEvent.java",
        "AnalysisResult.java",
        "AnalysisRootCauseEvent.java",
        "AnalysisUtils.java",
        "AspectCompleteEvent.java",
        "AspectResolver.java",
        "AspectValue.java",
        "BaseRuleClasses.java",
        "BashCommandConstructor.java",
        "CachingAnalysisEnvironment.java",
        "CommandConstructor.java",
        "CommandHelper.java",
        "ConfigurationMakeVariableContext.java",
        "ConfiguredAspect.java",
        "ConfiguredAspectFactory.java",
        "ConfiguredRuleClassProvider.java",
        "ConfiguredTargetFactory.java",
        "DefaultInfo.java",
        "DependencyResolver.java",
        "EmptyConfiguredTarget.java",
        "EventHandlingErrorReporter.java",
        "ExecGroupCollection.java",
        "Expander.java",
        "ExtraActionUtils.java",
        "ExtraActionsVisitor.java",
        "FilesToRunProvider.java",
        "LicensesProviderImpl.java",
        "LocationExpander.java",
        "LocationTemplateContext.java",
        "OutputGroupInfo.java",
        "PlatformConfiguration.java",
        "PrerequisiteArtifacts.java",
        "PseudoAction.java",
        "RuleConfiguredTargetBuilder.java",
        "RuleConfiguredTargetFactory.java",
        "RuleContext.java",
        "RuleDefinition.java",
        "Runfiles.java",
        "RunfilesProvider.java",
        "RunfilesSupplierImpl.java",
        "RunfilesSupport.java",
        "ShToolchain.java",
        "ShellConfiguration.java",
        "SourceManifestAction.java",
        "StarlarkProviderValidationUtil.java",
        "TargetAndConfiguration.java",
        "TargetCompleteEvent.java",
        "TargetContext.java",
        "TopLevelArtifactHelper.java",
        "Util.java",
        "Whitelist.java",
        "WindowsBatchCommandConstructor.java",
        "WindowsPowershellCommandConstructor.java",
        "actions/ActionConstructionContext.java",
        "actions/FileWriteAction.java",
        "actions/SpawnAction.java",
        "actions/StarlarkAction.java",
        "actions/SymlinkTreeAction.java",
        "actions/SymlinkTreeActionContext.java",
        "buildinfo/BuildInfoFactory.java",
        "config/BuildConfigurationCollection.java",
        "config/ConfigAwareRuleClassBuilder.java",
        "config/ConfigurationResolver.java",
        "config/ConvenienceSymlinks.java",
        "config/DependencyEvaluationException.java",
        "config/FragmentCollection.java",
        "config/TransitionResolver.java",
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
        "skylark/StarlarkActionFactory.java",
        "skylark/StarlarkAttrModule.java",
        "skylark/StarlarkAttributeTransitionProvider.java",
        "skylark/StarlarkAttributesCollection.java",
        "skylark/StarlarkModules.java",
        "skylark/StarlarkRuleClassFunctions.java",
        "skylark/StarlarkRuleConfiguredTargetUtil.java",
        "skylark/StarlarkRuleContext.java",
        "skylark/StarlarkRuleTransitionProvider.java",
        "skylark/StarlarkTransition.java",
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
        "test/TestTargetExecutionSettings.java",
        "test/TestTargetProperties.java",
    ],
    deps = [
        ":actions/abstract_file_write_action",
        ":actions/compression",
        ":actions/custom_command_line",
        ":actions/deterministic_writer",
        ":actions/lazy_write_nested_set_of_pair_action",
        ":actions/parameter_file_write_action",
        ":actions/substitution",
        ":actions/symlink_action",
        ":actions/template_expansion_action",
        ":actions_provider",
        ":artifacts_to_owner_labels",
        ":aspect_aware_attribute_mapper",
        ":aspect_collection",
        ":build_setting_provider",
        ":buildinfo/build_info_collection",
        ":buildinfo/build_info_key",
        ":config/build_configuration",
        ":config/build_options",
        ":config/config_matching_provider",
        ":config/core_options",
        ":config/execution_transition_factory",
        ":config/fragment",
        ":config/fragment_options",
        ":config/fragment_provider",
        ":config/host_transition",
        ":config/invalid_configuration_exception",
        ":config/per_label_options",
        ":config/run_under",
        ":config/starlark_defined_config_transition",
        ":config/transition_factories",
        ":config/transitions/composing_transition",
        ":config/transitions/composing_transition_factory",
        ":config/transitions/configuration_transition",
        ":config/transitions/no_transition",
        ":config/transitions/null_transition",
        ":config/transitions/patch_transition",
        ":config/transitions/split_transition",
        ":config/transitions/transition_factory",
        ":configurations_collector",
        ":configured_object_value",
        ":configured_target",
        ":constraints/constraint_constants",
        ":constraints/constraint_semantics",
        ":constraints/environment_collection",
        ":constraints/supported_environments",
        ":constraints/supported_environments_provider",
        ":dependency",
        ":dependency_key",
        ":dependency_kind",
        ":duplicate_exception",
        ":extra/extra_action_info_file_write_action",
        ":extra_action_artifacts_provider",
        ":file_provider",
        ":inconsistent_aspect_order_exception",
        ":label_and_location",
        ":label_expander",
        ":licenses_provider",
        ":make_variable_supplier",
        ":middleman_provider",
        ":options_diff_predicate",
        ":package_specification_provider",
        ":platform_options",
        ":provider_collection",
        ":required_config_fragments_provider",
        ":resolved_toolchain_context",
        ":rule_definition_context",
        ":rule_definition_environment",
        ":skylark/args",
        ":skylark/bazel_build_api_globals",
        ":skylark/function_transition_util",
        ":skylark/starlark_api_provider",
        ":skylark/starlark_command_line",
        ":skylark/starlark_late_bound_default",
        ":template_variable_info",
        ":test/analysis_failure",
        ":test/analysis_failure_info",
        ":test/analysis_test_result_info",
        ":test/baseline_coverage_result",
        ":test/execution_info",
        ":test/instrumented_files_info",
        ":test/test_configuration",
        ":test/test_environment_info",
        ":test/test_sharding_strategy",
        ":toolchain_collection",
        ":toolchain_context",
        ":top_level_artifact_context",
        ":transition_mode",
        ":transitive_info_collection",
        ":transitive_info_provider",
        ":transitive_info_provider_effective_class_helper",
        ":transitive_info_provider_map",
        ":transitive_info_provider_map_builder",
        ":visibility_provider",
        ":visibility_provider_impl",
        "//src/main/java/com/google/devtools/build/lib:build-request-options",
        "//src/main/java/com/google/devtools/build/lib:runtime/build_event_streamer_utils",
        "//src/main/java/com/google/devtools/build/lib/actions",
        "//src/main/java/com/google/devtools/build/lib/actions:localhost_capacity",
        "//src/main/java/com/google/devtools/build/lib/analysis/platform",
        "//src/main/java/com/google/devtools/build/lib/analysis/stringtemplate",
        "//src/main/java/com/google/devtools/build/lib/bugreport",
        "//src/main/java/com/google/devtools/build/lib/buildeventstream",
        "//src/main/java/com/google/devtools/build/lib/buildeventstream/proto:build_event_stream_java_proto",
        "//src/main/java/com/google/devtools/build/lib/causes",
        "//src/main/java/com/google/devtools/build/lib/cmdline",
        "//src/main/java/com/google/devtools/build/lib/cmdline:LabelValidator",
        "//src/main/java/com/google/devtools/build/lib/collect",
        "//src/main/java/com/google/devtools/build/lib/collect/compacthashset",
        "//src/main/java/com/google/devtools/build/lib/collect/nestedset",
        "//src/main/java/com/google/devtools/build/lib/concurrent",
        "//src/main/java/com/google/devtools/build/lib/events",
        "//src/main/java/com/google/devtools/build/lib/exec:bin_tools",
        "//src/main/java/com/google/devtools/build/lib/exec:execution_options",
        "//src/main/java/com/google/devtools/build/lib/exec:spawn_strategy_resolver",
        "//src/main/java/com/google/devtools/build/lib/exec:streamed_test_output",
        "//src/main/java/com/google/devtools/build/lib/exec:test_log_helper",
        "//src/main/java/com/google/devtools/build/lib/exec:test_xml_output_parser",
        "//src/main/java/com/google/devtools/build/lib/exec:test_xml_output_parser_exception",
        "//src/main/java/com/google/devtools/build/lib/graph",
        "//src/main/java/com/google/devtools/build/lib/packages",
        "//src/main/java/com/google/devtools/build/lib/packages:build_type",
        "//src/main/java/com/google/devtools/build/lib/packages:type",
        "//src/main/java/com/google/devtools/build/lib/profiler",
        "//src/main/java/com/google/devtools/build/lib/profiler:google-auto-profiler-utils",
        "//src/main/java/com/google/devtools/build/lib/profiler/memory:current_rule_tracker",
        "//src/main/java/com/google/devtools/build/lib/shell",
        "//src/main/java/com/google/devtools/build/lib/skyframe:aspect_creation_exception",
        "//src/main/java/com/google/devtools/build/lib/skyframe:aspect_value_key",
        "//src/main/java/com/google/devtools/build/lib/skyframe:build_configuration_value",
        "//src/main/java/com/google/devtools/build/lib/skyframe:build_info_collection_value",
        "//src/main/java/com/google/devtools/build/lib/skyframe:configured_target_and_data",
        "//src/main/java/com/google/devtools/build/lib/skyframe:configured_target_key",
        "//src/main/java/com/google/devtools/build/lib/skyframe:configured_value_creation_exception",
        "//src/main/java/com/google/devtools/build/lib/skyframe:package_value",
        "//src/main/java/com/google/devtools/build/lib/skyframe:precomputed_value",
        "//src/main/java/com/google/devtools/build/lib/skyframe:sane_analysis_exception",
        "//src/main/java/com/google/devtools/build/lib/skyframe:transitive_target_key",
        "//src/main/java/com/google/devtools/build/lib/skyframe:transitive_target_value",
        "//src/main/java/com/google/devtools/build/lib/skyframe:workspace_status_value",
        "//src/main/java/com/google/devtools/build/lib/skyframe/serialization/autocodec",
        "//src/main/java/com/google/devtools/build/lib/skylarkbuildapi",
        "//src/main/java/com/google/devtools/build/lib/skylarkbuildapi/core",
        "//src/main/java/com/google/devtools/build/lib/skylarkbuildapi/platform",
        "//src/main/java/com/google/devtools/build/lib/skylarkbuildapi/test",
        "//src/main/java/com/google/devtools/build/lib/syntax:evaluator",
        "//src/main/java/com/google/devtools/build/lib/syntax:frontend",
        "//src/main/java/com/google/devtools/build/lib/util",
        "//src/main/java/com/google/devtools/build/lib/util:detailed_exit_code",
        "//src/main/java/com/google/devtools/build/lib/util:filetype",
        "//src/main/java/com/google/devtools/build/lib/util:os",
        "//src/main/java/com/google/devtools/build/lib/util:shell_escaper",
        "//src/main/java/com/google/devtools/build/lib/util:string",
        "//src/main/java/com/google/devtools/build/lib/util/io:out-err",
        "//src/main/java/com/google/devtools/build/lib/vfs",
        "//src/main/java/com/google/devtools/build/lib/vfs:pathfragment",
        "//src/main/java/com/google/devtools/build/skyframe",
        "//src/main/java/com/google/devtools/build/skyframe:skyframe-objects",
        "//src/main/java/com/google/devtools/common/options",
        "//src/main/java/net/starlark/java/annot",
        "//src/main/protobuf:extra_actions_base_java_proto",
        "//src/main/protobuf:failure_details_java_proto",
        "//src/main/protobuf:test_status_java_proto",
        "//third_party:auto_value",
        "//third_party:guava",
        "//third_party:jsr305",
        "//third_party/protobuf:protobuf_java",
    ],
)

java_library(
    name = "actions_provider",
    srcs = ["ActionsProvider.java"],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/actions",
        "//src/main/java/com/google/devtools/build/lib/packages",
        "//src/main/java/com/google/devtools/build/lib/skylarkbuildapi",
        "//src/main/java/com/google/devtools/build/lib/syntax:evaluator",
        "//src/main/java/com/google/devtools/build/lib/syntax:frontend",
        "//third_party:guava",
    ],
)

java_library(
    name = "analysis_options",
    srcs = ["AnalysisOptions.java"],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/util",
        "//src/main/java/com/google/devtools/common/options",
    ],
)

java_library(
    name = "analysis_phase_complete_event",
    srcs = ["AnalysisPhaseCompleteEvent.java"],
    deps = [
        ":configured_target",
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
    name = "artifacts_to_owner_labels",
    srcs = ["ArtifactsToOwnerLabels.java"],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/actions",
        "//src/main/java/com/google/devtools/build/lib/cmdline",
        "//third_party:guava",
    ],
)

java_library(
    name = "aspect_aware_attribute_mapper",
    srcs = ["AspectAwareAttributeMapper.java"],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/cmdline",
        "//src/main/java/com/google/devtools/build/lib/packages",
        "//src/main/java/com/google/devtools/build/lib/packages:type",
        "//third_party:guava",
    ],
)

java_library(
    name = "aspect_collection",
    srcs = ["AspectCollection.java"],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/concurrent",
        "//src/main/java/com/google/devtools/build/lib/packages",
        "//third_party:guava",
    ],
)

java_library(
    name = "aspect_configured_event",
    srcs = ["AspectConfiguredEvent.java"],
    deps = [
        ":config/build_configuration",
        "//src/main/java/com/google/devtools/build/lib/buildeventstream",
        "//src/main/java/com/google/devtools/build/lib/buildeventstream/proto:build_event_stream_java_proto",
        "//src/main/java/com/google/devtools/build/lib/cmdline",
        "//third_party:guava",
    ],
)

java_library(
    name = "blaze_directories",
    srcs = ["BlazeDirectories.java"],
    deps = [
        ":server_directories",
        "//src/main/java/com/google/devtools/build/lib/actions",
        "//src/main/java/com/google/devtools/build/lib/concurrent",
        "//src/main/java/com/google/devtools/build/lib/skyframe/serialization/autocodec",
        "//src/main/java/com/google/devtools/build/lib/util:string",
        "//src/main/java/com/google/devtools/build/lib/vfs",
        "//third_party:guava",
        "//third_party:jsr305",
    ],
)

java_library(
    name = "blaze_version_info",
    srcs = ["BlazeVersionInfo.java"],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/util:string",
        "//third_party:flogger",
        "//third_party:guava",
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
        "//src/main/java/com/google/devtools/build/lib/packages:type",
        "//third_party:guava",
    ],
)

java_library(
    name = "build_view",
    srcs = ["BuildView.java"],
    deps = [
        ":analysis_cluster",
        ":analysis_options",
        ":analysis_phase_started_event",
        ":artifacts_to_owner_labels",
        ":aspect_configured_event",
        ":blaze_directories",
        ":config/build_configuration",
        ":config/build_options",
        ":config/invalid_configuration_exception",
        ":configured_target",
        ":constraints/top_level_constraint_semantics",
        ":extra_action_artifacts_provider",
        ":make_environment_event",
        ":target_configured_event",
        ":test/coverage_report_action_factory",
        ":test/instrumented_files_info",
        ":top_level_artifact_context",
        ":view_creation_failed_exception",
        "//src/main/java/com/google/devtools/build/lib/actions",
        "//src/main/java/com/google/devtools/build/lib/cmdline",
        "//src/main/java/com/google/devtools/build/lib/collect/nestedset",
        "//src/main/java/com/google/devtools/build/lib/concurrent",
        "//src/main/java/com/google/devtools/build/lib/events",
        "//src/main/java/com/google/devtools/build/lib/packages",
        "//src/main/java/com/google/devtools/build/lib/pkgcache",
        "//src/main/java/com/google/devtools/build/lib/profiler",
        "//src/main/java/com/google/devtools/build/lib/skyframe:aspect_value_key",
        "//src/main/java/com/google/devtools/build/lib/skyframe:build_configuration_value",
        "//src/main/java/com/google/devtools/build/lib/skyframe:configured_target_key",
        "//src/main/java/com/google/devtools/build/lib/skyframe:coverage_report_value",
        "//src/main/java/com/google/devtools/build/lib/skyframe:skyframe_cluster",
        "//src/main/java/com/google/devtools/build/lib/skyframe:target_pattern_phase_value",
        "//src/main/java/com/google/devtools/build/lib/util",
        "//src/main/java/com/google/devtools/build/skyframe",
        "//src/main/java/com/google/devtools/build/skyframe:skyframe-objects",
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
        "//src/main/java/com/google/devtools/build/lib/cmdline",
        "//src/main/java/com/google/devtools/build/lib/packages",
        "//src/main/java/com/google/devtools/build/lib/packages:type",
        "//src/main/java/com/google/devtools/build/lib/skyframe:configured_target_and_data",
    ],
)

java_library(
    name = "compilation_helper",
    srcs = ["CompilationHelper.java"],
    deps = [
        ":analysis_cluster",
        ":file_provider",
        ":transitive_info_collection",
        "//src/main/java/com/google/devtools/build/lib/actions",
        "//src/main/java/com/google/devtools/build/lib/collect/nestedset",
        "//third_party:guava",
    ],
)

java_library(
    name = "configurations_collector",
    srcs = [
        "ConfigurationsCollector.java",
        "ConfigurationsResult.java",
    ],
    deps = [
        ":config/build_configuration",
        ":config/build_options",
        ":config/invalid_configuration_exception",
        ":dependency_key",
        "//src/main/java/com/google/devtools/build/lib/events",
        "//third_party:guava",
    ],
)

java_library(
    name = "configured_object_value",
    srcs = ["ConfiguredObjectValue.java"],
    deps = [
        ":provider_collection",
        "//src/main/java/com/google/devtools/build/lib/actions",
        "//src/main/java/com/google/devtools/build/lib/collect/nestedset",
        "//src/main/java/com/google/devtools/build/lib/packages",
        "//src/main/java/com/google/devtools/build/skyframe",
    ],
)

java_library(
    name = "configured_target",
    srcs = ["ConfiguredTarget.java"],
    deps = [
        ":transitive_info_collection",
        "//src/main/java/com/google/devtools/build/lib/actions",
        "//src/main/java/com/google/devtools/build/lib/cmdline",
        "//src/main/java/com/google/devtools/build/lib/skyframe:build_configuration_value",
        "//src/main/java/com/google/devtools/build/lib/syntax:evaluator",
        "//third_party:guava",
        "//third_party:jsr305",
    ],
)

java_library(
    name = "dependency",
    srcs = ["Dependency.java"],
    deps = [
        ":aspect_collection",
        ":config/build_configuration",
        ":config/transitions/configuration_transition",
        "//src/main/java/com/google/devtools/build/lib/cmdline",
        "//src/main/java/com/google/devtools/build/lib/packages",
        "//src/main/java/com/google/devtools/build/lib/skyframe:configured_target_key",
        "//third_party:auto_value",
        "//third_party:guava",
        "//third_party:jsr305",
    ],
)

java_library(
    name = "dependency_key",
    srcs = ["DependencyKey.java"],
    deps = [
        ":aspect_collection",
        ":config/transitions/configuration_transition",
        "//src/main/java/com/google/devtools/build/lib/cmdline",
        "//third_party:auto_value",
    ],
)

java_library(
    name = "dependency_kind",
    srcs = ["DependencyKind.java"],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/packages",
        "//third_party:auto_value",
        "//third_party:guava",
        "//third_party:jsr305",
    ],
)

java_library(
    name = "duplicate_exception",
    srcs = ["DuplicateException.java"],
)

java_library(
    name = "extra_action_artifacts_provider",
    srcs = ["ExtraActionArtifactsProvider.java"],
    deps = [
        ":transitive_info_provider",
        "//src/main/java/com/google/devtools/build/lib/actions",
        "//src/main/java/com/google/devtools/build/lib/collect/nestedset",
        "//src/main/java/com/google/devtools/build/lib/concurrent",
        "//src/main/java/com/google/devtools/build/lib/skyframe/serialization/autocodec",
    ],
)

java_library(
    name = "file_provider",
    srcs = ["FileProvider.java"],
    deps = [
        ":transitive_info_provider",
        "//src/main/java/com/google/devtools/build/lib/actions",
        "//src/main/java/com/google/devtools/build/lib/collect/nestedset",
        "//src/main/java/com/google/devtools/build/lib/concurrent",
        "//src/main/java/com/google/devtools/build/lib/skyframe/serialization/autocodec",
        "//src/main/java/com/google/devtools/build/lib/skylarkbuildapi",
    ],
)

java_library(
    name = "inconsistent_aspect_order_exception",
    srcs = ["InconsistentAspectOrderException.java"],
    deps = [
        ":aspect_collection",
        "//src/main/java/com/google/devtools/build/lib/packages",
        "//src/main/java/com/google/devtools/build/lib/syntax:frontend",
    ],
)

java_library(
    name = "label_and_location",
    srcs = ["LabelAndLocation.java"],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/cmdline",
        "//src/main/java/com/google/devtools/build/lib/packages",
        "//src/main/java/com/google/devtools/build/lib/skyframe/serialization/autocodec",
        "//src/main/java/com/google/devtools/build/lib/syntax:frontend",
        "//third_party:auto_value",
    ],
)

java_library(
    name = "label_expander",
    srcs = ["LabelExpander.java"],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/actions",
        "//src/main/java/com/google/devtools/build/lib/cmdline",
        "//src/main/java/com/google/devtools/build/lib/vfs:pathfragment",
        "//third_party:guava",
        "//third_party:jsr305",
    ],
)

java_library(
    name = "licenses_provider",
    srcs = ["LicensesProvider.java"],
    deps = [
        ":transitive_info_provider",
        "//src/main/java/com/google/devtools/build/lib/cmdline",
        "//src/main/java/com/google/devtools/build/lib/collect/nestedset",
        "//src/main/java/com/google/devtools/build/lib/packages:build_type",
        "//src/main/java/com/google/devtools/build/lib/skyframe/serialization/autocodec",
        "//third_party:guava",
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
    name = "middleman_provider",
    srcs = ["MiddlemanProvider.java"],
    deps = [
        ":transitive_info_provider",
        "//src/main/java/com/google/devtools/build/lib/actions",
        "//src/main/java/com/google/devtools/build/lib/collect/nestedset",
        "//src/main/java/com/google/devtools/build/lib/concurrent",
        "//src/main/java/com/google/devtools/build/lib/skyframe/serialization/autocodec",
    ],
)

java_library(
    name = "no_build_event",
    srcs = ["NoBuildEvent.java"],
    deps = [
        ":blaze_version_info",
        "//src/main/java/com/google/devtools/build/lib/buildeventstream",
        "//src/main/java/com/google/devtools/build/lib/buildeventstream/proto:build_event_stream_java_proto",
        "//src/main/java/com/google/devtools/build/lib/util:process",
        "//third_party:guava",
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
        ":config/build_options",
        "//src/main/java/com/google/devtools/common/options",
    ],
)

java_library(
    name = "package_specification_provider",
    srcs = ["PackageSpecificationProvider.java"],
    deps = [
        ":transitive_info_provider",
        "//src/main/java/com/google/devtools/build/lib/collect/nestedset",
        "//src/main/java/com/google/devtools/build/lib/packages",
    ],
)

java_library(
    name = "platform_configuration",
    srcs = ["PlatformConfiguration.java"],
    deps = [
        ":config/build_options",
        ":config/fragment",
        ":platform_options",
        "//src/main/java/com/google/devtools/build/lib/cmdline",
        "//src/main/java/com/google/devtools/build/lib/concurrent",
        "//src/main/java/com/google/devtools/build/lib/events",
        "//src/main/java/com/google/devtools/build/lib/skylarkbuildapi/platform",
        "//src/main/java/com/google/devtools/build/lib/util",
        "//third_party:guava",
    ],
)

java_library(
    name = "platform_configuration_loader",
    srcs = ["PlatformConfigurationLoader.java"],
    deps = [
        ":config/build_options",
        ":config/fragment",
        ":config/fragment_options",
        ":config/fragment_provider",
        ":config/invalid_configuration_exception",
        ":platform_configuration",
        ":platform_options",
        "//third_party:guava",
    ],
)

java_library(
    name = "platform_options",
    srcs = ["PlatformOptions.java"],
    deps = [
        ":config/core_option_converters",
        ":config/fragment_options",
        "//src/main/java/com/google/devtools/build/lib/cmdline",
        "//src/main/java/com/google/devtools/build/lib/util",
        "//src/main/java/com/google/devtools/build/lib/vfs:pathfragment",
        "//src/main/java/com/google/devtools/common/options",
        "//third_party:guava",
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
    name = "required_config_fragments_provider",
    srcs = ["RequiredConfigFragmentsProvider.java"],
    deps = [
        ":config/fragment",
        ":transitive_info_provider",
        "//src/main/java/com/google/devtools/build/lib/concurrent",
        "//third_party:guava",
    ],
)

java_library(
    name = "resolved_toolchain_context",
    srcs = ["ResolvedToolchainContext.java"],
    deps = [
        ":template_variable_info",
        ":toolchain_context",
        "//src/main/java/com/google/devtools/build/lib/analysis/platform",
        "//src/main/java/com/google/devtools/build/lib/analysis/platform:utils",
        "//src/main/java/com/google/devtools/build/lib/cmdline",
        "//src/main/java/com/google/devtools/build/lib/concurrent",
        "//src/main/java/com/google/devtools/build/lib/skyframe:configured_target_and_data",
        "//src/main/java/com/google/devtools/build/lib/skyframe:toolchain_context_key",
        "//src/main/java/com/google/devtools/build/lib/skyframe:toolchain_exception",
        "//src/main/java/com/google/devtools/build/lib/skyframe:unloaded_toolchain_context",
        "//src/main/java/com/google/devtools/build/lib/skylarkbuildapi/platform",
        "//src/main/java/com/google/devtools/build/lib/syntax:evaluator",
        "//third_party:auto_value",
        "//third_party:guava",
        "//third_party:jsr305",
    ],
)

java_library(
    name = "rule_definition_context",
    srcs = ["RuleDefinitionContext.java"],
)

java_library(
    name = "rule_definition_environment",
    srcs = ["RuleDefinitionEnvironment.java"],
    deps = [
        ":rule_definition_context",
        "//src/main/java/com/google/devtools/build/lib/cmdline",
    ],
)

java_library(
    name = "server_directories",
    srcs = ["ServerDirectories.java"],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/concurrent",
        "//src/main/java/com/google/devtools/build/lib/vfs",
        "//third_party:guava",
        "//third_party:jsr305",
    ],
)

java_library(
    name = "statically_linked_marker_provider",
    srcs = ["StaticallyLinkedMarkerProvider.java"],
    deps = [
        ":transitive_info_provider",
        "//src/main/java/com/google/devtools/build/lib/concurrent",
    ],
)

java_library(
    name = "target_configured_event",
    srcs = ["TargetConfiguredEvent.java"],
    deps = [
        ":config/build_configuration",
        "//src/main/java/com/google/devtools/build/lib/buildeventstream",
        "//src/main/java/com/google/devtools/build/lib/buildeventstream/proto:build_event_stream_java_proto",
        "//src/main/java/com/google/devtools/build/lib/packages",
        "//src/main/java/com/google/devtools/build/lib/packages:type",
        "//third_party:guava",
    ],
)

java_library(
    name = "template_variable_info",
    srcs = ["TemplateVariableInfo.java"],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/concurrent",
        "//src/main/java/com/google/devtools/build/lib/packages",
        "//src/main/java/com/google/devtools/build/lib/skyframe/serialization/autocodec",
        "//src/main/java/com/google/devtools/build/lib/skylarkbuildapi",
        "//src/main/java/com/google/devtools/build/lib/syntax:evaluator",
        "//src/main/java/com/google/devtools/build/lib/syntax:frontend",
        "//third_party:guava",
    ],
)

java_library(
    name = "toolchain_collection",
    srcs = ["ToolchainCollection.java"],
    deps = [
        ":toolchain_context",
        "//src/main/java/com/google/devtools/build/lib/cmdline",
        "//third_party:auto_value",
        "//third_party:guava",
    ],
)

java_library(
    name = "toolchain_context",
    srcs = ["ToolchainContext.java"],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/analysis/platform",
        "//src/main/java/com/google/devtools/build/lib/cmdline",
        "//src/main/java/com/google/devtools/build/lib/skyframe:toolchain_context_key",
        "//third_party:guava",
    ],
)

java_library(
    name = "top_level_artifact_context",
    srcs = ["TopLevelArtifactContext.java"],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/concurrent",
        "//src/main/java/com/google/devtools/build/lib/skyframe/serialization/autocodec",
        "//third_party:guava",
    ],
)

java_library(
    name = "transition_mode",
    srcs = ["TransitionMode.java"],
)

java_library(
    name = "transitive_info_collection",
    srcs = ["TransitiveInfoCollection.java"],
    deps = [
        ":provider_collection",
        ":transitive_info_provider",
        "//src/main/java/com/google/devtools/build/lib/cmdline",
        "//src/main/java/com/google/devtools/build/lib/packages",
        "//src/main/java/com/google/devtools/build/lib/skylarkbuildapi/core",
        "//src/main/java/com/google/devtools/build/lib/syntax:evaluator",
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
        "//src/main/java/com/google/devtools/build/lib/collect",
        "//src/main/java/com/google/devtools/build/lib/packages",
        "//src/main/java/com/google/devtools/build/lib/skyframe/serialization/autocodec",
        "//third_party:guava",
        "//third_party:jsr305",
    ],
)

java_library(
    name = "view_creation_failed_exception",
    srcs = ["ViewCreationFailedException.java"],
)

java_library(
    name = "visibility_provider",
    srcs = ["VisibilityProvider.java"],
    deps = [
        ":transitive_info_provider",
        "//src/main/java/com/google/devtools/build/lib/collect/nestedset",
        "//src/main/java/com/google/devtools/build/lib/packages",
    ],
)

java_library(
    name = "visibility_provider_impl",
    srcs = ["VisibilityProviderImpl.java"],
    deps = [
        ":visibility_provider",
        "//src/main/java/com/google/devtools/build/lib/collect/nestedset",
        "//src/main/java/com/google/devtools/build/lib/concurrent",
        "//src/main/java/com/google/devtools/build/lib/packages",
        "//src/main/java/com/google/devtools/build/lib/skyframe/serialization/autocodec",
    ],
)

java_library(
    name = "workspace_status_action",
    srcs = ["WorkspaceStatusAction.java"],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/actions",
        "//src/main/java/com/google/devtools/build/lib/collect/nestedset",
        "//src/main/java/com/google/devtools/build/lib/shell",
        "//src/main/java/com/google/devtools/build/lib/util",
        "//src/main/java/com/google/devtools/build/lib/vfs",
        "//src/main/java/com/google/devtools/build/lib/vfs:pathfragment",
        "//src/main/java/com/google/devtools/common/options",
        "//third_party:guava",
    ],
)

java_library(
    name = "xcode_config_event",
    srcs = ["XcodeConfigEvent.java"],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/events",
        "//src/main/protobuf:xcode_java_proto",
    ],
)

# TODO(b/144899336): This should be analysis/actions/BUILD
java_library(
    name = "actions/abstract_file_write_action",
    srcs = ["actions/AbstractFileWriteAction.java"],
    deps = [
        ":actions/deterministic_writer",
        ":actions/file_write_action_context",
        "//src/main/java/com/google/devtools/build/lib/actions",
        "//src/main/java/com/google/devtools/build/lib/cmdline",
        "//src/main/java/com/google/devtools/build/lib/collect/nestedset",
        "//third_party:guava",
        "//third_party:jsr305",
    ],
)

java_library(
    name = "actions/binary_file_write_action",
    srcs = ["actions/BinaryFileWriteAction.java"],
    deps = [
        ":actions/abstract_file_write_action",
        ":actions/deterministic_writer",
        "//src/main/java/com/google/devtools/build/lib/actions",
        "//src/main/java/com/google/devtools/build/lib/collect/nestedset",
        "//src/main/java/com/google/devtools/build/lib/concurrent",
        "//src/main/java/com/google/devtools/build/lib/util",
        "//third_party:guava",
    ],
)

java_library(
    name = "actions/byte_string_deterministic_writer",
    srcs = ["actions/ByteStringDeterministicWriter.java"],
    deps = [
        ":actions/deterministic_writer",
        "//third_party/protobuf:protobuf_java",
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
        "//src/main/java/com/google/devtools/build/lib/actions:commandline_item",
        "//src/main/java/com/google/devtools/build/lib/cmdline",
        "//src/main/java/com/google/devtools/build/lib/collect/nestedset",
        "//src/main/java/com/google/devtools/build/lib/concurrent",
        "//src/main/java/com/google/devtools/build/lib/skyframe/serialization/autocodec",
        "//src/main/java/com/google/devtools/build/lib/util",
        "//src/main/java/com/google/devtools/build/lib/vfs:pathfragment",
        "//third_party:auto_value",
        "//third_party:guava",
        "//third_party:jsr305",
    ],
)

java_library(
    name = "actions/deterministic_writer",
    srcs = ["actions/DeterministicWriter.java"],
    deps = ["//third_party/protobuf:protobuf_java"],
)

java_library(
    name = "actions/file_write_action_context",
    srcs = ["actions/FileWriteActionContext.java"],
    deps = [
        ":actions/deterministic_writer",
        "//src/main/java/com/google/devtools/build/lib/actions",
    ],
)

java_library(
    name = "actions/launcher_file_write_action",
    srcs = ["actions/LauncherFileWriteAction.java"],
    deps = [
        ":actions/abstract_file_write_action",
        ":actions/deterministic_writer",
        ":analysis_cluster",
        ":transition_mode",
        "//src/main/java/com/google/devtools/build/lib/actions",
        "//src/main/java/com/google/devtools/build/lib/collect/nestedset",
        "//src/main/java/com/google/devtools/build/lib/util",
        "//src/main/java/com/google/devtools/build/lib/util:os",
        "//third_party:guava",
        "//third_party:jsr305",
    ],
)

java_library(
    name = "actions/lazy_write_nested_set_of_pair_action",
    srcs = ["actions/LazyWriteNestedSetOfPairAction.java"],
    deps = [
        ":actions/abstract_file_write_action",
        ":actions/deterministic_writer",
        "//src/main/java/com/google/devtools/build/lib/actions",
        "//src/main/java/com/google/devtools/build/lib/collect/nestedset",
        "//src/main/java/com/google/devtools/build/lib/util",
    ],
)

java_library(
    name = "actions/lazy_write_paths_file_action",
    srcs = ["actions/LazyWritePathsFileAction.java"],
    deps = [
        ":actions/abstract_file_write_action",
        ":actions/deterministic_writer",
        "//src/main/java/com/google/devtools/build/lib/actions",
        "//src/main/java/com/google/devtools/build/lib/collect/nestedset",
        "//src/main/java/com/google/devtools/build/lib/util",
        "//third_party:guava",
    ],
)

java_library(
    name = "actions/parameter_file_write_action",
    srcs = ["actions/ParameterFileWriteAction.java"],
    deps = [
        ":actions/abstract_file_write_action",
        ":actions/deterministic_writer",
        "//src/main/java/com/google/devtools/build/lib/actions",
        "//src/main/java/com/google/devtools/build/lib/collect/nestedset",
        "//src/main/java/com/google/devtools/build/lib/concurrent",
        "//src/main/java/com/google/devtools/build/lib/skyframe/serialization/autocodec",
        "//src/main/java/com/google/devtools/build/lib/syntax:evaluator",
        "//src/main/java/com/google/devtools/build/lib/util",
        "//third_party:guava",
    ],
)

java_library(
    name = "actions/proto_deterministic_writer",
    srcs = ["actions/ProtoDeterministicWriter.java"],
    deps = [
        ":actions/deterministic_writer",
        "//third_party/protobuf:protobuf_java",
    ],
)

java_library(
    name = "actions/spawn_action_template",
    srcs = ["actions/SpawnActionTemplate.java"],
    deps = [
        ":actions/custom_command_line",
        ":analysis_cluster",
        "//src/main/java/com/google/devtools/build/lib/actions",
        "//src/main/java/com/google/devtools/build/lib/collect/nestedset",
        "//src/main/java/com/google/devtools/build/lib/skyframe:action_template_expansion_value",
        "//src/main/java/com/google/devtools/build/lib/util",
        "//src/main/java/com/google/devtools/build/lib/vfs:pathfragment",
        "//third_party:guava",
    ],
)

java_library(
    name = "actions/substitution",
    srcs = ["actions/Substitution.java"],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/actions",
        "//src/main/java/com/google/devtools/build/lib/cmdline",
        "//src/main/java/com/google/devtools/build/lib/collect/nestedset",
        "//src/main/java/com/google/devtools/build/lib/concurrent",
        "//src/main/java/com/google/devtools/build/lib/skyframe/serialization/autocodec",
        "//src/main/java/com/google/devtools/build/lib/vfs:pathfragment",
        "//third_party:guava",
    ],
)

java_library(
    name = "actions/symlink_action",
    srcs = ["actions/SymlinkAction.java"],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/actions",
        "//src/main/java/com/google/devtools/build/lib/collect/nestedset",
        "//src/main/java/com/google/devtools/build/lib/skyframe/serialization/autocodec",
        "//src/main/java/com/google/devtools/build/lib/util",
        "//src/main/java/com/google/devtools/build/lib/vfs",
        "//src/main/java/com/google/devtools/build/lib/vfs:pathfragment",
        "//third_party:guava",
        "//third_party:jsr305",
    ],
)

java_library(
    name = "actions/template",
    srcs = ["actions/Template.java"],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/actions",
        "//src/main/java/com/google/devtools/build/lib/concurrent",
        "//src/main/java/com/google/devtools/build/lib/skyframe/serialization/autocodec",
        "//src/main/java/com/google/devtools/build/lib/util",
        "//src/main/java/com/google/devtools/build/lib/vfs",
        "//third_party:jsr305",
    ],
)

java_library(
    name = "actions/template_expansion_action",
    srcs = [
        "actions/LocalTemplateExpansionStrategy.java",
        "actions/TemplateExpansionAction.java",
        "actions/TemplateExpansionContext.java",
    ],
    deps = [
        ":actions/deterministic_writer",
        ":actions/file_write_action_context",
        ":actions/substitution",
        ":actions/template",
        "//src/main/java/com/google/devtools/build/lib/actions",
        "//src/main/java/com/google/devtools/build/lib/cmdline",
        "//src/main/java/com/google/devtools/build/lib/collect/nestedset",
        "//src/main/java/com/google/devtools/build/lib/concurrent",
        "//src/main/java/com/google/devtools/build/lib/skyframe/serialization/autocodec",
        "//src/main/java/com/google/devtools/build/lib/syntax:evaluator",
        "//src/main/java/com/google/devtools/build/lib/util",
        "//src/main/java/com/google/devtools/build/lib/util:string",
        "//third_party:guava",
        "//third_party:jsr305",
    ],
)

# TODO(b/144899336): This should be analysis/buildinfo/BUILD
java_library(
    name = "buildinfo/build_info_collection",
    srcs = ["buildinfo/BuildInfoCollection.java"],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/actions",
        "//src/main/java/com/google/devtools/build/lib/skyframe/serialization/autocodec",
        "//third_party:guava",
    ],
)

java_library(
    name = "buildinfo/build_info_key",
    srcs = ["buildinfo/BuildInfoKey.java"],
)

# TODO(b/144899336):  This should be analysis/config/BUILD
java_library(
    name = "config/auto_cpu_converter",
    srcs = ["config/AutoCpuConverter.java"],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/util",
        "//src/main/java/com/google/devtools/build/lib/util:os",
        "//src/main/java/com/google/devtools/common/options",
    ],
)

java_library(
    name = "config/build_configuration",
    srcs = [
        "config/BuildConfiguration.java",
        "config/FragmentClassSet.java",
        "config/OutputDirectories.java",
    ],
    deps = [
        ":actions/compression",
        ":blaze_directories",
        ":config/build_options",
        ":config/compilation_mode",
        ":config/core_options",
        ":config/fragment",
        ":config/fragment_options",
        ":config/fragment_provider",
        ":config/run_under",
        ":config/transitive_option_details",
        "//src/main/java/com/google/devtools/build/lib/actions",
        "//src/main/java/com/google/devtools/build/lib/buildeventstream",
        "//src/main/java/com/google/devtools/build/lib/buildeventstream/proto:build_event_stream_java_proto",
        "//src/main/java/com/google/devtools/build/lib/cmdline",
        "//src/main/java/com/google/devtools/build/lib/concurrent",
        "//src/main/java/com/google/devtools/build/lib/events",
        "//src/main/java/com/google/devtools/build/lib/packages",
        "//src/main/java/com/google/devtools/build/lib/skyframe/serialization/autocodec",
        "//src/main/java/com/google/devtools/build/lib/skylarkbuildapi",
        "//src/main/java/com/google/devtools/build/lib/util",
        "//src/main/java/com/google/devtools/build/lib/util:os",
        "//src/main/java/com/google/devtools/build/lib/vfs",
        "//src/main/java/com/google/devtools/build/lib/vfs:pathfragment",
        "//src/main/java/net/starlark/java/annot",
        "//third_party:guava",
        "//third_party:jsr305",
    ],
)

java_library(
    name = "config/build_configuration_option_details",
    srcs = ["config/BuildConfigurationOptionDetails.java"],
    deps = [
        ":config/build_configuration",
        ":config/transitive_option_details",
    ],
)

java_library(
    name = "config/build_options",
    srcs = [
        "config/BuildOptions.java",
        "config/BuildOptionsView.java",
    ],
    deps = [
        ":config/fragment_options",
        "//src/main/java/com/google/devtools/build/lib/cmdline",
        "//src/main/java/com/google/devtools/build/lib/skyframe/serialization/autocodec",
        "//src/main/java/com/google/devtools/build/lib/skyframe/trimming:trimmed_configuration_cache",
        "//src/main/java/com/google/devtools/build/lib/util",
        "//src/main/java/com/google/devtools/common/options",
        "//third_party:flogger",
        "//third_party:guava",
        "//third_party:jsr305",
        "//third_party/protobuf:protobuf_java",
    ],
)

java_library(
    name = "config/build_options_cache",
    srcs = ["config/BuildOptionsCache.java"],
    deps = [
        ":config/build_options",
        "//third_party:guava",
    ],
)

java_library(
    name = "config/compilation_mode",
    srcs = ["config/CompilationMode.java"],
    deps = ["//src/main/java/com/google/devtools/common/options"],
)

java_library(
    name = "config/config_aware_aspect_builder",
    srcs = ["config/ConfigAwareAspectBuilder.java"],
    deps = [
        ":config/fragment",
        ":config/host_transition",
        "//src/main/java/com/google/devtools/build/lib/packages",
    ],
)

java_library(
    name = "config/config_matching_provider",
    srcs = ["config/ConfigMatchingProvider.java"],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/analysis:transitive_info_provider",
        "//src/main/java/com/google/devtools/build/lib/cmdline",
        "//src/main/java/com/google/devtools/build/lib/concurrent",
        "//src/main/java/com/google/devtools/build/lib/skyframe/serialization/autocodec",
        "//third_party:guava",
    ],
)

java_library(
    name = "config/core_option_converters",
    srcs = ["config/CoreOptionConverters.java"],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/cmdline",
        "//src/main/java/com/google/devtools/build/lib/packages:build_type",
        "//src/main/java/com/google/devtools/build/lib/packages:type",
        "//src/main/java/com/google/devtools/common/options",
        "//third_party:guava",
    ],
)

java_library(
    name = "config/core_options",
    srcs = ["config/CoreOptions.java"],
    deps = [
        ":config/auto_cpu_converter",
        ":config/compilation_mode",
        ":config/core_option_converters",
        ":config/execution_info_modifier",
        ":config/fragment_options",
        ":config/run_under",
        ":config/run_under_converter",
        "//src/main/java/com/google/devtools/build/lib/cmdline",
        "//src/main/java/com/google/devtools/build/lib/util",
        "//src/main/java/com/google/devtools/common/options",
        "//third_party:guava",
    ],
)

java_library(
    name = "config/execution_info_modifier",
    srcs = ["config/ExecutionInfoModifier.java"],
    deps = [
        "//src/main/java/com/google/devtools/common/options",
        "//third_party:auto_value",
        "//third_party:guava",
    ],
)

java_library(
    name = "config/execution_transition_factory",
    srcs = ["config/ExecutionTransitionFactory.java"],
    deps = [
        ":config/build_options",
        ":config/build_options_cache",
        ":config/core_options",
        ":config/transitions/patch_transition",
        ":config/transitions/transition_factory",
        ":platform_options",
        ":toolchain_collection",
        "//src/main/java/com/google/devtools/build/lib/cmdline",
        "//src/main/java/com/google/devtools/build/lib/events",
        "//src/main/java/com/google/devtools/build/lib/packages",
        "//src/main/java/com/google/devtools/build/lib/skylarkbuildapi",
        "//third_party:guava",
        "//third_party:jsr305",
    ],
)

java_library(
    name = "config/fragment",
    srcs = ["config/Fragment.java"],
    deps = [
        ":config/build_options",
        "//src/main/java/com/google/devtools/build/lib/concurrent",
        "//src/main/java/com/google/devtools/build/lib/events",
        "//src/main/java/com/google/devtools/build/lib/syntax:evaluator",
        "//third_party:jsr305",
    ],
)

java_library(
    name = "config/fragment_options",
    srcs = ["config/FragmentOptions.java"],
    deps = [
        "//src/main/java/com/google/devtools/common/options",
        "//third_party:guava",
        "//third_party:jsr305",
    ],
)

java_library(
    name = "config/fragment_provider",
    srcs = [
        "config/ConfigurationFragmentFactory.java",
        "config/FragmentProvider.java",
    ],
    deps = [
        ":config/build_options",
        ":config/fragment",
        ":config/fragment_options",
        ":config/invalid_configuration_exception",
        "//src/main/java/com/google/devtools/build/lib/packages",
        "//third_party:guava",
        "//third_party:jsr305",
    ],
)

java_library(
    name = "config/host_transition",
    srcs = ["config/HostTransition.java"],
    deps = [
        ":config/build_options",
        ":config/core_options",
        ":config/fragment_options",
        ":config/transitions/patch_transition",
        ":config/transitions/transition_factory",
        "//src/main/java/com/google/devtools/build/lib/events",
        "//src/main/java/com/google/devtools/build/lib/skyframe/serialization/autocodec",
        "//third_party:auto_value",
        "//third_party:guava",
    ],
)

java_library(
    name = "config/invalid_configuration_exception",
    srcs = ["config/InvalidConfigurationException.java"],
    deps = [
        "//src/main/protobuf:failure_details_java_proto",
        "//third_party:jsr305",
    ],
)

java_library(
    name = "config/per_label_options",
    srcs = ["config/PerLabelOptions.java"],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/actions",
        "//src/main/java/com/google/devtools/build/lib/cmdline",
        "//src/main/java/com/google/devtools/build/lib/skyframe/serialization/autocodec",
        "//src/main/java/com/google/devtools/build/lib/util",
        "//src/main/java/com/google/devtools/common/options",
        "//third_party:guava",
    ],
)

java_library(
    name = "config/run_under",
    srcs = ["config/RunUnder.java"],
    deps = ["//src/main/java/com/google/devtools/build/lib/cmdline"],
)

java_library(
    name = "config/run_under_converter",
    srcs = ["config/RunUnderConverter.java"],
    deps = [
        ":config/run_under",
        "//src/main/java/com/google/devtools/build/lib/cmdline",
        "//src/main/java/com/google/devtools/build/lib/shell",
        "//src/main/java/com/google/devtools/build/lib/skyframe/serialization/autocodec",
        "//src/main/java/com/google/devtools/common/options",
        "//third_party:guava",
    ],
)

java_library(
    name = "config/starlark_defined_config_transition",
    srcs = ["config/StarlarkDefinedConfigTransition.java"],
    deps = [
        ":config/transitions/configuration_transition",
        "//src/main/java/com/google/devtools/build/lib/events",
        "//src/main/java/com/google/devtools/build/lib/packages",
        "//src/main/java/com/google/devtools/build/lib/skylarkbuildapi/config",
        "//src/main/java/com/google/devtools/build/lib/syntax:evaluator",
        "//src/main/java/com/google/devtools/build/lib/syntax:frontend",
        "//third_party:guava",
    ],
)

java_library(
    name = "config/transition_factories",
    srcs = ["config/TransitionFactories.java"],
    deps = [
        ":config/host_transition",
        ":config/transitions/configuration_transition",
        ":config/transitions/no_transition",
        ":config/transitions/null_transition",
        ":config/transitions/split_transition",
        ":config/transitions/transition_factory",
        "//third_party:auto_value",
    ],
)

java_library(
    name = "config/transitive_option_details",
    srcs = ["config/TransitiveOptionDetails.java"],
    deps = [
        ":config/fragment_options",
        "//src/main/java/com/google/devtools/build/lib/cmdline",
        "//src/main/java/com/google/devtools/common/options",
        "//third_party:guava",
        "//third_party:jsr305",
    ],
)

# TODO(b/144899336): This should be config/transitions/BUILD
java_library(
    name = "config/transitions/composing_transition",
    srcs = ["config/transitions/ComposingTransition.java"],
    deps = [
        ":config/build_options",
        ":config/fragment_options",
        ":config/transitions/configuration_transition",
        ":config/transitions/no_transition",
        ":config/transitions/null_transition",
        "//src/main/java/com/google/devtools/build/lib/events",
        "//src/main/java/com/google/devtools/build/lib/skyframe/serialization/autocodec",
        "//third_party:guava",
    ],
)

java_library(
    name = "config/transitions/composing_transition_factory",
    srcs = ["config/transitions/ComposingTransitionFactory.java"],
    deps = [
        ":config/transitions/composing_transition",
        ":config/transitions/configuration_transition",
        ":config/transitions/no_transition",
        ":config/transitions/null_transition",
        ":config/transitions/transition_factory",
        "//third_party:auto_value",
        "//third_party:guava",
    ],
)

java_library(
    name = "config/transitions/configuration_transition",
    srcs = [
        "config/transitions/ConfigurationTransition.java",
        "config/transitions/TransitionUtil.java",
    ],
    deps = [
        ":config/build_options",
        ":config/fragment_options",
        "//src/main/java/com/google/devtools/build/lib/events",
        "//third_party:guava",
    ],
)

java_library(
    name = "config/transitions/no_transition",
    srcs = ["config/transitions/NoTransition.java"],
    deps = [
        ":config/build_options",
        ":config/transitions/configuration_transition",
        ":config/transitions/patch_transition",
        ":config/transitions/transition_factory",
        "//src/main/java/com/google/devtools/build/lib/events",
        "//src/main/java/com/google/devtools/build/lib/skyframe/serialization/autocodec",
        "//third_party:auto_value",
    ],
)

java_library(
    name = "config/transitions/null_transition",
    srcs = ["config/transitions/NullTransition.java"],
    deps = [
        ":config/build_options",
        ":config/transitions/configuration_transition",
        ":config/transitions/patch_transition",
        ":config/transitions/transition_factory",
        "//src/main/java/com/google/devtools/build/lib/events",
        "//src/main/java/com/google/devtools/build/lib/skyframe/serialization/autocodec",
        "//third_party:auto_value",
    ],
)

java_library(
    name = "config/transitions/patch_transition",
    srcs = ["config/transitions/PatchTransition.java"],
    deps = [
        ":config/build_options",
        ":config/transitions/configuration_transition",
        "//src/main/java/com/google/devtools/build/lib/events",
    ],
)

java_library(
    name = "config/transitions/split_transition",
    srcs = ["config/transitions/SplitTransition.java"],
    deps = [
        ":config/build_options",
        ":config/transitions/configuration_transition",
        "//src/main/java/com/google/devtools/build/lib/concurrent",
        "//src/main/java/com/google/devtools/build/lib/events",
        "//third_party:guava",
    ],
)

java_library(
    name = "config/transitions/transition_factory",
    srcs = ["config/transitions/TransitionFactory.java"],
    deps = [":config/transitions/configuration_transition"],
)

# TODO(b/144899336): This should be analysis/constraints/BUILD
java_library(
    name = "constraints/constraint_constants",
    srcs = ["constraints/ConstraintConstants.java"],
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
    ],
)

java_library(
    name = "constraints/environment_collection",
    srcs = ["constraints/EnvironmentCollection.java"],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/cmdline",
        "//src/main/java/com/google/devtools/build/lib/concurrent",
        "//src/main/java/com/google/devtools/build/lib/packages",
        "//src/main/java/com/google/devtools/build/lib/skyframe/serialization/autocodec",
        "//third_party:auto_value",
        "//third_party:guava",
    ],
)

java_library(
    name = "constraints/environment_rule",
    srcs = ["constraints/EnvironmentRule.java"],
    deps = [
        ":analysis_cluster",
        ":config/host_transition",
        ":constraints/constraint_constants",
        ":constraints/environment",
        ":rule_definition_environment",
        "//src/main/java/com/google/devtools/build/lib/packages",
        "//src/main/java/com/google/devtools/build/lib/packages:build_type",
        "//src/main/java/com/google/devtools/build/lib/packages:type",
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
        "//src/main/java/com/google/devtools/build/lib/skyframe/serialization/autocodec",
        "//third_party:guava",
    ],
)

java_library(
    name = "constraints/supported_environments_provider",
    srcs = ["constraints/SupportedEnvironmentsProvider.java"],
    deps = [
        ":constraints/environment_collection",
        ":label_and_location",
        "//src/main/java/com/google/devtools/build/lib/analysis:transitive_info_provider",
        "//src/main/java/com/google/devtools/build/lib/cmdline",
        "//src/main/java/com/google/devtools/build/lib/skyframe/serialization/autocodec",
        "//third_party:auto_value",
    ],
)

java_library(
    name = "constraints/top_level_constraint_semantics",
    srcs = ["constraints/TopLevelConstraintSemantics.java"],
    deps = [
        ":analysis_cluster",
        ":config/build_configuration",
        ":configured_target",
        ":constraints/constraint_semantics",
        ":constraints/environment_collection",
        ":constraints/supported_environments_provider",
        ":transitive_info_collection",
        ":view_creation_failed_exception",
        "//src/main/java/com/google/devtools/build/lib/cmdline",
        "//src/main/java/com/google/devtools/build/lib/events",
        "//src/main/java/com/google/devtools/build/lib/packages",
        "//src/main/java/com/google/devtools/build/lib/pkgcache",
        "//src/main/java/com/google/devtools/build/lib/skyframe:build_configuration_value",
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
        ":actions/deterministic_writer",
        ":actions/proto_deterministic_writer",
        "//src/main/java/com/google/devtools/build/lib/actions",
        "//src/main/java/com/google/devtools/build/lib/collect/nestedset",
        "//src/main/java/com/google/devtools/build/lib/concurrent",
        "//src/main/java/com/google/devtools/build/lib/skyframe/serialization/autocodec",
        "//src/main/java/com/google/devtools/build/lib/util",
        "//third_party:guava",
    ],
)

# TODO(b/144899336): This should be analysis/skylark/BUILD
java_library(
    name = "skylark/args",
    srcs = ["skylark/Args.java"],
    deps = [
        ":skylark/starlark_custom_command_line",
        "//src/main/java/com/google/devtools/build/lib/actions",
        "//src/main/java/com/google/devtools/build/lib/collect/nestedset",
        "//src/main/java/com/google/devtools/build/lib/concurrent",
        "//src/main/java/com/google/devtools/build/lib/skylarkbuildapi",
        "//src/main/java/com/google/devtools/build/lib/syntax:evaluator",
        "//src/main/java/com/google/devtools/build/lib/syntax:frontend",
        "//third_party:guava",
        "//third_party:jsr305",
    ],
)

java_library(
    name = "skylark/bazel_build_api_globals",
    srcs = ["skylark/BazelBuildApiGlobals.java"],
    deps = [
        ":skylark/starlark_late_bound_default",
        "//src/main/java/com/google/devtools/build/lib/packages",
        "//src/main/java/com/google/devtools/build/lib/skylarkbuildapi",
        "//src/main/java/com/google/devtools/build/lib/syntax:evaluator",
    ],
)

java_library(
    name = "skylark/function_transition_util",
    srcs = ["skylark/FunctionTransitionUtil.java"],
    deps = [
        ":config/build_options",
        ":config/core_options",
        ":config/fragment_options",
        ":config/starlark_defined_config_transition",
        "//src/main/java/com/google/devtools/build/lib/cmdline",
        "//src/main/java/com/google/devtools/build/lib/events",
        "//src/main/java/com/google/devtools/build/lib/packages",
        "//src/main/java/com/google/devtools/build/lib/syntax:evaluator",
        "//src/main/java/com/google/devtools/build/lib/syntax:frontend",
        "//src/main/java/com/google/devtools/build/lib/util",
        "//src/main/java/com/google/devtools/common/options",
        "//third_party:guava",
    ],
)

java_library(
    name = "skylark/starlark_api_provider",
    srcs = ["skylark/StarlarkApiProvider.java"],
    deps = [
        ":provider_collection",
        "//third_party:guava",
    ],
)

java_library(
    name = "skylark/starlark_command_line",
    srcs = ["skylark/StarlarkCommandLine.java"],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/actions",
        "//src/main/java/com/google/devtools/build/lib/collect/nestedset",
        "//src/main/java/com/google/devtools/build/lib/skylarkbuildapi",
        "//src/main/java/com/google/devtools/build/lib/syntax:evaluator",
    ],
)

java_library(
    name = "skylark/starlark_custom_command_line",
    srcs = ["skylark/StarlarkCustomCommandLine.java"],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/actions",
        "//src/main/java/com/google/devtools/build/lib/actions:commandline_item",
        "//src/main/java/com/google/devtools/build/lib/cmdline",
        "//src/main/java/com/google/devtools/build/lib/collect/nestedset",
        "//src/main/java/com/google/devtools/build/lib/concurrent",
        "//src/main/java/com/google/devtools/build/lib/skyframe/serialization/autocodec",
        "//src/main/java/com/google/devtools/build/lib/skylarkbuildapi",
        "//src/main/java/com/google/devtools/build/lib/syntax:evaluator",
        "//src/main/java/com/google/devtools/build/lib/syntax:frontend",
        "//src/main/java/com/google/devtools/build/lib/util",
        "//src/main/java/com/google/devtools/build/lib/vfs:pathfragment",
        "//third_party:guava",
        "//third_party:jsr305",
    ],
)

java_library(
    name = "skylark/starlark_error_reporter",
    srcs = ["skylark/StarlarkErrorReporter.java"],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/packages",
        "//src/main/java/com/google/devtools/build/lib/syntax:evaluator",
    ],
)

java_library(
    name = "skylark/starlark_late_bound_default",
    srcs = ["skylark/StarlarkLateBoundDefault.java"],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/analysis/skylark/annotations",
        "//src/main/java/com/google/devtools/build/lib/cmdline",
        "//src/main/java/com/google/devtools/build/lib/packages",
        "//src/main/java/com/google/devtools/build/lib/skyframe/serialization/autocodec",
        "//src/main/java/com/google/devtools/build/lib/skylarkbuildapi",
        "//src/main/java/com/google/devtools/build/lib/syntax:evaluator",
        "//src/main/java/net/starlark/java/annot",
        "//third_party:guava",
        "//third_party:jsr305",
    ],
)

java_library(
    name = "skylark/starlark_config",
    srcs = ["skylark/StarlarkConfig.java"],
    deps = [
        ":config/execution_transition_factory",
        "//src/main/java/com/google/devtools/build/lib/packages",
        "//src/main/java/com/google/devtools/build/lib/packages:type",
        "//src/main/java/com/google/devtools/build/lib/skylarkbuildapi",
        "//src/main/java/com/google/devtools/build/lib/syntax:evaluator",
    ],
)

# TODO(b/144899336): This should be lib/analysis/test/BUILD
java_library(
    name = "test/analysis_failure",
    srcs = ["test/AnalysisFailure.java"],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/cmdline",
        "//src/main/java/com/google/devtools/build/lib/collect/nestedset",
        "//src/main/java/com/google/devtools/build/lib/skylarkbuildapi/test",
        "//src/main/java/com/google/devtools/build/lib/syntax:evaluator",
    ],
)

java_library(
    name = "test/analysis_failure_info",
    srcs = ["test/AnalysisFailureInfo.java"],
    deps = [
        ":test/analysis_failure",
        "//src/main/java/com/google/devtools/build/lib/collect/nestedset",
        "//src/main/java/com/google/devtools/build/lib/packages",
        "//src/main/java/com/google/devtools/build/lib/skylarkbuildapi/test",
    ],
)

java_library(
    name = "test/analysis_test_result_info",
    srcs = ["test/AnalysisTestResultInfo.java"],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/packages",
        "//src/main/java/com/google/devtools/build/lib/skylarkbuildapi/test",
    ],
)

java_library(
    name = "test/baseline_coverage_result",
    srcs = ["test/BaselineCoverageResult.java"],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/actions",
        "//src/main/java/com/google/devtools/build/lib/events",
        "//third_party:guava",
    ],
)

java_library(
    name = "test/coverage_report_action_factory",
    srcs = ["test/CoverageReportActionFactory.java"],
    deps = [
        ":blaze_directories",
        ":configured_target",
        "//src/main/java/com/google/devtools/build/lib/actions",
        "//src/main/java/com/google/devtools/build/lib/collect/nestedset",
        "//src/main/java/com/google/devtools/build/lib/events",
        "//src/main/java/com/google/devtools/build/lib/skyframe:coverage_report_value",
        "//third_party:guava",
        "//third_party:jsr305",
    ],
)

java_library(
    name = "test/execution_info",
    srcs = ["test/ExecutionInfo.java"],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/concurrent",
        "//src/main/java/com/google/devtools/build/lib/packages",
        "//src/main/java/com/google/devtools/build/lib/skylarkbuildapi/test",
        "//third_party:guava",
    ],
)

java_library(
    name = "test/instrumented_files_info",
    srcs = ["test/InstrumentedFilesInfo.java"],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/actions",
        "//src/main/java/com/google/devtools/build/lib/collect/nestedset",
        "//src/main/java/com/google/devtools/build/lib/packages",
        "//src/main/java/com/google/devtools/build/lib/skyframe/serialization/autocodec",
        "//src/main/java/com/google/devtools/build/lib/skylarkbuildapi/test",
        "//src/main/java/com/google/devtools/build/lib/util",
    ],
)

java_library(
    name = "test/test_configuration",
    srcs = ["test/TestConfiguration.java"],
    deps = [
        ":config/build_options",
        ":config/core_option_converters",
        ":config/fragment",
        ":config/fragment_options",
        ":config/fragment_provider",
        ":config/invalid_configuration_exception",
        ":config/per_label_options",
        ":options_diff_predicate",
        ":test/test_sharding_strategy",
        "//src/main/java/com/google/devtools/build/lib/cmdline",
        "//src/main/java/com/google/devtools/build/lib/packages",
        "//src/main/java/com/google/devtools/build/lib/util",
        "//src/main/java/com/google/devtools/common/options",
        "//third_party:guava",
    ],
)

java_library(
    name = "test/test_environment_info",
    srcs = ["test/TestEnvironmentInfo.java"],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/concurrent",
        "//src/main/java/com/google/devtools/build/lib/packages",
        "//src/main/java/com/google/devtools/build/lib/skylarkbuildapi/test",
        "//third_party:guava",
    ],
)

java_library(
    name = "test/test_sharding_strategy",
    srcs = ["test/TestShardingStrategy.java"],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/packages",
        "//src/main/java/com/google/devtools/common/options",
    ],
)

java_library(
    name = "test/test_trimming_transition_factory",
    srcs = ["test/TestTrimmingTransitionFactory.java"],
    deps = [
        ":config/build_options",
        ":config/fragment_options",
        ":config/transitions/no_transition",
        ":config/transitions/patch_transition",
        ":config/transitions/transition_factory",
        ":test/test_configuration",
        "//src/main/java/com/google/devtools/build/lib/events",
        "//src/main/java/com/google/devtools/build/lib/packages",
        "//src/main/java/com/google/devtools/common/options",
        "//third_party:guava",
    ],
)
