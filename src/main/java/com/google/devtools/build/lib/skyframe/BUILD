load("@rules_java//java:defs.bzl", "java_library")

package(default_visibility = ["//src:__subpackages__"])

filegroup(
    name = "srcs",
    srcs = glob(["**"]) + [
        "//src/main/java/com/google/devtools/build/lib/skyframe/packages:srcs",
        "//src/main/java/com/google/devtools/build/lib/skyframe/proto:srcs",
        "//src/main/java/com/google/devtools/build/lib/skyframe/serialization:srcs",
        "//src/main/java/com/google/devtools/build/lib/skyframe/trimming:srcs",
    ],
    visibility = ["//src:__subpackages__"],
)

# TODO(b/144899336): Split this further.
java_library(
    name = "skyframe_cluster",
    srcs = [
        "ASTFileLookupFunction.java",
        "ActionArtifactCycleReporter.java",
        "ActionExecutionFunction.java",
        "ActionLookupConflictFindingFunction.java",
        "AspectCompletionValue.java",
        "AspectCompletor.java",
        "AspectFunction.java",
        "BazelSkyframeExecutorConstants.java",
        "BuildConfigurationFunction.java",
        "BuildInfoCollectionFunction.java",
        "BzlLoadFunction.java",
        "CompletionFunction.java",
        "ConfiguredTargetCycleReporter.java",
        "ConfiguredTargetFunction.java",
        "ConfiguredTargetValue.java",
        "ConstraintValueLookupUtil.java",
        "CoverageReportFunction.java",
        "DirectoryListingStateFunction.java",
        "DirtinessCheckerUtils.java",
        "EnvironmentBackedRecursivePackageProvider.java",
        "ExternalFilesHelper.java",
        "ExternalPackageFunction.java",
        "FileStateFunction.java",
        "LocalRepositoryLookupFunction.java",
        "NonRuleConfiguredTargetValue.java",
        "PackageFunction.java",
        "PlatformLookupUtil.java",
        "PlatformMappingFunction.java",
        "PrepareAnalysisPhaseFunction.java",
        "PrepareAnalysisPhaseValue.java",
        "PrepareDepsOfPatternFunction.java",
        "RegisteredExecutionPlatformsFunction.java",
        "RegisteredExecutionPlatformsValue.java",
        "RegisteredToolchainsCycleReporter.java",
        "RegisteredToolchainsFunction.java",
        "RegisteredToolchainsValue.java",
        "RuleConfiguredTargetValue.java",
        "SequencedSkyframeExecutor.java",
        "SequencedSkyframeExecutorFactory.java",
        "SingleToolchainResolutionFunction.java",
        "SingleToolchainResolutionValue.java",
        "SkyFunctionEnvironmentForTesting.java",
        "SkyframeActionExecutor.java",
        "SkyframeAnalysisResult.java",
        "SkyframeBuildView.java",
        "SkyframeDependencyResolver.java",
        "SkyframeExecutor.java",
        "SkyframeExecutorFactory.java",
        "SkyframeExecutorWrappingWalkableGraph.java",
        "SkyframeLabelVisitor.java",
        "SkyframePackageManager.java",
        "SkyframeTargetPatternEvaluator.java",
        "SkyframeTransitivePackageLoader.java",
        "StarlarkAspectFactory.java",
        "StarlarkBuiltinsFunction.java",
        "StarlarkBuiltinsValue.java",
        "TargetCompletionValue.java",
        "TargetCompletor.java",
        "TargetPatternFunction.java",
        "TargetPatternPhaseFunction.java",
        "TestCompletionFunction.java",
        "TestCompletionValue.java",
        "ToolchainResolutionFunction.java",
        "ToolchainTypeLookupUtil.java",
        "TopLevelActionLookupConflictFindingFunction.java",
        "ToplevelStarlarkAspectFunction.java",
        "TransitiveTargetFunction.java",
        "TrimmedConfigurationProgressReceiver.java",
        "UnloadedToolchainContextImpl.java",
        "WorkspaceASTFunction.java",
        "WorkspaceFileFunction.java",
        "actiongraph/ActionGraphDump.java",
        "actiongraph/v2/ActionGraphDump.java",
    ],
    deps = [
        ":abstract_label_cycle_reporter",
        ":action_environment_function",
        ":action_execution_state",
        ":action_execution_value",
        ":action_input_map_helper",
        ":action_lookup_conflict_finding_value",
        ":action_metadata_handler",
        ":action_rewind_strategy",
        ":action_sketch_function",
        ":action_template_expansion_function",
        ":action_utils",
        ":actiongraph/known_artifacts",
        ":actiongraph/known_aspect_descriptors",
        ":actiongraph/known_configurations",
        ":actiongraph/known_nested_sets",
        ":actiongraph/known_rule_class_strings",
        ":actiongraph/known_targets",
        ":actiongraph/v2/aquery_output_handler",
        ":actiongraph/v2/known_artifacts",
        ":actiongraph/v2/known_aspect_descriptors",
        ":actiongraph/v2/known_configurations",
        ":actiongraph/v2/known_nested_sets",
        ":actiongraph/v2/known_rule_class_strings",
        ":actiongraph/v2/known_targets",
        ":artifact_conflict_finder",
        ":artifact_function",
        ":artifact_nested_set_function",
        ":artifact_nested_set_key",
        ":aspect_creation_exception",
        ":aspect_value_key",
        ":ast_file_lookup_value",
        ":blacklisted_package_prefixes_function",
        ":blacklisted_package_prefixes_value",
        ":build_configuration_value",
        ":build_info_collection_value",
        ":bzl_load_value",
        ":cached_bzl_load_value_and_deps",
        ":cached_bzl_load_value_and_deps_builder_factory",
        ":client_environment_function",
        ":client_environment_value",
        ":collect_packages_under_directory_function",
        ":collect_packages_under_directory_value",
        ":collect_targets_in_package_function",
        ":collect_test_suites_in_package_function",
        ":configuration_phase_started_event",
        ":configured_target_and_data",
        ":configured_target_key",
        ":configured_target_progress_receiver",
        ":configured_value_creation_exception",
        ":containing_package_lookup_function",
        ":containing_package_lookup_value",
        ":coverage_report_value",
        ":cycle_utils",
        ":diff_awareness",
        ":diff_awareness_manager",
        ":directory_listing_function",
        ":directory_listing_state_value",
        ":error_reading_skylark_extension_exception",
        ":execution_finished_event",
        ":file_function",
        ":file_symlink_cycle_uniqueness_function",
        ":file_symlink_exception",
        ":file_symlink_infinite_expansion_uniqueness_function",
        ":fileset_entry_function",
        ":filesystem_value_checker",
        ":glob_descriptor",
        ":glob_function",
        ":glob_value",
        ":loading_phase_started_event",
        ":local_repository_lookup_value",
        ":managed_directories_knowledge",
        ":map_as_package_roots",
        ":minimal_output_store",
        ":output_store",
        ":package_error_function",
        ":package_error_message_function",
        ":package_lookup_function",
        ":package_lookup_value",
        ":package_progress_receiver",
        ":package_roots_no_symlink_creation",
        ":package_value",
        ":pattern_expanding_error",
        ":per_build_syscall_cache",
        ":precomputed_function",
        ":precomputed_value",
        ":prepare_deps_of_pattern_value",
        ":prepare_deps_of_patterns_function",
        ":prepare_deps_of_patterns_value",
        ":prepare_deps_of_targets_under_directory_function",
        ":prepare_deps_of_targets_under_directory_value",
        ":prepare_test_suites_under_directory_function",
        ":prepare_test_suites_under_directory_value",
        ":progress_event_suppressing_environment",
        ":progress_suppressing_event_handler",
        ":recursive_filesystem_traversal",
        ":recursive_package_provider_backed_target_pattern_resolver",
        ":recursive_pkg_function",
        ":recursive_pkg_value",
        ":repository_mapping_function",
        ":repository_mapping_value",
        ":sane_analysis_exception",
        ":sky_functions",
        ":sky_value_dirtiness_checker",
        ":skyframe_aware_action",
        ":skyframe_incremental_build_monitor",
        ":skylark_module_cycle_reporter",
        ":starlark_file_dependency",
        ":state_informing_sky_function_environment",
        ":target_pattern_error_function",
        ":target_pattern_phase_value",
        ":target_pattern_util",
        ":target_pattern_value",
        ":target_patterns_result_builder",
        ":test_expansion_cycle_reporter",
        ":test_expansion_function",
        ":tests_for_target_pattern_function",
        ":tests_for_target_pattern_value",
        ":toolchain_context_key",
        ":toolchain_exception",
        ":top_down_action_cache",
        ":transitive_base_traversal_function",
        ":transitive_target_cycle_reporter",
        ":transitive_target_key",
        ":transitive_target_value",
        ":transitive_traversal_function",
        ":transitive_traversal_value",
        ":tree_artifact_value",
        ":unloaded_toolchain_context",
        ":workspace_ast_value",
        ":workspace_name_function",
        ":workspace_name_value",
        ":workspace_status_function",
        ":workspace_status_value",
        "//src/main/java/com/google/devtools/build/lib:build-request-options",
        "//src/main/java/com/google/devtools/build/lib:keep-going-option",
        "//src/main/java/com/google/devtools/build/lib/actions",
        "//src/main/java/com/google/devtools/build/lib/actionsketch:action_sketch",
        "//src/main/java/com/google/devtools/build/lib/analysis:actions/parameter_file_write_action",
        "//src/main/java/com/google/devtools/build/lib/analysis:analysis_cluster",
        "//src/main/java/com/google/devtools/build/lib/analysis:blaze_directories",
        "//src/main/java/com/google/devtools/build/lib/analysis:buildinfo/build_info_collection",
        "//src/main/java/com/google/devtools/build/lib/analysis:buildinfo/build_info_key",
        "//src/main/java/com/google/devtools/build/lib/analysis:config/build_configuration",
        "//src/main/java/com/google/devtools/build/lib/analysis:config/build_options",
        "//src/main/java/com/google/devtools/build/lib/analysis:config/config_matching_provider",
        "//src/main/java/com/google/devtools/build/lib/analysis:config/core_options",
        "//src/main/java/com/google/devtools/build/lib/analysis:config/fragment",
        "//src/main/java/com/google/devtools/build/lib/analysis:config/fragment_provider",
        "//src/main/java/com/google/devtools/build/lib/analysis:config/host_transition",
        "//src/main/java/com/google/devtools/build/lib/analysis:config/invalid_configuration_exception",
        "//src/main/java/com/google/devtools/build/lib/analysis:config/transitions/configuration_transition",
        "//src/main/java/com/google/devtools/build/lib/analysis:config/transitions/no_transition",
        "//src/main/java/com/google/devtools/build/lib/analysis:config/transitions/null_transition",
        "//src/main/java/com/google/devtools/build/lib/analysis:configurations_collector",
        "//src/main/java/com/google/devtools/build/lib/analysis:configured_object_value",
        "//src/main/java/com/google/devtools/build/lib/analysis:configured_target",
        "//src/main/java/com/google/devtools/build/lib/analysis:dependency",
        "//src/main/java/com/google/devtools/build/lib/analysis:dependency_key",
        "//src/main/java/com/google/devtools/build/lib/analysis:dependency_kind",
        "//src/main/java/com/google/devtools/build/lib/analysis:duplicate_exception",
        "//src/main/java/com/google/devtools/build/lib/analysis:inconsistent_aspect_order_exception",
        "//src/main/java/com/google/devtools/build/lib/analysis:platform_options",
        "//src/main/java/com/google/devtools/build/lib/analysis:resolved_toolchain_context",
        "//src/main/java/com/google/devtools/build/lib/analysis:toolchain_collection",
        "//src/main/java/com/google/devtools/build/lib/analysis:toolchain_context",
        "//src/main/java/com/google/devtools/build/lib/analysis:top_level_artifact_context",
        "//src/main/java/com/google/devtools/build/lib/analysis:view_creation_failed_exception",
        "//src/main/java/com/google/devtools/build/lib/analysis:workspace_status_action",
        "//src/main/java/com/google/devtools/build/lib/analysis/platform",
        "//src/main/java/com/google/devtools/build/lib/analysis/platform:utils",
        "//src/main/java/com/google/devtools/build/lib/bugreport",
        "//src/main/java/com/google/devtools/build/lib/buildeventstream",
        "//src/main/java/com/google/devtools/build/lib/buildeventstream/proto:build_event_stream_java_proto",
        "//src/main/java/com/google/devtools/build/lib/causes",
        "//src/main/java/com/google/devtools/build/lib/clock",
        "//src/main/java/com/google/devtools/build/lib/cmdline",
        "//src/main/java/com/google/devtools/build/lib/collect/compacthashset",
        "//src/main/java/com/google/devtools/build/lib/collect/nestedset",
        "//src/main/java/com/google/devtools/build/lib/concurrent",
        "//src/main/java/com/google/devtools/build/lib/events",
        "//src/main/java/com/google/devtools/build/lib/packages",
        "//src/main/java/com/google/devtools/build/lib/packages:build_type",
        "//src/main/java/com/google/devtools/build/lib/packages:starlark_semantics_options",
        "//src/main/java/com/google/devtools/build/lib/packages:type",
        "//src/main/java/com/google/devtools/build/lib/pkgcache",
        "//src/main/java/com/google/devtools/build/lib/profiler",
        "//src/main/java/com/google/devtools/build/lib/profiler:google-auto-profiler-utils",
        "//src/main/java/com/google/devtools/build/lib/profiler/memory:current_rule_tracker",
        "//src/main/java/com/google/devtools/build/lib/query2:aquery-utils",
        "//src/main/java/com/google/devtools/build/lib/remote/options",
        "//src/main/java/com/google/devtools/build/lib/repository:external_package_helper",
        "//src/main/java/com/google/devtools/build/lib/rules:repository/local_repository_rule",
        "//src/main/java/com/google/devtools/build/lib/rules:repository/repository_directory_value",
        "//src/main/java/com/google/devtools/build/lib/rules:repository/repository_function",
        "//src/main/java/com/google/devtools/build/lib/rules:repository/resolved_file_value",
        "//src/main/java/com/google/devtools/build/lib/rules:repository/workspace_file_helper",
        "//src/main/java/com/google/devtools/build/lib/rules/cpp:cpp_interface",
        "//src/main/java/com/google/devtools/build/lib/skyframe/serialization/autocodec",
        "//src/main/java/com/google/devtools/build/lib/skyframe/trimming:trimmed_configuration_cache",
        "//src/main/java/com/google/devtools/build/lib/syntax:evaluator",
        "//src/main/java/com/google/devtools/build/lib/syntax:frontend",
        "//src/main/java/com/google/devtools/build/lib/util",
        "//src/main/java/com/google/devtools/build/lib/util:TestType",
        "//src/main/java/com/google/devtools/build/lib/util:abrupt_exit_exception",
        "//src/main/java/com/google/devtools/build/lib/util:detailed_exit_code",
        "//src/main/java/com/google/devtools/build/lib/util:exit_code",
        "//src/main/java/com/google/devtools/build/lib/util:resource_usage",
        "//src/main/java/com/google/devtools/build/lib/util/io",
        "//src/main/java/com/google/devtools/build/lib/vfs",
        "//src/main/java/com/google/devtools/build/lib/vfs:output_service",
        "//src/main/java/com/google/devtools/build/lib/vfs:pathfragment",
        "//src/main/java/com/google/devtools/build/skyframe",
        "//src/main/java/com/google/devtools/build/skyframe:skyframe-objects",
        "//src/main/java/com/google/devtools/common/options",
        "//src/main/protobuf:analysis_java_proto",
        "//src/main/protobuf:analysis_v2_java_proto",
        "//src/main/protobuf:failure_details_java_proto",
        "//third_party:auto_value",
        "//third_party:error_prone_annotations",
        "//third_party:flogger",
        "//third_party:guava",
        "//third_party:jsr305",
    ],
)

java_library(
    name = "abstract_chain_uniqueness_function",
    srcs = ["AbstractChainUniquenessFunction.java"],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/events",
        "//src/main/java/com/google/devtools/build/skyframe",
        "//src/main/java/com/google/devtools/build/skyframe:skyframe-objects",
        "//third_party:guava",
    ],
)

java_library(
    name = "abstract_label_cycle_reporter",
    srcs = ["AbstractLabelCycleReporter.java"],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/cmdline",
        "//src/main/java/com/google/devtools/build/lib/concurrent",
        "//src/main/java/com/google/devtools/build/lib/events",
        "//src/main/java/com/google/devtools/build/lib/packages",
        "//src/main/java/com/google/devtools/build/lib/pkgcache",
        "//src/main/java/com/google/devtools/build/skyframe",
        "//src/main/java/com/google/devtools/build/skyframe:skyframe-objects",
        "//third_party:guava",
    ],
)

java_library(
    name = "action_environment_function",
    srcs = ["ActionEnvironmentFunction.java"],
    deps = [
        ":client_environment_function",
        ":client_environment_value",
        ":precomputed_value",
        ":sky_functions",
        "//src/main/java/com/google/devtools/build/lib/concurrent",
        "//src/main/java/com/google/devtools/build/lib/skyframe/serialization/autocodec",
        "//src/main/java/com/google/devtools/build/skyframe",
        "//src/main/java/com/google/devtools/build/skyframe:skyframe-objects",
        "//third_party:guava",
        "//third_party:jsr305",
    ],
)

java_library(
    name = "action_execution_inactivity_watchdog",
    srcs = ["ActionExecutionInactivityWatchdog.java"],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/actions",
        "//third_party:guava",
    ],
)

java_library(
    name = "action_execution_state",
    srcs = ["ActionExecutionState.java"],
    deps = [
        ":action_execution_value",
        "//src/main/java/com/google/devtools/build/lib/actions",
        "//src/main/java/com/google/devtools/build/skyframe",
        "//third_party:guava",
        "//third_party:jsr305",
    ],
)

java_library(
    name = "action_execution_value",
    srcs = ["ActionExecutionValue.java"],
    deps = [
        ":output_store",
        ":tree_artifact_value",
        "//src/main/java/com/google/devtools/build/lib/actions",
        "//src/main/java/com/google/devtools/build/lib/collect/nestedset",
        "//src/main/java/com/google/devtools/build/lib/concurrent",
        "//src/main/java/com/google/devtools/build/skyframe:skyframe-objects",
        "//third_party:guava",
        "//third_party:jsr305",
    ],
)

java_library(
    name = "action_input_map_helper",
    srcs = ["ActionInputMapHelper.java"],
    deps = [
        ":action_execution_value",
        ":aggregating_artifact_value",
        ":runfiles_artifact_value",
        ":tree_artifact_value",
        "//src/main/java/com/google/devtools/build/lib/actions",
        "//src/main/java/com/google/devtools/build/lib/analysis:actions/symlink_action",
        "//src/main/java/com/google/devtools/build/lib/util",
        "//src/main/java/com/google/devtools/build/skyframe",
        "//src/main/java/com/google/devtools/build/skyframe:skyframe-objects",
        "//third_party:guava",
    ],
)

java_library(
    name = "action_lookup_conflict_finding_value",
    srcs = ["ActionLookupConflictFindingValue.java"],
    deps = [
        ":sky_functions",
        "//src/main/java/com/google/devtools/build/lib/actions",
        "//src/main/java/com/google/devtools/build/lib/concurrent",
        "//src/main/java/com/google/devtools/build/lib/skyframe/serialization/autocodec",
        "//src/main/java/com/google/devtools/build/skyframe:skyframe-objects",
        "//third_party:guava",
    ],
)

java_library(
    name = "action_metadata_handler",
    srcs = ["ActionMetadataHandler.java"],
    deps = [
        ":output_store",
        ":tree_artifact_value",
        "//src/main/java/com/google/devtools/build/lib/actions",
        "//src/main/java/com/google/devtools/build/lib/util/io",
        "//src/main/java/com/google/devtools/build/lib/vfs",
        "//src/main/java/com/google/devtools/build/lib/vfs:pathfragment",
        "//third_party:flogger",
        "//third_party:guava",
        "//third_party:jsr305",
    ],
)

java_library(
    name = "action_rewind_strategy",
    srcs = [
        "ActionRewindStrategy.java",
        "ActionRewindingStats.java",
    ],
    deps = [
        ":action_utils",
        ":artifact_function",
        ":skyframe_aware_action",
        "//src/main/java/com/google/devtools/build/lib/actions",
        "//src/main/java/com/google/devtools/build/lib/bugreport",
        "//src/main/java/com/google/devtools/build/lib/events",
        "//src/main/java/com/google/devtools/build/lib/skyframe/proto:action_rewind_event_java_proto",
        "//src/main/java/com/google/devtools/build/skyframe",
        "//src/main/java/com/google/devtools/build/skyframe:skyframe-objects",
        "//third_party:auto_value",
        "//third_party:flogger",
        "//third_party:guava",
        "//third_party:jsr305",
    ],
)

java_library(
    name = "action_sketch_function",
    srcs = ["ActionSketchFunction.java"],
    deps = [
        ":artifact_function",
        ":sky_functions",
        "//src/main/java/com/google/devtools/build/lib/actions",
        "//src/main/java/com/google/devtools/build/lib/actionsketch:action_sketch",
        "//src/main/java/com/google/devtools/build/lib/concurrent",
        "//src/main/java/com/google/devtools/build/lib/skyframe/serialization/autocodec",
        "//src/main/java/com/google/devtools/build/lib/util",
        "//src/main/java/com/google/devtools/build/skyframe",
        "//src/main/java/com/google/devtools/build/skyframe:skyframe-objects",
        "//third_party:guava",
        "//third_party:jsr305",
    ],
)

java_library(
    name = "action_template_expansion_function",
    srcs = ["ActionTemplateExpansionFunction.java"],
    deps = [
        ":action_template_expansion_value",
        ":tree_artifact_value",
        "//src/main/java/com/google/devtools/build/lib/actions",
        "//src/main/java/com/google/devtools/build/lib/bugreport",
        "//src/main/java/com/google/devtools/build/lib/events",
        "//src/main/java/com/google/devtools/build/lib/vfs:pathfragment",
        "//src/main/java/com/google/devtools/build/skyframe",
        "//src/main/java/com/google/devtools/build/skyframe:skyframe-objects",
        "//third_party:guava",
        "//third_party:jsr305",
    ],
)

java_library(
    name = "action_template_expansion_value",
    srcs = ["ActionTemplateExpansionValue.java"],
    deps = [
        ":sky_functions",
        "//src/main/java/com/google/devtools/build/lib/actions",
        "//src/main/java/com/google/devtools/build/lib/cmdline",
        "//src/main/java/com/google/devtools/build/lib/concurrent",
        "//src/main/java/com/google/devtools/build/lib/skyframe/serialization/autocodec",
        "//src/main/java/com/google/devtools/build/skyframe:skyframe-objects",
        "//third_party:guava",
    ],
)

java_library(
    name = "action_utils",
    srcs = ["ActionUtils.java"],
    deps = [
        ":artifact_function",
        "//src/main/java/com/google/devtools/build/lib/actions",
        "//src/main/java/com/google/devtools/build/skyframe",
        "//third_party:jsr305",
    ],
)

java_library(
    name = "actiongraph/base_cache",
    srcs = ["actiongraph/BaseCache.java"],
    deps = ["//src/main/protobuf:analysis_java_proto"],
)

java_library(
    name = "actiongraph/known_artifacts",
    srcs = ["actiongraph/KnownArtifacts.java"],
    deps = [
        ":actiongraph/base_cache",
        "//src/main/java/com/google/devtools/build/lib/actions",
        "//src/main/protobuf:analysis_java_proto",
    ],
)

java_library(
    name = "actiongraph/known_aspect_descriptors",
    srcs = ["actiongraph/KnownAspectDescriptors.java"],
    deps = [
        ":actiongraph/base_cache",
        "//src/main/java/com/google/devtools/build/lib/packages",
        "//src/main/protobuf:analysis_java_proto",
    ],
)

java_library(
    name = "actiongraph/known_configurations",
    srcs = ["actiongraph/KnownConfigurations.java"],
    deps = [
        ":actiongraph/base_cache",
        "//src/main/java/com/google/devtools/build/lib/buildeventstream",
        "//src/main/java/com/google/devtools/build/lib/buildeventstream/proto:build_event_stream_java_proto",
        "//src/main/protobuf:analysis_java_proto",
    ],
)

java_library(
    name = "actiongraph/known_nested_sets",
    srcs = ["actiongraph/KnownNestedSets.java"],
    deps = [
        ":actiongraph/base_cache",
        ":actiongraph/known_artifacts",
        "//src/main/java/com/google/devtools/build/lib/actions",
        "//src/main/java/com/google/devtools/build/lib/collect/nestedset",
        "//src/main/protobuf:analysis_java_proto",
    ],
)

java_library(
    name = "actiongraph/known_rule_class_strings",
    srcs = ["actiongraph/KnownRuleClassStrings.java"],
    deps = [
        ":actiongraph/base_cache",
        "//src/main/protobuf:analysis_java_proto",
    ],
)

java_library(
    name = "actiongraph/known_targets",
    srcs = ["actiongraph/KnownTargets.java"],
    deps = [
        ":actiongraph/base_cache",
        ":actiongraph/known_rule_class_strings",
        "//src/main/java/com/google/devtools/build/lib/util",
        "//src/main/protobuf:analysis_java_proto",
    ],
)

java_library(
    name = "actiongraph/v2/aquery_output_handler",
    srcs = ["actiongraph/v2/AqueryOutputHandler.java"],
    deps = ["//src/main/protobuf:analysis_v2_java_proto"],
)

java_library(
    name = "actiongraph/v2/base_cache",
    srcs = ["actiongraph/v2/BaseCache.java"],
    deps = [":actiongraph/v2/aquery_output_handler"],
)

java_library(
    name = "actiongraph/v2/known_artifacts",
    srcs = ["actiongraph/v2/KnownArtifacts.java"],
    deps = [
        ":actiongraph/v2/aquery_output_handler",
        ":actiongraph/v2/base_cache",
        ":actiongraph/v2/known_path_fragments",
        "//src/main/java/com/google/devtools/build/lib/actions",
        "//src/main/protobuf:analysis_v2_java_proto",
    ],
)

java_library(
    name = "actiongraph/v2/known_aspect_descriptors",
    srcs = ["actiongraph/v2/KnownAspectDescriptors.java"],
    deps = [
        ":actiongraph/v2/aquery_output_handler",
        ":actiongraph/v2/base_cache",
        "//src/main/java/com/google/devtools/build/lib/packages",
        "//src/main/protobuf:analysis_v2_java_proto",
    ],
)

java_library(
    name = "actiongraph/v2/known_configurations",
    srcs = ["actiongraph/v2/KnownConfigurations.java"],
    deps = [
        ":actiongraph/v2/aquery_output_handler",
        ":actiongraph/v2/base_cache",
        "//src/main/java/com/google/devtools/build/lib/buildeventstream",
        "//src/main/java/com/google/devtools/build/lib/buildeventstream/proto:build_event_stream_java_proto",
        "//src/main/protobuf:analysis_v2_java_proto",
    ],
)

java_library(
    name = "actiongraph/v2/known_nested_sets",
    srcs = ["actiongraph/v2/KnownNestedSets.java"],
    deps = [
        ":actiongraph/v2/aquery_output_handler",
        ":actiongraph/v2/base_cache",
        ":actiongraph/v2/known_artifacts",
        "//src/main/java/com/google/devtools/build/lib/actions",
        "//src/main/java/com/google/devtools/build/lib/collect/nestedset",
        "//src/main/protobuf:analysis_v2_java_proto",
    ],
)

java_library(
    name = "actiongraph/v2/known_path_fragments",
    srcs = ["actiongraph/v2/KnownPathFragments.java"],
    deps = [
        ":actiongraph/v2/aquery_output_handler",
        ":actiongraph/v2/base_cache",
        "//src/main/java/com/google/devtools/build/lib/vfs:pathfragment",
        "//src/main/protobuf:analysis_v2_java_proto",
    ],
)

java_library(
    name = "actiongraph/v2/known_rule_class_strings",
    srcs = ["actiongraph/v2/KnownRuleClassStrings.java"],
    deps = [
        ":actiongraph/v2/aquery_output_handler",
        ":actiongraph/v2/base_cache",
        "//src/main/protobuf:analysis_v2_java_proto",
    ],
)

java_library(
    name = "actiongraph/v2/known_targets",
    srcs = ["actiongraph/v2/KnownTargets.java"],
    deps = [
        ":actiongraph/v2/aquery_output_handler",
        ":actiongraph/v2/base_cache",
        ":actiongraph/v2/known_rule_class_strings",
        "//src/main/java/com/google/devtools/build/lib/util",
        "//src/main/protobuf:analysis_v2_java_proto",
    ],
)

java_library(
    name = "actiongraph/v2/monolithic_output_handler",
    srcs = ["actiongraph/v2/MonolithicOutputHandler.java"],
    deps = [
        ":actiongraph/v2/aquery_output_handler",
        "//src/main/protobuf:analysis_v2_java_proto",
        "//third_party/protobuf:protobuf_java_util",
    ],
)

java_library(
    name = "actiongraph/v2/streamed_output_handler",
    srcs = ["actiongraph/v2/StreamedOutputHandler.java"],
    deps = [
        ":actiongraph/v2/aquery_output_handler",
        "//src/main/protobuf:analysis_v2_java_proto",
        "//third_party:guava",
        "//third_party/protobuf:protobuf_java",
    ],
)

java_library(
    name = "aggregating_artifact_value",
    srcs = ["AggregatingArtifactValue.java"],
    deps = [
        ":tree_artifact_value",
        "//src/main/java/com/google/devtools/build/lib/actions",
        "//src/main/java/com/google/devtools/build/lib/util",
        "//src/main/java/com/google/devtools/build/skyframe:skyframe-objects",
        "//third_party:guava",
    ],
)

java_library(
    name = "artifact_conflict_finder",
    srcs = ["ArtifactConflictFinder.java"],
    deps = [
        ":precomputed_value",
        "//src/main/java/com/google/devtools/build/lib/actions",
        "//src/main/java/com/google/devtools/build/lib/concurrent",
        "//src/main/java/com/google/devtools/build/lib/util",
        "//src/main/java/com/google/devtools/build/lib/vfs:pathfragment",
        "//third_party:guava",
        "//third_party:jsr305",
    ],
)

java_library(
    name = "artifact_function",
    srcs = ["ArtifactFunction.java"],
    deps = [
        ":action_execution_value",
        ":action_template_expansion_value",
        ":aggregating_artifact_value",
        ":coverage_report_value",
        ":recursive_filesystem_traversal",
        ":runfiles_artifact_value",
        ":track_source_directories_flag",
        ":tree_artifact_value",
        "//src/main/java/com/google/devtools/build/lib/actions",
        "//src/main/java/com/google/devtools/build/lib/bugreport",
        "//src/main/java/com/google/devtools/build/lib/cmdline",
        "//src/main/java/com/google/devtools/build/lib/events",
        "//src/main/java/com/google/devtools/build/lib/util",
        "//src/main/java/com/google/devtools/build/lib/vfs",
        "//src/main/java/com/google/devtools/build/skyframe",
        "//src/main/java/com/google/devtools/build/skyframe:skyframe-objects",
        "//third_party:guava",
        "//third_party:jsr305",
    ],
)

java_library(
    name = "artifact_nested_set_function",
    srcs = ["ArtifactNestedSetFunction.java"],
    deps = [
        ":artifact_nested_set_key",
        ":artifact_nested_set_value",
        "//src/main/java/com/google/devtools/build/lib/actions",
        "//src/main/java/com/google/devtools/build/lib/collect/nestedset",
        "//src/main/java/com/google/devtools/build/lib/util",
        "//src/main/java/com/google/devtools/build/skyframe",
        "//src/main/java/com/google/devtools/build/skyframe:skyframe-objects",
        "//third_party:guava",
    ],
)

java_library(
    name = "artifact_nested_set_key",
    srcs = ["ArtifactNestedSetKey.java"],
    deps = [
        ":sky_functions",
        "//src/main/java/com/google/devtools/build/lib/actions",
        "//src/main/java/com/google/devtools/build/lib/collect/nestedset",
        "//src/main/java/com/google/devtools/build/skyframe:skyframe-objects",
        "//third_party:guava",
    ],
)

java_library(
    name = "artifact_nested_set_value",
    srcs = ["ArtifactNestedSetValue.java"],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/concurrent",
        "//src/main/java/com/google/devtools/build/skyframe:skyframe-objects",
    ],
)

java_library(
    name = "aspect_creation_exception",
    srcs = ["AspectCreationException.java"],
    deps = [
        ":sane_analysis_exception",
        "//src/main/java/com/google/devtools/build/lib/analysis:config/build_configuration",
        "//src/main/java/com/google/devtools/build/lib/buildeventstream/proto:build_event_stream_java_proto",
        "//src/main/java/com/google/devtools/build/lib/causes",
        "//src/main/java/com/google/devtools/build/lib/cmdline",
        "//src/main/java/com/google/devtools/build/lib/collect/nestedset",
        "//third_party:jsr305",
    ],
)

java_library(
    name = "aspect_value_key",
    srcs = ["AspectValueKey.java"],
    deps = [
        ":build_configuration_value",
        ":configured_target_key",
        ":sky_functions",
        "//src/main/java/com/google/devtools/build/lib/actions",
        "//src/main/java/com/google/devtools/build/lib/analysis:config/build_configuration",
        "//src/main/java/com/google/devtools/build/lib/cmdline",
        "//src/main/java/com/google/devtools/build/lib/concurrent",
        "//src/main/java/com/google/devtools/build/lib/packages",
        "//src/main/java/com/google/devtools/build/lib/skyframe/serialization/autocodec",
        "//src/main/java/com/google/devtools/build/skyframe:skyframe-objects",
        "//third_party:guava",
        "//third_party:jsr305",
    ],
)

java_library(
    name = "ast_file_lookup_value",
    srcs = ["ASTFileLookupValue.java"],
    deps = [
        ":sky_functions",
        "//src/main/java/com/google/devtools/build/lib/cmdline",
        "//src/main/java/com/google/devtools/build/lib/concurrent",
        "//src/main/java/com/google/devtools/build/lib/skyframe/serialization/autocodec",
        "//src/main/java/com/google/devtools/build/lib/syntax:frontend",
        "//src/main/java/com/google/devtools/build/skyframe",
        "//src/main/java/com/google/devtools/build/skyframe:skyframe-objects",
        "//third_party:error_prone_annotations",
        "//third_party:guava",
    ],
)

java_library(
    name = "blacklisted_package_prefixes_function",
    srcs = ["BlacklistedPackagePrefixesFunction.java"],
    deps = [
        ":blacklisted_package_prefixes_value",
        ":precomputed_value",
        "//src/main/java/com/google/devtools/build/lib/actions",
        "//src/main/java/com/google/devtools/build/lib/cmdline",
        "//src/main/java/com/google/devtools/build/lib/pkgcache",
        "//src/main/java/com/google/devtools/build/lib/rules:repository/repository_directory_value",
        "//src/main/java/com/google/devtools/build/lib/vfs",
        "//src/main/java/com/google/devtools/build/lib/vfs:pathfragment",
        "//src/main/java/com/google/devtools/build/skyframe",
        "//src/main/java/com/google/devtools/build/skyframe:skyframe-objects",
        "//third_party:guava",
        "//third_party:jsr305",
    ],
)

java_library(
    name = "blacklisted_package_prefixes_value",
    srcs = ["BlacklistedPackagePrefixesValue.java"],
    deps = [
        ":sky_functions",
        "//src/main/java/com/google/devtools/build/lib/cmdline",
        "//src/main/java/com/google/devtools/build/lib/concurrent",
        "//src/main/java/com/google/devtools/build/lib/skyframe/serialization/autocodec",
        "//src/main/java/com/google/devtools/build/lib/vfs:pathfragment",
        "//src/main/java/com/google/devtools/build/skyframe:skyframe-objects",
        "//third_party:guava",
    ],
)

java_library(
    name = "broken_diff_awareness_exception",
    srcs = ["BrokenDiffAwarenessException.java"],
    deps = ["//third_party:guava"],
)

java_library(
    name = "build_configuration_value",
    srcs = [
        "BuildConfigurationValue.java",
        "PlatformMappingValue.java",
    ],
    deps = [
        ":sky_functions",
        "//src/main/java/com/google/devtools/build/lib/analysis:config/build_configuration",
        "//src/main/java/com/google/devtools/build/lib/analysis:config/build_options",
        "//src/main/java/com/google/devtools/build/lib/analysis:config/fragment",
        "//src/main/java/com/google/devtools/build/lib/analysis:platform_options",
        "//src/main/java/com/google/devtools/build/lib/cmdline",
        "//src/main/java/com/google/devtools/build/lib/concurrent",
        "//src/main/java/com/google/devtools/build/lib/skyframe/serialization/autocodec",
        "//src/main/java/com/google/devtools/build/lib/vfs:pathfragment",
        "//src/main/java/com/google/devtools/build/skyframe:skyframe-objects",
        "//src/main/java/com/google/devtools/common/options",
        "//third_party:guava",
        "//third_party:jsr305",
    ],
)

java_library(
    name = "build_info_collection_value",
    srcs = ["BuildInfoCollectionValue.java"],
    deps = [
        ":build_configuration_value",
        ":sky_functions",
        "//src/main/java/com/google/devtools/build/lib/actions",
        "//src/main/java/com/google/devtools/build/lib/analysis:buildinfo/build_info_collection",
        "//src/main/java/com/google/devtools/build/lib/analysis:buildinfo/build_info_key",
        "//src/main/java/com/google/devtools/build/lib/analysis:config/build_configuration",
        "//src/main/java/com/google/devtools/build/lib/cmdline",
        "//src/main/java/com/google/devtools/build/lib/concurrent",
        "//src/main/java/com/google/devtools/build/lib/skyframe/serialization/autocodec",
        "//src/main/java/com/google/devtools/build/skyframe:skyframe-objects",
        "//third_party:guava",
        "//third_party:jsr305",
    ],
)

java_library(
    name = "builder",
    srcs = ["Builder.java"],
    deps = [
        ":aspect_value_key",
        ":configured_target_key",
        "//src/main/java/com/google/devtools/build/lib/actions",
        "//src/main/java/com/google/devtools/build/lib/analysis:analysis_cluster",
        "//src/main/java/com/google/devtools/build/lib/analysis:configured_target",
        "//src/main/java/com/google/devtools/build/lib/analysis:top_level_artifact_context",
        "//src/main/java/com/google/devtools/build/lib/concurrent",
        "//src/main/java/com/google/devtools/build/lib/events",
        "//src/main/java/com/google/devtools/build/lib/util:abrupt_exit_exception",
        "//src/main/java/com/google/devtools/common/options",
        "//third_party:guava",
        "//third_party:jsr305",
    ],
)

java_library(
    name = "cached_bzl_load_value_and_deps",
    srcs = ["CachedBzlLoadData.java"],
    deps = [
        ":bzl_load_value",
        "//src/main/java/com/google/devtools/build/skyframe:skyframe-objects",
        "//third_party:auto_value",
        "//third_party:guava",
    ],
)

java_library(
    name = "cached_bzl_load_value_and_deps_builder_factory",
    srcs = ["CachedBzlLoadDataBuilderFactory.java"],
    deps = [
        ":cached_bzl_load_value_and_deps",
        "//src/main/java/com/google/devtools/build/lib/concurrent",
        "//third_party:guava",
    ],
)

java_library(
    name = "chain_uniqueness_utils",
    srcs = ["ChainUniquenessUtils.java"],
    deps = ["//third_party:guava"],
)

java_library(
    name = "client_environment_function",
    srcs = ["ClientEnvironmentFunction.java"],
    deps = [
        ":client_environment_value",
        ":sky_functions",
        "//src/main/java/com/google/devtools/build/lib/concurrent",
        "//src/main/java/com/google/devtools/build/lib/skyframe/serialization/autocodec",
        "//src/main/java/com/google/devtools/build/skyframe",
        "//src/main/java/com/google/devtools/build/skyframe:skyframe-objects",
        "//third_party:guava",
        "//third_party:jsr305",
    ],
)

java_library(
    name = "client_environment_value",
    srcs = ["ClientEnvironmentValue.java"],
    deps = [
        "//src/main/java/com/google/devtools/build/skyframe:skyframe-objects",
        "//third_party:jsr305",
    ],
)

java_library(
    name = "collect_packages_under_directory_function",
    srcs = ["CollectPackagesUnderDirectoryFunction.java"],
    deps = [
        ":collect_packages_under_directory_value",
        ":recursive_directory_traversal_function",
        ":recursive_pkg_key",
        "//src/main/java/com/google/devtools/build/lib/analysis:blaze_directories",
        "//src/main/java/com/google/devtools/build/lib/cmdline",
        "//src/main/java/com/google/devtools/build/lib/vfs",
        "//src/main/java/com/google/devtools/build/lib/vfs:pathfragment",
        "//src/main/java/com/google/devtools/build/skyframe",
        "//src/main/java/com/google/devtools/build/skyframe:skyframe-objects",
        "//third_party:guava",
        "//third_party:jsr305",
    ],
)

java_library(
    name = "collect_packages_under_directory_value",
    srcs = ["CollectPackagesUnderDirectoryValue.java"],
    deps = [
        ":recursive_pkg_sky_key",
        ":sky_functions",
        "//src/main/java/com/google/devtools/build/lib/cmdline",
        "//src/main/java/com/google/devtools/build/lib/concurrent",
        "//src/main/java/com/google/devtools/build/lib/skyframe/serialization/autocodec",
        "//src/main/java/com/google/devtools/build/lib/vfs",
        "//src/main/java/com/google/devtools/build/lib/vfs:pathfragment",
        "//src/main/java/com/google/devtools/build/skyframe:skyframe-objects",
        "//third_party:guava",
        "//third_party:jsr305",
    ],
)

java_library(
    name = "collect_targets_in_package_function",
    srcs = ["CollectTargetsInPackageFunction.java"],
    deps = [
        ":collect_targets_in_package_value",
        ":package_value",
        ":transitive_traversal_value",
        "//src/main/java/com/google/devtools/build/lib/cmdline",
        "//src/main/java/com/google/devtools/build/lib/events",
        "//src/main/java/com/google/devtools/build/lib/packages",
        "//src/main/java/com/google/devtools/build/lib/pkgcache",
        "//src/main/java/com/google/devtools/build/skyframe",
        "//src/main/java/com/google/devtools/build/skyframe:skyframe-objects",
        "//third_party:guava",
        "//third_party:jsr305",
    ],
)

java_library(
    name = "collect_targets_in_package_value",
    srcs = ["CollectTargetsInPackageValue.java"],
    deps = [
        ":sky_functions",
        "//src/main/java/com/google/devtools/build/lib/cmdline",
        "//src/main/java/com/google/devtools/build/lib/concurrent",
        "//src/main/java/com/google/devtools/build/lib/pkgcache",
        "//src/main/java/com/google/devtools/build/lib/skyframe/serialization/autocodec",
        "//src/main/java/com/google/devtools/build/skyframe:skyframe-objects",
        "//third_party:auto_value",
        "//third_party:guava",
    ],
)

java_library(
    name = "collect_test_suites_in_package_function",
    srcs = ["CollectTestSuitesInPackageFunction.java"],
    deps = [
        ":collect_test_suites_in_package_value",
        ":package_value",
        "//src/main/java/com/google/devtools/build/lib/cmdline",
        "//src/main/java/com/google/devtools/build/lib/events",
        "//src/main/java/com/google/devtools/build/lib/packages",
        "//src/main/java/com/google/devtools/build/skyframe",
        "//src/main/java/com/google/devtools/build/skyframe:skyframe-objects",
        "//third_party:guava",
        "//third_party:jsr305",
    ],
)

java_library(
    name = "collect_test_suites_in_package_value",
    srcs = ["CollectTestSuitesInPackageValue.java"],
    deps = [
        ":sky_functions",
        "//src/main/java/com/google/devtools/build/lib/cmdline",
        "//src/main/java/com/google/devtools/build/lib/concurrent",
        "//src/main/java/com/google/devtools/build/lib/skyframe/serialization/autocodec",
        "//src/main/java/com/google/devtools/build/skyframe:skyframe-objects",
        "//third_party:guava",
    ],
)

java_library(
    name = "configuration_phase_started_event",
    srcs = ["ConfigurationPhaseStartedEvent.java"],
    deps = [
        ":configured_target_progress_receiver",
        "//src/main/java/com/google/devtools/build/lib/events",
    ],
)

java_library(
    name = "configured_target_and_data",
    srcs = ["ConfiguredTargetAndData.java"],
    deps = [
        ":build_configuration_value",
        ":package_value",
        "//src/main/java/com/google/devtools/build/lib/analysis:config/build_configuration",
        "//src/main/java/com/google/devtools/build/lib/analysis:configured_target",
        "//src/main/java/com/google/devtools/build/lib/packages",
        "//src/main/java/com/google/devtools/build/skyframe",
        "//src/main/java/com/google/devtools/build/skyframe:skyframe-objects",
        "//third_party:guava",
        "//third_party:jsr305",
    ],
)

java_library(
    name = "configured_target_key",
    srcs = ["ConfiguredTargetKey.java"],
    deps = [
        ":build_configuration_value",
        ":sky_functions",
        ":toolchain_context_key",
        "//src/main/java/com/google/devtools/build/lib/actions",
        "//src/main/java/com/google/devtools/build/lib/analysis:config/build_configuration",
        "//src/main/java/com/google/devtools/build/lib/analysis:configured_target",
        "//src/main/java/com/google/devtools/build/lib/cmdline",
        "//src/main/java/com/google/devtools/build/lib/concurrent",
        "//src/main/java/com/google/devtools/build/lib/skyframe/serialization/autocodec",
        "//src/main/java/com/google/devtools/build/skyframe:skyframe-objects",
        "//third_party:guava",
        "//third_party:jsr305",
    ],
)

java_library(
    name = "configured_target_progress_receiver",
    srcs = ["ConfiguredTargetProgressReceiver.java"],
)

java_library(
    name = "configured_value_creation_exception",
    srcs = ["ConfiguredValueCreationException.java"],
    deps = [
        ":sane_analysis_exception",
        "//src/main/java/com/google/devtools/build/lib/analysis:config/build_configuration",
        "//src/main/java/com/google/devtools/build/lib/buildeventstream/proto:build_event_stream_java_proto",
        "//src/main/java/com/google/devtools/build/lib/causes",
        "//src/main/java/com/google/devtools/build/lib/cmdline",
        "//src/main/java/com/google/devtools/build/lib/collect/nestedset",
        "//src/main/java/com/google/devtools/build/lib/skyframe/serialization/autocodec",
        "//third_party:jsr305",
    ],
)

java_library(
    name = "containing_package_lookup_function",
    srcs = ["ContainingPackageLookupFunction.java"],
    deps = [
        ":containing_package_lookup_value",
        ":package_lookup_value",
        "//src/main/java/com/google/devtools/build/lib/cmdline",
        "//src/main/java/com/google/devtools/build/lib/vfs:pathfragment",
        "//src/main/java/com/google/devtools/build/skyframe",
        "//src/main/java/com/google/devtools/build/skyframe:skyframe-objects",
        "//third_party:jsr305",
    ],
)

java_library(
    name = "containing_package_lookup_value",
    srcs = ["ContainingPackageLookupValue.java"],
    deps = [
        ":sky_functions",
        "//src/main/java/com/google/devtools/build/lib/cmdline",
        "//src/main/java/com/google/devtools/build/lib/concurrent",
        "//src/main/java/com/google/devtools/build/lib/skyframe/serialization/autocodec",
        "//src/main/java/com/google/devtools/build/lib/vfs",
        "//src/main/java/com/google/devtools/build/lib/vfs:pathfragment",
        "//src/main/java/com/google/devtools/build/skyframe:skyframe-objects",
        "//third_party:guava",
        "//third_party:jsr305",
    ],
)

java_library(
    name = "coverage_report_value",
    srcs = ["CoverageReportValue.java"],
    deps = [
        ":sky_functions",
        "//src/main/java/com/google/devtools/build/lib/actions",
        "//src/main/java/com/google/devtools/build/lib/skyframe/serialization/autocodec",
        "//src/main/java/com/google/devtools/build/skyframe:skyframe-objects",
    ],
)

java_library(
    name = "cycle_utils",
    srcs = ["CycleUtils.java"],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/util",
        "//third_party:guava",
    ],
)

java_library(
    name = "diff_awareness",
    srcs = ["DiffAwareness.java"],
    deps = [
        ":broken_diff_awareness_exception",
        ":incompatible_view_exception",
        "//src/main/java/com/google/devtools/build/lib/vfs",
        "//src/main/java/com/google/devtools/common/options",
        "//third_party:jsr305",
    ],
)

java_library(
    name = "diff_awareness_manager",
    srcs = ["DiffAwarenessManager.java"],
    deps = [
        ":broken_diff_awareness_exception",
        ":diff_awareness",
        ":incompatible_view_exception",
        "//src/main/java/com/google/devtools/build/lib/events",
        "//src/main/java/com/google/devtools/build/lib/vfs",
        "//src/main/java/com/google/devtools/common/options",
        "//third_party:flogger",
        "//third_party:guava",
        "//third_party:jsr305",
    ],
)

java_library(
    name = "directory_listing_function",
    srcs = ["DirectoryListingFunction.java"],
    deps = [
        ":directory_listing_state_value",
        ":directory_listing_value",
        "//src/main/java/com/google/devtools/build/lib/actions",
        "//src/main/java/com/google/devtools/build/lib/vfs",
        "//src/main/java/com/google/devtools/build/skyframe",
        "//src/main/java/com/google/devtools/build/skyframe:skyframe-objects",
        "//third_party:jsr305",
    ],
)

java_library(
    name = "directory_listing_state_value",
    srcs = ["DirectoryListingStateValue.java"],
    deps = [
        ":dirents",
        ":sky_functions",
        "//src/main/java/com/google/devtools/build/lib/concurrent",
        "//src/main/java/com/google/devtools/build/lib/skyframe/serialization/autocodec",
        "//src/main/java/com/google/devtools/build/lib/vfs",
        "//src/main/java/com/google/devtools/build/skyframe:skyframe-objects",
        "//third_party:guava",
        "//third_party:jsr305",
    ],
)

java_library(
    name = "directory_listing_value",
    srcs = ["DirectoryListingValue.java"],
    deps = [
        ":directory_listing_state_value",
        ":dirents",
        ":sky_functions",
        "//src/main/java/com/google/devtools/build/lib/actions",
        "//src/main/java/com/google/devtools/build/lib/concurrent",
        "//src/main/java/com/google/devtools/build/lib/skyframe/serialization/autocodec",
        "//src/main/java/com/google/devtools/build/lib/vfs",
        "//src/main/java/com/google/devtools/build/skyframe:skyframe-objects",
        "//third_party:guava",
    ],
)

java_library(
    name = "dirents",
    srcs = ["Dirents.java"],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/vfs",
        "//third_party:jsr305",
    ],
)

java_library(
    name = "error_reading_skylark_extension_exception",
    srcs = ["ErrorReadingStarlarkExtensionException.java"],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/packages",
        "//src/main/java/com/google/devtools/build/skyframe",
    ],
)

java_library(
    name = "execution_finished_event",
    srcs = ["ExecutionFinishedEvent.java"],
    deps = ["//third_party:auto_value"],
)

java_library(
    name = "file_function",
    srcs = ["FileFunction.java"],
    deps = [
        ":cycle_utils",
        ":file_symlink_cycle_exception",
        ":file_symlink_cycle_uniqueness_function",
        ":file_symlink_exception",
        ":file_symlink_infinite_expansion_exception",
        ":file_symlink_infinite_expansion_uniqueness_function",
        "//src/main/java/com/google/devtools/build/lib/actions",
        "//src/main/java/com/google/devtools/build/lib/pkgcache",
        "//src/main/java/com/google/devtools/build/lib/util",
        "//src/main/java/com/google/devtools/build/lib/vfs",
        "//src/main/java/com/google/devtools/build/lib/vfs:pathfragment",
        "//src/main/java/com/google/devtools/build/skyframe",
        "//src/main/java/com/google/devtools/build/skyframe:skyframe-objects",
        "//third_party:guava",
        "//third_party:jsr305",
    ],
)

java_library(
    name = "file_symlink_cycle_exception",
    srcs = ["FileSymlinkCycleException.java"],
    deps = [
        ":file_symlink_exception",
        "//src/main/java/com/google/devtools/build/lib/vfs",
        "//third_party:guava",
    ],
)

java_library(
    name = "file_symlink_cycle_uniqueness_function",
    srcs = ["FileSymlinkCycleUniquenessFunction.java"],
    deps = [
        ":abstract_chain_uniqueness_function",
        ":chain_uniqueness_utils",
        ":sky_functions",
        "//src/main/java/com/google/devtools/build/lib/concurrent",
        "//src/main/java/com/google/devtools/build/lib/skyframe/serialization/autocodec",
        "//src/main/java/com/google/devtools/build/lib/vfs",
        "//src/main/java/com/google/devtools/build/skyframe:skyframe-objects",
        "//third_party:guava",
    ],
)

java_library(
    name = "file_symlink_exception",
    srcs = ["FileSymlinkException.java"],
)

java_library(
    name = "file_symlink_infinite_expansion_exception",
    srcs = ["FileSymlinkInfiniteExpansionException.java"],
    deps = [
        ":file_symlink_exception",
        "//src/main/java/com/google/devtools/build/lib/skyframe/serialization/autocodec",
        "//src/main/java/com/google/devtools/build/lib/vfs",
        "//third_party:guava",
    ],
)

java_library(
    name = "file_symlink_infinite_expansion_uniqueness_function",
    srcs = ["FileSymlinkInfiniteExpansionUniquenessFunction.java"],
    deps = [
        ":abstract_chain_uniqueness_function",
        ":chain_uniqueness_utils",
        ":sky_functions",
        "//src/main/java/com/google/devtools/build/lib/concurrent",
        "//src/main/java/com/google/devtools/build/lib/skyframe/serialization/autocodec",
        "//src/main/java/com/google/devtools/build/lib/vfs",
        "//src/main/java/com/google/devtools/build/skyframe:skyframe-objects",
        "//third_party:guava",
    ],
)

java_library(
    name = "fileset_entry_function",
    srcs = ["FilesetEntryFunction.java"],
    deps = [
        ":fileset_entry_key",
        ":fileset_entry_value",
        ":recursive_filesystem_traversal",
        ":workspace_name_value",
        "//src/main/java/com/google/devtools/build/lib/actions",
        "//src/main/java/com/google/devtools/build/lib/vfs",
        "//src/main/java/com/google/devtools/build/lib/vfs:pathfragment",
        "//src/main/java/com/google/devtools/build/skyframe",
        "//src/main/java/com/google/devtools/build/skyframe:skyframe-objects",
        "//third_party:guava",
    ],
)

java_library(
    name = "fileset_entry_key",
    srcs = ["FilesetEntryKey.java"],
    deps = [
        ":sky_functions",
        "//src/main/java/com/google/devtools/build/lib/actions",
        "//src/main/java/com/google/devtools/build/lib/concurrent",
        "//src/main/java/com/google/devtools/build/skyframe:skyframe-objects",
        "//third_party:auto_value",
        "//third_party:guava",
    ],
)

java_library(
    name = "fileset_entry_value",
    srcs = ["FilesetEntryValue.java"],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/actions",
        "//src/main/java/com/google/devtools/build/skyframe:skyframe-objects",
        "//third_party:guava",
    ],
)

java_library(
    name = "filesystem_value_checker",
    srcs = ["FilesystemValueChecker.java"],
    deps = [
        ":action_execution_value",
        ":action_metadata_handler",
        ":sky_functions",
        ":sky_value_dirtiness_checker",
        ":tree_artifact_value",
        "//src/main/java/com/google/devtools/build/lib/actions",
        "//src/main/java/com/google/devtools/build/lib/concurrent",
        "//src/main/java/com/google/devtools/build/lib/profiler",
        "//src/main/java/com/google/devtools/build/lib/util",
        "//src/main/java/com/google/devtools/build/lib/util/io",
        "//src/main/java/com/google/devtools/build/lib/vfs",
        "//src/main/java/com/google/devtools/build/lib/vfs:pathfragment",
        "//src/main/java/com/google/devtools/build/skyframe",
        "//src/main/java/com/google/devtools/build/skyframe:skyframe-objects",
        "//third_party:flogger",
        "//third_party:guava",
        "//third_party:jsr305",
    ],
)

java_library(
    name = "glob_descriptor",
    srcs = ["GlobDescriptor.java"],
    deps = [
        ":sky_functions",
        "//src/main/java/com/google/devtools/build/lib/cmdline",
        "//src/main/java/com/google/devtools/build/lib/concurrent",
        "//src/main/java/com/google/devtools/build/lib/skyframe/serialization/autocodec",
        "//src/main/java/com/google/devtools/build/lib/util:string",
        "//src/main/java/com/google/devtools/build/lib/vfs",
        "//src/main/java/com/google/devtools/build/lib/vfs:pathfragment",
        "//src/main/java/com/google/devtools/build/skyframe:skyframe-objects",
        "//third_party:guava",
    ],
)

java_library(
    name = "glob_function",
    srcs = ["GlobFunction.java"],
    deps = [
        ":blacklisted_package_prefixes_value",
        ":directory_listing_value",
        ":glob_descriptor",
        ":glob_value",
        ":package_lookup_value",
        "//src/main/java/com/google/devtools/build/lib/actions",
        "//src/main/java/com/google/devtools/build/lib/cmdline",
        "//src/main/java/com/google/devtools/build/lib/collect/nestedset",
        "//src/main/java/com/google/devtools/build/lib/vfs",
        "//src/main/java/com/google/devtools/build/lib/vfs:pathfragment",
        "//src/main/java/com/google/devtools/build/skyframe",
        "//src/main/java/com/google/devtools/build/skyframe:skyframe-objects",
        "//third_party:guava",
        "//third_party:jsr305",
    ],
)

java_library(
    name = "glob_value",
    srcs = ["GlobValue.java"],
    deps = [
        ":glob_descriptor",
        "//src/main/java/com/google/devtools/build/lib/cmdline",
        "//src/main/java/com/google/devtools/build/lib/collect/nestedset",
        "//src/main/java/com/google/devtools/build/lib/concurrent",
        "//src/main/java/com/google/devtools/build/lib/vfs",
        "//src/main/java/com/google/devtools/build/lib/vfs:pathfragment",
        "//src/main/java/com/google/devtools/build/skyframe:skyframe-objects",
        "//third_party:guava",
    ],
)

java_library(
    name = "graph_backed_recursive_package_provider",
    srcs = ["GraphBackedRecursivePackageProvider.java"],
    deps = [
        ":package_lookup_value",
        ":package_value",
        ":root_package_extractor",
        "//src/main/java/com/google/devtools/build/lib/actions",
        "//src/main/java/com/google/devtools/build/lib/cmdline",
        "//src/main/java/com/google/devtools/build/lib/concurrent",
        "//src/main/java/com/google/devtools/build/lib/events",
        "//src/main/java/com/google/devtools/build/lib/packages",
        "//src/main/java/com/google/devtools/build/lib/pkgcache",
        "//src/main/java/com/google/devtools/build/lib/rules:repository/repository_directory_value",
        "//src/main/java/com/google/devtools/build/lib/vfs",
        "//src/main/java/com/google/devtools/build/lib/vfs:pathfragment",
        "//src/main/java/com/google/devtools/build/skyframe",
        "//src/main/java/com/google/devtools/build/skyframe:skyframe-objects",
        "//third_party:flogger",
        "//third_party:guava",
    ],
)

java_library(
    name = "incompatible_view_exception",
    srcs = ["IncompatibleViewException.java"],
    deps = ["//third_party:guava"],
)

java_library(
    name = "interner_with_presence_check",
    srcs = ["InternerWithPresenceCheck.java"],
    deps = [
        "//third_party:guava",
        "//third_party:jsr305",
    ],
)

java_library(
    name = "loading_phase_started_event",
    srcs = ["LoadingPhaseStartedEvent.java"],
    deps = [
        ":package_progress_receiver",
        "//src/main/java/com/google/devtools/build/lib/events",
    ],
)

java_library(
    name = "local_diff_awareness",
    srcs = [
        "LocalDiffAwareness.java",
        "MacOSXFsEventsDiffAwareness.java",
        "WatchServiceDiffAwareness.java",
    ],
    deps = [
        ":broken_diff_awareness_exception",
        ":diff_awareness",
        ":incompatible_view_exception",
        "//src/main/java/com/google/devtools/build/lib/unix/jni",
        "//src/main/java/com/google/devtools/build/lib/util:os",
        "//src/main/java/com/google/devtools/build/lib/vfs",
        "//src/main/java/com/google/devtools/build/lib/vfs:pathfragment",
        "//src/main/java/com/google/devtools/common/options",
        "//third_party:guava",
    ],
)

java_library(
    name = "local_repository_lookup_value",
    srcs = ["LocalRepositoryLookupValue.java"],
    deps = [
        ":sky_functions",
        "//src/main/java/com/google/devtools/build/lib/cmdline",
        "//src/main/java/com/google/devtools/build/lib/concurrent",
        "//src/main/java/com/google/devtools/build/lib/skyframe/serialization/autocodec",
        "//src/main/java/com/google/devtools/build/lib/vfs",
        "//src/main/java/com/google/devtools/build/lib/vfs:pathfragment",
        "//src/main/java/com/google/devtools/build/skyframe:skyframe-objects",
        "//third_party:guava",
    ],
)

java_library(
    name = "managed_directories_knowledge",
    srcs = ["ManagedDirectoriesKnowledge.java"],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/cmdline",
        "//src/main/java/com/google/devtools/build/lib/packages",
        "//src/main/java/com/google/devtools/build/lib/util:abrupt_exit_exception",
        "//src/main/java/com/google/devtools/build/lib/vfs:pathfragment",
        "//third_party:guava",
        "//third_party:jsr305",
    ],
)

java_library(
    name = "map_as_package_roots",
    srcs = ["MapAsPackageRoots.java"],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/actions",
        "//src/main/java/com/google/devtools/build/lib/cmdline",
        "//src/main/java/com/google/devtools/build/lib/vfs",
        "//third_party:guava",
    ],
)

java_library(
    name = "minimal_output_store",
    srcs = ["MinimalOutputStore.java"],
    deps = [
        ":output_store",
        "//src/main/java/com/google/devtools/build/lib/actions",
    ],
)

java_library(
    name = "mutable_supplier",
    srcs = ["MutableSupplier.java"],
    deps = ["//third_party:guava"],
)

java_library(
    name = "output_store",
    srcs = ["OutputStore.java"],
    deps = [
        ":tree_artifact_value",
        "//src/main/java/com/google/devtools/build/lib/actions",
        "//src/main/java/com/google/devtools/build/lib/concurrent",
        "//third_party:guava",
        "//third_party:jsr305",
    ],
)

java_library(
    name = "package_error_function",
    srcs = ["PackageErrorFunction.java"],
    deps = [
        ":package_value",
        ":sky_functions",
        "//src/main/java/com/google/devtools/build/lib/cmdline",
        "//src/main/java/com/google/devtools/build/lib/concurrent",
        "//src/main/java/com/google/devtools/build/lib/packages",
        "//src/main/java/com/google/devtools/build/lib/skyframe/serialization/autocodec",
        "//src/main/java/com/google/devtools/build/skyframe",
        "//src/main/java/com/google/devtools/build/skyframe:skyframe-objects",
        "//third_party:guava",
        "//third_party:jsr305",
    ],
)

java_library(
    name = "package_error_message_function",
    srcs = ["PackageErrorMessageFunction.java"],
    deps = [
        ":package_error_message_value",
        ":package_value",
        "//src/main/java/com/google/devtools/build/lib/cmdline",
        "//src/main/java/com/google/devtools/build/lib/packages",
        "//src/main/java/com/google/devtools/build/skyframe",
        "//src/main/java/com/google/devtools/build/skyframe:skyframe-objects",
        "//third_party:jsr305",
    ],
)

java_library(
    name = "package_error_message_value",
    srcs = ["PackageErrorMessageValue.java"],
    deps = [
        ":sky_functions",
        "//src/main/java/com/google/devtools/build/lib/cmdline",
        "//src/main/java/com/google/devtools/build/lib/concurrent",
        "//src/main/java/com/google/devtools/build/lib/skyframe/serialization/autocodec",
        "//src/main/java/com/google/devtools/build/skyframe:skyframe-objects",
        "//third_party:guava",
    ],
)

java_library(
    name = "package_identifier_batching_callback",
    srcs = ["PackageIdentifierBatchingCallback.java"],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/cmdline",
        "//src/main/java/com/google/devtools/build/lib/concurrent",
        "//third_party:guava",
        "//third_party:jsr305",
    ],
)

java_library(
    name = "package_lookup_function",
    srcs = ["PackageLookupFunction.java"],
    deps = [
        ":blacklisted_package_prefixes_value",
        ":file_symlink_exception",
        ":local_repository_lookup_value",
        ":package_lookup_value",
        ":precomputed_value",
        ":repository_value",
        "//src/main/java/com/google/devtools/build/lib/actions",
        "//src/main/java/com/google/devtools/build/lib/cmdline",
        "//src/main/java/com/google/devtools/build/lib/cmdline:LabelValidator",
        "//src/main/java/com/google/devtools/build/lib/packages",
        "//src/main/java/com/google/devtools/build/lib/pkgcache",
        "//src/main/java/com/google/devtools/build/lib/repository:external_package_helper",
        "//src/main/java/com/google/devtools/build/lib/syntax:evaluator",
        "//src/main/java/com/google/devtools/build/lib/vfs",
        "//src/main/java/com/google/devtools/build/lib/vfs:pathfragment",
        "//src/main/java/com/google/devtools/build/skyframe",
        "//src/main/java/com/google/devtools/build/skyframe:skyframe-objects",
        "//third_party:guava",
        "//third_party:jsr305",
    ],
)

java_library(
    name = "package_lookup_value",
    srcs = ["PackageLookupValue.java"],
    deps = [
        ":repository_value",
        ":sky_functions",
        "//src/main/java/com/google/devtools/build/lib/cmdline",
        "//src/main/java/com/google/devtools/build/lib/concurrent",
        "//src/main/java/com/google/devtools/build/lib/packages",
        "//src/main/java/com/google/devtools/build/lib/skyframe/serialization/autocodec",
        "//src/main/java/com/google/devtools/build/lib/vfs",
        "//src/main/java/com/google/devtools/build/lib/vfs:pathfragment",
        "//src/main/java/com/google/devtools/build/skyframe:skyframe-objects",
        "//third_party:guava",
        "//third_party:jsr305",
    ],
)

java_library(
    name = "package_progress_receiver",
    srcs = ["PackageProgressReceiver.java"],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/cmdline",
        "//src/main/java/com/google/devtools/build/lib/util",
        "//third_party:guava",
    ],
)

java_library(
    name = "package_roots_no_symlink_creation",
    srcs = ["PackageRootsNoSymlinkCreation.java"],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/actions",
        "//src/main/java/com/google/devtools/build/lib/cmdline",
        "//src/main/java/com/google/devtools/build/lib/vfs",
        "//third_party:guava",
    ],
)

java_library(
    name = "package_value",
    srcs = ["PackageValue.java"],
    deps = [
        ":sky_functions",
        "//src/main/java/com/google/devtools/build/lib/cmdline",
        "//src/main/java/com/google/devtools/build/lib/concurrent",
        "//src/main/java/com/google/devtools/build/lib/packages",
        "//src/main/java/com/google/devtools/build/lib/skyframe/serialization/autocodec",
        "//src/main/java/com/google/devtools/build/skyframe",
        "//src/main/java/com/google/devtools/build/skyframe:skyframe-objects",
        "//third_party:guava",
    ],
)

java_library(
    name = "path_casing_lookup_function",
    srcs = ["PathCasingLookupFunction.java"],
    deps = [
        ":directory_listing_value",
        ":path_casing_lookup_value",
        "//src/main/java/com/google/devtools/build/lib/actions",
        "//src/main/java/com/google/devtools/build/lib/vfs",
        "//src/main/java/com/google/devtools/build/skyframe",
        "//src/main/java/com/google/devtools/build/skyframe:skyframe-objects",
        "//third_party:guava",
        "//third_party:jsr305",
    ],
)

java_library(
    name = "path_casing_lookup_value",
    srcs = ["PathCasingLookupValue.java"],
    deps = [
        ":sky_functions",
        "//src/main/java/com/google/devtools/build/lib/concurrent",
        "//src/main/java/com/google/devtools/build/lib/skyframe/serialization/autocodec",
        "//src/main/java/com/google/devtools/build/lib/vfs",
        "//src/main/java/com/google/devtools/build/skyframe:skyframe-objects",
        "//third_party:guava",
    ],
)

java_library(
    name = "pattern_expanding_error",
    srcs = ["PatternExpandingError.java"],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/buildeventstream",
        "//src/main/java/com/google/devtools/build/lib/buildeventstream/proto:build_event_stream_java_proto",
        "//third_party:guava",
    ],
)

java_library(
    name = "per_build_syscall_cache",
    srcs = ["PerBuildSyscallCache.java"],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/util",
        "//src/main/java/com/google/devtools/build/lib/vfs",
        "//third_party:guava",
    ],
)

java_library(
    name = "precomputed_function",
    srcs = ["PrecomputedFunction.java"],
    deps = [
        "//src/main/java/com/google/devtools/build/skyframe",
        "//src/main/java/com/google/devtools/build/skyframe:skyframe-objects",
    ],
)

java_library(
    name = "precomputed_value",
    srcs = [
        "PrecomputedValue.java",
    ],
    deps = [
        ":sky_functions",
        "//src/main/java/com/google/devtools/build/lib/concurrent",
        "//src/main/java/com/google/devtools/build/lib/packages",
        "//src/main/java/com/google/devtools/build/lib/pkgcache",
        "//src/main/java/com/google/devtools/build/lib/remote/options",
        "//src/main/java/com/google/devtools/build/lib/skyframe/serialization/autocodec",
        "//src/main/java/com/google/devtools/build/lib/syntax:evaluator",
        "//src/main/java/com/google/devtools/build/skyframe",
        "//src/main/java/com/google/devtools/build/skyframe:skyframe-objects",
        "//third_party:guava",
        "//third_party:jsr305",
    ],
)

java_library(
    name = "prepare_deps_of_pattern_value",
    srcs = ["PrepareDepsOfPatternValue.java"],
    deps = [
        ":sky_functions",
        ":target_pattern_value",
        "//src/main/java/com/google/devtools/build/lib/cmdline",
        "//src/main/java/com/google/devtools/build/lib/concurrent",
        "//src/main/java/com/google/devtools/build/lib/pkgcache",
        "//src/main/java/com/google/devtools/build/lib/skyframe/serialization/autocodec",
        "//src/main/java/com/google/devtools/build/skyframe:skyframe-objects",
        "//third_party:guava",
    ],
)

java_library(
    name = "prepare_deps_of_patterns_function",
    srcs = ["PrepareDepsOfPatternsFunction.java"],
    deps = [
        ":prepare_deps_of_pattern_value",
        ":prepare_deps_of_patterns_value",
        ":target_pattern_value",
        "//src/main/java/com/google/devtools/build/lib/cmdline",
        "//src/main/java/com/google/devtools/build/lib/events",
        "//src/main/java/com/google/devtools/build/lib/pkgcache",
        "//src/main/java/com/google/devtools/build/skyframe",
        "//src/main/java/com/google/devtools/build/skyframe:skyframe-objects",
        "//third_party:guava",
        "//third_party:jsr305",
    ],
)

java_library(
    name = "prepare_deps_of_patterns_value",
    srcs = ["PrepareDepsOfPatternsValue.java"],
    deps = [
        ":sky_functions",
        ":target_pattern_value",
        "//src/main/java/com/google/devtools/build/lib/concurrent",
        "//src/main/java/com/google/devtools/build/lib/skyframe/serialization/autocodec",
        "//src/main/java/com/google/devtools/build/skyframe:skyframe-objects",
        "//third_party:guava",
    ],
)

java_library(
    name = "prepare_deps_of_targets_under_directory_function",
    srcs = ["PrepareDepsOfTargetsUnderDirectoryFunction.java"],
    deps = [
        ":collect_targets_in_package_value",
        ":prepare_deps_of_targets_under_directory_value",
        ":process_package_directory",
        ":process_package_directory_result",
        ":recursive_pkg_key",
        "//src/main/java/com/google/devtools/build/lib/analysis:blaze_directories",
        "//src/main/java/com/google/devtools/build/lib/cmdline",
        "//src/main/java/com/google/devtools/build/lib/packages",
        "//src/main/java/com/google/devtools/build/lib/pkgcache",
        "//src/main/java/com/google/devtools/build/skyframe",
        "//src/main/java/com/google/devtools/build/skyframe:skyframe-objects",
        "//third_party:guava",
        "//third_party:jsr305",
    ],
)

java_library(
    name = "prepare_deps_of_targets_under_directory_value",
    srcs = ["PrepareDepsOfTargetsUnderDirectoryValue.java"],
    deps = [
        ":recursive_pkg_key",
        ":sky_functions",
        "//src/main/java/com/google/devtools/build/lib/cmdline",
        "//src/main/java/com/google/devtools/build/lib/concurrent",
        "//src/main/java/com/google/devtools/build/lib/pkgcache",
        "//src/main/java/com/google/devtools/build/lib/skyframe/serialization/autocodec",
        "//src/main/java/com/google/devtools/build/lib/vfs",
        "//src/main/java/com/google/devtools/build/lib/vfs:pathfragment",
        "//src/main/java/com/google/devtools/build/skyframe:skyframe-objects",
        "//third_party:guava",
    ],
)

java_library(
    name = "prepare_test_suites_under_directory_function",
    srcs = ["PrepareTestSuitesUnderDirectoryFunction.java"],
    deps = [
        ":collect_test_suites_in_package_value",
        ":prepare_test_suites_under_directory_value",
        ":process_package_directory",
        ":process_package_directory_result",
        "//src/main/java/com/google/devtools/build/lib/analysis:blaze_directories",
        "//src/main/java/com/google/devtools/build/lib/cmdline",
        "//src/main/java/com/google/devtools/build/lib/packages",
        "//src/main/java/com/google/devtools/build/skyframe",
        "//src/main/java/com/google/devtools/build/skyframe:skyframe-objects",
        "//third_party:guava",
        "//third_party:jsr305",
    ],
)

java_library(
    name = "prepare_test_suites_under_directory_value",
    srcs = ["PrepareTestSuitesUnderDirectoryValue.java"],
    deps = [
        ":recursive_pkg_sky_key",
        ":sky_functions",
        "//src/main/java/com/google/devtools/build/lib/cmdline",
        "//src/main/java/com/google/devtools/build/lib/concurrent",
        "//src/main/java/com/google/devtools/build/lib/skyframe/serialization/autocodec",
        "//src/main/java/com/google/devtools/build/lib/vfs",
        "//src/main/java/com/google/devtools/build/lib/vfs:pathfragment",
        "//src/main/java/com/google/devtools/build/skyframe:skyframe-objects",
        "//third_party:guava",
    ],
)

java_library(
    name = "process_package_directory",
    srcs = ["ProcessPackageDirectory.java"],
    deps = [
        ":directory_listing_value",
        ":dirents",
        ":file_symlink_exception",
        ":package_lookup_value",
        ":process_package_directory_result",
        "//src/main/java/com/google/devtools/build/lib/actions",
        "//src/main/java/com/google/devtools/build/lib/analysis:blaze_directories",
        "//src/main/java/com/google/devtools/build/lib/cmdline",
        "//src/main/java/com/google/devtools/build/lib/events",
        "//src/main/java/com/google/devtools/build/lib/packages",
        "//src/main/java/com/google/devtools/build/lib/vfs",
        "//src/main/java/com/google/devtools/build/lib/vfs:pathfragment",
        "//src/main/java/com/google/devtools/build/skyframe",
        "//src/main/java/com/google/devtools/build/skyframe:skyframe-objects",
        "//third_party:guava",
        "//third_party:jsr305",
    ],
)

java_library(
    name = "process_package_directory_result",
    srcs = ["ProcessPackageDirectoryResult.java"],
    deps = [
        "//src/main/java/com/google/devtools/build/skyframe:skyframe-objects",
        "//third_party:guava",
    ],
)

java_library(
    name = "progress_event_suppressing_environment",
    srcs = ["ProgressEventSuppressingEnvironment.java"],
    deps = [
        ":progress_suppressing_event_handler",
        "//src/main/java/com/google/devtools/build/lib/events",
        "//src/main/java/com/google/devtools/build/lib/util",
        "//src/main/java/com/google/devtools/build/skyframe",
        "//src/main/java/com/google/devtools/build/skyframe:skyframe-objects",
        "//third_party:guava",
        "//third_party:jsr305",
    ],
)

java_library(
    name = "progress_suppressing_event_handler",
    srcs = ["ProgressSuppressingEventHandler.java"],
    deps = ["//src/main/java/com/google/devtools/build/lib/events"],
)

java_library(
    name = "recursive_directory_traversal_function",
    srcs = ["RecursiveDirectoryTraversalFunction.java"],
    deps = [
        ":package_error_message_value",
        ":process_package_directory",
        ":process_package_directory_result",
        ":recursive_pkg_key",
        "//src/main/java/com/google/devtools/build/lib/analysis:blaze_directories",
        "//src/main/java/com/google/devtools/build/lib/cmdline",
        "//src/main/java/com/google/devtools/build/lib/events",
        "//src/main/java/com/google/devtools/build/lib/vfs",
        "//src/main/java/com/google/devtools/build/lib/vfs:pathfragment",
        "//src/main/java/com/google/devtools/build/skyframe",
        "//src/main/java/com/google/devtools/build/skyframe:skyframe-objects",
        "//third_party:guava",
        "//third_party:jsr305",
    ],
)

java_library(
    name = "recursive_filesystem_traversal",
    srcs = [
        "RecursiveFilesystemTraversalFunction.java",
        "RecursiveFilesystemTraversalValue.java",
    ],
    deps = [
        ":action_execution_value",
        ":directory_listing_value",
        ":package_lookup_value",
        ":sky_functions",
        ":tree_artifact_value",
        "//src/main/java/com/google/devtools/build/lib/actions",
        "//src/main/java/com/google/devtools/build/lib/collect/nestedset",
        "//src/main/java/com/google/devtools/build/lib/concurrent",
        "//src/main/java/com/google/devtools/build/lib/events",
        "//src/main/java/com/google/devtools/build/lib/skyframe/serialization/autocodec",
        "//src/main/java/com/google/devtools/build/lib/util",
        "//src/main/java/com/google/devtools/build/lib/vfs",
        "//src/main/java/com/google/devtools/build/lib/vfs:pathfragment",
        "//src/main/java/com/google/devtools/build/skyframe",
        "//src/main/java/com/google/devtools/build/skyframe:skyframe-objects",
        "//third_party:guava",
        "//third_party:jsr305",
    ],
)

java_library(
    name = "recursive_package_provider_backed_target_pattern_resolver",
    srcs = ["RecursivePackageProviderBackedTargetPatternResolver.java"],
    deps = [
        ":package_identifier_batching_callback",
        "//src/main/java/com/google/devtools/build/lib/cmdline",
        "//src/main/java/com/google/devtools/build/lib/concurrent",
        "//src/main/java/com/google/devtools/build/lib/events",
        "//src/main/java/com/google/devtools/build/lib/packages",
        "//src/main/java/com/google/devtools/build/lib/pkgcache",
        "//src/main/java/com/google/devtools/build/lib/vfs:pathfragment",
        "//third_party:guava",
    ],
)

java_library(
    name = "recursive_pkg_function",
    srcs = ["RecursivePkgFunction.java"],
    deps = [
        ":recursive_directory_traversal_function",
        ":recursive_pkg_key",
        ":recursive_pkg_value",
        "//src/main/java/com/google/devtools/build/lib/analysis:blaze_directories",
        "//src/main/java/com/google/devtools/build/lib/cmdline",
        "//src/main/java/com/google/devtools/build/lib/collect/nestedset",
        "//src/main/java/com/google/devtools/build/lib/vfs",
        "//src/main/java/com/google/devtools/build/lib/vfs:pathfragment",
        "//src/main/java/com/google/devtools/build/skyframe",
        "//src/main/java/com/google/devtools/build/skyframe:skyframe-objects",
        "//third_party:guava",
        "//third_party:jsr305",
    ],
)

java_library(
    name = "recursive_pkg_key",
    srcs = ["RecursivePkgKey.java"],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/cmdline",
        "//src/main/java/com/google/devtools/build/lib/concurrent",
        "//src/main/java/com/google/devtools/build/lib/skyframe/serialization/autocodec",
        "//src/main/java/com/google/devtools/build/lib/vfs",
        "//src/main/java/com/google/devtools/build/lib/vfs:pathfragment",
        "//third_party:guava",
    ],
)

java_library(
    name = "recursive_pkg_sky_key",
    srcs = ["RecursivePkgSkyKey.java"],
    deps = [
        ":recursive_pkg_key",
        "//src/main/java/com/google/devtools/build/lib/cmdline",
        "//src/main/java/com/google/devtools/build/lib/vfs",
        "//src/main/java/com/google/devtools/build/lib/vfs:pathfragment",
        "//src/main/java/com/google/devtools/build/skyframe:skyframe-objects",
        "//third_party:guava",
    ],
)

java_library(
    name = "recursive_pkg_value",
    srcs = ["RecursivePkgValue.java"],
    deps = [
        ":recursive_pkg_sky_key",
        ":sky_functions",
        "//src/main/java/com/google/devtools/build/lib/cmdline",
        "//src/main/java/com/google/devtools/build/lib/collect/nestedset",
        "//src/main/java/com/google/devtools/build/lib/concurrent",
        "//src/main/java/com/google/devtools/build/lib/skyframe/serialization/autocodec",
        "//src/main/java/com/google/devtools/build/lib/vfs",
        "//src/main/java/com/google/devtools/build/lib/vfs:pathfragment",
        "//src/main/java/com/google/devtools/build/skyframe:skyframe-objects",
        "//third_party:guava",
    ],
)

java_library(
    name = "recursive_pkg_value_root_package_extractor",
    srcs = ["RecursivePkgValueRootPackageExtractor.java"],
    deps = [
        ":recursive_pkg_value",
        ":root_package_extractor",
        "//src/main/java/com/google/devtools/build/lib/cmdline",
        "//src/main/java/com/google/devtools/build/lib/concurrent",
        "//src/main/java/com/google/devtools/build/lib/events",
        "//src/main/java/com/google/devtools/build/lib/vfs",
        "//src/main/java/com/google/devtools/build/lib/vfs:pathfragment",
        "//src/main/java/com/google/devtools/build/skyframe",
        "//third_party:guava",
    ],
)

java_library(
    name = "repository_mapping_function",
    srcs = ["RepositoryMappingFunction.java"],
    deps = [
        ":package_value",
        ":repository_mapping_value",
        "//src/main/java/com/google/devtools/build/lib/cmdline",
        "//src/main/java/com/google/devtools/build/lib/packages",
        "//src/main/java/com/google/devtools/build/skyframe",
        "//src/main/java/com/google/devtools/build/skyframe:skyframe-objects",
        "//third_party:guava",
        "//third_party:jsr305",
    ],
)

java_library(
    name = "repository_mapping_value",
    srcs = ["RepositoryMappingValue.java"],
    deps = [
        ":sky_functions",
        "//src/main/java/com/google/devtools/build/lib/cmdline",
        "//src/main/java/com/google/devtools/build/lib/concurrent",
        "//src/main/java/com/google/devtools/build/lib/skyframe/serialization/autocodec",
        "//src/main/java/com/google/devtools/build/skyframe:skyframe-objects",
        "//third_party:guava",
    ],
)

java_library(
    name = "repository_value",
    srcs = ["RepositoryValue.java"],
    deps = [
        ":sky_functions",
        "//src/main/java/com/google/devtools/build/lib/cmdline",
        "//src/main/java/com/google/devtools/build/lib/concurrent",
        "//src/main/java/com/google/devtools/build/lib/rules:repository/repository_directory_value",
        "//src/main/java/com/google/devtools/build/lib/skyframe/serialization/autocodec",
        "//src/main/java/com/google/devtools/build/lib/vfs",
        "//src/main/java/com/google/devtools/build/skyframe:skyframe-objects",
        "//third_party:guava",
    ],
)

java_library(
    name = "root_package_extractor",
    srcs = ["RootPackageExtractor.java"],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/cmdline",
        "//src/main/java/com/google/devtools/build/lib/concurrent",
        "//src/main/java/com/google/devtools/build/lib/events",
        "//src/main/java/com/google/devtools/build/lib/vfs",
        "//src/main/java/com/google/devtools/build/lib/vfs:pathfragment",
        "//src/main/java/com/google/devtools/build/skyframe",
        "//third_party:guava",
    ],
)

java_library(
    name = "runfiles_artifact_value",
    srcs = ["RunfilesArtifactValue.java"],
    deps = [
        ":aggregating_artifact_value",
        ":tree_artifact_value",
        "//src/main/java/com/google/devtools/build/lib/actions",
        "//src/main/java/com/google/devtools/build/lib/util",
        "//third_party:guava",
    ],
)

java_library(
    name = "sane_analysis_exception",
    srcs = ["SaneAnalysisException.java"],
)

java_library(
    name = "sky_functions",
    srcs = ["SkyFunctions.java"],
    deps = [
        "//src/main/java/com/google/devtools/build/skyframe:skyframe-objects",
        "//third_party:guava",
    ],
)

java_library(
    name = "sky_value_dirtiness_checker",
    srcs = ["SkyValueDirtinessChecker.java"],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/util/io",
        "//src/main/java/com/google/devtools/build/skyframe:skyframe-objects",
        "//third_party:guava",
        "//third_party:jsr305",
    ],
)

java_library(
    name = "skyframe_aware_action",
    srcs = ["SkyframeAwareAction.java"],
    deps = [
        "//src/main/java/com/google/devtools/build/skyframe",
        "//src/main/java/com/google/devtools/build/skyframe:skyframe-objects",
        "//third_party:guava",
    ],
)

java_library(
    name = "skyframe_incremental_build_monitor",
    srcs = ["SkyframeIncrementalBuildMonitor.java"],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/actions",
        "//src/main/java/com/google/devtools/build/lib/concurrent",
        "//src/main/java/com/google/devtools/build/lib/vfs",
        "//src/main/java/com/google/devtools/build/lib/vfs:pathfragment",
        "//src/main/java/com/google/devtools/build/skyframe:skyframe-objects",
        "//third_party:guava",
    ],
)

java_library(
    name = "skylark_module_cycle_reporter",
    srcs = ["StarlarkModuleCycleReporter.java"],
    deps = [
        ":abstract_label_cycle_reporter",
        ":bzl_load_value",
        ":repository_value",
        ":sky_functions",
        "//src/main/java/com/google/devtools/build/lib/cmdline",
        "//src/main/java/com/google/devtools/build/lib/events",
        "//src/main/java/com/google/devtools/build/lib/packages",
        "//src/main/java/com/google/devtools/build/lib/repository:request_repository_information_event",
        "//src/main/java/com/google/devtools/build/skyframe",
        "//src/main/java/com/google/devtools/build/skyframe:skyframe-objects",
        "//third_party:guava",
    ],
)

java_library(
    name = "starlark_file_dependency",
    srcs = ["StarlarkFileDependency.java"],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/cmdline",
        "//src/main/java/com/google/devtools/build/lib/skyframe/serialization/autocodec",
        "//third_party:guava",
    ],
)

java_library(
    name = "bzl_load_value",
    srcs = ["BzlLoadValue.java"],
    deps = [
        ":sky_functions",
        ":starlark_file_dependency",
        "//src/main/java/com/google/devtools/build/lib/cmdline",
        "//src/main/java/com/google/devtools/build/lib/concurrent",
        "//src/main/java/com/google/devtools/build/lib/skyframe/serialization/autocodec",
        "//src/main/java/com/google/devtools/build/lib/syntax:evaluator",
        "//src/main/java/com/google/devtools/build/lib/vfs",
        "//src/main/java/com/google/devtools/build/skyframe:skyframe-objects",
        "//third_party:guava",
    ],
)

java_library(
    name = "state_informing_sky_function_environment",
    srcs = ["StateInformingSkyFunctionEnvironment.java"],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/events",
        "//src/main/java/com/google/devtools/build/lib/util",
        "//src/main/java/com/google/devtools/build/skyframe",
        "//src/main/java/com/google/devtools/build/skyframe:skyframe-objects",
        "//third_party:guava",
        "//third_party:jsr305",
    ],
)

java_library(
    name = "target_excluding_filtering_policy",
    srcs = ["TargetExcludingFilteringPolicy.java"],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/cmdline",
        "//src/main/java/com/google/devtools/build/lib/packages",
        "//src/main/java/com/google/devtools/build/lib/pkgcache",
        "//third_party:guava",
    ],
)

java_library(
    name = "target_pattern_error_function",
    srcs = ["TargetPatternErrorFunction.java"],
    deps = [
        ":sky_functions",
        "//src/main/java/com/google/devtools/build/lib/cmdline",
        "//src/main/java/com/google/devtools/build/lib/concurrent",
        "//src/main/java/com/google/devtools/build/lib/skyframe/serialization/autocodec",
        "//src/main/java/com/google/devtools/build/skyframe",
        "//src/main/java/com/google/devtools/build/skyframe:skyframe-objects",
        "//third_party:guava",
        "//third_party:jsr305",
    ],
)

java_library(
    name = "target_pattern_phase_value",
    srcs = ["TargetPatternPhaseValue.java"],
    deps = [
        ":sky_functions",
        "//src/main/java/com/google/devtools/build/lib/cmdline",
        "//src/main/java/com/google/devtools/build/lib/concurrent",
        "//src/main/java/com/google/devtools/build/lib/events",
        "//src/main/java/com/google/devtools/build/lib/packages",
        "//src/main/java/com/google/devtools/build/lib/pkgcache",
        "//src/main/java/com/google/devtools/build/lib/skyframe/serialization/autocodec",
        "//src/main/java/com/google/devtools/build/skyframe:skyframe-objects",
        "//third_party:guava",
        "//third_party:jsr305",
    ],
)

java_library(
    name = "target_pattern_util",
    srcs = ["TargetPatternUtil.java"],
    deps = [
        ":target_pattern_value",
        "//src/main/java/com/google/devtools/build/lib/cmdline",
        "//src/main/java/com/google/devtools/build/lib/pkgcache",
        "//src/main/java/com/google/devtools/build/skyframe",
        "//src/main/java/com/google/devtools/build/skyframe:skyframe-objects",
        "//third_party:guava",
        "//third_party:jsr305",
    ],
)

java_library(
    name = "target_pattern_value",
    srcs = ["TargetPatternValue.java"],
    deps = [
        ":sky_functions",
        ":target_excluding_filtering_policy",
        "//src/main/java/com/google/devtools/build/lib/cmdline",
        "//src/main/java/com/google/devtools/build/lib/concurrent",
        "//src/main/java/com/google/devtools/build/lib/pkgcache",
        "//src/main/java/com/google/devtools/build/lib/supplier",
        "//src/main/java/com/google/devtools/build/lib/vfs:pathfragment",
        "//src/main/java/com/google/devtools/build/skyframe:skyframe-objects",
        "//third_party:guava",
    ],
)

java_library(
    name = "target_patterns_result_builder",
    srcs = ["TargetPatternsResultBuilder.java"],
    deps = [
        ":package_value",
        "//src/main/java/com/google/devtools/build/lib/cmdline",
        "//src/main/java/com/google/devtools/build/lib/collect/compacthashset",
        "//src/main/java/com/google/devtools/build/lib/packages",
        "//src/main/java/com/google/devtools/build/skyframe",
        "//third_party:guava",
    ],
)

java_library(
    name = "test_expansion_cycle_reporter",
    srcs = ["TestExpansionCycleReporter.java"],
    deps = [
        ":abstract_label_cycle_reporter",
        ":test_expansion_value",
        "//src/main/java/com/google/devtools/build/lib/cmdline",
        "//src/main/java/com/google/devtools/build/lib/pkgcache",
        "//src/main/java/com/google/devtools/build/skyframe",
        "//src/main/java/com/google/devtools/build/skyframe:skyframe-objects",
    ],
)

java_library(
    name = "test_expansion_function",
    srcs = ["TestExpansionFunction.java"],
    deps = [
        ":package_error_function",
        ":package_value",
        ":test_expansion_value",
        "//src/main/java/com/google/devtools/build/lib/cmdline",
        "//src/main/java/com/google/devtools/build/lib/events",
        "//src/main/java/com/google/devtools/build/lib/packages",
        "//src/main/java/com/google/devtools/build/skyframe",
        "//src/main/java/com/google/devtools/build/skyframe:skyframe-objects",
        "//third_party:guava",
        "//third_party:jsr305",
    ],
)

java_library(
    name = "test_expansion_value",
    srcs = ["TestExpansionValue.java"],
    deps = [
        ":sky_functions",
        "//src/main/java/com/google/devtools/build/lib/cmdline",
        "//src/main/java/com/google/devtools/build/lib/concurrent",
        "//src/main/java/com/google/devtools/build/lib/packages",
        "//src/main/java/com/google/devtools/build/lib/skyframe/serialization/autocodec",
        "//src/main/java/com/google/devtools/build/skyframe:skyframe-objects",
        "//third_party:guava",
    ],
)

java_library(
    name = "tests_for_target_pattern_function",
    srcs = ["TestsForTargetPatternFunction.java"],
    deps = [
        ":package_value",
        ":test_expansion_value",
        ":tests_for_target_pattern_value",
        "//src/main/java/com/google/devtools/build/lib/cmdline",
        "//src/main/java/com/google/devtools/build/lib/packages",
        "//src/main/java/com/google/devtools/build/skyframe",
        "//src/main/java/com/google/devtools/build/skyframe:skyframe-objects",
        "//third_party:guava",
        "//third_party:jsr305",
    ],
)

java_library(
    name = "tests_for_target_pattern_value",
    srcs = ["TestsForTargetPatternValue.java"],
    deps = [
        ":sky_functions",
        "//src/main/java/com/google/devtools/build/lib/cmdline",
        "//src/main/java/com/google/devtools/build/lib/concurrent",
        "//src/main/java/com/google/devtools/build/lib/skyframe/serialization/autocodec",
        "//src/main/java/com/google/devtools/build/skyframe:skyframe-objects",
        "//third_party:guava",
    ],
)

java_library(
    name = "toolchain_context_key",
    srcs = ["ToolchainContextKey.java"],
    deps = [
        ":build_configuration_value",
        ":sky_functions",
        "//src/main/java/com/google/devtools/build/lib/cmdline",
        "//src/main/java/com/google/devtools/build/skyframe:skyframe-objects",
        "//third_party:auto_value",
        "//third_party:guava",
    ],
)

java_library(
    name = "toolchain_exception",
    srcs = ["ToolchainException.java"],
)

java_library(
    name = "top_down_action_cache",
    srcs = ["TopDownActionCache.java"],
    deps = [
        ":action_execution_value",
        "//src/main/java/com/google/devtools/build/lib/actionsketch:action_sketch",
        "//third_party:jsr305",
    ],
)

java_library(
    name = "track_source_directories_flag",
    srcs = ["TrackSourceDirectoriesFlag.java"],
)

java_library(
    name = "transitive_base_traversal_function",
    srcs = ["TransitiveBaseTraversalFunction.java"],
    deps = [
        ":containing_package_lookup_value",
        ":package_value",
        "//src/main/java/com/google/devtools/build/lib/actions",
        "//src/main/java/com/google/devtools/build/lib/cmdline",
        "//src/main/java/com/google/devtools/build/lib/events",
        "//src/main/java/com/google/devtools/build/lib/packages",
        "//src/main/java/com/google/devtools/build/lib/vfs:pathfragment",
        "//src/main/java/com/google/devtools/build/skyframe",
        "//src/main/java/com/google/devtools/build/skyframe:skyframe-objects",
        "//third_party:guava",
        "//third_party:jsr305",
    ],
)

java_library(
    name = "transitive_target_cycle_reporter",
    srcs = ["TransitiveTargetCycleReporter.java"],
    deps = [
        ":abstract_label_cycle_reporter",
        ":sky_functions",
        ":transitive_target_key",
        "//src/main/java/com/google/devtools/build/lib/cmdline",
        "//src/main/java/com/google/devtools/build/lib/events",
        "//src/main/java/com/google/devtools/build/lib/packages",
        "//src/main/java/com/google/devtools/build/lib/pkgcache",
        "//src/main/java/com/google/devtools/build/skyframe",
        "//src/main/java/com/google/devtools/build/skyframe:skyframe-objects",
        "//third_party:guava",
    ],
)

java_library(
    name = "transitive_target_key",
    srcs = ["TransitiveTargetKey.java"],
    deps = [
        ":sky_functions",
        "//src/main/java/com/google/devtools/build/lib/cmdline",
        "//src/main/java/com/google/devtools/build/lib/concurrent",
        "//src/main/java/com/google/devtools/build/skyframe:skyframe-objects",
        "//third_party:guava",
    ],
)

java_library(
    name = "transitive_target_value",
    srcs = ["TransitiveTargetValue.java"],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/analysis:config/fragment",
        "//src/main/java/com/google/devtools/build/lib/cmdline",
        "//src/main/java/com/google/devtools/build/lib/collect/nestedset",
        "//src/main/java/com/google/devtools/build/lib/concurrent",
        "//src/main/java/com/google/devtools/build/lib/packages",
        "//src/main/java/com/google/devtools/build/skyframe:skyframe-objects",
        "//third_party:jsr305",
    ],
)

java_library(
    name = "transitive_traversal_function",
    srcs = ["TransitiveTraversalFunction.java"],
    deps = [
        ":sky_functions",
        ":transitive_base_traversal_function",
        ":transitive_traversal_value",
        "//src/main/java/com/google/devtools/build/lib/cmdline",
        "//src/main/java/com/google/devtools/build/lib/events",
        "//src/main/java/com/google/devtools/build/lib/packages",
        "//src/main/java/com/google/devtools/build/lib/util",
        "//src/main/java/com/google/devtools/build/skyframe",
        "//src/main/java/com/google/devtools/build/skyframe:skyframe-objects",
        "//third_party:guava",
        "//third_party:jsr305",
    ],
)

java_library(
    name = "transitive_traversal_value",
    srcs = ["TransitiveTraversalValue.java"],
    deps = [
        ":interner_with_presence_check",
        "//src/main/java/com/google/devtools/build/lib/cmdline",
        "//src/main/java/com/google/devtools/build/lib/concurrent",
        "//src/main/java/com/google/devtools/build/lib/packages",
        "//src/main/java/com/google/devtools/build/lib/skyframe/serialization/autocodec",
        "//src/main/java/com/google/devtools/build/lib/util:string",
        "//src/main/java/com/google/devtools/build/skyframe:skyframe-objects",
        "//third_party:guava",
        "//third_party:jsr305",
    ],
)

java_library(
    name = "traversal_info_root_package_extractor",
    srcs = ["TraversalInfoRootPackageExtractor.java"],
    deps = [
        ":collect_packages_under_directory_value",
        ":recursive_package_provider_backed_target_pattern_resolver",
        ":root_package_extractor",
        "//src/main/java/com/google/devtools/build/lib/cmdline",
        "//src/main/java/com/google/devtools/build/lib/concurrent",
        "//src/main/java/com/google/devtools/build/lib/events",
        "//src/main/java/com/google/devtools/build/lib/vfs",
        "//src/main/java/com/google/devtools/build/lib/vfs:pathfragment",
        "//src/main/java/com/google/devtools/build/skyframe",
        "//src/main/java/com/google/devtools/build/skyframe:skyframe-objects",
        "//third_party:guava",
    ],
)

java_library(
    name = "tree_artifact_value",
    srcs = ["TreeArtifactValue.java"],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/actions",
        "//src/main/java/com/google/devtools/build/lib/skyframe/serialization/autocodec",
        "//src/main/java/com/google/devtools/build/lib/skyframe/serialization/autocodec:serialization-constant",
        "//src/main/java/com/google/devtools/build/lib/vfs",
        "//src/main/java/com/google/devtools/build/lib/vfs:pathfragment",
        "//src/main/java/com/google/devtools/build/skyframe:skyframe-objects",
        "//third_party:guava",
        "//third_party:jsr305",
    ],
)

java_library(
    name = "unloaded_toolchain_context",
    srcs = ["UnloadedToolchainContext.java"],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/analysis:toolchain_context",
        "//src/main/java/com/google/devtools/build/lib/analysis/platform",
        "//src/main/java/com/google/devtools/build/lib/cmdline",
        "//src/main/java/com/google/devtools/build/skyframe:skyframe-objects",
        "//third_party:guava",
    ],
)

java_library(
    name = "workspace_ast_value",
    srcs = ["WorkspaceASTValue.java"],
    deps = [
        ":sky_functions",
        "//src/main/java/com/google/devtools/build/lib/concurrent",
        "//src/main/java/com/google/devtools/build/lib/skyframe/serialization/autocodec",
        "//src/main/java/com/google/devtools/build/lib/syntax:frontend",
        "//src/main/java/com/google/devtools/build/lib/vfs",
        "//src/main/java/com/google/devtools/build/skyframe:skyframe-objects",
        "//third_party:guava",
    ],
)

java_library(
    name = "workspace_name_function",
    srcs = ["WorkspaceNameFunction.java"],
    deps = [
        ":package_value",
        ":workspace_name_value",
        "//src/main/java/com/google/devtools/build/lib/cmdline",
        "//src/main/java/com/google/devtools/build/lib/packages",
        "//src/main/java/com/google/devtools/build/skyframe",
        "//src/main/java/com/google/devtools/build/skyframe:skyframe-objects",
        "//third_party:jsr305",
    ],
)

java_library(
    name = "workspace_name_value",
    srcs = ["WorkspaceNameValue.java"],
    deps = [
        ":sky_functions",
        "//src/main/java/com/google/devtools/build/lib/skyframe/serialization/autocodec",
        "//src/main/java/com/google/devtools/build/skyframe:skyframe-objects",
        "//third_party:guava",
    ],
)

java_library(
    name = "workspace_status_function",
    srcs = ["WorkspaceStatusFunction.java"],
    deps = [
        ":workspace_name_value",
        ":workspace_status_value",
        "//src/main/java/com/google/devtools/build/lib/analysis:workspace_status_action",
        "//src/main/java/com/google/devtools/build/skyframe",
        "//src/main/java/com/google/devtools/build/skyframe:skyframe-objects",
        "//third_party:guava",
    ],
)

java_library(
    name = "workspace_status_value",
    srcs = ["WorkspaceStatusValue.java"],
    deps = [
        ":sky_functions",
        "//src/main/java/com/google/devtools/build/lib/actions",
        "//src/main/java/com/google/devtools/build/lib/analysis:workspace_status_action",
        "//src/main/java/com/google/devtools/build/lib/skyframe/serialization/autocodec",
        "//src/main/java/com/google/devtools/build/skyframe:skyframe-objects",
    ],
)
