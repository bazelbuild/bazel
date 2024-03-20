load("@rules_java//java:defs.bzl", "java_library")

package(
    default_applicable_licenses = ["//:license"],
    default_visibility = ["//src:__subpackages__"],
)

filegroup(
    name = "srcs",
    srcs = glob(["**"]),
    visibility = ["//src:__subpackages__"],
)

java_library(
    name = "execution_requirements",
    srcs = ["ExecutionRequirements.java"],
    deps = [
        "//third_party:auto_value",
        "//third_party:guava",
        "//third_party:jsr305",
    ],
)

java_library(
    name = "resource_estimator",
    srcs = ["ResourceEstimator.java"],
)

# Action graph and cache.
java_library(
    name = "actions",
    srcs = [
        "AbstractAction.java",
        "AbstractCommandLine.java",
        "Action.java",
        "ActionAnalysisMetadata.java",
        "ActionCacheAwareAction.java",
        "ActionCacheChecker.java",
        "ActionCacheUtils.java",
        "ActionCompletionEvent.java",
        "ActionContext.java",
        "ActionContextMarker.java",
        "ActionEnvironment.java",
        "ActionExecutedEvent.java",
        "ActionExecutionContext.java",
        "ActionExecutionException.java",
        "ActionExecutionMetadata.java",
        "ActionExecutionStatusReporter.java",
        "ActionGraph.java",
        "ActionGraphVisitor.java",
        "ActionInputDepOwnerMap.java",
        "ActionInputDepOwners.java",
        "ActionInputMap.java",
        "ActionInputMapSink.java",
        "ActionInputPrefetcher.java",
        "ActionKeyCacher.java",
        "ActionKeyContext.java",
        "ActionLogBufferPathGenerator.java",
        "ActionLookupValue.java",
        "ActionMiddlemanEvent.java",
        "ActionOwner.java",
        "ActionProgressEvent.java",
        "ActionRegistry.java",
        "ActionResult.java",
        "ActionResultReceivedEvent.java",
        "ActionScanningCompletedEvent.java",
        "ActionStartedEvent.java",
        "ActionTemplate.java",
        "ActionUploadFinishedEvent.java",
        "ActionUploadStartedEvent.java",
        "Actions.java",
        "AggregatedSpawnMetrics.java",
        "AlreadyReportedActionExecutionException.java",
        "BaseSpawn.java",
        "BasicActionLookupValue.java",
        "BipartiteVisitor.java",
        "BuildConfigurationEvent.java",
        "BuildFailedException.java",
        "CachedActionEvent.java",
        "CachingActionEvent.java",
        "ChangedFilesMessage.java",
        "CommandAction.java",
        "CommandLine.java",
        "CommandLineLimits.java",
        "CommandLines.java",
        "CompletionContext.java",
        "DelegateSpawn.java",
        "DelegatingPairInputMetadataProvider.java",
        "DigestOfDirectoryException.java",
        "DiscoveredInputsEvent.java",
        "DiscoveredModulesPruner.java",
        "DynamicStrategyRegistry.java",
        "EnvironmentalExecException.java",
        "EventReportingArtifacts.java",
        "ExecException.java",
        "Executor.java",
        "FailAction.java",
        "FilesetManifest.java",
        "FilesetTraversalParams.java",
        "FilesetTraversalParamsFactory.java",
        "ForbiddenActionInputException.java",
        "FullSpawnMetrics.java",
        "HasDigest.java",
        "InputFileErrorException.java",
        "InputMetadataProvider.java",
        "LostInputsActionExecutionException.java",
        "LostInputsExecException.java",
        "MapBasedActionGraph.java",
        "MiddlemanAction.java",
        "MiddlemanFactory.java",
        "MissingDepExecException.java",
        "MissingInputFileException.java",
        "MutableActionGraph.java",
        "NotifyOnActionCacheHit.java",
        "ParamFileInfo.java",
        "ParameterFile.java",
        "PathMapper.java",
        "PathStrippable.java",
        "RemoteArtifactChecker.java",
        "ResourceEstimator.java",
        "RunningActionEvent.java",
        "SandboxedSpawnStrategy.java",
        "ScanningActionEvent.java",
        "SchedulingActionEvent.java",
        "SimpleSpawn.java",
        "SingleStringArgFormatter.java",
        "Spawn.java",
        "SpawnActionExecutionException.java",
        "SpawnExecutedEvent.java",
        "SpawnMetrics.java",
        "SpawnResult.java",
        "SpawnStrategy.java",
        "Spawns.java",
        "StaticInputMetadataProvider.java",
        "StoppedScanningActionEvent.java",
        "TestExecException.java",
        "TestMiddlemanObserver.java",
        "TopLevelOutputException.java",
        "TotalAndConfiguredTargetOnlyMetric.java",
        "UserExecException.java",
        "cache/ActionCache.java",
        "cache/CompactPersistentActionCache.java",
        "cache/MetadataInjector.java",
        "cache/OutputMetadataStore.java",
        "cache/PersistentStringIndexer.java",
        "cache/PostableActionCacheStats.java",
        "cache/VirtualActionInput.java",
    ],
    deps = [
        ":action_input_helper",
        ":action_lookup_data",
        ":action_lookup_key",
        ":artifact_owner",
        ":artifacts",
        ":commandline_item",
        ":execution_requirements",
        ":file_metadata",
        ":fileset_output_symlink",
        ":localhost_capacity",
        ":middleman_type",
        ":package_roots",
        ":runfiles_metadata",
        ":runfiles_tree",
        ":thread_state_receiver",
        "//src/main/java/com/google/devtools/build/lib/analysis:config/build_configuration",
        "//src/main/java/com/google/devtools/build/lib/analysis/platform",
        "//src/main/java/com/google/devtools/build/lib/bugreport",
        "//src/main/java/com/google/devtools/build/lib/buildeventstream",
        "//src/main/java/com/google/devtools/build/lib/buildeventstream/proto:build_event_stream_java_proto",
        "//src/main/java/com/google/devtools/build/lib/causes",
        "//src/main/java/com/google/devtools/build/lib/clock",
        "//src/main/java/com/google/devtools/build/lib/cmdline",
        "//src/main/java/com/google/devtools/build/lib/collect/nestedset",
        "//src/main/java/com/google/devtools/build/lib/collect/nestedset:fingerprint_cache",
        "//src/main/java/com/google/devtools/build/lib/concurrent",
        "//src/main/java/com/google/devtools/build/lib/events",
        "//src/main/java/com/google/devtools/build/lib/packages",
        "//src/main/java/com/google/devtools/build/lib/profiler",
        "//src/main/java/com/google/devtools/build/lib/profiler:google-auto-profiler-utils",
        "//src/main/java/com/google/devtools/build/lib/remote:store",
        "//src/main/java/com/google/devtools/build/lib/shell",
        "//src/main/java/com/google/devtools/build/lib/skyframe:detailed_exceptions",
        "//src/main/java/com/google/devtools/build/lib/skyframe:sane_analysis_exception",
        "//src/main/java/com/google/devtools/build/lib/skyframe:skyframe_aware_action",
        "//src/main/java/com/google/devtools/build/lib/skyframe:tree_artifact_value",
        "//src/main/java/com/google/devtools/build/lib/skyframe/serialization:visible-for-serialization",
        "//src/main/java/com/google/devtools/build/lib/skyframe/serialization/autocodec",
        "//src/main/java/com/google/devtools/build/lib/skyframe/serialization/autocodec:serialization-constant",
        "//src/main/java/com/google/devtools/build/lib/starlarkbuildapi",
        "//src/main/java/com/google/devtools/build/lib/unsafe:string",
        "//src/main/java/com/google/devtools/build/lib/util",
        "//src/main/java/com/google/devtools/build/lib/util:command",
        "//src/main/java/com/google/devtools/build/lib/util:describable_execution_unit",
        "//src/main/java/com/google/devtools/build/lib/util:detailed_exit_code",
        "//src/main/java/com/google/devtools/build/lib/util:exit_code",
        "//src/main/java/com/google/devtools/build/lib/util:filetype",
        "//src/main/java/com/google/devtools/build/lib/util:os",
        "//src/main/java/com/google/devtools/build/lib/util:shell_escaper",
        "//src/main/java/com/google/devtools/build/lib/util:string",
        "//src/main/java/com/google/devtools/build/lib/util:var_int",
        "//src/main/java/com/google/devtools/build/lib/util/io",
        "//src/main/java/com/google/devtools/build/lib/vfs",
        "//src/main/java/com/google/devtools/build/lib/vfs:ospathpolicy",
        "//src/main/java/com/google/devtools/build/lib/vfs:pathfragment",
        "//src/main/java/com/google/devtools/build/lib/vfs/inmemoryfs",
        "//src/main/java/com/google/devtools/build/skyframe",
        "//src/main/java/com/google/devtools/build/skyframe:skyframe-objects",
        "//src/main/java/com/google/devtools/common/options",
        "//src/main/java/net/starlark/java/eval",
        "//src/main/java/net/starlark/java/syntax",
        "//src/main/protobuf:action_cache_java_proto",
        "//src/main/protobuf:extra_actions_base_java_proto",
        "//src/main/protobuf:failure_details_java_proto",
        "//src/main/protobuf:spawn_java_proto",
        "//third_party:auto_value",
        "//third_party:checker_framework_annotations",
        "//third_party:flogger",
        "//third_party:guava",
        "//third_party:jsr305",
        "//third_party/protobuf:protobuf_java",
        "@remoteapis//:build_bazel_remote_execution_v2_remote_execution_java_proto",
    ],
)

java_library(
    name = "action_environment",
    srcs = ["ActionEnvironment.java"],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/util",
        "//third_party:guava",
    ],
)

java_library(
    name = "action_lookup_data",
    srcs = ["ActionLookupData.java"],
    deps = [
        ":action_lookup_key",
        "//src/main/java/com/google/devtools/build/lib/cmdline",
        "//src/main/java/com/google/devtools/build/lib/skyframe:sky_functions",
        "//src/main/java/com/google/devtools/build/skyframe:execution_phase_skykey",
        "//src/main/java/com/google/devtools/build/skyframe:skyframe-objects",
        "//third_party:guava",
    ],
)

java_library(
    name = "action_input_helper",
    srcs = ["ActionInputHelper.java"],
    deps = [
        ":artifacts",
        "//src/main/java/com/google/devtools/build/lib/collect/nestedset",
        "//src/main/java/com/google/devtools/build/lib/vfs",
        "//src/main/java/com/google/devtools/build/lib/vfs:pathfragment",
        "//third_party:guava",
    ],
)

java_library(
    name = "action_lookup_key",
    srcs = [
        "ActionLookupKey.java",
    ],
    deps = [
        ":artifact_owner",
        "//src/main/java/com/google/devtools/build/lib/skyframe/config",
        "//src/main/java/com/google/devtools/build/skyframe:cpu_heavy_skykey",
        "//src/main/java/com/google/devtools/build/skyframe:skyframe-objects",
        "//third_party:jsr305",
    ],
)

java_library(
    name = "action_output_directory_helper",
    srcs = ["ActionOutputDirectoryHelper.java"],
    deps = [
        ":artifacts",
        "//src/main/java/com/google/devtools/build/lib/vfs",
        "//src/main/java/com/google/devtools/build/lib/vfs:pathfragment",
        "//third_party:caffeine",
        "//third_party:guava",
    ],
)

java_library(
    name = "analysis_graph_stats_event",
    srcs = ["AnalysisGraphStatsEvent.java"],
    deps = ["//src/main/java/com/google/devtools/build/lib/buildeventstream/proto:build_event_stream_java_proto"],
)

java_library(
    name = "artifact_owner",
    srcs = ["ArtifactOwner.java"],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/cmdline",
        "//src/main/java/com/google/devtools/build/lib/skyframe/serialization/autocodec:serialization-constant",
        "//third_party:jsr305",
    ],
)

java_library(
    name = "artifacts",
    srcs = [
        "ActionInput.java",
        "Artifact.java",
        "ArtifactCodecs.java",
        "ArtifactFactory.java",
        "ArtifactPathResolver.java",
        "ArtifactPrefixConflictException.java",
        "ArtifactResolver.java",
        "ArtifactRoot.java",
        "Artifacts.java",
    ],
    deps = [
        ":action_lookup_data",
        ":action_lookup_key",
        ":artifact_owner",
        ":commandline_item",
        ":fileset_output_symlink",
        ":package_roots",
        ":path_strippable",
        "//src/main/java/com/google/devtools/build/lib/cmdline",
        "//src/main/java/com/google/devtools/build/lib/collect/nestedset",
        "//src/main/java/com/google/devtools/build/lib/concurrent",
        "//src/main/java/com/google/devtools/build/lib/skyframe:detailed_exceptions",
        "//src/main/java/com/google/devtools/build/lib/skyframe:sky_functions",
        "//src/main/java/com/google/devtools/build/lib/skyframe/serialization",
        "//src/main/java/com/google/devtools/build/lib/skyframe/serialization:visible-for-serialization",
        "//src/main/java/com/google/devtools/build/lib/skyframe/serialization/autocodec",
        "//src/main/java/com/google/devtools/build/lib/skyframe/serialization/autocodec:serialization-constant",
        "//src/main/java/com/google/devtools/build/lib/starlarkbuildapi",
        "//src/main/java/com/google/devtools/build/lib/util",
        "//src/main/java/com/google/devtools/build/lib/util:detailed_exit_code",
        "//src/main/java/com/google/devtools/build/lib/util:filetype",
        "//src/main/java/com/google/devtools/build/lib/util:hash_codes",
        "//src/main/java/com/google/devtools/build/lib/vfs",
        "//src/main/java/com/google/devtools/build/lib/vfs:pathfragment",
        "//src/main/java/com/google/devtools/build/skyframe:execution_phase_skykey",
        "//src/main/java/com/google/devtools/build/skyframe:skyframe-objects",
        "//src/main/java/net/starlark/java/eval",
        "//src/main/protobuf:failure_details_java_proto",
        "//third_party:guava",
        "//third_party:jsr305",
        "//third_party/protobuf:protobuf_java",
    ],
)

java_library(
    name = "build_configuration_event",
    srcs = ["BuildConfigurationEvent.java"],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/buildeventstream",
        "//src/main/java/com/google/devtools/build/lib/buildeventstream/proto:build_event_stream_java_proto",
        "//third_party:guava",
    ],
)

java_library(
    name = "commandline_item",
    srcs = [
        "CommandLineExpansionException.java",
        "CommandLineItem.java",
    ],
    deps = ["//third_party:jsr305"],
)

java_library(
    name = "commandline_limits",
    srcs = ["CommandLineLimits.java"],
)

java_library(
    name = "exec_exception",
    srcs = [
        "ExecException.java",
        "UserExecException.java",
    ],
    deps = ["//src/main/protobuf:failure_details_java_proto"],
)

java_library(
    name = "fileset_output_symlink",
    srcs = ["FilesetOutputSymlink.java"],
    deps = [
        ":has_digest",
        "//src/main/java/com/google/devtools/build/lib/vfs:pathfragment",
        "//third_party:auto_value",
        "//third_party:guava",
    ],
)

java_library(
    name = "runfiles_metadata",
    srcs = ["RunfilesArtifactValue.java"],
    deps = [
        ":artifacts",
        ":file_metadata",
        ":runfiles_tree",
        "//src/main/java/com/google/devtools/build/lib/skyframe:tree_artifact_value",
        "//src/main/java/com/google/devtools/build/lib/util:hash_codes",
        "//src/main/java/com/google/devtools/build/skyframe:skyframe-objects",
        "//third_party:guava",
    ],
)

java_library(
    name = "file_metadata",
    srcs = [
        "FileArtifactValue.java",
        "FileContentsProxy.java",
        "FileStateType.java",
        "FileStateValue.java",
        "FileStatusWithMetadata.java",
        "FileValue.java",
        "RemoteArtifactChecker.java",
        "cache/MetadataDigestUtils.java",
    ],
    deps = [
        ":artifacts",
        ":has_digest",
        "//src/main/java/com/google/devtools/build/lib/concurrent",
        "//src/main/java/com/google/devtools/build/lib/io:inconsistent_filesystem_exception",
        "//src/main/java/com/google/devtools/build/lib/skyframe/serialization:visible-for-serialization",
        "//src/main/java/com/google/devtools/build/lib/skyframe/serialization/autocodec",
        "//src/main/java/com/google/devtools/build/lib/skyframe/serialization/autocodec:serialization-constant",
        "//src/main/java/com/google/devtools/build/lib/util",
        "//src/main/java/com/google/devtools/build/lib/util:var_int",
        "//src/main/java/com/google/devtools/build/lib/util/io",
        "//src/main/java/com/google/devtools/build/lib/vfs",
        "//src/main/java/com/google/devtools/build/lib/vfs:pathfragment",
        "//src/main/java/com/google/devtools/build/skyframe:skyframe-objects",
        "//third_party:guava",
        "//third_party:jsr305",
    ],
)

java_library(
    name = "has_digest",
    srcs = ["HasDigest.java"],
    deps = ["//third_party/protobuf:protobuf_java"],
)

java_library(
    name = "important_output_handler",
    srcs = ["ImportantOutputHandler.java"],
    deps = [
        ":actions",
        ":artifacts",
        "//src/main/java/com/google/devtools/build/lib/skyframe:detailed_exceptions",
        "//src/main/java/com/google/devtools/build/lib/util:detailed_exit_code",
        "//src/main/protobuf:failure_details_java_proto",
        "//third_party:guava",
    ],
)

java_library(
    name = "localhost_capacity",
    srcs = [
        "LocalHostCapacity.java",
        "LocalHostResource.java",
        "ResourceSet.java",
        "ResourceSetOrBuilder.java",
    ],
    deps = [
        ":exec_exception",
        "//src/main/java/com/google/devtools/build/lib/concurrent",
        "//src/main/java/com/google/devtools/build/lib/util:os",
        "//src/main/java/com/google/devtools/build/lib/worker:worker_key",
        "//src/main/java/com/google/devtools/common/options",
        "//third_party:flogger",
        "//third_party:guava",
        "//third_party:jsr305",
    ],
)

java_library(
    name = "middleman_type",
    srcs = ["MiddlemanType.java"],
)

java_library(
    name = "package_roots",
    srcs = [
        "PackageRootResolver.java",
        "PackageRoots.java",
    ],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/bugreport",
        "//src/main/java/com/google/devtools/build/lib/cmdline",
        "//src/main/java/com/google/devtools/build/lib/io:inconsistent_filesystem_exception",
        "//src/main/java/com/google/devtools/build/lib/packages",
        "//src/main/java/com/google/devtools/build/lib/util:detailed_exit_code",
        "//src/main/java/com/google/devtools/build/lib/vfs",
        "//src/main/java/com/google/devtools/build/lib/vfs:pathfragment",
        "//src/main/protobuf:failure_details_java_proto",
        "//third_party:guava",
        "//third_party:jsr305",
    ],
)

java_library(
    name = "path_strippable",
    srcs = ["PathStrippable.java"],
    deps = [
        ":commandline_item",
        "//src/main/java/com/google/devtools/build/lib/vfs:pathfragment",
    ],
)

java_library(
    name = "resource_manager",
    srcs = [
        "ResourceManager.java",
    ],
    deps = [
        ":actions",
        ":localhost_capacity",
        "//src/main/java/com/google/devtools/build/lib/concurrent",
        "//src/main/java/com/google/devtools/build/lib/profiler",
        "//src/main/java/com/google/devtools/build/lib/util",
        "//src/main/java/com/google/devtools/build/lib/worker",
        "//src/main/java/com/google/devtools/build/lib/worker:worker_key",
        "//src/main/java/com/google/devtools/build/lib/worker:worker_pool",
        "//src/main/java/com/google/devtools/build/lib/worker:worker_process_status",
        "//third_party:guava",
        "//third_party:jsr305",
    ],
)

java_library(
    name = "runfiles_tree",
    srcs = ["RunfilesTree.java"],
    deps = [
        ":artifacts",
        "//src/main/java/com/google/devtools/build/lib/analysis:config/build_configuration",
        "//src/main/java/com/google/devtools/build/lib/collect/nestedset",
        "//src/main/java/com/google/devtools/build/lib/vfs:pathfragment",
    ],
)

java_library(
    name = "shared_action_event",
    srcs = ["SharedActionEvent.java"],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/events",
        "//src/main/java/com/google/devtools/build/lib/skyframe:action_execution_value",
        "//third_party:guava",
    ],
)

java_library(
    name = "thread_state_receiver",
    srcs = ["ThreadStateReceiver.java"],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/profiler",
        "//third_party:error_prone_annotations",
    ],
)

java_library(
    name = "total_and_configured_target_only_metric",
    srcs = ["TotalAndConfiguredTargetOnlyMetric.java"],
    deps = [
        "//third_party:auto_value",
    ],
)
