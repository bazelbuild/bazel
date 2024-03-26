load("@rules_java//java:defs.bzl", "java_library")

package(
    default_applicable_licenses = ["//:license"],
    default_visibility = ["//src:__subpackages__"],
)

filegroup(
    name = "srcs",
    srcs = glob(["*"]) + [
        "//src/main/java/com/google/devtools/build/lib/bazel/bzlmod/modcommand:srcs",
    ],
    visibility = ["//src:__subpackages__"],
)

java_library(
    name = "common",
    srcs = [
        "ArchiveRepoSpecBuilder.java",
        "AttributeValues.java",
        "GitRepoSpecBuilder.java",
        "ModuleFile.java",
        "ModuleKey.java",
        "RepoSpec.java",
        "Version.java",
    ],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/cmdline",
        "//src/main/java/com/google/devtools/build/lib/vfs:pathfragment",
        "//src/main/java/com/google/devtools/build/skyframe:skyframe-objects",
        "//src/main/java/net/starlark/java/eval",
        "//third_party:auto_value",
        "//third_party:gson",
        "//third_party:guava",
        "//third_party:jsr305",
    ],
)

java_library(
    name = "exception",
    srcs = ["ExternalDepsException.java"],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/skyframe:detailed_exceptions",
        "//src/main/java/com/google/devtools/build/lib/util:detailed_exit_code",
        "//src/main/protobuf:failure_details_java_proto",
        "//third_party:error_prone_annotations",
        "//third_party:jsr305",
    ],
)

java_library(
    name = "module_extension",
    srcs = [
        "ModuleExtension.java",
        "ModuleExtensionId.java",
        "ModuleExtensionUsage.java",
        "Tag.java",
        "TagClass.java",
    ],
    deps = [
        ":common",
        "//src/main/java/com/google/devtools/build/lib/cmdline",
        "//src/main/java/com/google/devtools/build/lib/packages",
        "//src/main/java/com/google/devtools/build/lib/starlarkbuildapi/repository",
        "//src/main/java/net/starlark/java/eval",
        "//src/main/java/net/starlark/java/syntax",
        "//third_party:auto_value",
        "//third_party:gson",
        "//third_party:guava",
    ],
)

java_library(
    name = "registry",
    srcs = [
        "IndexRegistry.java",
        "Registry.java",
        "RegistryFactory.java",
        "RegistryFactoryImpl.java",
    ],
    deps = [
        ":common",
        "//src/main/java/com/google/devtools/build/lib/bazel/repository/downloader",
        "//src/main/java/com/google/devtools/build/lib/cmdline",
        "//src/main/java/com/google/devtools/build/lib/events",
        "//src/main/java/com/google/devtools/build/lib/profiler",
        "//src/main/java/com/google/devtools/build/lib/util:os",
        "//src/main/java/com/google/devtools/build/lib/vfs",
        "//src/main/java/com/google/devtools/build/lib/vfs:pathfragment",
        "//third_party:caffeine",
        "//third_party:gson",
        "//third_party:guava",
    ],
)

java_library(
    name = "bazel_lockfile_module",
    srcs = ["BazelLockFileModule.java"],
    deps = [
        ":common",
        ":exception",
        ":resolution",
        ":resolution_impl",
        "//src/main/java/com/google/devtools/build/lib:runtime",
        "//src/main/java/com/google/devtools/build/lib/bazel/bzlmod:module_extension",
        "//src/main/java/com/google/devtools/build/lib/bazel/repository:repository_options",
        "//src/main/java/com/google/devtools/build/lib/cmdline",
        "//src/main/java/com/google/devtools/build/lib/util:abrupt_exit_exception",
        "//src/main/java/com/google/devtools/build/lib/vfs",
        "//third_party:flogger",
        "//third_party:guava",
        "//third_party:jsr305",
    ],
)

java_library(
    name = "resolution",
    srcs = [
        "AbridgedModule.java",
        "ArchiveOverride.java",
        "BazelDepGraphValue.java",
        "BazelFetchAllValue.java",
        "BazelLockFileValue.java",
        "BazelModuleResolutionEvent.java",
        "BazelModuleResolutionValue.java",
        "BzlmodFlagsAndEnvVars.java",
        "GitOverride.java",
        "InterimModule.java",
        "LocalPathOverride.java",
        "LockFileModuleExtension.java",
        "Module.java",
        "ModuleBase.java",
        "ModuleExtensionEvalFactors.java",
        "ModuleExtensionEvalStarlarkThreadContext.java",
        "ModuleExtensionRepoMappingEntriesValue.java",
        "ModuleExtensionResolutionEvent.java",
        "ModuleFileGlobals.java",
        "ModuleFileValue.java",
        "ModuleOverride.java",
        "ModuleThreadContext.java",
        "MultipleVersionOverride.java",
        "NonRegistryOverride.java",
        "RegistryOverride.java",
        "RepoSpecKey.java",
        "SingleExtensionEvalValue.java",
        "SingleExtensionUsagesValue.java",
        "SingleVersionOverride.java",
    ],
    deps = [
        ":common",
        ":inspection",
        ":module_extension",
        ":module_extension_metadata",
        ":registry",
        ":repo_rule_creator",
        "//src/main/java/com/google/devtools/build/docgen/annot",
        "//src/main/java/com/google/devtools/build/lib/analysis:blaze_directories",
        "//src/main/java/com/google/devtools/build/lib/cmdline",
        "//src/main/java/com/google/devtools/build/lib/events",
        "//src/main/java/com/google/devtools/build/lib/packages",
        "//src/main/java/com/google/devtools/build/lib/rules:repository/repo_recorded_input",
        "//src/main/java/com/google/devtools/build/lib/skyframe:sky_functions",
        "//src/main/java/com/google/devtools/build/lib/skyframe/serialization:visible-for-serialization",
        "//src/main/java/com/google/devtools/build/lib/skyframe/serialization/autocodec",
        "//src/main/java/com/google/devtools/build/lib/skyframe/serialization/autocodec:serialization-constant",
        "//src/main/java/com/google/devtools/build/lib/vfs:pathfragment",
        "//src/main/java/com/google/devtools/build/skyframe:skyframe-objects",
        "//src/main/java/net/starlark/java/annot",
        "//src/main/java/net/starlark/java/eval",
        "//src/main/java/net/starlark/java/syntax",
        "//third_party:auto_value",
        "//third_party:gson",
        "//third_party:guava",
        "//third_party:jsr305",
    ],
)

java_library(
    name = "resolution_impl",
    srcs = [
        "AttributeValuesAdapter.java",
        "BazelDepGraphFunction.java",
        "BazelFetchAllFunction.java",
        "BazelLockFileFunction.java",
        "BazelModuleResolutionFunction.java",
        "BzlmodFlagsAndEnvVars.java",
        "DelegateTypeAdapterFactory.java",
        "Discovery.java",
        "GsonTypeAdapterUtil.java",
        "ModuleExtensionContext.java",
        "ModuleExtensionEvaluationProgress.java",
        "ModuleExtensionRepoMappingEntriesFunction.java",
        "ModuleFileFunction.java",
        "RepoSpecFunction.java",
        "Selection.java",
        "SingleExtensionEvalFunction.java",
        "SingleExtensionUsagesFunction.java",
        "StarlarkBazelModule.java",
        "TypeCheckedTag.java",
        "YankedVersionsUtil.java",
    ],
    deps = [
        ":common",
        ":exception",
        ":inspection",
        ":module_extension",
        ":module_extension_metadata",
        ":registry",
        ":repo_rule_creator",
        ":resolution",
        "//src/main/java/com/google/devtools/build/docgen/annot",
        "//src/main/java/com/google/devtools/build/lib:runtime",
        "//src/main/java/com/google/devtools/build/lib/actions:file_metadata",
        "//src/main/java/com/google/devtools/build/lib/analysis:blaze_directories",
        "//src/main/java/com/google/devtools/build/lib/analysis:blaze_version_info",
        "//src/main/java/com/google/devtools/build/lib/bazel:bazel_version",
        "//src/main/java/com/google/devtools/build/lib/bazel/bzlmod:repo_rule_value",
        "//src/main/java/com/google/devtools/build/lib/bazel/repository:repository_options",
        "//src/main/java/com/google/devtools/build/lib/bazel/repository/downloader",
        "//src/main/java/com/google/devtools/build/lib/bazel/repository/starlark",
        "//src/main/java/com/google/devtools/build/lib/cmdline",
        "//src/main/java/com/google/devtools/build/lib/events",
        "//src/main/java/com/google/devtools/build/lib/packages",
        "//src/main/java/com/google/devtools/build/lib/profiler",
        "//src/main/java/com/google/devtools/build/lib/rules:repository/repo_recorded_input",
        "//src/main/java/com/google/devtools/build/lib/rules:repository/repository_directory_value",
        "//src/main/java/com/google/devtools/build/lib/rules:repository/repository_function",
        "//src/main/java/com/google/devtools/build/lib/skyframe:bzl_load_failed_exception",
        "//src/main/java/com/google/devtools/build/lib/skyframe:bzl_load_value",
        "//src/main/java/com/google/devtools/build/lib/skyframe:client_environment_function",
        "//src/main/java/com/google/devtools/build/lib/skyframe:client_environment_value",
        "//src/main/java/com/google/devtools/build/lib/skyframe:precomputed_value",
        "//src/main/java/com/google/devtools/build/lib/skyframe:repository_mapping_value",
        "//src/main/java/com/google/devtools/build/lib/skyframe:skyframe_cluster",
        "//src/main/java/com/google/devtools/build/lib/util",
        "//src/main/java/com/google/devtools/build/lib/util:os",
        "//src/main/java/com/google/devtools/build/lib/vfs",
        "//src/main/java/com/google/devtools/build/lib/vfs:pathfragment",
        "//src/main/java/com/google/devtools/build/skyframe",
        "//src/main/java/com/google/devtools/build/skyframe:skyframe-objects",
        "//src/main/java/net/starlark/java/annot",
        "//src/main/java/net/starlark/java/eval",
        "//src/main/java/net/starlark/java/spelling",
        "//src/main/java/net/starlark/java/syntax",
        "//src/main/protobuf:failure_details_java_proto",
        "//third_party:auto_value",
        "//third_party:gson",
        "//third_party:guava",
        "//third_party:jsr305",
    ],
)

java_library(
    name = "repo_rule_value",
    srcs = [
        "BzlmodRepoRuleValue.java",
    ],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/cmdline",
        "//src/main/java/com/google/devtools/build/lib/packages",
        "//src/main/java/com/google/devtools/build/lib/skyframe/serialization:visible-for-serialization",
        "//src/main/java/com/google/devtools/build/lib/skyframe/serialization/autocodec",
        "//src/main/java/com/google/devtools/build/skyframe:skyframe-objects",
    ],
)

java_library(
    name = "repo_rule_creator",
    srcs = ["BzlmodRepoRuleCreator.java"],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/analysis:blaze_directories",
        "//src/main/java/com/google/devtools/build/lib/cmdline",
        "//src/main/java/com/google/devtools/build/lib/events",
        "//src/main/java/com/google/devtools/build/lib/packages",
        "//src/main/java/com/google/devtools/build/lib/packages/semantics",
        "//src/main/java/com/google/devtools/build/lib/vfs",
        "//src/main/java/net/starlark/java/eval",
        "//src/main/java/net/starlark/java/syntax",
        "//third_party:guava",
    ],
)

java_library(
    name = "tidy",
    srcs = [
        "BazelModTidyValue.java",
    ],
    deps = [
        ":resolution",
        "//src/main/java/com/google/devtools/build/lib/bazel/repository:repository_options",
        "//src/main/java/com/google/devtools/build/lib/skyframe:sky_functions",
        "//src/main/java/com/google/devtools/build/lib/skyframe/serialization/autocodec:serialization-constant",
        "//src/main/java/com/google/devtools/build/lib/vfs",
        "//src/main/java/com/google/devtools/build/skyframe:skyframe-objects",
        "//src/main/java/net/starlark/java/eval",
        "//third_party:auto_value",
        "//third_party:guava",
    ],
)

java_library(
    name = "tidy_impl",
    srcs = [
        "BazelModTidyFunction.java",
    ],
    deps = [
        ":common",
        ":exception",
        ":resolution",
        ":resolution_impl",
        ":tidy",
        "//src/main/java/com/google/devtools/build/lib/cmdline",
        "//src/main/java/com/google/devtools/build/lib/rules:repository/repository_function",
        "//src/main/java/com/google/devtools/build/lib/skyframe:precomputed_value",
        "//src/main/java/com/google/devtools/build/lib/skyframe:repository_mapping_value",
        "//src/main/java/com/google/devtools/build/lib/vfs",
        "//src/main/java/com/google/devtools/build/skyframe",
        "//src/main/java/com/google/devtools/build/skyframe:skyframe-objects",
        "//src/main/java/net/starlark/java/eval",
        "//src/main/protobuf:failure_details_java_proto",
        "//third_party:guava",
        "//third_party:jsr305",
    ],
)

java_library(
    name = "inspection",
    srcs = [
        "BazelModuleInspectorValue.java",
    ],
    deps = [
        ":common",
        ":module_extension",
        "//src/main/java/com/google/devtools/build/lib/cmdline",
        "//src/main/java/com/google/devtools/build/lib/skyframe:sky_functions",
        "//src/main/java/com/google/devtools/build/lib/skyframe/serialization/autocodec:serialization-constant",
        "//src/main/java/com/google/devtools/build/skyframe:skyframe-objects",
        "//third_party:auto_value",
        "//third_party:guava",
    ],
)

java_library(
    name = "inspection_impl",
    srcs = [
        "BazelModuleInspectorFunction.java",
    ],
    deps = [
        ":common",
        ":inspection",
        ":module_extension",
        ":resolution",
        "//src/main/java/com/google/devtools/build/skyframe",
        "//src/main/java/com/google/devtools/build/skyframe:skyframe-objects",
        "//third_party:guava",
        "//third_party:jsr305",
    ],
)

java_library(
    name = "module_extension_metadata",
    srcs = [
        "ModuleExtensionMetadata.java",
    ],
    deps = [
        ":common",
        ":module_extension",
        ":module_file_fixup_event",
        "//src/main/java/com/google/devtools/build/docgen/annot",
        "//src/main/java/com/google/devtools/build/lib/cmdline",
        "//src/main/java/com/google/devtools/build/lib/events",
        "//src/main/java/net/starlark/java/annot",
        "//src/main/java/net/starlark/java/eval",
        "//src/main/java/net/starlark/java/syntax",
        "//third_party:auto_value",
        "//third_party:gson",
        "//third_party:guava",
        "//third_party:jsr305",
    ],
)

java_library(
    name = "module_file_fixup_event",
    srcs = [
        "RootModuleFileFixupEvent.java",
    ],
    deps = [
        ":module_extension",
        "//src/main/java/com/google/devtools/build/lib/events",
        "//third_party:guava",
    ],
)
