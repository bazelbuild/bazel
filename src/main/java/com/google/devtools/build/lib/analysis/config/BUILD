load("@rules_java//java:defs.bzl", "java_library")

package(
    default_applicable_licenses = ["//:license"],
    default_visibility = ["//src:__subpackages__"],
)

filegroup(
    name = "srcs",
    srcs = glob(["**"]) + [
        "//src/main/java/com/google/devtools/build/lib/analysis/config/output:srcs",
        "//src/main/java/com/google/devtools/build/lib/analysis/config/transitions:srcs",
    ],
    visibility = ["//src:__subpackages__"],
)

java_library(
    name = "auto_cpu_converter",
    srcs = ["AutoCpuConverter.java"],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/util",
        "//src/main/java/com/google/devtools/build/lib/util:os",
        "//src/main/java/com/google/devtools/common/options",
    ],
)

java_library(
    name = "fragment_class_set",
    srcs = ["FragmentClassSet.java"],
    deps = [
        ":fragment",
        "//src/main/java/com/google/devtools/build/lib/concurrent",
        "//src/main/java/com/google/devtools/build/lib/skyframe/serialization/autocodec:serialization-constant",
        "//src/main/java/com/google/devtools/build/lib/util",
        "//third_party:guava",
        "//third_party:jsr305",
    ],
)

java_library(
    name = "build_configuration",
    srcs = [
        "BuildConfigurationInfo.java",
        "BuildConfigurationValue.java",
        "ConfigRequestedEvent.java",
        "OutputDirectories.java",
        "OutputPathMnemonicComputer.java",
    ],
    deps = [
        ":build_option_details",
        ":build_options",
        ":compilation_mode",
        ":core_options",
        ":execution_info_modifier",
        ":feature_set",
        ":fragment",
        ":fragment_class_set",
        ":fragment_factory",
        ":fragment_options",
        ":fragment_registry",
        ":invalid_configuration_exception",
        ":optioninfo",
        ":options_diff",
        ":run_under",
        "//src/main/java/com/google/devtools/build/lib/actions:action_environment",
        "//src/main/java/com/google/devtools/build/lib/actions:artifacts",
        "//src/main/java/com/google/devtools/build/lib/actions:build_configuration_event",
        "//src/main/java/com/google/devtools/build/lib/actions:commandline_limits",
        "//src/main/java/com/google/devtools/build/lib/analysis:blaze_directories",
        "//src/main/java/com/google/devtools/build/lib/analysis:platform_options",
        "//src/main/java/com/google/devtools/build/lib/analysis:test/test_configuration",
        "//src/main/java/com/google/devtools/build/lib/analysis:test/test_trimming_logic",
        "//src/main/java/com/google/devtools/build/lib/buildeventstream",
        "//src/main/java/com/google/devtools/build/lib/buildeventstream/proto:build_event_stream_java_proto",
        "//src/main/java/com/google/devtools/build/lib/cmdline",
        "//src/main/java/com/google/devtools/build/lib/concurrent",
        "//src/main/java/com/google/devtools/build/lib/events",
        "//src/main/java/com/google/devtools/build/lib/packages",
        "//src/main/java/com/google/devtools/build/lib/skyframe/config",
        "//src/main/java/com/google/devtools/build/lib/skyframe/serialization/autocodec",
        "//src/main/java/com/google/devtools/build/lib/starlarkbuildapi",
        "//src/main/java/com/google/devtools/build/lib/util",
        "//src/main/java/com/google/devtools/build/lib/util:os",
        "//src/main/java/com/google/devtools/build/lib/vfs",
        "//src/main/java/com/google/devtools/build/lib/vfs:pathfragment",
        "//src/main/java/com/google/devtools/build/skyframe:skyframe-objects",
        "//src/main/java/com/google/devtools/common/options",
        "//src/main/java/net/starlark/java/annot",
        "//src/main/java/net/starlark/java/eval",
        "//src/main/protobuf:failure_details_java_proto",
        "//third_party:auto_value",
        "//third_party:guava",
        "//third_party:jsr305",
    ],
)

java_library(
    name = "build_options",
    srcs = [
        "BuildOptions.java",
        "BuildOptionsView.java",
    ],
    deps = [
        ":fragment_options",
        ":scope",
        "//src/main/java/com/google/devtools/build/lib/cmdline",
        "//src/main/java/com/google/devtools/build/lib/skyframe/serialization:visible-for-serialization",
        "//src/main/java/com/google/devtools/build/lib/skyframe/serialization/autocodec",
        "//src/main/java/com/google/devtools/build/lib/skyframe/serialization/autocodec:serialization-constant",
        "//src/main/java/com/google/devtools/build/lib/util",
        "//src/main/java/com/google/devtools/common/options",
        "//third_party:error_prone_annotations",
        "//third_party:guava",
        "//third_party:jsr305",
        "@com_google_protobuf//:protobuf_java",
    ],
)

java_library(
    name = "build_options_cache",
    srcs = ["BuildOptionsCache.java"],
    deps = [
        ":build_options",
        "//src/main/java/com/google/devtools/build/lib/events",
        "//third_party:auto_value",
        "//third_party:caffeine",
        "//third_party:guava",
        "//third_party:jsr305",
    ],
)

java_library(
    name = "common_options",
    srcs = ["CommonOptions.java"],
    deps = [
        ":build_options",
        ":core_options",
        "//src/main/java/com/google/devtools/common/options",
    ],
)

java_library(
    name = "compilation_mode",
    srcs = ["CompilationMode.java"],
    deps = [
        "//src/main/java/com/google/devtools/common/options",
        "//src/main/java/net/starlark/java/eval",
    ],
)

java_library(
    name = "config_aware_aspect_builder",
    srcs = ["ConfigAwareAspectBuilder.java"],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/packages",
    ],
)

java_library(
    name = "config_matching_provider",
    srcs = ["ConfigMatchingProvider.java"],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/analysis:transitive_info_provider",
        "//src/main/java/com/google/devtools/build/lib/cmdline",
        "//src/main/java/com/google/devtools/build/lib/concurrent",
        "//src/main/java/com/google/devtools/build/lib/concurrent:thread_safety",
        "//src/main/java/com/google/devtools/build/lib/skyframe/serialization/autocodec",
        "//third_party:auto_value",
        "//third_party:guava",
    ],
)

java_library(
    name = "config_conditions",
    srcs = ["ConfigConditions.java"],
    deps = [
        ":config_matching_provider",
        "//src/main/java/com/google/devtools/build/lib/analysis:configured_target",
        "//src/main/java/com/google/devtools/build/lib/analysis/platform",
        "//src/main/java/com/google/devtools/build/lib/cmdline",
        "//src/main/java/com/google/devtools/build/lib/skyframe:configured_target_and_data",
        "//third_party:auto_value",
        "//third_party:guava",
    ],
)

java_library(
    name = "configuration_transition_event",
    srcs = ["ConfigurationTransitionEvent.java"],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/events",
        "//third_party:auto_value",
    ],
)

java_library(
    name = "additional_configuration_change_event",
    srcs = ["AdditionalConfigurationChangeEvent.java"],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/events",
        "//third_party:guava",
    ],
)

java_library(
    name = "configuration_value_event",
    srcs = ["ConfigurationValueEvent.java"],
    deps = [
        ":build_configuration",
        "//src/main/java/com/google/devtools/build/lib/events",
        "//third_party:auto_value",
    ],
)

java_library(
    name = "core_option_converters",
    srcs = ["CoreOptionConverters.java"],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/cmdline",
        "//src/main/java/com/google/devtools/build/lib/packages",
        "//src/main/java/com/google/devtools/common/options",
        "//src/main/java/net/starlark/java/eval",
        "//third_party:guava",
        "//third_party:jsr305",
    ],
)

java_library(
    name = "core_options",
    srcs = ["CoreOptions.java"],
    deps = [
        ":auto_cpu_converter",
        ":compilation_mode",
        ":core_option_converters",
        ":execution_info_modifier",
        ":fragment_options",
        ":run_under",
        ":run_under_converter",
        "//src/main/java/com/google/devtools/build/lib/cmdline",
        "//src/main/java/com/google/devtools/build/lib/util",
        "//src/main/java/com/google/devtools/common/options",
        "//third_party:guava",
    ],
)

java_library(
    name = "execution_info_modifier",
    srcs = ["ExecutionInfoModifier.java"],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/util:string_encoding",
        "//src/main/java/com/google/devtools/common/options",
        "//third_party:auto_value",
        "//third_party:guava",
    ],
)

java_library(
    name = "starlark_exec_transition_loader",
    srcs = ["StarlarkExecTransitionLoader.java"],
    deps = [
        ":build_options",
        ":common_options",
        ":core_options",
        ":starlark_defined_config_transition",
        "//src/main/java/com/google/devtools/build/lib/analysis:analysis_cluster",
        "//src/main/java/com/google/devtools/build/lib/cmdline",
        "//src/main/java/com/google/devtools/build/lib/skyframe:bzl_load_failed_exception",
        "//src/main/java/com/google/devtools/build/lib/skyframe:bzl_load_value",
        "//src/main/java/com/google/devtools/build/lib/skyframe:starlark_builtins_value",
        "//third_party:auto_value",
        "//third_party:guava",
        "//third_party:jsr305",
    ],
)

java_library(
    name = "execution_transition_factory",
    srcs = ["ExecutionTransitionFactory.java"],
    deps = [
        ":build_options",
        ":core_options",
        ":fragment_options",
        "//src/main/java/com/google/devtools/build/lib/analysis:platform_options",
        "//src/main/java/com/google/devtools/build/lib/analysis:starlark/function_transition_util",
        "//src/main/java/com/google/devtools/build/lib/analysis/config/transitions:configuration_transition",
        "//src/main/java/com/google/devtools/build/lib/analysis/config/transitions:no_config_transition",
        "//src/main/java/com/google/devtools/build/lib/analysis/config/transitions:patch_transition",
        "//src/main/java/com/google/devtools/build/lib/analysis/config/transitions:transition_factory",
        "//src/main/java/com/google/devtools/build/lib/cmdline",
        "//src/main/java/com/google/devtools/build/lib/events",
        "//src/main/java/com/google/devtools/build/lib/packages",
        "//src/main/java/com/google/devtools/build/lib/packages:declared_exec_group",
        "//src/main/java/com/google/devtools/build/lib/rules/config:feature_flag_value",
        "//src/main/java/com/google/devtools/build/lib/starlarkbuildapi/config",
        "//src/main/java/com/google/devtools/build/lib/util:pair",
        "//third_party:caffeine",
        "//third_party:guava",
        "//third_party:jsr305",
    ],
)

java_library(
    name = "feature_set",
    srcs = ["FeatureSet.java"],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/skyframe/serialization/autocodec",
        "//third_party:auto_value",
        "//third_party:guava",
    ],
)

java_library(
    name = "fragment",
    srcs = ["Fragment.java"],
    deps = [
        ":build_options",
        ":fragment_options",
        "//src/main/java/com/google/devtools/build/lib/concurrent:thread_safety",
        "//src/main/java/com/google/devtools/build/lib/events",
        "//src/main/java/net/starlark/java/eval",
        "//third_party:error_prone_annotations",
        "//third_party:guava",
        "//third_party:jsr305",
    ],
)

java_library(
    name = "fragment_factory",
    srcs = ["FragmentFactory.java"],
    deps = [
        ":build_options",
        ":core_options",
        ":fragment",
        ":fragment_options",
        ":invalid_configuration_exception",
        "//third_party:auto_value",
        "//third_party:caffeine",
        "//third_party:error_prone_annotations",
        "//third_party:guava",
        "//third_party:jsr305",
    ],
)

java_library(
    name = "fragment_registry",
    srcs = ["FragmentRegistry.java"],
    deps = [
        ":build_options",
        ":fragment",
        ":fragment_class_set",
        ":fragment_options",
        "//third_party:guava",
    ],
)

java_library(
    name = "fragment_options",
    srcs = [
        "FragmentOptions.java",
        "RequiresOptions.java",
    ],
    deps = [
        "//src/main/java/com/google/devtools/common/options",
        "//third_party:guava",
        "//third_party:jsr305",
    ],
)

java_library(
    name = "invalid_configuration_exception",
    srcs = ["InvalidConfigurationException.java"],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/skyframe:detailed_exceptions",
        "//src/main/java/com/google/devtools/build/lib/util:detailed_exit_code",
        "//src/main/protobuf:failure_details_java_proto",
        "//third_party:guava",
        "//third_party:jsr305",
    ],
)

java_library(
    name = "options_diff",
    srcs = ["OptionsDiff.java"],
    deps = [
        ":build_options",
        ":fragment_options",
        "//src/main/java/com/google/devtools/build/lib/cmdline",
        "//src/main/java/com/google/devtools/build/lib/util",
        "//src/main/java/com/google/devtools/common/options",
        "//third_party:guava",
    ],
)

java_library(
    name = "per_label_options",
    srcs = ["PerLabelOptions.java"],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/actions:artifacts",
        "//src/main/java/com/google/devtools/build/lib/cmdline",
        "//src/main/java/com/google/devtools/build/lib/util",
        "//src/main/java/com/google/devtools/common/options",
        "//third_party:guava",
    ],
)

java_library(
    name = "run_under",
    srcs = ["RunUnder.java"],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/cmdline",
        "//src/main/java/com/google/devtools/build/lib/skyframe/serialization/autocodec",
        "//third_party:guava",
        "//third_party:jsr305",
    ],
)

java_library(
    name = "run_under_converter",
    srcs = ["RunUnderConverter.java"],
    deps = [
        ":core_option_converters",
        ":run_under",
        "//src/main/java/com/google/devtools/build/lib/cmdline",
        "//src/main/java/com/google/devtools/build/lib/shell",
        "//src/main/java/com/google/devtools/common/options",
        "//third_party:guava",
    ],
)

java_library(
    name = "starlark_defined_config_transition",
    srcs = ["StarlarkDefinedConfigTransition.java"],
    deps = [
        ":core_options",
        ":optioninfo",
        "//src/main/java/com/google/devtools/build/lib/analysis/config/transitions:configuration_transition",
        "//src/main/java/com/google/devtools/build/lib/analysis/config/transitions:patch_transition",
        "//src/main/java/com/google/devtools/build/lib/cmdline",
        "//src/main/java/com/google/devtools/build/lib/events",
        "//src/main/java/com/google/devtools/build/lib/packages",
        "//src/main/java/com/google/devtools/build/lib/starlarkbuildapi/config:configuration_transition_api",
        "//src/main/java/com/google/devtools/build/lib/util",
        "//src/main/java/com/google/devtools/build/lib/util:hash_codes",
        "//src/main/java/com/google/devtools/build/lib/vfs:pathfragment",
        "//src/main/java/com/google/devtools/common/options",
        "//src/main/java/net/starlark/java/eval",
        "//src/main/java/net/starlark/java/syntax",
        "//third_party:caffeine",
        "//third_party:error_prone_annotations",
        "//third_party:guava",
        "//third_party:jsr305",
    ],
)

java_library(
    name = "toolchain_type_requirement",
    srcs = ["ToolchainTypeRequirement.java"],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/cmdline",
        "//src/main/java/com/google/devtools/build/lib/skyframe/serialization/autocodec",
        "//src/main/java/com/google/devtools/build/lib/starlarkbuildapi/config:starlark_toolchain_type_requirement",
        "//third_party:auto_value",
        "//third_party:guava",
    ],
)

java_library(
    name = "build_option_details",
    srcs = ["BuildOptionDetails.java"],
    deps = [
        ":fragment_options",
        "//src/main/java/com/google/devtools/build/lib/cmdline",
        "//src/main/java/com/google/devtools/common/options",
        "//third_party:guava",
        "//third_party:jsr305",
    ],
)

java_library(
    name = "optioninfo",
    srcs = ["OptionInfo.java"],
    deps = [
        ":build_options",
        ":fragment_options",
        "//src/main/java/com/google/devtools/common/options",
        "//third_party:guava",
    ],
)

java_library(
    name = "starlark_transition_cache",
    srcs = ["StarlarkTransitionCache.java"],
    deps = [
        ":build_options",
        "//src/main/java/com/google/devtools/build/lib/analysis:starlark/starlark_build_settings_details_value",
        "//src/main/java/com/google/devtools/build/lib/analysis:starlark/starlark_transition",
        "//src/main/java/com/google/devtools/build/lib/analysis/config/transitions:configuration_transition",
        "//src/main/java/com/google/devtools/build/lib/events",
        "//third_party:caffeine",
        "//third_party:jsr305",
    ],
)

java_library(
    name = "scope",
    srcs = ["Scope.java"],
    deps = [
        "//third_party:guava",
        "//third_party:jsr305",
    ],
)

java_library(
    name = "dependency_evaluation_exception",
    srcs = ["DependencyEvaluationException.java"],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/analysis:inconsistent_aspect_order_exception",
        "//src/main/java/com/google/devtools/build/lib/skyframe:configured_value_creation_exception",
        "//src/main/java/com/google/devtools/build/lib/util:detailed_exit_code",
        "//src/main/java/net/starlark/java/eval",
        "//src/main/java/net/starlark/java/syntax",
        "//third_party:jsr305",
    ],
)

java_library(
    name = "symlink_definition",
    srcs = ["SymlinkDefinition.java"],
    deps = [
        ":build_configuration",
        ":build_options",
        "//src/main/java/com/google/devtools/build/lib:build-request-options",
        "//src/main/java/com/google/devtools/build/lib/cmdline",
        "//src/main/java/com/google/devtools/build/lib/vfs",
        "//third_party:guava",
    ],
)
