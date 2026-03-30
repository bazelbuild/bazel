load("@rules_java//java:defs.bzl", "java_library")

package(
    default_applicable_licenses = ["//:license"],
    default_visibility = ["//src:__subpackages__"],
)

filegroup(
    name = "srcs",
    srcs = glob(["**"]) + [
    ],
    visibility = ["//src:__subpackages__"],
)

java_library(
    name = "baseline_options_value",
    srcs = [
        "BaselineOptionsValue.java",
    ],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/analysis/config:build_options",
        "//src/main/java/com/google/devtools/build/lib/cmdline",
        "//src/main/java/com/google/devtools/build/lib/concurrent:thread_safety",
        "//src/main/java/com/google/devtools/build/lib/skyframe:sky_functions",
        "//src/main/java/com/google/devtools/build/lib/skyframe/serialization/autocodec",
        "//src/main/java/com/google/devtools/build/skyframe:skyframe-objects",
        "//third_party:auto_value",
        "//third_party:error_prone_annotations",
        "//third_party:jsr305",
    ],
)

java_library(
    name = "composing_transition_factory",
    srcs = ["ComposingTransitionFactory.java"],
    deps = [
        ":configuration_transition",
        ":no_transition",
        ":transition_factory",
        "//src/main/java/com/google/devtools/build/lib/analysis:required_config_fragments_provider",
        "//src/main/java/com/google/devtools/build/lib/analysis/config:build_option_details",
        "//src/main/java/com/google/devtools/build/lib/analysis/config:build_options",
        "//src/main/java/com/google/devtools/build/lib/events",
        "//third_party:auto_value",
        "//third_party:guava",
    ],
)

java_library(
    name = "configuration_transition",
    srcs = [
        "ConfigurationTransition.java",
        "TransitionUtil.java",
    ],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/analysis:required_config_fragments_provider",
        "//src/main/java/com/google/devtools/build/lib/analysis/config:build_option_details",
        "//src/main/java/com/google/devtools/build/lib/analysis/config:build_options",
        "//src/main/java/com/google/devtools/build/lib/analysis/config:fragment_options",
        "//src/main/java/com/google/devtools/build/lib/events",
        "//third_party:guava",
    ],
)

java_library(
    name = "no_transition",
    srcs = ["NoTransition.java"],
    deps = [
        ":configuration_transition",
        ":patch_transition",
        ":transition_factory",
        "//src/main/java/com/google/devtools/build/lib/analysis/config:build_options",
        "//src/main/java/com/google/devtools/build/lib/events",
        "//src/main/java/com/google/devtools/build/lib/skyframe/serialization/autocodec:serialization-constant",
        "//src/main/java/com/google/devtools/build/lib/starlarkbuildapi/config:configuration_transition_api",
        "//third_party:auto_value",
    ],
)

java_library(
    name = "no_config_transition",
    srcs = ["NoConfigTransition.java"],
    deps = [
        ":patch_transition",
        ":transition_factory",
        "//src/main/java/com/google/devtools/build/lib/analysis/config:build_options",
        "//src/main/java/com/google/devtools/build/lib/analysis/config:common_options",
        "//src/main/java/com/google/devtools/build/lib/analysis/config:core_options",
        "//src/main/java/com/google/devtools/build/lib/analysis/config:fragment_options",
        "//src/main/java/com/google/devtools/build/lib/events",
        "//src/main/java/com/google/devtools/build/lib/skyframe/serialization/autocodec:serialization-constant",
        "//src/main/java/com/google/devtools/build/lib/starlarkbuildapi/config:configuration_transition_api",
        "//third_party:guava",
    ],
)

java_library(
    name = "patch_transition",
    srcs = ["PatchTransition.java"],
    deps = [
        ":configuration_transition",
        "//src/main/java/com/google/devtools/build/lib/analysis/config:build_options",
        "//src/main/java/com/google/devtools/build/lib/events",
    ],
)

java_library(
    name = "split_transition",
    srcs = ["SplitTransition.java"],
    deps = [
        ":configuration_transition",
        "//src/main/java/com/google/devtools/build/lib/analysis/config:build_options",
        "//src/main/java/com/google/devtools/build/lib/concurrent:thread_safety",
        "//src/main/java/com/google/devtools/build/lib/events",
        "//third_party:guava",
    ],
)

java_library(
    name = "transition_collector",
    srcs = ["TransitionCollector.java"],
    deps = [
        ":configuration_transition",
        "//src/main/java/com/google/devtools/build/lib/analysis:dependency_kind",
        "//src/main/java/com/google/devtools/build/lib/cmdline",
    ],
)

java_library(
    name = "transition_factory",
    srcs = ["TransitionFactory.java"],
    deps = [":configuration_transition"],
)

java_library(
    name = "starlark_exposed_rule_transition_factory",
    srcs = ["StarlarkExposedRuleTransitionFactory.java"],
    deps = [
        ":transition_factory",
        "//src/main/java/com/google/devtools/build/lib/analysis:rule_definition_environment",
        "//src/main/java/com/google/devtools/build/lib/packages",
    ],
)

java_library(
    name = "comparing_transition",
    srcs = ["ComparingTransition.java"],
    deps = [
        ":configuration_transition",
        ":patch_transition",
        "//src/main/java/com/google/devtools/build/lib/analysis/config:build_options",
        "//src/main/java/com/google/devtools/build/lib/analysis/config:core_options",
        "//src/main/java/com/google/devtools/build/lib/analysis/config:fragment_options",
        "//src/main/java/com/google/devtools/build/lib/events",
        "//src/main/java/com/google/devtools/build/lib/util:pair",
        "//third_party:guava",
    ],
)
