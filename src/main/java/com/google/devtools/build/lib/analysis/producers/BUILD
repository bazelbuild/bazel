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
    name = "producers",
    srcs = glob(["*.java"]),
    deps = [
        "//src/main/java/com/google/devtools/build/lib/analysis:config/build_configuration",
        "//src/main/java/com/google/devtools/build/lib/analysis:config/config_conditions",
        "//src/main/java/com/google/devtools/build/lib/analysis:config/config_matching_provider",
        "//src/main/java/com/google/devtools/build/lib/analysis:configured_target",
        "//src/main/java/com/google/devtools/build/lib/analysis:configured_target_value",
        "//src/main/java/com/google/devtools/build/lib/analysis:constraints/incompatible_target_checker",
        "//src/main/java/com/google/devtools/build/lib/analysis:exec_group_collection",
        "//src/main/java/com/google/devtools/build/lib/analysis:platform_configuration",
        "//src/main/java/com/google/devtools/build/lib/analysis:target_and_configuration",
        "//src/main/java/com/google/devtools/build/lib/analysis:toolchain_collection",
        "//src/main/java/com/google/devtools/build/lib/analysis/platform",
        "//src/main/java/com/google/devtools/build/lib/analysis/platform:utils",
        "//src/main/java/com/google/devtools/build/lib/causes",
        "//src/main/java/com/google/devtools/build/lib/cmdline",
        "//src/main/java/com/google/devtools/build/lib/collect/nestedset",
        "//src/main/java/com/google/devtools/build/lib/events",
        "//src/main/java/com/google/devtools/build/lib/packages",
        "//src/main/java/com/google/devtools/build/lib/packages:configured_attribute_mapper",
        "//src/main/java/com/google/devtools/build/lib/packages:exec_group",
        "//src/main/java/com/google/devtools/build/lib/skyframe:configured_target_and_data",
        "//src/main/java/com/google/devtools/build/lib/skyframe:configured_target_key",
        "//src/main/java/com/google/devtools/build/lib/skyframe:configured_value_creation_exception",
        "//src/main/java/com/google/devtools/build/lib/skyframe:package_value",
        "//src/main/java/com/google/devtools/build/lib/skyframe:rule_configured_target_value",
        "//src/main/java/com/google/devtools/build/lib/skyframe/toolchains:no_matching_platform_exception",
        "//src/main/java/com/google/devtools/build/lib/skyframe/toolchains:platform_lookup_util",
        "//src/main/java/com/google/devtools/build/lib/skyframe/toolchains:toolchain_context_key",
        "//src/main/java/com/google/devtools/build/lib/skyframe/toolchains:toolchain_exception",
        "//src/main/java/com/google/devtools/build/lib/skyframe/toolchains:unloaded_toolchain_context",
        "//src/main/java/com/google/devtools/build/lib/util:detailed_exit_code",
        "//src/main/java/com/google/devtools/build/skyframe",
        "//src/main/java/com/google/devtools/build/skyframe:skyframe-objects",
        "//third_party:auto_value",
        "//third_party:guava",
        "//third_party:jsr305",
    ],
)
