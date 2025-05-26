load("@rules_java//java:defs.bzl", "java_test")

package(
    default_applicable_licenses = ["//:license"],
    default_testonly = 1,
    default_visibility = ["//src:__subpackages__"],
)

filegroup(
    name = "srcs",
    testonly = 0,
    srcs = glob(["*"]) + [
        "//src/test/java/com/google/devtools/build/lib/analysis/config/transitions:srcs",
    ],
    visibility = ["//src:__subpackages__"],
)

java_test(
    name = "BuildConfigurationFunctionTest",
    srcs = ["BuildConfigurationFunctionTest.java"],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/analysis:configured_target",
        "//src/main/java/com/google/devtools/build/lib/analysis/config:build_configuration",
        "//src/main/java/com/google/devtools/build/lib/analysis/config:core_options",
        "//src/main/java/com/google/devtools/build/lib/cmdline",
        "//src/main/java/com/google/devtools/build/lib/packages",
        "//src/main/java/com/google/devtools/build/lib/rules/cpp",
        "//src/main/java/com/google/devtools/build/lib/rules/java:java-compilation",
        "//src/main/java/com/google/devtools/build/lib/skyframe:bzl_load_value",
        "//src/test/java/com/google/devtools/build/lib/analysis/util",
        "//third_party:guava",
        "//third_party:junit4",
        "//third_party:truth",
    ],
)

java_test(
    name = "BuildConfigurationStarlarkTest",
    srcs = ["BuildConfigurationStarlarkTest.java"],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/analysis:configured_target",
        "//src/main/java/com/google/devtools/build/lib/cmdline",
        "//src/main/java/com/google/devtools/build/lib/packages",
        "//src/main/java/com/google/devtools/build/lib/skyframe:bzl_load_value",
        "//src/main/java/net/starlark/java/eval",
        "//src/test/java/com/google/devtools/build/lib/analysis/util",
        "//third_party:junit4",
        "//third_party:truth",
    ],
)

java_test(
    name = "BuildConfigurationValueTest",
    srcs = ["BuildConfigurationValueTest.java"],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/analysis/config:build_configuration",
        "//src/main/java/com/google/devtools/build/lib/analysis/config:build_options",
        "//src/main/java/com/google/devtools/build/lib/analysis/config:compilation_mode",
        "//src/main/java/com/google/devtools/build/lib/analysis/config:core_options",
        "//src/main/java/com/google/devtools/build/lib/cmdline",
        "//src/main/java/com/google/devtools/build/lib/rules/objc",
        "//src/main/java/com/google/devtools/build/lib/skyframe/serialization/testutils",
        "//src/main/java/com/google/devtools/build/lib/vfs",
        "//src/main/java/com/google/devtools/common/options:options_internal",
        "//src/test/java/com/google/devtools/build/lib/analysis/util",
        "//src/test/java/com/google/devtools/build/lib/testutil:TestConstants",
        "//third_party:guava",
        "//third_party:guava-testlib",
        "//third_party:junit4",
        "//third_party:truth",
    ],
)

java_test(
    name = "BuildOptionDetailsTest",
    srcs = ["BuildOptionDetailsTest.java"],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/analysis/config:build_option_details",
        "//src/main/java/com/google/devtools/build/lib/analysis/config:fragment_options",
        "//src/main/java/com/google/devtools/build/lib/cmdline",
        "//src/main/java/com/google/devtools/common/options:options_internal",
        "//third_party:guava",
        "//third_party:junit4",
        "//third_party:truth",
    ],
)

java_test(
    name = "BuildOptionsCacheTest",
    srcs = ["BuildOptionsCacheTest.java"],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/analysis/config:build_options",
        "//src/main/java/com/google/devtools/build/lib/analysis/config:build_options_cache",
        "//src/main/java/com/google/devtools/build/lib/analysis/config:core_options",
        "//src/main/java/com/google/devtools/common/options:options_internal",
        "//third_party:guava",
        "//third_party:guava-testlib",
        "//third_party:junit4",
        "//third_party:truth",
    ],
)

java_test(
    name = "BuildOptionsTest",
    srcs = ["BuildOptionsTest.java"],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/analysis/config:build_options",
        "//src/main/java/com/google/devtools/build/lib/analysis/config:fragment_options",
        "//src/main/java/com/google/devtools/build/lib/cmdline",
        "//src/main/java/com/google/devtools/build/lib/skyframe/serialization",
        "//src/main/java/com/google/devtools/build/lib/skyframe/serialization/testutils",
        "//src/main/java/com/google/devtools/common/options:options_internal",
        "//third_party:guava",
        "//third_party:junit4",
        "//third_party:truth",
        "@com_google_protobuf//:protobuf_java",
        "@maven//:com_google_testparameterinjector_test_parameter_injector",
    ],
)

java_test(
    name = "BuildOptionsViewTest",
    srcs = ["BuildOptionsViewTest.java"],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/analysis/config:build_options",
        "//src/main/java/com/google/devtools/build/lib/analysis/config:core_options",
        "//src/main/java/com/google/devtools/build/lib/analysis/config:fragment_options",
        "//src/main/java/com/google/devtools/build/lib/rules/cpp",
        "//src/main/java/com/google/devtools/common/options:options_internal",
        "//third_party:guava",
        "//third_party:junit4",
        "//third_party:truth",
    ],
)

java_test(
    name = "CoreOptionsTest",
    srcs = ["CoreOptionsTest.java"],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/analysis/config:core_options",
        "//src/test/java/com/google/devtools/build/lib/analysis/util",
        "//third_party:junit4",
    ],
)

java_test(
    name = "ExecutionInfoModifierTest",
    srcs = ["ExecutionInfoModifierTest.java"],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/analysis/config:execution_info_modifier",
        "//src/main/java/com/google/devtools/common/options:options_internal",
        "//third_party:guava",
        "//third_party:guava-testlib",
        "//third_party:junit4",
        "//third_party:truth",
    ],
)

java_test(
    name = "ExecutionTransitionFactoryTest",
    srcs = ["ExecutionTransitionFactoryTest.java"],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/analysis:platform_options",
        "//src/main/java/com/google/devtools/build/lib/analysis/config:build_configuration",
        "//src/main/java/com/google/devtools/build/lib/analysis/config:build_options",
        "//src/main/java/com/google/devtools/build/lib/analysis/config:core_options",
        "//src/main/java/com/google/devtools/build/lib/analysis/config:execution_transition_factory",
        "//src/main/java/com/google/devtools/build/lib/analysis/config:fragment_options",
        "//src/main/java/com/google/devtools/build/lib/analysis/config:invalid_configuration_exception",
        "//src/main/java/com/google/devtools/build/lib/analysis/config:optioninfo",
        "//src/main/java/com/google/devtools/build/lib/analysis/config/transitions:patch_transition",
        "//src/main/java/com/google/devtools/build/lib/cmdline",
        "//src/main/java/com/google/devtools/build/lib/events",
        "//src/main/java/com/google/devtools/build/lib/packages",
        "//src/main/java/com/google/devtools/common/options:options_internal",
        "//src/test/java/com/google/devtools/build/lib/analysis/util",
        "//src/test/java/com/google/devtools/build/lib/testutil",
        "//third_party:guava",
        "//third_party:junit4",
        "//third_party:truth",
        "@maven//:com_google_testparameterinjector_test_parameter_injector",
    ],
)

java_test(
    name = "FeatureSetTest",
    srcs = ["FeatureSetTest.java"],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/analysis/config:feature_set",
        "//third_party:junit4",
        "//third_party:truth",
    ],
)

java_test(
    name = "FragmentRegistryTest",
    srcs = ["FragmentRegistryTest.java"],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/analysis/config:fragment",
        "//src/main/java/com/google/devtools/build/lib/analysis/config:fragment_options",
        "//src/main/java/com/google/devtools/build/lib/analysis/config:fragment_registry",
        "//third_party:guava",
        "//third_party:junit4",
        "//third_party:truth",
    ],
)

java_test(
    name = "OptionsDiffTest",
    srcs = ["OptionsDiffTest.java"],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/analysis/config:build_options",
        "//src/main/java/com/google/devtools/build/lib/analysis/config:core_options",
        "//src/main/java/com/google/devtools/build/lib/analysis/config:fragment_options",
        "//src/main/java/com/google/devtools/build/lib/analysis/config:options_diff",
        "//src/main/java/com/google/devtools/build/lib/cmdline",
        "//src/main/java/com/google/devtools/build/lib/rules/cpp",
        "//third_party:guava",
        "//third_party:junit4",
        "//third_party:truth",
    ],
)

java_test(
    name = "PerLabelOptionsTest",
    srcs = ["PerLabelOptionsTest.java"],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/analysis/config:per_label_options",
        "//src/main/java/com/google/devtools/build/lib/util",
        "//src/main/java/com/google/devtools/common/options:options_internal",
        "//third_party:guava-testlib",
        "//third_party:junit4",
        "//third_party:truth",
    ],
)

java_test(
    name = "RunUnderConverterTest",
    srcs = ["RunUnderConverterTest.java"],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/analysis/config:run_under",
        "//src/main/java/com/google/devtools/build/lib/analysis/config:run_under_converter",
        "//src/main/java/com/google/devtools/build/lib/cmdline",
        "//src/main/java/com/google/devtools/common/options:options_internal",
        "//third_party:guava",
        "//third_party:junit4",
        "//third_party:truth",
    ],
)

java_test(
    name = "StampTest",
    srcs = ["StampTest.java"],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/packages",
        "//src/test/java/com/google/devtools/build/lib/analysis/util",
        "//third_party:junit4",
        "//third_party:truth",
    ],
)
