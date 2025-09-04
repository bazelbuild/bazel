load("@rules_java//java:defs.bzl", "java_library", "java_test")

package(
    default_applicable_licenses = ["//:license"],
    default_testonly = 1,
    default_visibility = ["//src:__subpackages__"],
)

filegroup(
    name = "srcs",
    testonly = 0,
    srcs = glob(["*"]),
    visibility = ["//src:__subpackages__"],
)

java_test(
    name = "ConstraintCollectionTest",
    srcs = ["ConstraintCollectionTest.java"],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/analysis/platform",
        "//src/main/java/com/google/devtools/build/lib/cmdline",
        "//src/test/java/com/google/devtools/build/lib/analysis/util",
        "//third_party:guava",
        "//third_party:truth",
    ],
)

java_test(
    name = "ConstraintSettingInfoTest",
    srcs = ["ConstraintSettingInfoTest.java"],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/analysis/platform",
        "//src/main/java/com/google/devtools/build/lib/cmdline",
        "//src/test/java/com/google/devtools/build/lib/analysis/util",
        "//third_party:guava-testlib",
    ],
)

java_test(
    name = "ConstraintValueInfoTest",
    srcs = ["ConstraintValueInfoTest.java"],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/analysis/platform",
        "//src/main/java/com/google/devtools/build/lib/cmdline",
        "//src/test/java/com/google/devtools/build/lib/analysis/util",
        "//third_party:guava-testlib",
    ],
)

java_test(
    name = "DeclaredToolchainInfoTest",
    srcs = ["DeclaredToolchainInfoTest.java"],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/analysis/platform",
        "//src/main/java/com/google/devtools/build/lib/cmdline",
        "//src/test/java/com/google/devtools/build/lib/analysis/util",
        "//third_party:guava",
        "//third_party:guava-testlib",
        "//third_party:truth",
    ],
)

java_test(
    name = "PlatformInfoTest",
    srcs = ["PlatformInfoTest.java"],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/analysis/config:config_matching_provider",
        "//src/main/java/com/google/devtools/build/lib/analysis/platform",
        "//src/main/java/com/google/devtools/build/lib/cmdline",
        "//src/test/java/com/google/devtools/build/lib/analysis/util",
        "//third_party:guava",
        "//third_party:guava-testlib",
        "//third_party:truth",
    ],
)

java_test(
    name = "PlatformPropertiesTest",
    srcs = ["PlatformPropertiesTest.java"],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/analysis/platform",
        "//third_party:guava",
        "//third_party:truth",
    ],
)

java_test(
    name = "PlatformUtilsTest",
    srcs = ["PlatformUtilsTest.java"],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/actions",
        "//src/main/java/com/google/devtools/build/lib/analysis/platform",
        "//src/main/java/com/google/devtools/build/lib/analysis/platform:platform_utils",
        "//src/main/java/com/google/devtools/build/lib/remote/options",
        "//src/main/java/com/google/devtools/common/options",
        "//src/test/java/com/google/devtools/build/lib/exec/util",
        "//third_party:guava",
        "//third_party:truth",
        "@remoteapis//:build_bazel_remote_execution_v2_remote_execution_java_proto",
    ],
)

java_test(
    name = "TemplateVariableInfoTest",
    srcs = ["TemplateVariableInfoTest.java"],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/analysis:analysis_cluster",
        "//src/main/java/com/google/devtools/build/lib/analysis:configured_target",
        "//src/main/java/com/google/devtools/build/lib/analysis:template_variable_info",
        "//src/main/java/com/google/devtools/build/lib/analysis/platform",
        "//src/main/java/com/google/devtools/build/lib/packages:provider",
        "//src/test/java/com/google/devtools/build/lib/analysis/util",
        "//third_party:truth",
    ],
)

java_test(
    name = "ToolchainInfoTest",
    srcs = ["ToolchainInfoTest.java"],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/analysis:configured_target",
        "//src/main/java/com/google/devtools/build/lib/analysis/platform",
        "//src/main/java/com/google/devtools/build/lib/analysis/platform:utils",
        "//src/main/java/com/google/devtools/build/lib/cmdline",
        "//src/main/java/net/starlark/java/syntax",
        "//src/test/java/com/google/devtools/build/lib/analysis/util",
        "//third_party:guava",
        "//third_party:guava-testlib",
        "//third_party:truth",
    ],
)

test_suite(
    name = "PlatformAnalysisTests",
)
