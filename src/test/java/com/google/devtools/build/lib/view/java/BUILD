load("@rules_java//java:defs.bzl", "java_test")

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
    name = "JavaConfigurationTest",
    srcs = ["JavaConfigurationTest.java"],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/analysis:analysis_cluster",
        "//src/main/java/com/google/devtools/build/lib/analysis/config:build_configuration",
        "//src/main/java/com/google/devtools/build/lib/cmdline",
        "//src/main/java/com/google/devtools/build/lib/rules/cpp",
        "//src/main/java/com/google/devtools/build/lib/rules/java:java-compilation",
        "//src/test/java/com/google/devtools/build/lib/analysis/util",
        "//src/test/java/com/google/devtools/build/lib/testutil:TestConstants",
        "//third_party:guava",
        "//third_party:junit4",
        "//third_party:truth",
    ],
)

java_test(
    name = "JavaCompileOnlyTest",
    srcs = ["JavaCompileOnlyTest.java"],
    runtime_deps = ["//src/test/java/com/google/devtools/build/lib/analysis/util"],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/analysis:configured_target",
        "//src/test/java/com/google/devtools/build/lib/analysis/util",
        "//third_party:junit4",
        "//third_party:truth",
    ],
)

java_test(
    name = "JavaPluginConfiguredTargetTest",
    srcs = ["JavaPluginConfiguredTargetTest.java"],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/actions:artifacts",
        "//src/main/java/com/google/devtools/build/lib/analysis:analysis_cluster",
        "//src/main/java/com/google/devtools/build/lib/analysis:configured_target",
        "//src/main/java/com/google/devtools/build/lib/cmdline",
        "//src/main/java/com/google/devtools/build/lib/collect/nestedset",
        "//src/main/java/com/google/devtools/build/lib/packages",
        "//src/main/java/com/google/devtools/build/lib/rules/java:java-compilation",
        "//src/main/java/com/google/devtools/build/lib/skyframe:bzl_load_value",
        "//src/test/java/com/google/devtools/build/lib/actions/util",
        "//src/test/java/com/google/devtools/build/lib/analysis/util",
        "//src/test/java/com/google/devtools/build/lib/rules/java:java_compile_action_test_helper",
        "//third_party:junit4",
        "//third_party:truth",
    ],
)
