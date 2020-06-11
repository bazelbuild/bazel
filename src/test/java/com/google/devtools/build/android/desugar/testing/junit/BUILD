load("@rules_java//java:defs.bzl", "java_library", "java_test")

# Description:
#   Tests for the Java 8 desugaring tool for Android.
package(
    default_testonly = 1,
    default_visibility = ["//src/test/java/com/google/devtools/build/android/desugar:__subpackages__"],
)

licenses(["notice"])  # Apache 2.0

java_test(
    name = "DesugarRuleTest",
    srcs = ["DesugarRuleTest.java"],
    data = [
        ":desugar_rule_test_target",
        ":test_source",
        "//src/tools/android/java/com/google/devtools/build/android/desugar/testing/junit:android_jar_for_testing",
        "//src/tools/android/java/com/google/devtools/build/android/desugar/testing/junit:jacoco_agent_jar",
    ],
    jvm_flags = [
        "-Dinput_jar=$(location :desugar_rule_test_target)",
        "-Dinput_srcs=$(location :test_source)",
        # Required by Desugar#verifyLambdaDumpDirectoryRegistered
        "-Djdk.internal.lambda.dumpProxyClasses=$$(mktemp -d)",
        "-Dandroid_runtime_jar=$(location //src/tools/android/java/com/google/devtools/build/android/desugar/testing/junit:android_jar_for_testing)",
        "-Djacoco_agent_jar=$(location //src/tools/android/java/com/google/devtools/build/android/desugar/testing/junit:jacoco_agent_jar)",
    ],
    deps = [
        "//src/tools/android/java/com/google/devtools/build/android/desugar/io",
        "//src/tools/android/java/com/google/devtools/build/android/desugar/testing/junit:desugar_rule",
        "//third_party:asm",
        "//third_party:asm-tree",
        "//third_party:guava",
        "//third_party:jsr305",
        "//third_party:jsr330_inject",
        "//third_party:junit4",
        "//third_party:truth",
    ],
)

java_library(
    name = "desugar_rule_test_target",
    testonly = 1,
    srcs = ["DesugarRuleTestTarget.java"],
)

# DO NOT depend on this target, kept for build verification only.
java_library(
    name = "test_source_lib",
    srcs = [":test_source"],
    javacopts = [
        "-source 11",
        "-target 11",
    ],
    neverlink = 1,
    visibility = ["//visibility:private"],
)

filegroup(
    name = "test_source",
    srcs = ["DesugarRuleTestSourceTarget.java"],
)

filegroup(
    name = "srcs",
    testonly = 0,
    srcs = glob(["*"]),
)
