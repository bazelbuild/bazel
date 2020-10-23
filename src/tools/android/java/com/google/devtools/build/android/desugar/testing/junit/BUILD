load("@rules_java//java:defs.bzl", "java_library")

package(
    default_visibility = [
        "//src/test/java/com/google/devtools/build/android/desugar:__subpackages__",
        "//src/tools/android/java/com/google/devtools/build/android/desugar:__subpackages__",
    ],
)

java_library(
    name = "desugar_rule",
    srcs = glob(["*.java"]),
    deps = [
        "//src/tools/android/java/com/google/devtools/build/android/desugar",
        "//src/tools/android/java/com/google/devtools/build/android/desugar/io",
        "//src/tools/android/java/com/google/devtools/build/android/desugar/langmodel",
        "//third_party:asm",
        "//third_party:asm-commons",
        "//third_party:asm-tree",
        "//third_party:auto_value",
        "//third_party:flogger",
        "//third_party:guava",
        "//third_party:jsr305",
        "//third_party:jsr330_inject",
        "//third_party:junit4",
    ],
)

filegroup(
    name = "android_jar_for_testing",
    srcs = select({
        # TODO(ajmichael): Use @bazel_tools//tools/android:android_jar here once it supplies runfiles.
        "//external:has_androidsdk": ["@androidsdk//:platforms/android-24/android.jar"],
        "//conditions:default": ["@bazel_tools//tools/android:error_message.jar"],
    }),
)

genrule(
    name = "jacoco_agent_jar",
    srcs = ["//third_party/java/jacoco:blaze-agent-0.8.3"],
    outs = ["jacoco_agent.jar"],
    cmd = """
        cp $(location //third_party/java/jacoco:blaze-agent-0.8.3) $(OUTS)
    """,
)

config_setting(
    name = "jdk11-config",
    constraint_values = [
        "@rules_java//java/constraints:jdk11",
    ],
)

filegroup(
    name = "srcs",
    srcs = glob(["**"]),
)
