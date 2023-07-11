load("@rules_java//java:defs.bzl", "java_test")

# Description:
#   Tests for the Java 8 desugaring tool for Android.
package(
    default_testonly = 1,
    default_visibility = ["//src/test/java/com/google/devtools/build/android/desugar:__subpackages__"],
)

licenses(["notice"])  # Apache 2.0

java_test(
    name = "NioBufferRefConverterTest",
    size = "medium",
    srcs = ["NioBufferRefConverterTest.java"],
    data = [
        ":nio_buffer_invocations_src",
        "//src/tools/android/java/com/google/devtools/build/android/desugar/testing/junit:android_jar_for_testing",
        "//src/tools/android/java/com/google/devtools/build/android/desugar/testing/junit:jacoco_agent_jar",
    ],
    jvm_flags = [
        "-Dinput_srcs='$(locations :nio_buffer_invocations_src)'",
        "-Djdk.internal.lambda.dumpProxyClasses=$$(mktemp -d)",
        "-Dandroid_runtime_jar=$(location //src/tools/android/java/com/google/devtools/build/android/desugar/testing/junit:android_jar_for_testing)",
        "-Djacoco_agent_jar=$(location //src/tools/android/java/com/google/devtools/build/android/desugar/testing/junit:jacoco_agent_jar)",
    ],
    test_class = "com.google.devtools.build.android.desugar.covariantreturn.NioBufferRefConverterTest",
    deps = [
        "//src/tools/android/java/com/google/devtools/build/android/desugar/testing/junit:desugar_rule",
        "//third_party:asm",
        "//third_party:asm-tree",
        "//third_party:guava",
        "//third_party:guava-testlib",
        "//third_party:jsr330_inject",
        "//third_party:junit4",
        "//third_party:truth",
    ],
)

filegroup(
    name = "nio_buffer_invocations_src",
    srcs = ["NioBufferInvocations.java"],
)

filegroup(
    name = "srcs",
    testonly = 0,
    srcs = glob(["**"]),
)
