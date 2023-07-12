load("@rules_java//java:defs.bzl", "java_test")

package(
    default_testonly = 1,
    default_visibility = ["//src/test/java/com/google/devtools/build/android/desugar:__subpackages__"],
)

licenses(["notice"])  # Apache 2.0

java_test(
    name = "LocalTypeAnnotationUseTest",
    size = "medium",
    srcs = ["LocalTypeAnnotationUseTest.java"],
    data = [
        ":annotation_test_src",
        "//src/tools/android/java/com/google/devtools/build/android/desugar/testing/junit:android_jar_for_testing",
        "//src/tools/android/java/com/google/devtools/build/android/desugar/testing/junit:jacoco_agent_jar",
    ],
    jvm_flags = [
        "-Dinput_srcs=$(locations :annotation_test_src)",
        # Required by Desugar#verifyLambdaDumpDirectoryRegistered
        "-Djdk.internal.lambda.dumpProxyClasses=$$(mktemp -d)",
        "-Dandroid_runtime_jar=$(location //src/tools/android/java/com/google/devtools/build/android/desugar/testing/junit:android_jar_for_testing)",
        "-Djacoco_agent_jar=$(location //src/tools/android/java/com/google/devtools/build/android/desugar/testing/junit:jacoco_agent_jar)",
    ],
    test_class = "com.google.devtools.build.android.desugar.typeannotation.LocalTypeAnnotationUseTest",
    deps = [
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

filegroup(
    name = "annotation_test_src",
    srcs = ["AnnotationUser.java"],
)

filegroup(
    name = "srcs",
    testonly = 0,
    srcs = glob(["*"]),
)
