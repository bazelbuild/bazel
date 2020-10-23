load("@rules_java//java:defs.bzl", "java_library", "java_test")

package(
    default_testonly = 1,
    default_visibility = ["//src/test/java/com/google/devtools/build/android/desugar:__subpackages__"],
)

licenses(["notice"])  # Apache 2.0

java_test(
    name = "IndyStringConcatDesugaringlTest",
    size = "medium",
    srcs = ["IndyStringConcatDesugaringlTest.java"],
    data = [
        ":string_concat_cases_srcs",
        "//src/tools/android/java/com/google/devtools/build/android/desugar/testing/junit:android_jar_for_testing",
        "//src/tools/android/java/com/google/devtools/build/android/desugar/testing/junit:jacoco_agent_jar",
    ],
    jvm_flags = [
        "-Dinput_srcs=$(locations :string_concat_cases_srcs)",
        # Required by Desugar#verifyLambdaDumpDirectoryRegistered
        "-Djdk.internal.lambda.dumpProxyClasses=$$(mktemp -d)",
        "-Dandroid_runtime_jar=$(location //src/tools/android/java/com/google/devtools/build/android/desugar/testing/junit:android_jar_for_testing)",
        "-Djacoco_agent_jar=$(location //src/tools/android/java/com/google/devtools/build/android/desugar/testing/junit:jacoco_agent_jar)",
        "-Djava.lang.invoke.stringConcat=MH_INLINE_SIZED_EXACT",
    ],
    test_class = "com.google.devtools.build.android.desugar.stringconcat.IndyStringConcatDesugaringlTest",
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

java_library(
    name = "string_concat_cases",
    srcs = [":string_concat_cases_srcs"],
    javacopts = select({
        "//src/tools/android/java/com/google/devtools/build/android/desugar/testing/junit:jdk11-config": [
            "-source 11",
            "-target 11",
            "-XDstringConcat=indyWithConstants",  # Enable OpenJDK version of String concat.
        ],
        "//conditions:default": [],
    }),
)

filegroup(
    name = "string_concat_cases_srcs",
    srcs = ["StringConcatTestCases.java"],
)

filegroup(
    name = "srcs",
    testonly = 0,
    srcs = glob(["*"]),
)
