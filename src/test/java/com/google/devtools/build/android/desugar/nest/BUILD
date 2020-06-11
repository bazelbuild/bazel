load("@rules_java//java:defs.bzl", "java_test")

# Description:
#   Tests for the Java 8 desugaring tool for Android.
package(
    default_testonly = 1,
    default_visibility = ["//src/test/java/com/google/devtools/build/android/desugar:__subpackages__"],
)

licenses(["notice"])  # Apache 2.0

java_test(
    name = "ClassFileFormatTest",
    size = "medium",
    srcs = ["ClassFileFormatTest.java"],
    data = [
        "//src/test/java/com/google/devtools/build/android/desugar/nest/testsrc/simpleunit/classfileformat",
        "//src/tools/android/java/com/google/devtools/build/android/desugar/testing/junit:android_jar_for_testing",
        "//src/tools/android/java/com/google/devtools/build/android/desugar/testing/junit:jacoco_agent_jar",
    ],
    jvm_flags = [
        "-Dinput_srcs='$(locations //src/test/java/com/google/devtools/build/android/desugar/nest/testsrc/simpleunit/classfileformat)'",
        "-Djdk.internal.lambda.dumpProxyClasses=$$(mktemp -d)",
        "-Dandroid_runtime_jar=$(location //src/tools/android/java/com/google/devtools/build/android/desugar/testing/junit:android_jar_for_testing)",
        "-Djacoco_agent_jar=$(location //src/tools/android/java/com/google/devtools/build/android/desugar/testing/junit:jacoco_agent_jar)",
    ],
    test_class = "com.google.devtools.build.android.desugar.nest.ClassFileFormatTest",
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

java_test(
    name = "NestAnalyzerTest",
    size = "medium",
    srcs = ["NestAnalyzerTest.java"],
    data = [
        "//src/test/java/com/google/devtools/build/android/desugar/nest/testsrc/nestanalyzer:analyzed_target",
    ],
    jvm_flags = [
        "-Dinput_srcs=$(location //src/test/java/com/google/devtools/build/android/desugar/nest/testsrc/nestanalyzer:analyzed_target)",
    ],
    test_class = "com.google.devtools.build.android.desugar.nest.NestAnalyzerTest",
    deps = [
        "//src/tools/android/java/com/google/devtools/build/android/desugar/io",
        "//src/tools/android/java/com/google/devtools/build/android/desugar/langmodel",
        "//src/tools/android/java/com/google/devtools/build/android/desugar/nest",
        "//src/tools/android/java/com/google/devtools/build/android/desugar/preanalysis",
        "//src/tools/android/java/com/google/devtools/build/android/desugar/testing/junit:desugar_rule",
        "//third_party:asm",
        "//third_party:guava",
        "//third_party:junit4",
        "//third_party:truth",
    ],
)

java_test(
    name = "NestDigestTest",
    size = "small",
    srcs = ["NestDigestTest.java"],
    test_class = "com.google.devtools.build.android.desugar.nest.NestDigestTest",
    deps = [
        "//src/tools/android/java/com/google/devtools/build/android/desugar/langmodel",
        "//src/tools/android/java/com/google/devtools/build/android/desugar/nest",
        "//third_party:asm",
        "//third_party:asm-tree",
        "//third_party:junit4",
        "//third_party:truth",
    ],
)

java_test(
    name = "NestDesugaringConstructorAccessTest",
    size = "medium",
    srcs = ["NestDesugaringConstructorAccessTest.java"],
    data = [
        "//src/test/java/com/google/devtools/build/android/desugar/nest/testsrc/simpleunit/constructor",
        "//src/tools/android/java/com/google/devtools/build/android/desugar/testing/junit:android_jar_for_testing",
        "//src/tools/android/java/com/google/devtools/build/android/desugar/testing/junit:jacoco_agent_jar",
    ],
    jvm_flags = [
        "-Dinput_srcs='$(locations //src/test/java/com/google/devtools/build/android/desugar/nest/testsrc/simpleunit/constructor)'",
        # Required by Desugar#verifyLambdaDumpDirectoryRegistered
        "-Djdk.internal.lambda.dumpProxyClasses=$$(mktemp -d)",
        "-Dandroid_runtime_jar=$(location //src/tools/android/java/com/google/devtools/build/android/desugar/testing/junit:android_jar_for_testing)",
        "-Djacoco_agent_jar=$(location //src/tools/android/java/com/google/devtools/build/android/desugar/testing/junit:jacoco_agent_jar)",
    ],
    test_class = "com.google.devtools.build.android.desugar.nest.NestDesugaringConstructorAccessTest",
    deps = [
        "//src/tools/android/java/com/google/devtools/build/android/desugar/testing/junit:desugar_rule",
        "//third_party:asm-tree",
        "//third_party:guava",
        "//third_party:jsr330_inject",
        "//third_party:junit4",
        "//third_party:truth",
    ],
)

java_test(
    name = "NestDesugaringFieldAccessTest",
    size = "medium",
    srcs = ["NestDesugaringFieldAccessTest.java"],
    data = [
        "//src/test/java/com/google/devtools/build/android/desugar/nest/testsrc/simpleunit/field",
        "//src/tools/android/java/com/google/devtools/build/android/desugar/testing/junit:android_jar_for_testing",
        "//src/tools/android/java/com/google/devtools/build/android/desugar/testing/junit:jacoco_agent_jar",
    ],
    jvm_flags = [
        "-Dinput_srcs=$(location //src/test/java/com/google/devtools/build/android/desugar/nest/testsrc/simpleunit/field)",
        # Required by Desugar#verifyLambdaDumpDirectoryRegistered
        "-Djdk.internal.lambda.dumpProxyClasses=$$(mktemp -d)",
        "-Dandroid_runtime_jar=$(location //src/tools/android/java/com/google/devtools/build/android/desugar/testing/junit:android_jar_for_testing)",
        "-Djacoco_agent_jar=$(location //src/tools/android/java/com/google/devtools/build/android/desugar/testing/junit:jacoco_agent_jar)",
    ],
    test_class = "com.google.devtools.build.android.desugar.nest.NestDesugaringFieldAccessTest",
    deps = [
        "//src/tools/android/java/com/google/devtools/build/android/desugar/testing/junit:desugar_rule",
        "//third_party:asm",
        "//third_party:asm-tree",
        "//third_party:jsr330_inject",
        "//third_party:junit4",
        "//third_party:truth",
    ],
)

java_test(
    name = "NestDesugaringInterfaceMethodAccessTest",
    size = "medium",
    srcs = ["NestDesugaringInterfaceMethodAccessTest.java"],
    data = [
        "//src/test/java/com/google/devtools/build/android/desugar/nest/testsrc/simpleunit/interfacemethod",
        "//src/tools/android/java/com/google/devtools/build/android/desugar/testing/junit:android_jar_for_testing",
        "//src/tools/android/java/com/google/devtools/build/android/desugar/testing/junit:jacoco_agent_jar",
    ],
    jvm_flags = [
        "-Dinput_srcs=$(location //src/test/java/com/google/devtools/build/android/desugar/nest/testsrc/simpleunit/interfacemethod)",
        # Required by Desugar#verifyLambdaDumpDirectoryRegistered
        "-Djdk.internal.lambda.dumpProxyClasses=$$(mktemp -d)",
        "-Djava.lang.invoke.stringConcat=MH_INLINE_SIZED_EXACT",
        "-Dandroid_runtime_jar=$(location //src/tools/android/java/com/google/devtools/build/android/desugar/testing/junit:android_jar_for_testing)",
        "-Djacoco_agent_jar=$(location //src/tools/android/java/com/google/devtools/build/android/desugar/testing/junit:jacoco_agent_jar)",
    ],
    test_class = "com.google.devtools.build.android.desugar.nest.NestDesugaringInterfaceMethodAccessTest",
    deps = [
        "//src/tools/android/java/com/google/devtools/build/android/desugar/testing/junit:desugar_rule",
        "//third_party:asm-tree",
        "//third_party:guava",
        "//third_party:jsr330_inject",
        "//third_party:junit4",
        "//third_party:truth",
    ],
)

java_test(
    name = "NestDesugaringMethodAccessTest",
    size = "medium",
    srcs = ["NestDesugaringMethodAccessTest.java"],
    data = [
        "//src/test/java/com/google/devtools/build/android/desugar/nest/testsrc/simpleunit/method",
        "//src/tools/android/java/com/google/devtools/build/android/desugar/testing/junit:android_jar_for_testing",
        "//src/tools/android/java/com/google/devtools/build/android/desugar/testing/junit:jacoco_agent_jar",
    ],
    jvm_flags = [
        "-Dinput_srcs='$(locations //src/test/java/com/google/devtools/build/android/desugar/nest/testsrc/simpleunit/method)'",
        # Required by Desugar#verifyLambdaDumpDirectoryRegistered
        "-Djdk.internal.lambda.dumpProxyClasses=$$(mktemp -d)",
        "-Djava.lang.invoke.stringConcat=MH_INLINE_SIZED_EXACT",
        "-Dandroid_runtime_jar=$(location //src/tools/android/java/com/google/devtools/build/android/desugar/testing/junit:android_jar_for_testing)",
        "-Djacoco_agent_jar=$(location //src/tools/android/java/com/google/devtools/build/android/desugar/testing/junit:jacoco_agent_jar)",
    ],
    test_class = "com.google.devtools.build.android.desugar.nest.NestDesugaringMethodAccessTest",
    deps = [
        "//src/tools/android/java/com/google/devtools/build/android/desugar/langmodel",
        "//src/tools/android/java/com/google/devtools/build/android/desugar/testing/junit:desugar_rule",
        "//third_party:asm",
        "//third_party:asm-tree",
        "//third_party:guava",
        "//third_party:jsr330_inject",
        "//third_party:junit4",
        "//third_party:truth",
    ],
)

java_test(
    name = "NestDesugaringComplexCasesTest",
    size = "medium",
    srcs = ["NestDesugaringComplexCasesTest.java"],
    data = [
        "//src/test/java/com/google/devtools/build/android/desugar/nest/testsrc/complexcase",
        "//src/tools/android/java/com/google/devtools/build/android/desugar/testing/junit:android_jar_for_testing",
        "//src/tools/android/java/com/google/devtools/build/android/desugar/testing/junit:jacoco_agent_jar",
    ],
    jvm_flags = [
        "-Dinput_srcs='$(locations //src/test/java/com/google/devtools/build/android/desugar/nest/testsrc/complexcase)'",
        # Required by Desugar#verifyLambdaDumpDirectoryRegistered
        "-Djdk.internal.lambda.dumpProxyClasses=$$(mktemp -d)",
        "-Dandroid_runtime_jar=$(location //src/tools/android/java/com/google/devtools/build/android/desugar/testing/junit:android_jar_for_testing)",
        "-Djacoco_agent_jar=$(location //src/tools/android/java/com/google/devtools/build/android/desugar/testing/junit:jacoco_agent_jar)",
    ],
    test_class = "com.google.devtools.build.android.desugar.nest.NestDesugaringComplexCasesTest",
    deps = [
        "//src/tools/android/java/com/google/devtools/build/android/desugar/testing/junit:desugar_rule",
        "//third_party:asm-tree",
        "//third_party:guava",
        "//third_party:junit4",
        "//third_party:truth",
    ],
)

filegroup(
    name = "srcs",
    testonly = 0,
    srcs = glob(["*"]) + [
        "//src/test/java/com/google/devtools/build/android/desugar/nest/testsrc/complexcase:srcs",
        "//src/test/java/com/google/devtools/build/android/desugar/nest/testsrc/nestanalyzer:srcs",
        "//src/test/java/com/google/devtools/build/android/desugar/nest/testsrc/simpleunit/constructor:srcs",
        "//src/test/java/com/google/devtools/build/android/desugar/nest/testsrc/simpleunit/core/javadesugar/testing:srcs",
        "//src/test/java/com/google/devtools/build/android/desugar/nest/testsrc/simpleunit/classfileformat:srcs",
        "//src/test/java/com/google/devtools/build/android/desugar/nest/testsrc/simpleunit/field:srcs",
        "//src/test/java/com/google/devtools/build/android/desugar/nest/testsrc/simpleunit/interfacemethod:srcs",
        "//src/test/java/com/google/devtools/build/android/desugar/nest/testsrc/simpleunit/method:srcs",
    ],
)

test_suite(
    name = "AllTests",
)
