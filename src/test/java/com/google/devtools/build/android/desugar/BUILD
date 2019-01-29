# Description:
#   Tests for the Java 8 desugaring tool for Android.
package(
    default_testonly = 1,
    default_visibility = ["//src/test/java/com/google/devtools/build/android/desugar:__subpackages__"],
)

licenses(["notice"])  # Apache 2.0

filegroup(
    name = "srcs",
    testonly = 0,
    srcs = glob(["**"]) + [
        "//src/test/java/com/google/devtools/build/android/desugar/classes_for_testing_type_inference:srcs",
        "//src/test/java/com/google/devtools/build/android/desugar/dependencies:srcs",
        "//src/test/java/com/google/devtools/build/android/desugar/io:srcs",
        "//src/test/java/com/google/devtools/build/android/desugar/runtime:srcs",
        "//src/test/java/com/google/devtools/build/android/desugar/scan:srcs",
    ],
    visibility = ["//src/test/java/com/google/devtools/build/android:__pkg__"],
)

filegroup(
    name = "android_jar_for_testing",
    srcs = select({
        # TODO(ajmichael): Use @bazel_tools//tools/android:android_jar here once it supplies runfiles.
        "//external:has_androidsdk": ["@androidsdk//:platforms/android-24/android.jar"],
        "//conditions:default": ["@bazel_tools//tools/android:error_message.jar"],
    }),
)

java_test(
    name = "DesugarFunctionalTest",
    size = "small",
    srcs = [
        "DesugarFunctionalTest.java",
    ],
    tags = ["no_windows"],
    deps = [
        ":testdata_desugared",  # Make tests run against desugared library
        "//third_party:asm",
        "//third_party:guava",
        "//third_party:jsr305",
        "//third_party:junit4",
        "//third_party:truth",
    ],
)

java_test(
    name = "DesugarIdempotencyFunctionalTest",
    size = "small",
    srcs = [
        "DesugarFunctionalTest.java",
    ],
    tags = ["no_windows"],
    test_class = "com.google.devtools.build.android.desugar.DesugarFunctionalTest",
    deps = [
        ":testdata_desugared_twice",  # Make tests run against twice-desugared library
        "//third_party:asm",
        "//third_party:guava",
        "//third_party:jsr305",
        "//third_party:junit4",
        "//third_party:truth",
    ],
)

# Test for b/62060793. Verifies constant lambda arguments that were pushed using *CONST_0
# instructions.
java_test(
    name = "DesugarFunctionalTestForConstantArgumentsInLambdas",
    size = "small",
    srcs = ["DesugarLambdaTest.java"],
    tags = ["no_windows"],
    test_class = "com.google.devtools.build.android.desugar.DesugarLambdaTest",
    deps = [
        ":desugar_lambda_with_constant_arguments_lib",
        "//third_party:guava",
        "//third_party:junit4",
        "//third_party:truth",
    ],
)

# Test for b/62456849. If a synthetic method is named "lambda$XXX", but not used in invokedynamic,
# then Desugar should keep it in the class, rather than renaming it.
java_test(
    name = "DesugarFunctionalTestForSyntheticMethodsWithLambdaNames",
    size = "small",
    srcs = ["DesugarFunctionalTest.java"],
    tags = ["no_windows"],
    test_class = "com.google.devtools.build.android.desugar.DesugarFunctionalTest",
    deps = [
        ":desugar_testdata_with_synthetic_methods_with_lambda_names",
        ":separate",
        "//third_party:asm",
        "//third_party:guava",
        "//third_party:jsr305",
        "//third_party:junit4",
        "//third_party:truth",
    ],
)

java_test(
    name = "DesugarLongCompareTest",
    size = "small",
    srcs = [
        "DesugarLongCompareTest.java",
    ],
    tags = ["no_windows"],
    deps = [
        ":testdata_desugared",  # Make tests run against desugared library
        "//third_party:asm",
        "//third_party:guava",
        "//third_party:jsr305",
        "//third_party:junit4",
        "//third_party:truth",
    ],
)

java_test(
    name = "DesugarObjectsRequireNonNullTest",
    size = "small",
    srcs = [
        "DesugarObjectsRequireNonNullTest.java",
    ],
    tags = ["no_windows"],
    deps = [
        ":testdata_desugared",  # Make tests run against desugared library
        "//third_party:asm",
        "//third_party:guava",
        "//third_party:jsr305",
        "//third_party:junit4",
        "//third_party:truth",
    ],
)

java_test(
    name = "DesugarObjectsRequireNonNullTestForAndroidLintMode",
    size = "small",
    srcs = ["DesugarObjectsRequireNonNullTest.java"],
    tags = ["no_windows"],
    test_class = "com.google.devtools.build.android.desugar.DesugarObjectsRequireNonNullTest",
    deps = [
        ":desugar_testdata_by_disabling_lambda_desugaring",  # Make tests run against desugared library
        "//third_party:asm",
        "//third_party:guava",
        "//third_party:jsr305",
        "//third_party:junit4",
        "//third_party:truth",
    ],
)

java_test(
    name = "DesugarJava8FunctionalTest",
    size = "small",
    srcs = [
        "DesugarFunctionalTest.java",
        "DesugarJava8FunctionalTest.java",
    ],
    tags = ["no_windows"],
    deps = [
        ":testdata_desugared_java8",  # Make tests run against desugared library
        "//src/test/java/com/google/devtools/build/lib:testutil",
        "//third_party:guava",
        "//third_party:jsr305",
        "//third_party:junit4",
        "//third_party:truth",
    ],
)

java_test(
    name = "DesugarJava8LikeAndroidStudioFunctionalTest",
    size = "small",
    srcs = [
        "DesugarFunctionalTest.java",
        "DesugarJava8FunctionalTest.java",
        "DesugarJava8LikeAndroidStudioFunctionalTest.java",
    ],
    tags = ["no_windows"],
    deps = [
        ":testdata_desugared_java8_like_in_android_studio",  # Make tests run against desugared library
        "//src/test/java/com/google/devtools/build/lib:testutil",
        "//third_party:guava",
        "//third_party:jsr305",
        "//third_party:junit4",
        "//third_party:truth",
    ],
)

java_test(
    name = "DesugarDefaultMethodsFunctionalTest",
    size = "small",
    srcs = [
        "DesugarDefaultMethodsFunctionalTest.java",
        "DesugarFunctionalTest.java",
        "DesugarJava8FunctionalTest.java",
    ],
    tags = ["no_windows"],
    deps = [
        ":testdata_desugared_default_methods",  # Make tests run against desugared library
        "//src/test/java/com/google/devtools/build/lib:testutil",
        "//third_party:guava",
        "//third_party:jsr305",
        "//third_party:junit4",
        "//third_party:truth",
    ],
)

java_test(
    name = "DesugarDefaultMethodsIdempotencyFunctionalTest",
    size = "small",
    srcs = [
        "DesugarDefaultMethodsFunctionalTest.java",
        "DesugarFunctionalTest.java",
        "DesugarJava8FunctionalTest.java",
    ],
    tags = ["no_windows"],
    test_class = "com.google.devtools.build.android.desugar.DesugarDefaultMethodsFunctionalTest",
    deps = [
        ":testdata_desugared_default_methods_twice",  # Make tests run against 2x desugared library
        "//src/test/java/com/google/devtools/build/lib:testutil",
        "//third_party:guava",
        "//third_party:jsr305",
        "//third_party:junit4",
        "//third_party:truth",
    ],
)

java_test(
    name = "DesugarLikeAndroidStudioFunctionalTest",
    size = "small",
    srcs = [
        "DesugarFunctionalTest.java",
    ],
    tags = ["no_windows"],
    test_class = "com.google.devtools.build.android.desugar.DesugarFunctionalTest",
    deps = [
        ":testdata_desugared_like_in_android_studio",  # Make tests run against desugared library
        "//third_party:guava",
        "//third_party:jsr305",
        "//third_party:junit4",
        "//third_party:truth",
    ],
)

java_test(
    name = "DesugarFunctionalTestWithMultipleInputs",
    size = "small",
    srcs = [
        "DesugarFunctionalTest.java",
    ],
    tags = ["no_windows"],
    test_class = "com.google.devtools.build.android.desugar.DesugarFunctionalTest",
    deps = [
        ":testdata_desugared_with_multiple_inputs",  # Make tests run against desugared library
        "//third_party:guava",
        "//third_party:jsr305",
        "//third_party:junit4",
        "//third_party:truth",
    ],
)

java_test(
    name = "DesugarFunctionalTestFromDirectoryToJar",
    size = "small",
    srcs = [
        "DesugarFunctionalTest.java",
    ],
    tags = ["no_windows"],
    test_class = "com.google.devtools.build.android.desugar.DesugarFunctionalTest",
    deps = [
        ":testdata_desugared_from_directory_to_jar",  # Make tests run against desugared library
        "//third_party:guava",
        "//third_party:jsr305",
        "//third_party:junit4",
        "//third_party:truth",
    ],
)

java_test(
    name = "DesugarFunctionalTestFromDirectoryToDirectory",
    size = "small",
    srcs = [
        "DesugarFunctionalTest.java",
    ],
    tags = ["no_windows"],
    test_class = "com.google.devtools.build.android.desugar.DesugarFunctionalTest",
    deps = [
        ":testdata_desugared_from_directory_to_directory",  # Make tests run against desugared library
        "//third_party:guava",
        "//third_party:jsr305",
        "//third_party:junit4",
        "//third_party:truth",
    ],
)

java_test(
    name = "DesugarFunctionalTestWithClasspathDirectory",
    size = "small",
    srcs = [
        "DesugarFunctionalTest.java",
    ],
    tags = ["no_windows"],
    test_class = "com.google.devtools.build.android.desugar.DesugarFunctionalTest",
    deps = [
        ":testdata_desugared_with_classpath_directory",  # Make tests run against desugared library
        "//third_party:guava",
        "//third_party:jsr305",
        "//third_party:junit4",
        "//third_party:truth",
    ],
)

java_test(
    name = "DesugarCoreLibraryFunctionalTest",
    size = "small",
    srcs = [
        "DesugarCoreLibraryFunctionalTest.java",
    ],
    jvm_flags = [
        # TODO (b/72181101): -Xbootclasspath/p is removed in JDK 9.
        "-XX:+IgnoreUnrecognizedVMOptions",
        "-Xbootclasspath/a:$(location :testdata_desugared_core_library):$(location //third_party/java/jacoco:blaze-agent)",
        "--patch-module=java.base=$(location :testdata_desugared_core_library)",
    ],
    tags = ["no_windows"],
    deps = [
        ":testdata_desugared_core_library",  # Make tests run against desugared library
        "//third_party:guava",
        "//third_party:jsr305",
        "//third_party:junit4",
        "//third_party:truth",
        # Depend on Jacoco runtime in case testdata was built with coverage
        # instrumentation
        "//third_party/java/jacoco:blaze-agent",
    ],
)

java_test(
    name = "StackMapBugTest",
    size = "small",
    srcs = [
        "StackMapBugTest.java",
    ],
    jvm_flags = ["-Xbootclasspath/a:$(location //third_party/java/jacoco:blaze-agent)"],
    tags = ["no_windows"],
    deps = [
        ":testdata_desugared_core_library",  # Make tests run against desugared library
        "//third_party:guava",
        "//third_party:junit4",
        "//third_party:truth",
        # Depend on Jacoco runtime in case testdata was built with coverage
        # instrumentation
        "//third_party/java/jacoco:blaze-agent",
    ],
)

java_test(
    name = "DefaultMethodClassFixerTest",
    size = "small",
    srcs = ["DefaultMethodClassFixerTest.java"],
    data = [
        ":android_jar_for_testing",
        ":separate",
        ":testdata_java8",
        "//third_party:guava-jars",
    ],
    jvm_flags = [
        "-DDefaultMethodClassFixerTest.bootclasspath=$(location :android_jar_for_testing)",
        "-DDefaultMethodClassFixerTest.classpath=$(location :separate):$(location //third_party:guava-jars):$(location //third_party/java/jacoco:blaze-agent)",
        "-DDefaultMethodClassFixerTest.input=$(location :testdata_java8)",
    ],
    tags = ["no_windows"],
    deps = [
        "//src/tools/android/java/com/google/devtools/build/android/desugar",
        "//src/tools/android/java/com/google/devtools/build/android/desugar:deps_collector_api",
        "//src/tools/android/java/com/google/devtools/build/android/desugar/io",
        "//third_party:asm",
        "//third_party:asm-tree",
        "//third_party:guava",
        "//third_party:junit4",
        "//third_party:truth",
        "//third_party/java/jacoco:blaze-agent",
    ],
)

java_test(
    name = "MethodInfoTest",
    size = "small",
    srcs = ["MethodInfoTest.java"],
    deps = [
        "//src/tools/android/java/com/google/devtools/build/android/desugar",
        "//third_party:junit4",
        "//third_party:truth",
    ],
)

java_test(
    name = "Java7CompatibilityTest",
    size = "small",
    srcs = ["Java7CompatibilityTest.java"],
    deps = [
        "//src/tools/android/java/com/google/devtools/build/android/desugar",
        "//src/tools/android/java/com/google/devtools/build/android/desugar/io",
        "//third_party:asm",
        "//third_party:junit4",
        "//third_party:truth",
    ],
)

java_test(
    name = "TryWithResourcesRewriterUnitTestWithReuseStrategy",
    size = "small",
    srcs = [
        "TryWithResourcesRewriterTest.java",
    ],
    jvm_flags = [
        "-Dfortest.simulated.android.sdk_int=19",
        "'-Dexpected.strategy=com.google.devtools.build.android.desugar.runtime.ThrowableExtension$$ReuseDesugaringStrategy'",
    ],
    tags = ["no_windows"],
    test_class = "com.google.devtools.build.android.desugar.TryWithResourcesRewriterTest",
    deps = [
        ":mocked_android_os_sdk_for_testing",
        ":testdata",
        "//src/test/java/com/google/devtools/build/android/desugar/runtime:throwable_extension_test_utility",
        "//src/tools/android/java/com/google/devtools/build/android/desugar",
        "//src/tools/android/java/com/google/devtools/build/android/desugar/io",
        "//src/tools/android/java/com/google/devtools/build/android/desugar/runtime:throwable_extension",
        "//third_party:asm",
        "//third_party:guava",
        "//third_party:junit4",
        "//third_party:truth",
    ],
)

java_test(
    name = "TryWithResourcesRewriterUnitTestWithMimicStrategy",
    size = "small",
    srcs = [
        "TryWithResourcesRewriterTest.java",
    ],
    jvm_flags = [
        "-Dfortest.simulated.android.sdk_int=18",
        "'-Dexpected.strategy=com.google.devtools.build.android.desugar.runtime.ThrowableExtension$$MimicDesugaringStrategy'",
    ],
    tags = ["no_windows"],
    test_class = "com.google.devtools.build.android.desugar.TryWithResourcesRewriterTest",
    deps = [
        ":mocked_android_os_sdk_for_testing",
        ":testdata",
        "//src/test/java/com/google/devtools/build/android/desugar/runtime:throwable_extension_test_utility",
        "//src/tools/android/java/com/google/devtools/build/android/desugar",
        "//src/tools/android/java/com/google/devtools/build/android/desugar/io",
        "//src/tools/android/java/com/google/devtools/build/android/desugar/runtime:throwable_extension",
        "//third_party:asm",
        "//third_party:guava",
        "//third_party:junit4",
        "//third_party:truth",
    ],
)

java_test(
    name = "TryWithResourcesRewriterUnitTestWithNullStrategy",
    size = "small",
    srcs = [
        "TryWithResourcesRewriterTest.java",
    ],
    jvm_flags = [
        "-Dfortest.simulated.android.sdk_int=18",
        "-Dcom.google.devtools.build.android.desugar.runtime.twr_disable_mimic=true",
        "'-Dexpected.strategy=com.google.devtools.build.android.desugar.runtime.ThrowableExtension$$NullDesugaringStrategy'",
    ],
    tags = ["no_windows"],
    test_class = "com.google.devtools.build.android.desugar.TryWithResourcesRewriterTest",
    deps = [
        ":mocked_android_os_sdk_for_testing",
        ":testdata",
        "//src/test/java/com/google/devtools/build/android/desugar/runtime:throwable_extension_test_utility",
        "//src/tools/android/java/com/google/devtools/build/android/desugar",
        "//src/tools/android/java/com/google/devtools/build/android/desugar/io",
        "//src/tools/android/java/com/google/devtools/build/android/desugar/runtime:throwable_extension",
        "//third_party:asm",
        "//third_party:guava",
        "//third_party:junit4",
        "//third_party:truth",
    ],
)

# This test sets the lambda dump directory via the system property in the command line
# (i.e., jvm_flags). So the dump directory is expected to succeed.
java_test(
    name = "DesugarMainClassTestLambdaDirectoryCorrectlySet",
    size = "small",
    srcs = ["DesugarMainClassTest.java"],
    jvm_flags = ["-Djdk.internal.lambda.dumpProxyClasses=$$(mktemp -d)"],
    tags = ["no_windows"],
    test_class = "com.google.devtools.build.android.desugar.DesugarMainClassTest",
    deps = [
        "//src/tools/android/java/com/google/devtools/build/android/desugar",
        "//third_party:guava",
        "//third_party:junit4",
        "//third_party:truth",
    ],
)

# Different from the test target above, this test does not set the lambda dump directory in the
# command line. Instead, it sets the system property in the test code, which is expected to fail.
java_test(
    name = "DesugarMainClassTestLambdaDirectoryIncorrectlySet",
    size = "small",
    srcs = ["DesugarMainClassTest.java"],
    tags = ["no_windows"],
    test_class = "com.google.devtools.build.android.desugar.DesugarMainClassTest",
    deps = [
        "//src/tools/android/java/com/google/devtools/build/android/desugar",
        "//third_party:guava",
        "//third_party:junit4",
        "//third_party:truth",
    ],
)

java_test(
    name = "BytecodeTypeInferenceTest",
    srcs = ["BytecodeTypeInferenceTest.java"],
    data = [
        "BytecodeTypeInferenceTest.golden.txt",
        "//src/test/java/com/google/devtools/build/android/desugar/classes_for_testing_type_inference:test_subjects",
    ],
    jvm_flags = [
        "-Djar_path=$(location //src/test/java/com/google/devtools/build/android/desugar/classes_for_testing_type_inference:test_subjects)",
        "-Dgolden_file=$(location BytecodeTypeInferenceTest.golden.txt)",
    ],
    tags = ["no_windows"],
    deps = [
        ":bytecode_type_printer",
        "//src/tools/android/java/com/google/devtools/build/android/desugar",
        "//third_party:asm",
        "//third_party:guava",
        "//third_party:junit4",
        "//third_party:truth",
    ],
)

sh_test(
    name = "testdata_desugared_jar_test",
    size = "small",
    srcs = ["testdata_desugared_jar_test.sh"],
    args = [
        "$(location testdata_desugared.jar)",
        "$(location testdata_desugared_jar_toc_golden.txt)",
        "$(JAVABASE)",
    ],
    data = [
        "testdata_desugared.jar",
        "testdata_desugared_jar_toc_golden.txt",  # Golden file
        "@bazel_tools//tools/jdk",
    ],
    tags = ["no_windows"],
    toolchains = ["@bazel_tools//tools/jdk:current_java_runtime"],
)

# Make sure desugaring twice doesn't add any files
sh_test(
    name = "testdata_desugared_twice_jar_test",
    size = "small",
    srcs = ["testdata_desugared_jar_test.sh"],
    args = [
        "$(location testdata_desugared_twice.jar)",
        "$(location testdata_desugared_jar_toc_golden.txt)",
        "$(JAVABASE)",
    ],
    data = [
        "testdata_desugared_jar_toc_golden.txt",
        "testdata_desugared_twice.jar",
        "@bazel_tools//tools/jdk",
    ],
    tags = ["no_windows"],
    toolchains = ["@bazel_tools//tools/jdk:current_java_runtime"],
)

sh_test(
    name = "testdata_desugared_java8_jar_test",
    size = "small",
    srcs = ["testdata_desugared_jar_test.sh"],
    args = [
        "$(location testdata_desugared_java8.jar)",
        "$(location testdata_desugared_java8_jar_toc_golden.txt)",
        "$(JAVABASE)",
    ],
    data = [
        "testdata_desugared_java8.jar",
        "testdata_desugared_java8_jar_toc_golden.txt",  # Golden file
        "@bazel_tools//tools/jdk",
    ],
    tags = ["no_windows"],
    toolchains = ["@bazel_tools//tools/jdk:current_java_runtime"],
)

# TODO(b/37110951): Add jar content test for desugared default methods that works with coverage on

sh_test(
    name = "testdata_desugared_core_library_jar_test",
    size = "small",
    srcs = ["testdata_desugared_jar_test.sh"],
    args = [
        "$(location testdata_desugared_core_library.jar)",
        "$(location testdata_desugared_core_library_jar_toc_golden.txt)",
        "$(JAVABASE)",
    ],
    data = [
        "testdata_desugared_core_library.jar",
        "testdata_desugared_core_library_jar_toc_golden.txt",  # Golden file
        "@bazel_tools//tools/jdk",
    ],
    tags = ["no_windows"],
    toolchains = ["@bazel_tools//tools/jdk:current_java_runtime"],
)

sh_test(
    name = "testdata_desugared_for_try_with_resources_test",
    size = "small",
    srcs = ["testdata_desugared_jar_test.sh"],
    args = [
        "$(location testdata_desugared_for_try_with_resources.jar)",
        "$(location testdata_desugared_for_try_with_resources_jar_toc_golden.txt)",
        "$(JAVABASE)",
    ],
    data = [
        "testdata_desugared_for_try_with_resources.jar",
        "testdata_desugared_for_try_with_resources_jar_toc_golden.txt",  # Golden file
        "@bazel_tools//tools/jdk",
    ],
    tags = ["no_windows"],
    toolchains = ["@bazel_tools//tools/jdk:current_java_runtime"],
)

sh_test(
    name = "testdata_desugared_for_try_with_resources_test_twice",
    size = "small",
    srcs = ["testdata_desugared_jar_test.sh"],
    args = [
        "$(location testdata_desugared_for_try_with_resources_twice.jar)",
        "$(location testdata_desugared_for_try_with_resources_jar_toc_golden.txt)",
        "$(JAVABASE)",
    ],
    data = [
        "testdata_desugared_for_try_with_resources_jar_toc_golden.txt",  # Golden file
        "testdata_desugared_for_try_with_resources_twice.jar",
        "@bazel_tools//tools/jdk",
    ],
    tags = ["no_windows"],
    toolchains = ["@bazel_tools//tools/jdk:current_java_runtime"],
)

# Test for b/63900665
sh_test(
    name = "testdata_desugared_for_disabling_try_with_resources_with_large_minsdkversion_test",
    size = "small",
    srcs = ["testdata_desugared_jar_test.sh"],
    args = [
        "$(location testdata_desugared_for_NO_desugaring_try_with_resources.jar)",
        "$(location testdata_desugared_for_disabling_twr_with_large_minsdkversion_jar_toc_golden.txt)",
        "$(JAVABASE)",
    ],
    data = [
        "testdata_desugared_for_NO_desugaring_try_with_resources.jar",
        "testdata_desugared_for_disabling_twr_with_large_minsdkversion_jar_toc_golden.txt",  # Golden file
        "@bazel_tools//tools/jdk",
    ],
    tags = ["no_windows"],
    toolchains = ["@bazel_tools//tools/jdk:current_java_runtime"],
)

# Test for b/62047432
sh_test(
    name = "initializer_of_functional_interface_should_not_execute",
    size = "small",
    srcs = ["static_initializer_of_functional_interface_should_not_execute.sh"],
    args = [
        "$(location :desugar_testdata_with_default_methods_for_static_initializer_test) ",
    ],
    data = [
        ":desugar_testdata_with_default_methods_for_static_initializer_test",
    ],
    tags = ["no_windows"],
)

java_library(
    name = "testdata",
    srcs = glob(["testdata/*.java"]),
    javacopts = ["-Xep:SelfAssignment:OFF"],
    resources = ["testdata/testresource.txt"],
    deps = [
        ":separate",
        "//third_party:guava",
    ],
)

java_library(
    name = "testdata_java8",
    srcs = glob([
        "testdata/*.java",
        "testdata/java8/**/*.java",
    ]),
    javacopts = ["-Xep:SelfAssignment:OFF"],
    resources = ["testdata/testresource.txt"],
    deps = [
        ":separate",
        ":separate_java8",
        "//third_party:guava",
    ],
)

java_library(
    name = "testdata_core_library",
    srcs = glob([
        "testdata/core_library/**/*.java",
    ]),
    resources = ["testdata/testresource.txt"],
    deps = [
        "//third_party:error_prone",
        "//third_party:guava",
    ],
)

java_library(
    name = "separate",
    srcs = glob(["testdata/separate/*.java"]),
)

java_library(
    name = "separate_java8",
    srcs = glob(["testdata/separate8/*.java"]),
)

java_library(
    name = "testdata_like_in_android_studio",
    srcs = glob(["testdata/*.java"]),
    javacopts = ["-Xep:SelfAssignment:OFF"],
    resources = ["testdata/testresource.txt"],
    deps = [
        ":libseparate.jar",  # puts full Jar instead of header Jar on the classpath
        "//third_party:guava",
    ],
)

java_library(
    name = "testdata_java8_like_in_android_studio",
    srcs = glob([
        "testdata/*.java",
        "testdata/java8/**/*.java",
    ]),
    javacopts = ["-Xep:SelfAssignment:OFF"],
    resources = ["testdata/testresource.txt"],
    deps = [
        ":libseparate.jar",  # puts full Jar instead of header Jar on the classpath
        ":libseparate_java8.jar",  # puts full Jar instead of header Jar on the classpath
        "//third_party:guava",
    ],
)

java_library(
    name = "bytecode_type_printer",
    srcs = ["ByteCodeTypePrinter.java"],
    deps = [
        "//src/tools/android/java/com/google/devtools/build/android/desugar",
        "//third_party:asm",
        "//third_party:asm-tree",
        "//third_party:asm-util",
        "//third_party:guava",
    ],
)

# The golden file for this target should NEVER contain generated lambda classes.
sh_test(
    name = "desugar_testdata_by_disabling_lambda_desugaring_test",
    size = "small",
    srcs = ["testdata_desugared_jar_test.sh"],
    args = [
        "$(location testdata_desugared_without_lambda_desugared.jar)",
        "$(location testdata_desugared_without_lambda_desugared_jar_toc_golden.txt)",
        "$(JAVABASE)",
    ],
    data = [
        "testdata_desugared_without_lambda_desugared.jar",
        "testdata_desugared_without_lambda_desugared_jar_toc_golden.txt",  # Golden file
        "@bazel_tools//tools/jdk",
    ],
    tags = ["no_windows"],
    toolchains = ["@bazel_tools//tools/jdk:current_java_runtime"],
)

# The following genrules run the code tested here as a build tool.  While that's
# very similar to how Bazel will invoke it natively, the downside is that
# running the tested tool in a genrule doesn't contribute code coverage
# information for the tested tool.  Note that the code in :testdata doesn't
# appear in coverage reports when depending on this target regardless because
# the needed metadata doesn't survive these genrules.
genrule(
    name = "desugar_testdata",
    srcs = [
        "//third_party:guava-jars",
        ":separate",
        ":testdata",
        # Depend on Jacoco runtime in case testdata was built with coverage
        # instrumentation
        "//third_party/java/jacoco:blaze-agent",
        "@bazel_tools//tools/android:android_jar",
    ],
    outs = ["testdata_desugared.jar"],
    cmd = "$(location //src/tools/android/java/com/google/devtools/build/android/desugar:Desugar) " +
          "--nodesugar_interface_method_bodies_if_needed -i $(location :testdata) -o $@ " +
          "--classpath_entry $(location :separate) " +
          "--classpath_entry $(location //third_party:guava-jars) " +
          "--classpath_entry $(location //third_party/java/jacoco:blaze-agent) " +
          "--bootclasspath_entry $(location @bazel_tools//tools/android:android_jar)",
    tags = ["no_windows"],
    tools = ["//src/tools/android/java/com/google/devtools/build/android/desugar:Desugar"],
)

# Generates special lambda invocation so we can test for b/62060793.
genrule(
    name = "generate_lambda_with_constant_arguments_in_test_data",
    outs = ["testdata_generate_lambda_with_constant_arguments.jar"],
    cmd = "$(location :generate_lambda_with_constant_arguments) $@",
    tags = ["no_windows"],
    tools = [":generate_lambda_with_constant_arguments"],
)

# Desugar the test data for b/62456849
genrule(
    name = "desugar_lambda_with_constant_arguments",
    srcs = [
        ":generate_lambda_with_constant_arguments_in_test_data",
        # Depend on Jacoco runtime in case testdata was built with coverage instrumentation
        "//third_party/java/jacoco:blaze-agent",
        "@bazel_tools//tools/android:android_jar",
    ],
    outs = ["testdata_desugar_generate_lambda_with_constant_arguments.jar"],
    cmd = "$(location //src/tools/android/java/com/google/devtools/build/android/desugar:Desugar) " +
          "-i $(location :generate_lambda_with_constant_arguments_in_test_data) -o $@ " +
          "--classpath_entry $(location //third_party/java/jacoco:blaze-agent) " +
          "--bootclasspath_entry $(location @bazel_tools//tools/android:android_jar)",
    tags = ["no_windows"],
    tools = ["//src/tools/android/java/com/google/devtools/build/android/desugar:Desugar"],
)

java_import(
    name = "desugar_lambda_with_constant_arguments_lib",
    jars = [":desugar_lambda_with_constant_arguments"],
    tags = ["no_windows"],
)

# Convert human-written methods whose names start with "lambda$XXX" to synthetic methods, so we can
# test for b/62456849.
genrule(
    name = "generate_synthetic_methods_with_lambda_names_in_test_data",
    srcs = [":testdata"],
    outs = ["testdata_with_generated_synthetic_methods_with_lambda_names.jar"],
    cmd = "$(location :generate_synthetic_method_with_lambda_name_convention) " +
          " $(location :testdata) " +
          " $@ ",
    tags = ["no_windows"],
    tools = [":generate_synthetic_method_with_lambda_name_convention"],
)

# Desugar the test data for b/62456849
genrule(
    name = "desugar_testdata_with_synthetic_methods_with_lambda_names",
    srcs = [
        "//third_party:guava-jars",
        ":separate",
        ":generate_synthetic_methods_with_lambda_names_in_test_data",
        # Depend on Jacoco runtime in case testdata was built with coverage instrumentation
        "//third_party/java/jacoco:blaze-agent",
        "@bazel_tools//tools/android:android_jar",
    ],
    outs = ["testdata_desugared_with_synthetic_methods_with_lambda_names.jar"],
    cmd = "$(location //src/tools/android/java/com/google/devtools/build/android/desugar:Desugar) " +
          "--nodesugar_interface_method_bodies_if_needed -i $(location :generate_synthetic_methods_with_lambda_names_in_test_data) -o $@ " +
          "--classpath_entry $(location :separate) " +
          "--classpath_entry $(location //third_party:guava-jars) " +
          "--classpath_entry $(location //third_party/java/jacoco:blaze-agent) " +
          "--bootclasspath_entry $(location @bazel_tools//tools/android:android_jar)",
    tags = ["no_windows"],
    tools = ["//src/tools/android/java/com/google/devtools/build/android/desugar:Desugar"],
)

genrule(
    name = "desugar_testdata_again",
    srcs = [
        "//third_party:guava-jars",
        ":separate",
        ":testdata_desugared.jar",
        # Depend on Jacoco runtime in case testdata was built with coverage
        # instrumentation
        "//third_party/java/jacoco:blaze-agent",
        "@bazel_tools//tools/android:android_jar",
    ],
    outs = ["testdata_desugared_twice.jar"],
    cmd = "$(location //src/tools/android/java/com/google/devtools/build/android/desugar:Desugar) " +
          "--nodesugar_interface_method_bodies_if_needed -i $(location :testdata_desugared.jar) -o $@ " +
          "--classpath_entry $(location :separate) " +
          "--classpath_entry $(location //third_party:guava-jars) " +
          "--classpath_entry $(location //third_party/java/jacoco:blaze-agent) " +
          "--bootclasspath_entry $(location @bazel_tools//tools/android:android_jar)",
    tags = ["no_windows"],
    tools = ["//src/tools/android/java/com/google/devtools/build/android/desugar:Desugar"],
)

# This target tests the flag --allow_desugaring_lambda_for_lint=true, which is introduced to solve
# the incompatibility issue with Android Lint. When the issue is addressed, this target and the
# flag will be removed.
genrule(
    name = "desugar_testdata_by_disabling_lambda_desugaring",
    srcs = [
        "//third_party:guava-jars",
        ":separate",
        ":testdata",
        # Depend on Jacoco runtime in case testdata was built with coverage
        # instrumentation
        "//third_party/java/jacoco:blaze-agent",
        "@bazel_tools//tools/android:android_jar",
    ],
    outs = ["testdata_desugared_without_lambda_desugared.jar"],
    cmd = "$(location //src/tools/android/java/com/google/devtools/build/android/desugar:Desugar) " +
          "--min_sdk_version 17 -i $(location :testdata) -o $@ " +
          "--nodesugar_try_with_resources_if_needed " +
          "--classpath_entry $(location :separate) " +
          "--classpath_entry $(location //third_party:guava-jars) " +
          "--classpath_entry $(location //third_party/java/jacoco:blaze-agent) " +
          "--bootclasspath_entry $(location @bazel_tools//tools/android:android_jar) " +
          "--only_desugar_javac9_for_lint",
    tags = ["no_windows"],
    tools = ["//src/tools/android/java/com/google/devtools/build/android/desugar:Desugar"],
)

java_test(
    name = "DesugarTryWithResourcesFunctionalTestWithNullStrategy",
    size = "small",
    srcs = [
        "DesugarTryWithResourcesFunctionalTest.java",
    ],
    jvm_flags = [
        "-Dfortest.simulated.android.sdk_int=18",
        "-Dcom.google.devtools.build.android.desugar.runtime.twr_disable_mimic=true",
        "'-Dexpected.strategy=com.google.devtools.build.android.desugar.runtime.ThrowableExtension$$NullDesugaringStrategy'",
    ],
    tags = ["no_windows"],
    test_class = "com.google.devtools.build.android.desugar.DesugarTryWithResourcesFunctionalTest",
    deps = [
        "mocked_android_os_sdk_for_testing",
        ":desugar_testdata_by_desugaring_try_with_resources",
        "//src/test/java/com/google/devtools/build/android/desugar/runtime:throwable_extension_test_utility",
        "//third_party:asm",
        "//third_party:guava",
        "//third_party:junit4",
        "//third_party:truth",
    ],
)

java_test(
    name = "DesugarTryWithResourcesFunctionalTestWithNullStrategyTwice",
    size = "small",
    srcs = [
        "DesugarTryWithResourcesFunctionalTest.java",
    ],
    jvm_flags = [
        "-Dfortest.simulated.android.sdk_int=18",
        "-Dcom.google.devtools.build.android.desugar.runtime.twr_disable_mimic=true",
        "'-Dexpected.strategy=com.google.devtools.build.android.desugar.runtime.ThrowableExtension$$NullDesugaringStrategy'",
    ],
    tags = ["no_windows"],
    test_class = "com.google.devtools.build.android.desugar.DesugarTryWithResourcesFunctionalTest",
    deps = [
        "mocked_android_os_sdk_for_testing",
        ":desugar_testdata_by_desugaring_try_with_resources_twice",  # the lib desugared twice
        "//src/test/java/com/google/devtools/build/android/desugar/runtime:throwable_extension_test_utility",
        "//third_party:asm",
        "//third_party:guava",
        "//third_party:junit4",
        "//third_party:truth",
    ],
)

java_test(
    name = "DesugarTryWithResourcesFunctionalTestWithMimicStrategy",
    size = "small",
    srcs = [
        "DesugarTryWithResourcesFunctionalTest.java",
    ],
    jvm_flags = [
        "-Dfortest.simulated.android.sdk_int=18",
        "'-Dexpected.strategy=com.google.devtools.build.android.desugar.runtime.ThrowableExtension$$MimicDesugaringStrategy'",
    ],
    tags = ["no_windows"],
    test_class = "com.google.devtools.build.android.desugar.DesugarTryWithResourcesFunctionalTest",
    deps = [
        ":desugar_testdata_by_desugaring_try_with_resources",
        ":mocked_android_os_sdk_for_testing",
        "//src/test/java/com/google/devtools/build/android/desugar/runtime:throwable_extension_test_utility",
        "//third_party:asm",
        "//third_party:guava",
        "//third_party:junit4",
        "//third_party:truth",
    ],
)

java_test(
    name = "DesugarTryWithResourcesFunctionalTestWithMimicStrategyTwice",
    size = "small",
    srcs = [
        "DesugarTryWithResourcesFunctionalTest.java",
    ],
    jvm_flags = [
        "-Dfortest.simulated.android.sdk_int=18",
        "'-Dexpected.strategy=com.google.devtools.build.android.desugar.runtime.ThrowableExtension$$MimicDesugaringStrategy'",
    ],
    tags = ["no_windows"],
    test_class = "com.google.devtools.build.android.desugar.DesugarTryWithResourcesFunctionalTest",
    deps = [
        ":desugar_testdata_by_desugaring_try_with_resources_twice",  # the lib desugared twice.
        ":mocked_android_os_sdk_for_testing",
        "//src/test/java/com/google/devtools/build/android/desugar/runtime:throwable_extension_test_utility",
        "//third_party:asm",
        "//third_party:guava",
        "//third_party:junit4",
        "//third_party:truth",
    ],
)

java_test(
    name = "DesugarTryWithResourcesFunctionalTestWithReuseStrategy",
    size = "small",
    srcs = [
        "DesugarTryWithResourcesFunctionalTest.java",
    ],
    jvm_flags = [
        "-Dfortest.simulated.android.sdk_int=19",
        "'-Dexpected.strategy=com.google.devtools.build.android.desugar.runtime.ThrowableExtension$$ReuseDesugaringStrategy'",
    ],
    tags = ["no_windows"],
    test_class = "com.google.devtools.build.android.desugar.DesugarTryWithResourcesFunctionalTest",
    deps = [
        ":desugar_testdata_by_desugaring_try_with_resources",
        ":mocked_android_os_sdk_for_testing",
        "//src/test/java/com/google/devtools/build/android/desugar/runtime:throwable_extension_test_utility",
        "//third_party:asm",
        "//third_party:guava",
        "//third_party:junit4",
        "//third_party:truth",
    ],
)

java_test(
    name = "DesugarTryWithResourcesFunctionalTestWithReuseStrategyTwice",
    size = "small",
    srcs = [
        "DesugarTryWithResourcesFunctionalTest.java",
    ],
    jvm_flags = [
        "-Dfortest.simulated.android.sdk_int=19",
        "'-Dexpected.strategy=com.google.devtools.build.android.desugar.runtime.ThrowableExtension$$ReuseDesugaringStrategy'",
    ],
    tags = ["no_windows"],
    test_class = "com.google.devtools.build.android.desugar.DesugarTryWithResourcesFunctionalTest",
    deps = [
        ":desugar_testdata_by_desugaring_try_with_resources_twice",  # the lib desugared twice.
        ":mocked_android_os_sdk_for_testing",
        "//src/test/java/com/google/devtools/build/android/desugar/runtime:throwable_extension_test_utility",
        "//third_party:asm",
        "//third_party:guava",
        "//third_party:junit4",
        "//third_party:truth",
    ],
)

java_library(
    name = "mocked_android_os_sdk_for_testing",
    srcs = ["mocked_android_framework/android/os/Build.java"],
)

genrule(
    name = "desugar_testdata_java8",
    srcs = [
        "//third_party:guava-jars",
        ":separate",
        ":separate_java8",
        ":testdata_java8",
        # Depend on Jacoco runtime in case testdata was built with coverage
        # instrumentation
        "//third_party/java/jacoco:blaze-agent",
        "@bazel_tools//tools/android:android_jar",
    ],
    outs = ["testdata_desugared_java8.jar"],
    cmd = "$(location //src/tools/android/java/com/google/devtools/build/android/desugar:Desugar) " +
          "--min_sdk_version 24 " +
          "-i $(location :testdata_java8) -o $@ " +
          "--classpath_entry $(location :separate) " +
          "--classpath_entry $(location :separate_java8) " +
          "--classpath_entry $(location //third_party:guava-jars) " +
          "--classpath_entry $(location //third_party/java/jacoco:blaze-agent) " +
          "--bootclasspath_entry $(location @bazel_tools//tools/android:android_jar)",
    tags = ["no_windows"],
    tools = ["//src/tools/android/java/com/google/devtools/build/android/desugar:Desugar"],
)

genrule(
    name = "desugar_testdata_java8_like_in_android_studio",
    srcs = [
        "//third_party:guava-jars",
        ":separate",
        ":separate_java8",
        ":testdata_java8_like_in_android_studio",
        # Depend on Jacoco runtime in case testdata was built with coverage
        # instrumentation
        "//third_party/java/jacoco:blaze-agent",
        "@bazel_tools//tools/android:android_jar",
    ],
    outs = ["testdata_desugared_java8_like_in_android_studio.jar"],
    cmd = "$(location //src/tools/android/java/com/google/devtools/build/android/desugar:Desugar) " +
          "--copy_bridges_from_classpath " +
          "--legacy_jacoco_fix " +
          "--min_sdk_version 24 " +
          "-i $(location :testdata_java8_like_in_android_studio) -o $@ " +
          "--classpath_entry $(location :separate) " +
          "--classpath_entry $(location :separate_java8) " +
          "--classpath_entry $(location //third_party:guava-jars) " +
          "--classpath_entry $(location //third_party/java/jacoco:blaze-agent) " +
          "--bootclasspath_entry $(location @bazel_tools//tools/android:android_jar)",
    tags = ["no_windows"],
    tools = ["//src/tools/android/java/com/google/devtools/build/android/desugar:Desugar"],
)

genrule(
    name = "desugar_testdata_with_default_methods",
    srcs = [
        "//third_party:guava-jars",
        ":separate",
        ":separate_java8",
        ":testdata_java8",
        # Depend on Jacoco runtime in case testdata was built with coverage
        # instrumentation
        "//third_party/java/jacoco:blaze-agent",
        "@bazel_tools//tools/android:android_jar",
    ],
    outs = ["testdata_desugared_default_methods.jar"],
    cmd = "$(location //src/tools/android/java/com/google/devtools/build/android/desugar:Desugar) " +
          "-i $(location :testdata_java8) -o $@ " +
          "--emit_dependency_metadata_as_needed " +
          "--classpath_entry $(location :separate) " +
          "--classpath_entry $(location :separate_java8) " +
          "--classpath_entry $(location //third_party:guava-jars) " +
          "--classpath_entry $(location //third_party/java/jacoco:blaze-agent) " +
          "--bootclasspath_entry $(location @bazel_tools//tools/android:android_jar)",
    tags = ["no_windows"],
    tools = ["//src/tools/android/java/com/google/devtools/build/android/desugar:Desugar"],
)

genrule(
    name = "desugar_separate_java8_with_default_methods",
    srcs = [
        ":separate_java8",
        # Depend on Jacoco runtime in case testdata was built with coverage
        # instrumentation
        "//third_party/java/jacoco:blaze-agent",
        "@bazel_tools//tools/android:android_jar",
    ],
    outs = ["separate_java8_desugared_default_methods.jar"],
    cmd = "$(location //src/tools/android/java/com/google/devtools/build/android/desugar:Desugar) " +
          "-i $(location :separate_java8) -o $@ " +
          "--emit_dependency_metadata_as_needed " +
          "--classpath_entry $(location //third_party/java/jacoco:blaze-agent) " +
          "--bootclasspath_entry $(location @bazel_tools//tools/android:android_jar)",
    tags = ["no_windows"],
    tools = ["//src/tools/android/java/com/google/devtools/build/android/desugar:Desugar"],
)

genrule(
    name = "desugar_testdata_with_default_methods_for_static_initializer_test",
    srcs = [
        "//third_party:guava-jars",
        ":separate",
        ":separate_java8",
        ":testdata_java8",
        # Depend on Jacoco runtime in case testdata was built with coverage
        # instrumentation
        "//third_party/java/jacoco:blaze-agent",
        "@bazel_tools//tools/android:android_jar",
    ],
    outs = ["testdata_desugared_default_methods.output.txt"],
    cmd = "$(location //src/tools/android/java/com/google/devtools/build/android/desugar:Desugar) " +
          "--desugar_interface_method_bodies_if_needed -i $(location :testdata_java8) " +
          "-o desugar_testdata_with_default_methods_for_static_initializer_test.jar " +
          "--classpath_entry $(location :separate) " +
          "--classpath_entry $(location :separate_java8) " +
          "--classpath_entry $(location //third_party:guava-jars) " +
          "--classpath_entry $(location //third_party/java/jacoco:blaze-agent) " +
          "--bootclasspath_entry $(location @bazel_tools//tools/android:android_jar) " +
          " &> $@",
    tags = ["no_windows"],
    tools = ["//src/tools/android/java/com/google/devtools/build/android/desugar:Desugar"],
)

genrule(
    name = "desugar_testdata_with_default_methods_again",
    srcs = [
        "//third_party:guava-jars",
        ":separate",
        ":separate_java8",
        ":testdata_desugared_default_methods.jar",
        # Depend on Jacoco runtime in case testdata was built with coverage
        # instrumentation
        "//third_party/java/jacoco:blaze-agent",
        "@bazel_tools//tools/android:android_jar",
    ],
    outs = ["testdata_desugared_default_methods_twice.jar"],
    cmd = "$(location //src/tools/android/java/com/google/devtools/build/android/desugar:Desugar) " +
          "--desugar_interface_method_bodies_if_needed -i $(location :testdata_desugared_default_methods.jar) -o $@ " +
          "--classpath_entry $(location :separate) " +
          "--classpath_entry $(location :separate_java8) " +
          "--classpath_entry $(location //third_party:guava-jars) " +
          "--classpath_entry $(location //third_party/java/jacoco:blaze-agent) " +
          "--bootclasspath_entry $(location @bazel_tools//tools/android:android_jar)",
    tags = ["no_windows"],
    tools = ["//src/tools/android/java/com/google/devtools/build/android/desugar:Desugar"],
)

genrule(
    name = "desugar_testdata_core_library",
    srcs = [
        ":testdata_core_library",
        # Depend on Jacoco runtime in case testdata was built with coverage
        # instrumentation
        "//third_party/java/jacoco:blaze-agent",
        "@bazel_tools//tools/android:android_jar",
    ],
    outs = ["testdata_desugared_core_library.jar"],
    cmd = "$(location //src/tools/android/java/com/google/devtools/build/android/desugar:Desugar) " +
          "--core_library -i $(location :testdata_core_library) -o $@ " +
          "--classpath_entry $(location //third_party/java/jacoco:blaze-agent) " +
          "--bootclasspath_entry $(location @bazel_tools//tools/android:android_jar)",
    tags = ["no_windows"],
    tools = ["//src/tools/android/java/com/google/devtools/build/android/desugar:Desugar"],
)

genrule(
    name = "desugar_testdata_like_in_android_studio",
    srcs = [
        "//third_party:guava-jars",
        ":separate",
        ":testdata_like_in_android_studio",
        # Depend on Jacoco runtime in case testdata was built with coverage
        # instrumentation
        "//third_party/java/jacoco:blaze-agent",
        "@bazel_tools//tools/android:android_jar",
    ],
    outs = ["testdata_desugared_like_in_android_studio.jar"],
    cmd = "$(location //src/tools/android/java/com/google/devtools/build/android/desugar:Desugar) " +
          "--min_sdk_version 23 " +
          "--legacy_jacoco_fix " +
          "--nodesugar_interface_method_bodies_if_needed " +
          "-i $(location :testdata_like_in_android_studio) -o $@ " +
          "--copy_bridges_from_classpath " +
          "--classpath_entry $(location :separate) " +
          "--classpath_entry $(location //third_party:guava-jars) " +
          "--classpath_entry $(location //third_party/java/jacoco:blaze-agent) " +
          "--bootclasspath_entry $(location @bazel_tools//tools/android:android_jar)",
    tags = ["no_windows"],
    tools = ["//src/tools/android/java/com/google/devtools/build/android/desugar:Desugar"],
)

genrule(
    name = "desugar_testdata_and_separate_with_multiple_inputs",
    srcs = [
        "//third_party:guava-jars",
        ":separate",
        ":testdata_like_in_android_studio",
        # Depend on Jacoco runtime in case testdata was built with coverage
        # instrumentation
        "//third_party/java/jacoco:blaze-agent",
        "@bazel_tools//tools/android:android_jar",
    ],
    outs = [
        "testdata_desugared_with_multiple_inputs.jar",
        "separate_desugared_with_multiple_inputs.jar",
    ],
    cmd = "$(location //src/tools/android/java/com/google/devtools/build/android/desugar:Desugar) " +
          "--min_sdk_version 23 --nodesugar_interface_method_bodies_if_needed " +
          "-i $(location :testdata_like_in_android_studio) -o $(location testdata_desugared_with_multiple_inputs.jar) " +
          "-i $(location :separate) -o $(location separate_desugared_with_multiple_inputs.jar) " +
          "--copy_bridges_from_classpath " +
          "--classpath_entry $(location :separate) " +
          "--classpath_entry $(location //third_party:guava-jars) " +
          "--classpath_entry $(location //third_party/java/jacoco:blaze-agent) " +
          "--bootclasspath_entry $(location @bazel_tools//tools/android:android_jar)",
    tags = ["no_windows"],
    tools = ["//src/tools/android/java/com/google/devtools/build/android/desugar:Desugar"],
)

genrule(
    name = "desugar_testdata_from_directory_to_jar",
    srcs = [
        "//third_party:guava-jars",
        ":separate",
        ":testdata_like_in_android_studio",
        # Depend on Jacoco runtime in case testdata was built with coverage
        # instrumentation
        "//third_party/java/jacoco:blaze-agent",
        "@bazel_tools//tools/android:android_jar",
    ],
    outs = [
        "testdata_desugared_from_directory_to_jar.jar",
    ],
    cmd = """
    tmpdir=$$(mktemp -d)
    # unzip input jar in order to pass a folder to desugar
    pwddir=$$PWD
    (cd $$tmpdir; $$pwddir/$(location //tools/zip:zipper) x $$pwddir/$(location :testdata_like_in_android_studio))
    $(location //src/tools/android/java/com/google/devtools/build/android/desugar:Desugar) \
          --min_sdk_version 23 --nodesugar_interface_method_bodies_if_needed \
          -i $$tmpdir -o $(location testdata_desugared_from_directory_to_jar.jar) \
          --copy_bridges_from_classpath \
          --classpath_entry $(location :separate) \
          --classpath_entry $(location //third_party:guava-jars) \
          --classpath_entry $(location //third_party/java/jacoco:blaze-agent) \
          --bootclasspath_entry $(location @bazel_tools//tools/android:android_jar)
     rm -rf $$tmpdir
    """,
    tags = ["no_windows"],
    tools = [
        "//src/tools/android/java/com/google/devtools/build/android/desugar:Desugar",
        "//tools/zip:zipper",
    ],
)

genrule(
    name = "desugar_testdata_from_directory_to_directory",
    srcs = [
        "//third_party:guava-jars",
        ":separate",
        ":testdata_like_in_android_studio",
        # Depend on Jacoco runtime in case testdata was built with coverage
        # instrumentation
        "//third_party/java/jacoco:blaze-agent",
        "@bazel_tools//tools/android:android_jar",
    ],
    outs = [
        "testdata_desugared_from_directory_to_directory.jar",
    ],
    cmd = """
    tmpdirIn=$$(mktemp -d)
    tmpdirOut=$$(mktemp -d)
    # unzip input jar in order to pass a folder to desugar
    pwddir=$$PWD
    (cd $$tmpdirIn; $$pwddir/$(location //tools/zip:zipper) x $$pwddir/$(location :testdata_like_in_android_studio))
    $(location //src/tools/android/java/com/google/devtools/build/android/desugar:Desugar) \
          --min_sdk_version 23 --nodesugar_interface_method_bodies_if_needed \
          -i $$tmpdirIn  -o $$tmpdirOut \
          --copy_bridges_from_classpath \
          --classpath_entry $(location :separate) \
          --classpath_entry $(location //third_party:guava-jars) \
          --classpath_entry $(location //third_party/java/jacoco:blaze-agent) \
          --bootclasspath_entry $(location @bazel_tools//tools/android:android_jar)
     pushd $$tmpdirOut
     $$pwddir/$(location //tools/zip:zipper) c $$pwddir/$(location testdata_desugared_from_directory_to_directory.jar) $$(find *)
     popd
     rm -rf $$tmpdirIn
     rm -rf $$tmpdirOut
    """,
    tags = ["no_windows"],
    tools = [
        "//src/tools/android/java/com/google/devtools/build/android/desugar:Desugar",
        "//tools/zip:zipper",
    ],
)

genrule(
    name = "desugar_testdata_with_classpath_directory",
    srcs = [
        "//third_party:guava-jars",
        ":separate",
        ":testdata_like_in_android_studio",
        # Depend on Jacoco runtime in case testdata was built with coverage
        # instrumentation
        "//third_party/java/jacoco:blaze-agent",
        "@bazel_tools//tools/android:android_jar",
    ],
    outs = [
        "testdata_desugared_with_classpath_directory.jar",
    ],
    cmd = """
    tmpdir=$$(mktemp -d)
    # unzip a classpath entry jar in order to pass a folder to desugar
    pwddir=$$PWD
    (cd $$tmpdir; $$pwddir/$(location //tools/zip:zipper) x $$pwddir/$(location :separate))
    $(location //src/tools/android/java/com/google/devtools/build/android/desugar:Desugar) \
          --min_sdk_version 23 --nodesugar_interface_method_bodies_if_needed \
          -i $(location :testdata_like_in_android_studio) -o $(location testdata_desugared_with_classpath_directory.jar) \
          --copy_bridges_from_classpath \
          --classpath_entry $$tmpdir \
          --classpath_entry $(location //third_party:guava-jars) \
          --classpath_entry $(location //third_party/java/jacoco:blaze-agent) \
          --bootclasspath_entry $(location @bazel_tools//tools/android:android_jar)
     rm -rf $$tmpdir
    """,
    tags = ["no_windows"],
    tools = [
        "//src/tools/android/java/com/google/devtools/build/android/desugar:Desugar",
        "//tools/zip:zipper",
    ],
)

# This target tests the flag -desugar_try_with_resources_if_needed, for desugaring
# try-with-resources statements.
genrule(
    name = "desugar_testdata_by_desugaring_try_with_resources",
    srcs = [
        "//third_party:guava-jars",
        ":separate",
        ":testdata",
        # Depend on Jacoco runtime in case testdata was built with coverage
        # instrumentation
        "//third_party/java/jacoco:blaze-agent",
        "@bazel_tools//tools/android:android_jar",
    ],
    outs = ["testdata_desugared_for_try_with_resources.jar"],
    cmd = "$(location //src/tools/android/java/com/google/devtools/build/android/desugar:Desugar) " +
          "--nodesugar_interface_method_bodies_if_needed -i $(location :testdata) -o $@ " +
          "--min_sdk_version 17 --desugar_try_with_resources_if_needed " +
          "--classpath_entry $(location :separate) " +
          "--classpath_entry $(location //third_party:guava-jars) " +
          "--classpath_entry $(location //third_party/java/jacoco:blaze-agent) " +
          "--bootclasspath_entry $(location @bazel_tools//tools/android:android_jar)",
    tags = ["no_windows"],
    tools = ["//src/tools/android/java/com/google/devtools/build/android/desugar:Desugar"],
)

# Desugars an interface with a default and a static interface method that was
# processed by a legacy version of JaCoCo that doesn't emit interface markers
# when calling methods it generated in interfaces (b/62623509).
genrule(
    name = "desugar_default_method_with_legacy_coverage",
    srcs = [
        "jacoco_0_7_5_default_method.jar",
        "//third_party/java/jacoco:blaze-agent",
        "@bazel_tools//tools/android:android_jar",
    ],
    outs = ["jacoco_0_7_5_default_method_desugared.jar"],
    cmd = "$(location //src/tools/android/java/com/google/devtools/build/android/desugar:Desugar) " +
          "--copy_bridges_from_classpath " +
          "--legacy_jacoco_fix " +
          "--min_sdk_version 19 " +
          "-i $(location :jacoco_0_7_5_default_method.jar) -o $@ " +
          "--classpath_entry $(location //third_party/java/jacoco:blaze-agent) " +
          "--bootclasspath_entry $(location @bazel_tools//tools/android:android_jar)",
    tags = ["no_windows"],
    tools = ["//src/tools/android/java/com/google/devtools/build/android/desugar:Desugar"],
)

# Disable desugaring try-with-resources by specifying a min_sdk_version above 18.
genrule(
    name = "desugar_testdata_by_disabling_desugaring_try_with_resources_with_minsdkversion_above_18",
    srcs = [
        "//third_party:guava-jars",
        ":separate",
        ":testdata",
        # Depend on Jacoco runtime in case testdata was built with coverage
        # instrumentation
        "//third_party/java/jacoco:blaze-agent",
        "@bazel_tools//tools/android:android_jar",
    ],
    outs = ["testdata_desugared_for_NO_desugaring_try_with_resources.jar"],
    cmd = "$(location //src/tools/android/java/com/google/devtools/build/android/desugar:Desugar) " +
          "--nodesugar_interface_method_bodies_if_needed -i $(location :testdata) -o $@ " +
          "--min_sdk_version 19 --desugar_try_with_resources_if_needed " +
          "--classpath_entry $(location :separate) " +
          "--classpath_entry $(location //third_party:guava-jars) " +
          "--classpath_entry $(location //third_party/java/jacoco:blaze-agent) " +
          "--bootclasspath_entry $(location @bazel_tools//tools/android:android_jar)",
    tags = ["no_windows"],
    tools = ["//src/tools/android/java/com/google/devtools/build/android/desugar:Desugar"],
)

genrule(
    name = "desugar_testdata_by_desugaring_try_with_resources_twice",
    srcs = [
        "//third_party:guava-jars",
        ":separate",
        ":desugar_testdata_by_desugaring_try_with_resources",
        # Depend on Jacoco runtime in case testdata was built with coverage
        # instrumentation
        "//third_party/java/jacoco:blaze-agent",
        "@bazel_tools//tools/android:android_jar",
    ],
    outs = ["testdata_desugared_for_try_with_resources_twice.jar"],
    cmd = "$(location //src/tools/android/java/com/google/devtools/build/android/desugar:Desugar) " +
          "--nodesugar_interface_method_bodies_if_needed -i $(location :desugar_testdata_by_desugaring_try_with_resources) -o $@ " +
          "--min_sdk_version 17 --desugar_try_with_resources_if_needed " +
          "--classpath_entry $(location :separate) " +
          "--classpath_entry $(location //third_party:guava-jars) " +
          "--classpath_entry $(location //third_party/java/jacoco:blaze-agent) " +
          "--bootclasspath_entry $(location @bazel_tools//tools/android:android_jar)",
    tags = ["no_windows"],
    tools = ["//src/tools/android/java/com/google/devtools/build/android/desugar:Desugar"],
)

# This target allows using the desugared jar as a regular dependency of the test
java_import(
    name = "testdata_desugared",
    jars = ["testdata_desugared.jar"],
    deps = [":separate"],
)

java_import(
    name = "testdata_desugared_twice",
    jars = ["testdata_desugared_twice.jar"],
    deps = [":separate"],
)

java_import(
    name = "testdata_desugared_java8",
    jars = ["testdata_desugared_java8.jar"],
    deps = [
        ":separate",
        ":separate_java8",  # use un-desugared since we're not desugaring default methods here
    ],
)

java_import(
    name = "testdata_desugared_java8_like_in_android_studio",
    jars = ["testdata_desugared_java8_like_in_android_studio.jar"],
    deps = [
        ":separate",
        ":separate_java8",  # use un-desugared since we're not desugaring default methods here
    ],
)

java_import(
    name = "testdata_desugared_default_methods",
    jars = ["testdata_desugared_default_methods.jar"],
    deps = [
        ":separate",
        ":separate_java8_desugared_default_methods",
    ],
)

java_import(
    name = "testdata_desugared_default_methods_twice",
    jars = ["testdata_desugared_default_methods_twice.jar"],
    deps = [
        ":separate",
        ":separate_java8_desugared_default_methods",
    ],
)

java_import(
    name = "separate_java8_desugared_default_methods",
    jars = ["separate_java8_desugared_default_methods.jar"],
)

java_import(
    name = "testdata_desugared_core_library",
    jars = ["testdata_desugared_core_library.jar"],
)

java_import(
    name = "separate_desugared_with_multiple_inputs",
    jars = ["separate_desugared_with_multiple_inputs.jar"],
)

java_import(
    name = "testdata_desugared_with_multiple_inputs",
    jars = ["testdata_desugared_with_multiple_inputs.jar"],
    deps = [":separate_desugared_with_multiple_inputs"],
)

java_import(
    name = "testdata_desugared_from_directory_to_jar",
    jars = ["testdata_desugared_from_directory_to_jar.jar"],
    deps = [":separate"],
)

java_import(
    name = "testdata_desugared_from_directory_to_directory",
    jars = ["testdata_desugared_from_directory_to_directory.jar"],
    deps = [":separate"],
)

java_import(
    name = "testdata_desugared_with_classpath_directory",
    jars = ["testdata_desugared_with_classpath_directory.jar"],
    deps = [":separate"],
)

java_import(
    name = "testdata_desugared_like_in_android_studio",
    jars = ["testdata_desugared_like_in_android_studio.jar"],
    deps = [":separate"],
)

sh_test(
    name = "stateless_lambda_has_no_factory_method_test",
    srcs = ["diff.sh"],
    args = [
        "$(location stateless_lambda_disassembled_golden.txt)",
        "$(location stateless_lambda_disassembled.txt)",
    ],
    data = [
        # Golden file should show a $instance field and *no* get$Lambda() method
        "stateless_lambda_disassembled_golden.txt",
        ":stateless_lambda_disassembled.txt",
    ],
    tags = ["no_windows"],
)

genrule(
    name = "dump_stateless_lambda",
    srcs = [":testdata_desugared.jar"],
    outs = ["stateless_lambda_disassembled.txt"],
    cmd = """
$(JAVABASE)/bin/javap -c -p -cp $< 'com.google.devtools.build.android.desugar.testdata.Lambda$$$$Lambda$$0' > $@
""",
    tags = ["no_windows"],
    toolchains = ["@bazel_tools//tools/jdk:current_host_java_runtime"],
    tools = ["@bazel_tools//tools/jdk"],
)

sh_test(
    name = "capture_lambda_has_no_factory_method_test",
    srcs = ["diff.sh"],
    args = [
        "$(location capture_lambda_disassembled_golden.txt)",
        "$(location capture_lambda_disassembled.txt)",
    ],
    data = [
        # Golden file should show a package-private constructor and *no* get$Lambda() method
        "capture_lambda_disassembled_golden.txt",
        ":capture_lambda_disassembled.txt",
    ],
    tags = ["no_windows"],
    toolchains = ["@bazel_tools//tools/jdk:current_java_runtime"],
)

genrule(
    name = "dump_capture_lambda",
    srcs = [":testdata_desugared.jar"],
    outs = ["capture_lambda_disassembled.txt"],
    cmd = """
$(JAVABASE)/bin/javap -c -p -cp $< 'com.google.devtools.build.android.desugar.testdata.CaptureLambda$$$$Lambda$$0' > $@
""",
    tags = ["no_windows"],
    toolchains = ["@bazel_tools//tools/jdk:current_java_runtime"],
    tools = ["@bazel_tools//tools/jdk"],
)

sh_test(
    name = "simple_method_reference_has_no_factory_method_test",
    srcs = ["diff.sh"],
    args = [
        "$(location simple_instance_method_reference_disassembled_golden.txt)",
        "$(location simple_instance_method_reference_disassembled.txt)",
    ],
    data = [
        # Golden file should show a package-private constructor and *no* get$Lambda() method
        "simple_instance_method_reference_disassembled_golden.txt",
        ":simple_instance_method_reference_disassembled.txt",
    ],
    tags = ["no_windows"],
)

genrule(
    name = "dump_simple_instance_method_reference",
    srcs = [":testdata_desugared.jar"],
    outs = ["simple_instance_method_reference_disassembled.txt"],
    cmd = """
$(JAVABASE)/bin/javap -c -p -cp $< 'com.google.devtools.build.android.desugar.testdata.MethodReferenceSuperclass$$$$Lambda$$0' > $@
""",
    tags = ["no_windows"],
    toolchains = ["@bazel_tools//tools/jdk:current_java_runtime"],
    tools = ["@bazel_tools//tools/jdk"],
)

sh_test(
    name = "interface_has_method_bodies_removed",
    srcs = ["diff.sh"],
    args = [
        "$(location interface_with_desugared_method_bodies_disassembled_golden.txt)",
        "$(location interface_with_desugared_method_bodies_disassembled.txt)",
    ],
    data = [
        # Golden file should show abstract methods only and *no* static methods
        "interface_with_desugared_method_bodies_disassembled_golden.txt",
        ":interface_with_desugared_method_bodies_disassembled.txt",
    ],
    tags = ["no_windows"],
)

genrule(
    name = "dump_interface_with_method_bodies",
    srcs = [":testdata_desugared_default_methods.jar"],
    outs = ["interface_with_desugared_method_bodies_disassembled.txt"],
    # drop jacoco fields and static initializers, which jacoco may add, so the output is stable
    # with or without coverage instrumentation
    cmd = """
$(JAVABASE)/bin/javap -p -cp $< 'com.google.devtools.build.android.desugar.testdata.java8.InterfaceMethod' | grep -v 'jacoco' | grep -v 'static {}' > $@
""",
    tags = ["no_windows"],
    toolchains = ["@bazel_tools//tools/jdk:current_java_runtime"],
    tools = ["@bazel_tools//tools/jdk"],
)

sh_test(
    name = "inherited_abstract_method_gets_no_default_method_stub",
    srcs = ["diff.sh"],
    args = [
        "$(location class_with_inherited_abstract_method_disassembled_golden.txt)",
        "$(location class_with_inherited_abstract_method_disassembled.txt)",
    ],
    data = [
        # Golden file should show *no* methods
        "class_with_inherited_abstract_method_disassembled_golden.txt",
        ":class_with_inherited_abstract_method_disassembled.txt",
    ],
    tags = ["no_windows"],
)

genrule(
    name = "dump_class_with_inherited_abstract_method",
    srcs = [":testdata_desugared_default_methods.jar"],
    outs = ["class_with_inherited_abstract_method_disassembled.txt"],
    # drop jacoco fields and static initializers, which jacoco may add, so the output is stable
    # with or without coverage instrumentation
    cmd = """
$(JAVABASE)/bin/javap -p -cp $< 'com.google.devtools.build.android.desugar.testdata.java8.Named$$AbstractName' | grep -v 'jacoco' | grep -v 'static {}' > $@
""",
    tags = ["no_windows"],
    toolchains = ["@bazel_tools//tools/jdk:current_java_runtime"],
    tools = ["@bazel_tools//tools/jdk"],
)

sh_test(
    name = "no_stubs_for_lambdas_in_implemented_interface",
    srcs = ["diff.sh"],
    args = [
        "$(location class_with_lambdas_in_implemented_interface_disassembled_golden.txt)",
        "$(location class_with_lambdas_in_implemented_interface_disassembled.txt)",
    ],
    data = [
        # Golden file should show *no* lambda$xxx methods (and 4 stubs for default methods)
        "class_with_lambdas_in_implemented_interface_disassembled_golden.txt",
        ":class_with_lambdas_in_implemented_interface_disassembled.txt",
    ],
    tags = ["no_windows"],
)

genrule(
    name = "dump_class_with_lambdas_in_implemented_interface",
    srcs = [":testdata_desugared_default_methods.jar"],
    outs = ["class_with_lambdas_in_implemented_interface_disassembled.txt"],
    # drop jacoco fields and static initializers, which jacoco may add, so the output is stable
    # with or without coverage instrumentation
    cmd = """
$(JAVABASE)/bin/javap -p -cp $< 'com.google.devtools.build.android.desugar.testdata.java8.InterfaceMethod$$Concrete' | grep -v 'jacoco' | grep -v 'static {}' > $@
""",
    tags = ["no_windows"],
    toolchains = ["@bazel_tools//tools/jdk:current_java_runtime"],
    tools = ["@bazel_tools//tools/jdk"],
)

sh_test(
    name = "desugar_idempotency_test",
    srcs = ["diff.sh"],
    args = [
        "$(location :capture_lambda_desugared.jar)",
        "$(location :capture_lambda_desugared_twice.jar)",
    ],
    data = [
        ":capture_lambda_desugared.jar",
        ":capture_lambda_desugared_twice.jar",
    ],
    tags = ["no_windows"],
)

# Smaller Jar used for idempotency test.  CaptureLambda can be desugared without
# any need to fix up the resulting lambda class.  Running the desugaring tool
# over lambda classes that required fixes somehow perturbs the constant pool
# (but nothing else) so for simplicity we use this Jar for the idempotency test
# we don't run into the perturbance issue.
java_library(
    name = "capture_lambda",
    srcs = ["testdata/CaptureLambda.java"],
)

genrule(
    name = "desugar_capture_lambda",
    srcs = [
        ":capture_lambda",
        # Depend on Jacoco runtime in case testdata was built with coverage
        # instrumentation
        "//third_party/java/jacoco:blaze-agent",
        "@bazel_tools//tools/android:android_jar",
    ],
    outs = ["capture_lambda_desugared.jar"],
    cmd = "$(location //src/tools/android/java/com/google/devtools/build/android/desugar:Desugar) " +
          "-i $(location :capture_lambda) -o $@ " +
          "--classpath_entry $(location //third_party/java/jacoco:blaze-agent) " +
          "--bootclasspath_entry $(location @bazel_tools//tools/android:android_jar)",
    tags = ["no_windows"],
    tools = ["//src/tools/android/java/com/google/devtools/build/android/desugar:Desugar"],
)

genrule(
    name = "desugar_capture_lambda_again",
    srcs = [
        ":capture_lambda_desugared.jar",
        # Depend on Jacoco runtime in case testdata was built with coverage
        # instrumentation
        "//third_party/java/jacoco:blaze-agent",
        "@bazel_tools//tools/android:android_jar",
    ],
    outs = ["capture_lambda_desugared_twice.jar"],
    cmd = "$(location //src/tools/android/java/com/google/devtools/build/android/desugar:Desugar) " +
          "-i $(location :capture_lambda_desugared.jar) -o $@ " +
          "--classpath_entry $(location //third_party/java/jacoco:blaze-agent) " +
          "--bootclasspath_entry $(location @bazel_tools//tools/android:android_jar)",
    tags = ["no_windows"],
    tools = ["//src/tools/android/java/com/google/devtools/build/android/desugar:Desugar"],
)

sh_test(
    name = "testdata_sanity_test",
    srcs = ["diff.sh"],
    args = [
        "$(location :baseclass_lambda_signature.txt)",
        "$(location :subclass_lambda_signature.txt)",
    ],
    data = [
        ":baseclass_lambda_signature.txt",
        ":subclass_lambda_signature.txt",
    ],
    tags = ["no_windows"],
)

genrule(
    name = "baseclass_lambda_signature",
    srcs = [":testdata"],
    outs = [
        "baseclass_disassembled.txt",
        "baseclass_lambda_signature.txt",
    ],
    cmd = """
$(JAVABASE)/bin/javap  -p -cp $< com.google.devtools.build.android.desugar.testdata.OuterReferenceLambda > $(location baseclass_disassembled.txt)
grep lambda $(location baseclass_disassembled.txt) > $(location baseclass_lambda_signature.txt)""",
    tags = ["no_windows"],
    toolchains = ["@bazel_tools//tools/jdk:current_java_runtime"],
    tools = ["@bazel_tools//tools/jdk"],
)

genrule(
    name = "subclass_lambda_signature",
    srcs = [":testdata"],
    outs = [
        "subclass_disassembled.txt",
        "subclass_lambda_signature.txt",
    ],
    cmd = """
$(JAVABASE)/bin/javap -p -cp $< com.google.devtools.build.android.desugar.testdata.LambdaInOverride > $(location subclass_disassembled.txt)
grep lambda $(location subclass_disassembled.txt) > $(location subclass_lambda_signature.txt)""",
    tags = ["no_windows"],
    toolchains = ["@bazel_tools//tools/jdk:current_java_runtime"],
    tools = [
        "@bazel_tools//tools/jdk",
    ],
)

_SINGLEJAR_IMPLS = {
    "singlejar": "//src/tools/singlejar:singlejar_local",
    "JavaSingleJar": "//src/java_tools/singlejar:SingleJar",
}

[sh_test(
    name = "desugar_deps_consistent_%s_test" % label,
    srcs = ["desugar_deps_consistency_test.sh"],
    args = [
        "$(location %s)" % singlejar,
        "$(location @bazel_tools//tools/jdk:jar)",
        "$(location :testdata_desugared_default_methods.jar)",
        "$(location :separate_java8_desugared_default_methods.jar)",
        "$(location :guava_at_head_desugared.jar)",
    ],
    data = [
        ":guava_at_head_desugared.jar",
        ":separate_java8_desugared_default_methods.jar",
        ":testdata_desugared_default_methods.jar",
        singlejar,
        "@bazel_tools//tools/jdk:jar",
    ],
    tags = ["no_windows"],
) for label, singlejar in _SINGLEJAR_IMPLS.items()]

# Tests that deps checking fails due to missing default method: libseparate_java8.jar defines
# default methods which testdata_desugared_default_methods.jar assumes have been moved to companion.
[sh_test(
    name = "desugar_deps_missing_%s_fail_test" % label,
    srcs = ["desugar_deps_consistency_test.sh"],
    args = [
        "$(location %s)" % singlejar,
        "$(location @bazel_tools//tools/jdk:jar)",
        "$(location :testdata_desugared_default_methods.jar)",
        "$(location :libseparate_java8.jar)",
        "$(location :guava_at_head_desugared.jar)",
    ],
    data = [
        ":guava_at_head_desugared.jar",
        ":libseparate_java8.jar",
        ":testdata_desugared_default_methods.jar",
        singlejar,
        "@bazel_tools//tools/jdk:jar",
    ],
    tags = ["no_windows"],
) for label, singlejar in _SINGLEJAR_IMPLS.items()]

# Regression test for b/68049457 with deps checking failing due to missing static method:
# b68049457_caller_desugared.jar assumes presence of companion class that's missing (in fact,
# the whole Jar is missing, similar to neverlink situations).
[sh_test(
    name = "desugar_deps_b68049457_%s_fail_test" % label,
    srcs = ["desugar_deps_consistency_test.sh"],
    args = [
        "$(location %s)" % singlejar,
        "$(location @bazel_tools//tools/jdk:jar)",
        "$(location :b68049457_caller_desugared.jar)",
    ],
    data = [
        ":b68049457_caller_desugared.jar",
        singlejar,
        "@bazel_tools//tools/jdk:jar",
    ],
    tags = ["no_windows"],
) for label, singlejar in _SINGLEJAR_IMPLS.items()]

[sh_test(
    name = "desugar_deps_missed_default_method_%s_fail_test" % label,
    srcs = ["desugar_deps_consistency_test.sh"],
    args = [
        "$(location %s)" % singlejar,
        "$(location @bazel_tools//tools/jdk:jar)",
        "$(location :testdata_desugared_with_missing_dep.jar)",
        "$(location :separate_java8_desugared_default_methods.jar)",
        "$(location :guava_at_head_desugared.jar)",
    ],
    data = [
        ":guava_at_head_desugared.jar",
        ":separate_java8_desugared_default_methods.jar",
        ":testdata_desugared_with_missing_dep.jar",
        singlejar,
        "@bazel_tools//tools/jdk:jar",
    ],
    tags = ["no_windows"],
) for label, singlejar in _SINGLEJAR_IMPLS.items()]

# Tests that any Jar entry starting with j$/ causes singlejar to fail when run
# with --check_desugar_deps.  This is a backstop against such entries being
# included in deploy.jars.
[sh_test(
    name = "bad_entry_jar_%s_fail_test" % label,
    srcs = ["desugar_deps_consistency_test.sh"],
    args = [
        "$(location %s)" % singlejar,
        "$(location @bazel_tools//tools/jdk:jar)",
        "$(location :mock_bad_entry_jar)",
    ],
    data = [
        ":mock_bad_entry_jar",
        singlejar,
        "@bazel_tools//tools/jdk:jar",
    ],
    tags = ["no_windows"],
) for label, singlejar in _SINGLEJAR_IMPLS.items()]

genrule(
    name = "desugar_testdata_with_missing_dep",
    srcs = [
        ":separate",
        ":testdata_java8",
        "//third_party:guava-jars",
        "//third_party/java/jacoco:blaze-agent",
        "@bazel_tools//tools/android:android_jar",
    ],
    outs = ["testdata_desugared_with_missing_dep.jar"],
    cmd = "$(location //src/tools/android/java/com/google/devtools/build/android/desugar:Desugar) " +
          "-i $(location :testdata_java8) -o $@ " +
          "--emit_dependency_metadata_as_needed " +
          "--classpath_entry $(location :separate) " +
          "--classpath_entry $(location //third_party:guava-jars) " +
          "--classpath_entry $(location //third_party/java/jacoco:blaze-agent) " +
          "--bootclasspath_entry $(location @bazel_tools//tools/android:android_jar)",
    tools = ["//src/tools/android/java/com/google/devtools/build/android/desugar:Desugar"],
)

genrule(
    name = "desugar_b68049457_caller",
    srcs = [
        ":b68049457_caller",
        ":b68049457_interface",
        "//third_party/java/jacoco:blaze-agent",
        "@bazel_tools//tools/android:android_jar",
    ],
    outs = ["b68049457_caller_desugared.jar"],
    cmd = "$(location //src/tools/android/java/com/google/devtools/build/android/desugar:Desugar) " +
          "-i $(location :b68049457_caller) -o $@ " +
          "--emit_dependency_metadata_as_needed " +
          "--classpath_entry $(location :b68049457_interface) " +
          "--bootclasspath_entry $(location @bazel_tools//tools/android:android_jar)",
    tools = ["//src/tools/android/java/com/google/devtools/build/android/desugar:Desugar"],
)

java_library(
    name = "b68049457_interface",
    srcs = ["testdata/b68049457/StaticInterfaceMethod.java"],
    neverlink = 1,
)

java_library(
    name = "b68049457_caller",
    srcs = ["testdata/b68049457/StaticInterfaceMethodCaller.java"],
    deps = [":b68049457_interface"],
)

# Creates simple .jar file with empty file entry called "j$/hello"
genrule(
    name = "mock_bad_entry_jar",
    outs = ["mock_bad_entry.jar"],
    cmd = """$(location //tools/zip:zipper) c $@ "j\$$/hello=" """,
    tools = ["//tools/zip:zipper"],
)

# b/62623509 regression test.  Calls to jacoco-generated method in interface
# need to be rewritten to call the companion class even though the original
# bytecode falsely doesn't indicate that the method is declared in an
# interface. Jacoco-generated field is *not* moved.
sh_test(
    name = "jacoco_legacy_default_method_regression_test",
    srcs = ["diff.sh"],
    args = [
        "$(location :jacoco_legacy_default_method_companion_disassembled_golden.txt)",
        "$(location :jacoco_legacy_default_method_companion_disassembled.txt)",
    ],
    data = [
        "jacoco_legacy_default_method_companion_disassembled_golden.txt",
        ":jacoco_legacy_default_method_companion_disassembled.txt",
    ],
    tags = ["no_windows"],
)

genrule(
    name = "dump_jacoco_legacy_default_method_companion",
    srcs = [":jacoco_0_7_5_default_method_desugared.jar"],
    outs = ["jacoco_legacy_default_method_companion_disassembled.txt"],
    cmd = """$(JAVABASE)/bin/javap -c -p -cp $< 'com/example/gavra/java8coverage/Defaults$$$$CC' > $@""",
    tags = ["no_windows"],
    toolchains = ["@bazel_tools//tools/jdk:current_java_runtime"],
    tools = [
        "@bazel_tools//tools/jdk",
    ],
)

# Regression test for b/70415451
genrule(
    name = "desugar_guava_at_head",
    srcs = [
        "//third_party:guava-jars",
        # Depend on Jacoco runtime in case testdata was built with coverage
        # instrumentation
        "//third_party/java/jacoco:blaze-agent",
        "@bazel_tools//tools/android:android_jar",
    ],
    outs = ["guava_at_head_desugared.jar"],
    cmd = "$(location //src/tools/android/java/com/google/devtools/build/android/desugar:Desugar) " +
          "-i $(location //third_party:guava-jars) -o $@ " +
          "--bootclasspath_entry $(location @bazel_tools//tools/android:android_jar)",
    tags = ["no_windows"],
    tools = ["//src/tools/android/java/com/google/devtools/build/android/desugar:Desugar"],
)

java_binary(
    name = "generate_lambda_with_constant_arguments",
    srcs = ["Bug62060793TestDataGenerator.java"],
    main_class = "com.google.devtools.build.android.desugar.Bug62060793TestDataGenerator",
    deps = [
        "//third_party:asm",
        "//third_party:guava",
    ],
)

java_binary(
    name = "generate_synthetic_method_with_lambda_name_convention",
    srcs = ["Bug62456849TestDataGenerator.java"],
    main_class = "com.google.devtools.build.android.desugar.Bug62456849TestDataGenerator",
    deps = [
        "//third_party:asm",
        "//third_party:guava",
    ],
)

test_suite(
    name = "windows_tests",
    tags = [
        "-no_windows",
        "-slow",
    ],
    visibility = ["//visibility:private"],
)

test_suite(
    name = "all_windows_tests",
    tests = [
        ":windows_tests",
        "//src/test/java/com/google/devtools/build/android/desugar/classes_for_testing_type_inference:all_windows_tests",
        "//src/test/java/com/google/devtools/build/android/desugar/dependencies:all_windows_tests",
        "//src/test/java/com/google/devtools/build/android/desugar/io:all_windows_tests",
        "//src/test/java/com/google/devtools/build/android/desugar/runtime:all_windows_tests",
    ],
    visibility = ["//src/test/java/com/google/devtools/build/android:__pkg__"],
)
