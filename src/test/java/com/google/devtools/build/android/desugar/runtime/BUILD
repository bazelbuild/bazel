# Description:
#   Tests for the extension classes for desugaring try-with-resources for Android.
package(
    default_testonly = 1,
    default_visibility = ["//src/test/java/com/google/devtools/build/android/desugar:__subpackages__"],
)

licenses(["notice"])  # Apache 2.0

filegroup(
    name = "srcs",
    testonly = 0,
    srcs = glob(["*"]),
)

java_library(
    name = "throwable_extension_test_utility",
    srcs = ["ThrowableExtensionTestUtility.java"],
    deps = [
        "//third_party:guava",
        "//third_party:truth",
    ],
)

java_test(
    name = "ThrowableExtensionTestWithMimicDesugaringStrategy",
    size = "small",
    srcs = ["ThrowableExtensionTest.java"],
    jvm_flags = [
        "-Dfortest.simulated.android.sdk_int=18",
        "-Dexpected.strategy='com.google.devtools.build.android.desugar.runtime.ThrowableExtension$$MimicDesugaringStrategy'",
    ],
    test_class = "com.google.devtools.build.android.desugar.runtime.ThrowableExtensionTest",
    deps = [
        ":throwable_extension_test_utility",
        "//src/test/java/com/google/devtools/build/android/desugar:mocked_android_os_sdk_for_testing",
        "//src/test/java/com/google/devtools/build/lib:testutil",
        "//src/tools/android/java/com/google/devtools/build/android/desugar/runtime:throwable_extension",
        "//third_party:guava",
        "//third_party:junit4",
        "//third_party:truth",
    ],
)

java_test(
    name = "ThrowableExtensionTestWithNullDesugaringStrategy",
    size = "small",
    srcs = ["ThrowableExtensionTest.java"],
    jvm_flags = [
        "-Dfortest.simulated.android.sdk_int=18",
        "-Dcom.google.devtools.build.android.desugar.runtime.twr_disable_mimic=true",
        "-Dexpected.strategy='com.google.devtools.build.android.desugar.runtime.ThrowableExtension$$NullDesugaringStrategy'",
    ],
    test_class = "com.google.devtools.build.android.desugar.runtime.ThrowableExtensionTest",
    deps = [
        ":throwable_extension_test_utility",
        "//src/test/java/com/google/devtools/build/android/desugar:mocked_android_os_sdk_for_testing",
        "//src/test/java/com/google/devtools/build/lib:testutil",
        "//src/tools/android/java/com/google/devtools/build/android/desugar/runtime:throwable_extension",
        "//third_party:guava",
        "//third_party:junit4",
        "//third_party:truth",
    ],
)

java_test(
    name = "ThrowableExtensionTestWithReuseDesugaringStrategy",
    size = "small",
    srcs = ["ThrowableExtensionTest.java"],
    jvm_flags = [
        "-Dfortest.simulated.android.sdk_int=19",
        "-Dexpected.strategy='com.google.devtools.build.android.desugar.runtime.ThrowableExtension$$ReuseDesugaringStrategy'",
    ],
    test_class = "com.google.devtools.build.android.desugar.runtime.ThrowableExtensionTest",
    deps = [
        ":throwable_extension_test_utility",
        "//src/test/java/com/google/devtools/build/android/desugar:mocked_android_os_sdk_for_testing",
        "//src/test/java/com/google/devtools/build/lib:testutil",
        "//src/tools/android/java/com/google/devtools/build/android/desugar/runtime:throwable_extension",
        "//third_party:guava",
        "//third_party:junit4",
        "//third_party:truth",
    ],
)

java_test(
    name = "ConcurrentWeakIdentityHashMapTest",
    timeout = "long",  # multi-threaded and rely on GC, so it could take long.
    srcs = ["ConcurrentWeakIdentityHashMapTest.java"],
    test_class = "com.google.devtools.build.android.desugar.runtime.ConcurrentWeakIdentityHashMapTest",
    deps = [
        "//src/tools/android/java/com/google/devtools/build/android/desugar/runtime:throwable_extension",
        "//third_party:guava",
        "//third_party:guava-testlib",
        "//third_party:junit4",
        "//third_party:truth",
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
    ],
    visibility = ["//src/test/java/com/google/devtools/build/android/desugar:__pkg__"],
)
