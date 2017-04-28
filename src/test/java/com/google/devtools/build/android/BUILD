filegroup(
    name = "srcs",
    srcs = glob(["**"]) + [
        "//src/test/java/com/google/devtools/build/android/idlclass:srcs",
        "//src/test/java/com/google/devtools/build/android/resources:srcs",
        "//src/test/java/com/google/devtools/build/android/testing/manifestmerge:srcs",
        "//src/test/java/com/google/devtools/build/android/ziputils:srcs",
    ],
    visibility = ["//src:__pkg__"],
)

java_test(
    name = "RClassGeneratorActionTest",
    size = "medium",
    srcs = ["RClassGeneratorActionTest.java"],
    deps = [
        "//src/tools/android/java/com/google/devtools/build/android:android_builder_lib",
        "//third_party:guava",
        "//third_party:junit4",
        "//third_party:truth",
    ],
)

java_test(
    name = "AndroidDataMergerTest",
    srcs = ["AndroidDataMergerTest.java"],
    tags = ["no_windows"],  # Test asserts forward slashes in android data xml files.
    deps = [
        ":test_utils",
        "//src/tools/android/java/com/google/devtools/build/android:android_builder_lib",
        "//third_party:android_common_25_0_0",
        "//third_party:guava",
        "//third_party:jimfs",
        "//third_party:junit4",
        "//third_party:truth",
    ],
)

java_test(
    name = "AndroidDataWriterTest",
    srcs = ["AndroidDataWriterTest.java"],
    tags = ["no_windows"],  # Test asserts forward slashes in android data xml files.
    deps = [
        ":test_utils",
        "//src/tools/android/java/com/google/devtools/build/android:android_builder_lib",
        "//third_party:android_common_25_0_0",
        "//third_party:guava",
        "//third_party:jimfs",
        "//third_party:junit4",
        "//third_party:truth",
    ],
)

java_test(
    name = "AndroidResourceClassWriterTest",
    srcs = ["AndroidResourceClassWriterTest.java"],
    tags = ["no_windows"],  # Test asserts forward slashes in android data xml files.
    deps = [
        ":test_utils",
        "//src/tools/android/java/com/google/devtools/build/android:android_builder_lib",
        "//third_party:guava",
        "//third_party:jimfs",
        "//third_party:junit4",
        "//third_party:truth",
    ],
)

java_test(
    name = "ConvertersTest",
    size = "small",
    srcs = ["ConvertersTest.java"],
    deps = [
        "//src/main/java/com/google/devtools/common/options",
        "//src/tools/android/java/com/google/devtools/build/android:android_builder_lib",
        "//third_party:guava",
        "//third_party:junit4",
        "//third_party:truth",
    ],
)

java_test(
    name = "DensitySpecificManifestProcessorTest",
    srcs = ["DensitySpecificManifestProcessorTest.java"],
    tags = ["no_windows"],  # Test asserts forward slashes in android data xml files.
    deps = [
        "//src/tools/android/java/com/google/devtools/build/android:android_builder_lib",
        "//third_party:guava",
        "//third_party:jimfs",
        "//third_party:junit4",
        "//third_party:truth",
    ],
)

java_test(
    name = "DependencyAndroidDataTest",
    srcs = ["DependencyAndroidDataTest.java"],
    deps = [
        "//src/tools/android/java/com/google/devtools/build/android:android_builder_lib",
        "//third_party:guava",
        "//third_party:jimfs",
        "//third_party:junit4",
        "//third_party:truth",
    ],
)

java_test(
    name = "ManifestMergerActionTest",
    srcs = ["ManifestMergerActionTest.java"],
    data = [
        "//src/test/java/com/google/devtools/build/android/testing/manifestmerge:test_data",
    ],
    jvm_flags = [
        "-Dtestdatadir=src/test/java/com/google/devtools/build/android/testing/manifestmerge",
    ],
    tags = ["no_windows"],
    deps = [
        ":test_utils",
        "//src/tools/android/java/com/google/devtools/build/android:android_builder_lib",
        "//third_party:guava",
        "//third_party:junit4",
        "//third_party:truth",
    ],
)

java_test(
    name = "ParsedAndroidDataTest",
    srcs = ["ParsedAndroidDataTest.java"],
    tags = ["manual"],  # Fails due to https://github.com/bazelbuild/bazel/issues/2709
    deps = [
        ":test_utils",
        "//src/tools/android/java/com/google/devtools/build/android:android_builder_lib",
        "//third_party:android_common_25_0_0",
        "//third_party:guava",
        "//third_party:jimfs",
        "//third_party:junit4",
        "//third_party:truth",
    ],
)

java_test(
    name = "SerializedAndroidDataTest",
    srcs = ["SerializedAndroidDataTest.java"],
    deps = [
        "//src/tools/android/java/com/google/devtools/build/android:android_builder_lib",
        "//third_party:guava",
        "//third_party:jimfs",
        "//third_party:junit4",
        "//third_party:truth",
    ],
)

java_test(
    name = "SplitConfigurationFilterTest",
    srcs = ["SplitConfigurationFilterTest.java"],
    deps = [
        "//src/tools/android/java/com/google/devtools/build/android:android_builder_lib",
        "//third_party:guava",
        "//third_party:guava-testlib",
        "//third_party:jsr305",
        "//third_party:junit4",
        "//third_party:truth",
    ],
)

java_test(
    name = "UnvalidatedAndroidDataTest",
    srcs = ["UnvalidatedAndroidDataTest.java"],
    deps = [
        "//src/tools/android/java/com/google/devtools/build/android:android_builder_lib",
        "//third_party:guava",
        "//third_party:jimfs",
        "//third_party:junit4",
        "//third_party:truth",
    ],
)

java_test(
    name = "UnvalidatedAndroidDirectoriesTest",
    srcs = ["UnvalidatedAndroidDirectoriesTest.java"],
    deps = [
        "//src/tools/android/java/com/google/devtools/build/android:android_builder_lib",
        "//third_party:guava",
        "//third_party:jimfs",
        "//third_party:junit4",
        "//third_party:truth",
    ],
)

java_library(
    name = "test_utils",
    testonly = 1,
    srcs = glob(
        ["*.java"],
        exclude = ["*Test.java"],
    ),
    deps = [
        "//src/tools/android/java/com/google/devtools/build/android:android_builder_lib",
        "//third_party:android_common_25_0_0",
        "//third_party:guava",
        "//third_party:truth",
    ],
)
