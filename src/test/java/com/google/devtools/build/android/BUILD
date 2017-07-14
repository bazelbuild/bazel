filegroup(
    name = "srcs",
    srcs = glob(["**"]) + [
        "//src/test/java/com/google/devtools/build/android/ideinfo:srcs",
        "//src/test/java/com/google/devtools/build/android/idlclass:srcs",
        "//src/test/java/com/google/devtools/build/android/dexer:srcs",
        "//src/test/java/com/google/devtools/build/android/junctions:srcs",
        "//src/test/java/com/google/devtools/build/android/resources:srcs",
        "//src/test/java/com/google/devtools/build/android/testing/manifestmerge:srcs",
        "//src/test/java/com/google/devtools/build/android/ziputils:srcs",
    ],
    visibility = ["//src:__pkg__"],
)

java_test(
    name = "AndroidResourceOutputsTest",
    size = "small",
    srcs = ["AndroidResourceOutputsTest.java"],
    deps = [
        "//src/tools/android/java/com/google/devtools/build/android:android_builder_lib",
        "//third_party:jimfs",
        "//third_party:junit4",
        "//third_party:truth",
    ],
)

java_test(
    name = "RClassGeneratorActionTest",
    size = "small",
    srcs = ["RClassGeneratorActionTest.java"],
    deps = [
        "//src/tools/android/java/com/google/devtools/build/android:android_builder_lib",
        "//third_party:guava",
        "//third_party:junit4",
        "//third_party:truth",
    ],
)

java_test(
    name = "AaptCommandBuilderTest",
    size = "small",
    srcs = ["AaptCommandBuilderTest.java"],
    deps = [
        "//src/tools/android/java/com/google/devtools/build/android:android_builder_lib",
        "//third_party:android_common_25_0_0",
        "//third_party:guava",
        "//third_party:guava-testlib",
        "//third_party:junit4",
        "//third_party:truth",
    ],
)

java_test(
    name = "AndroidDataMergerTest",
    size = "small",
    srcs = ["AndroidDataMergerTest.java"],
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
    name = "AndroidDataSerializerAndDeserializerTest",
    size = "small",
    srcs = ["AndroidDataSerializerAndDeserializerTest.java"],
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
    name = "AndroidDataWriterTest",
    size = "small",
    srcs = ["AndroidDataWriterTest.java"],
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
    size = "small",
    srcs = ["AndroidResourceClassWriterTest.java"],
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
    name = "DataResourceXmlTest",
    size = "small",
    srcs = ["DataResourceXmlTest.java"],
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
    name = "DensitySpecificManifestProcessorTest",
    size = "small",
    srcs = ["DensitySpecificManifestProcessorTest.java"],
    deps = [
        "//src/tools/android/java/com/google/devtools/build/android:android_builder_lib",
        "//third_party:guava",
        "//third_party:jimfs",
        "//third_party:junit4",
        "//third_party:truth",
    ],
)

java_test(
    name = "DensitySpecificResourceFilterTest",
    size = "small",
    srcs = ["DensitySpecificResourceFilterTest.java"],
    deps = [
        "//src/tools/android/java/com/google/devtools/build/android:android_builder_lib",
        "//third_party:guava",
        "//third_party:junit4",
        "//third_party:truth",
    ],
)

java_test(
    name = "DependencyAndroidDataTest",
    size = "small",
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
    size = "small",
    srcs = ["ManifestMergerActionTest.java"],
    data = [
        "//src/test/java/com/google/devtools/build/android/testing/manifestmerge:test_data",
    ],
    jvm_flags = [
        "-Dtestdatadir=io_bazel/src/test/java/com/google/devtools/build/android/testing/manifestmerge",
    ],
    deps = [
        ":test_utils",
        "//src/tools/android/java/com/google/devtools/build/android:android_builder_lib",
        "//third_party:guava",
        "//third_party:jsr305",
        "//third_party:junit4",
        "//third_party:truth",
    ],
)

java_test(
    name = "ParsedAndroidDataTest",
    size = "small",
    srcs = ["ParsedAndroidDataTest.java"],
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
    size = "small",
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
    size = "small",
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
    size = "small",
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
    size = "small",
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
