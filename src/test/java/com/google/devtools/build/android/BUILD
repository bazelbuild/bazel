load("@rules_java//java:defs.bzl", "java_library", "java_test")

filegroup(
    name = "srcs",
    srcs = glob(["**"]) + [
        "//src/test/java/com/google/devtools/build/android/desugar:srcs",
        "//src/test/java/com/google/devtools/build/android/dexer:srcs",
        "//src/test/java/com/google/devtools/build/android/idlclass:srcs",
        "//src/test/java/com/google/devtools/build/android/junctions:srcs",
        "//src/test/java/com/google/devtools/build/android/r8:srcs",
        "//src/test/java/com/google/devtools/build/android/resources:srcs",
        "//src/test/java/com/google/devtools/build/android/testing/manifestmerge:srcs",
        "//src/test/java/com/google/devtools/build/android/xml:srcs",
        "//src/test/java/com/google/devtools/build/android/ziputils:srcs",
    ],
    visibility = ["//src:__pkg__"],
)

java_test(
    name = "AarGeneratorActionTest",
    size = "small",
    srcs = ["AarGeneratorActionTest.java"],
    deps = [
        ":test_utils",
        "//src/java_tools/singlejar/java/com/google/devtools/build/zip",
        "//src/tools/android/java/com/google/devtools/build/android:android_builder_lib",
        "//src/tools/android/java/com/google/devtools/build/android:android_common_30_1_3",
        "//src/tools/android/java/com/google/devtools/build/android:android_options_utils",
        "//third_party:guava",
        "//third_party:junit4",
        "//third_party:truth",
        "//third_party/java/jcommander",
    ],
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
    name = "AndroidManifestTest",
    size = "small",
    srcs = ["AndroidManifestTest.java"],
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
        "//src/tools/android/java/com/google/devtools/build/android:android_common_30_1_3",
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
        "//src/tools/android/java/com/google/devtools/build/android:android_common_30_1_3",
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
        "//src/tools/android/java/com/google/devtools/build/android:android_common_30_1_3",
        "//third_party:guava",
        "//third_party:jimfs",
        "//third_party:junit4",
        "//third_party:truth",
    ],
)

java_library(
    name = "test_options",
    srcs = ["TestOptions.java"],
    deps = [
        "@maven//:com_beust_jcommander",
    ],
)

java_test(
    name = "AndroidOptionsUtilsTest",
    size = "small",
    srcs = ["AndroidOptionsUtilsTest.java"],
    deps = [
        ":test_options",
        "//src/tools/android/java/com/google/devtools/build/android:android_options_utils",
        "//third_party:junit4",
        "//third_party:truth",
        "@maven//:com_beust_jcommander",
    ],
)

java_test(
    name = "AndroidResourceClassWriterTest",
    size = "small",
    srcs = ["AndroidResourceClassWriterTest.java"],
    deps = [
        ":test_utils",
        "//src/tools/android/java/com/google/devtools/build/android:android_builder_lib",
        "//src/tools/android/java/com/google/devtools/build/android/resources",
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
        "//src/tools/android/java/com/google/devtools/build/android:android_builder_lib",
        "//third_party:guava",
        "//third_party:junit4",
        "//third_party:truth",
        "//third_party/java/jcommander",
    ],
)

java_test(
    name = "DataResourceXmlTest",
    size = "small",
    srcs = ["DataResourceXmlTest.java"],
    deps = [
        ":test_utils",
        "//src/tools/android/java/com/google/devtools/build/android:android_builder_lib",
        "//src/tools/android/java/com/google/devtools/build/android:android_common_30_1_3",
        "//src/tools/android/java/com/google/devtools/build/android/resources",
        "//third_party:guava",
        "//third_party:jimfs",
        "//third_party:junit4",
        "//third_party:truth",
    ],
)

java_test(
    name = "DataSourceTest",
    size = "small",
    srcs = ["DataSourceTest.java"],
    deps = [
        "//src/tools/android/java/com/google/devtools/build/android:android_builder_lib",
        "//third_party:jimfs",
        "//third_party:junit4",
        "//third_party:truth",
    ],
)

java_test(
    name = "DataValueFileTest",
    size = "small",
    srcs = ["DataValueFileTest.java"],
    deps = [
        "//src/tools/android/java/com/google/devtools/build/android:android_builder_lib",
        "//src/tools/android/java/com/google/devtools/build/android/resources",
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
        "//src/tools/android/java/com/google/devtools/build/android:android_common_30_1_3",
        "//src/tools/android/java/com/google/devtools/build/android/resources",
        "//third_party:guava",
        "//third_party:jimfs",
        "//third_party:junit4",
        "//third_party:truth",
    ],
)

java_test(
    name = "PlaceholderRTxtWriterTest",
    size = "small",
    srcs = ["PlaceholderRTxtWriterTest.java"],
    deps = [
        "//src/tools/android/java/com/google/devtools/build/android:android_builder_lib",
        "//src/tools/android/java/com/google/devtools/build/android:android_common_30_1_3",
        "//src/tools/android/java/com/google/devtools/build/android/resources",
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

java_test(
    name = "ZipFilterActionTest",
    size = "small",
    srcs = ["ZipFilterActionTest.java"],
    deps = [
        "//src/java_tools/singlejar/java/com/google/devtools/build/singlejar:libSingleJar",
        "//src/tools/android/java/com/google/devtools/build/android:android_builder_lib",
        "//third_party:guava",
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
        "//src/tools/android/java/com/google/devtools/build/android:android_common_30_1_3",
        "//src/tools/android/java/com/google/devtools/build/android/resources",
        "//third_party:guava",
        "//third_party:truth",
        "@maven//:com_beust_jcommander",
    ],
)
