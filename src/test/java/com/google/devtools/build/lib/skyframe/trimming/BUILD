# Tests for the trimming support classes.

package(
    default_testonly = 1,
)

filegroup(
    name = "srcs",
    testonly = 0,
    srcs = glob(["**"]),
    visibility = ["//src:__subpackages__"],
)

java_library(
    name = "test_key",
    srcs = ["TestKey.java"],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/skyframe/trimming:trimmed_configuration_cache",
        "//third_party:auto_value",
        "//third_party:guava",
    ],
)

java_test(
    name = "TrimmedConfigurationCacheTests",
    size = "small",
    srcs = [
        "TestKeyTest.java",
        "TrimmedConfigurationCacheTest.java",
    ],
    test_class = "com.google.devtools.build.lib.AllTests",
    runtime_deps = ["//src/test/java/com/google/devtools/build/lib:test_runner"],
    deps = [
        ":test_key",
        "//src/main/java/com/google/devtools/build/lib/skyframe/trimming:trimmed_configuration_cache",
        "//third_party:guava",
        "//third_party:guava-testlib",
        "//third_party:junit4",
        "//third_party:truth",
        "//third_party:truth8",
    ],
)
