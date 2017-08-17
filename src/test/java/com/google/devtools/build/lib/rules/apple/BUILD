filegroup(
    name = "srcs",
    srcs = glob(["**"]),
    visibility = ["//src/test/java/com/google/devtools/build/lib:__pkg__"],
)

SERIALIZATION_TEST_SRCS = ["AppleConfigurationSerializationTest.java"]

java_library(
    name = "AppleConfigurationSerializationTest_lib",
    testonly = 1,
    srcs = SERIALIZATION_TEST_SRCS,
    deps = [
        "//src/main/java/com/google/devtools/build/lib:build-base",
        "//src/main/java/com/google/devtools/build/lib/cmdline",
        "//src/main/java/com/google/devtools/build/lib/rules/apple",
        "//src/main/java/com/google/devtools/build/lib/skyframe/serialization",
        "//src/main/java/com/google/devtools/common/options",
        "//src/test/java/com/google/devtools/build/lib/skyframe/serialization:serialization-test-base",
        "//third_party:guava",
        "//third_party:junit4",
        "//third_party/protobuf:protobuf_java",
    ],
)

java_test(
    name = "AppleConfigurationSerializationTest",
    test_class = "com.google.devtools.build.lib.AllTests",
    runtime_deps = [
        ":AppleConfigurationSerializationTest_lib",
        "//src/test/java/com/google/devtools/build/lib:test_runner",
    ],
)

java_library(
    name = "AppleRulesTests_lib",
    srcs = glob(
        ["*.java"],
        exclude = SERIALIZATION_TEST_SRCS,
    ),
    deps = [
        "//src/main/java/com/google/devtools/build/lib:build-base",
        "//src/main/java/com/google/devtools/build/lib/rules/apple",
        "//src/test/java/com/google/devtools/build/lib:analysis_testutil",
        "//third_party:guava",
        "//third_party:guava-testlib",
        "//third_party:jsr305",
        "//third_party:junit4",
        "//third_party:truth",
    ],
)

java_test(
    name = "AppleRulesTests",
    test_class = "com.google.devtools.build.lib.AllTests",
    runtime_deps = [
        ":AppleRulesTests_lib",
        "//src/main/java/com/google/devtools/build/lib:util",
        "//src/test/java/com/google/devtools/build/lib:testutil",
    ],
)
