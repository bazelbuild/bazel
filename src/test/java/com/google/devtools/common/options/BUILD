filegroup(
    name = "srcs",
    srcs = glob(["**"]) + [
        "//src/test/java/com/google/devtools/common/options/testing:srcs",
        "//src/test/java/com/google/devtools/common/options/processor:srcs",
    ],
    visibility = ["//src:__pkg__"],
)

java_library(
    name = "testutils",
    testonly = 1,
    srcs = [
        "InvocationPolicyEnforcerTestBase.java",
        "TestOptions.java",
    ],
    visibility = [
        "//src/test/java:__subpackages__",
    ],
    deps = [
        "//src/main/java/com/google/devtools/build/lib:build-base",
        "//src/main/java/com/google/devtools/common/options",
        "//src/main/java/com/google/devtools/common/options:invocation_policy",
        "//src/main/protobuf:invocation_policy_java_proto",
        "//third_party:guava",
        "//third_party:junit4",
    ],
)

java_test(
    name = "options_test",
    srcs = glob(
        ["*.java"],
        exclude = [
            "TestOptions.java",
            "InvocationPolicyEnforcerTestBase.java",
        ],
    ),
    test_class = "com.google.devtools.common.options.AllTests",
    deps = [
        ":testutils",
        "//src/main/java/com/google/devtools/build/lib:build-base",
        "//src/main/java/com/google/devtools/build/lib:util",
        "//src/main/java/com/google/devtools/common/options",
        "//src/main/java/com/google/devtools/common/options:invocation_policy",
        "//src/main/protobuf:invocation_policy_java_proto",
        "//src/main/protobuf:option_filters_java_proto",
        "//src/test/java/com/google/devtools/build/lib:testutil",
        "//third_party:guava",
        "//third_party:guava-testlib",
        "//third_party:jimfs",
        "//third_party:jsr305",
        "//third_party:junit4",
        "//third_party:mockito2",
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
        "//src/test/java/com/google/devtools/common/options/processor:all_windows_tests",
        "//src/test/java/com/google/devtools/common/options/testing:all_windows_tests",
    ],
    visibility = ["//src:__pkg__"],
)
