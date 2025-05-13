load("@rules_fuzzing//fuzzing:java_defs.bzl", "java_fuzz_test")
load("@rules_java//java:defs.bzl", "java_binary", "java_library", "java_test")

package(
    default_applicable_licenses = ["//:license"],
    default_visibility = ["//src:__subpackages__"],
)

filegroup(
    name = "srcs",
    srcs = glob(["**"]) + [
        "//src/test/java/com/google/devtools/common/options/processor:srcs",
        "//src/test/java/com/google/devtools/common/options/testing:srcs",
    ],
    visibility = ["//src:__subpackages__"],
)

java_library(
    name = "testutils",
    testonly = 1,
    srcs = [
        "InvocationPolicyEnforcerTestBase.java",
        "TestOptions.java",
    ],
    deps = [
        "//src/main/java/com/google/devtools/build/lib:runtime",
        "//src/main/java/com/google/devtools/common/options",
        "//src/main/java/com/google/devtools/common/options:invocation_policy",
        "//src/main/protobuf:invocation_policy_java_proto",
        "//third_party:guava",
        "//third_party:junit4",
    ],
)

java_test(
    name = "AllTests",
    srcs = glob(
        ["*.java"],
        exclude = [
            "RegexPatternOptionBenchmark.java",
            "RegexPatternOptionFuzzTest.java",
            "TestOptions.java",
            "InvocationPolicyEnforcerTestBase.java",
        ],
    ),
    deps = [
        ":testutils",
        "//src/main/java/com/google/devtools/build/lib/util:classpath",
        "//src/main/java/com/google/devtools/common/options",
        "//src/main/java/com/google/devtools/common/options:invocation_policy",
        "//src/main/java/com/google/devtools/common/options/testing",
        "//src/main/protobuf:invocation_policy_java_proto",
        "//src/main/protobuf:option_filters_java_proto",
        "//src/test/java/com/google/devtools/build/lib/testutil:TestSuite",
        "//third_party:flogger",
        "//third_party:guava",
        "//third_party:guava-testlib",
        "//third_party:jimfs",
        "//third_party:jsr305",
        "//third_party:junit4",
        "//third_party:mockito",
        "//third_party:truth",
        "@maven//:com_google_testparameterinjector_test_parameter_injector",
    ],
)

java_binary(
    name = "RegexPatternOptionBenchmark",
    srcs = ["RegexPatternOptionBenchmark.java"],
    main_class = "org.openjdk.jmh.Main",
    deps = [
        "//src/main/java/com/google/devtools/build/lib:runtime",
        "//src/main/java/com/google/devtools/common/options:options_internal",
        "//third_party:jmh",
    ],
)

java_fuzz_test(
    name = "RegexPatternOptionFuzzTest",
    srcs = ["RegexPatternOptionFuzzTest.java"],
    jvm_flags = [
        "-Djazzer.disabled_hooks=com.code_intelligence.jazzer.sanitizers.RegexInjection",
    ],
    tags = ["no_windows"],
    deps = [
        "//src/main/java/com/google/devtools/build/lib:runtime",
        "//src/main/java/com/google/devtools/common/options:options_internal",
    ],
)
