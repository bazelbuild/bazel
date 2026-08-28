load("@rules_java//java:defs.bzl", "java_library", "java_test")

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
        "//src/main/java/com/google/devtools/build/lib/runtime:blaze_server_startup_options",
        "//src/main/java/com/google/devtools/common/options",
        "//src/main/java/com/google/devtools/common/options:invocation_policy",
        "//src/main/protobuf:invocation_policy_java_proto",
        "//third_party/java/guava:collect",
        "//third_party/java/guava:io",
        "//third_party/java/junit",
    ],
)

java_test(
    name = "AllTests",
    srcs = glob(
        ["*.java"],
        exclude = [
            "TestOptions.java",
            "InvocationPolicyEnforcerTestBase.java",
        ],
    ),
    deps = [
        ":testutils",
        "//src/main/java/com/google/devtools/build/lib/util:classpath",
        "//src/main/java/com/google/devtools/build/lib/util:string_encoding",
        "//src/main/java/com/google/devtools/common/options",
        "//src/main/java/com/google/devtools/common/options:html_utils",
        "//src/main/java/com/google/devtools/common/options:invocation_policy",
        "//src/main/java/com/google/devtools/common/options/testing",
        "//src/main/protobuf:invocation_policy_java_proto",
        "//src/main/protobuf:option_filters_java_proto",
        "//src/test/java/com/google/devtools/build/lib/testutil:TestSuite",
        "//third_party/java/flogger",
        "//third_party/java/guava:collect",
        "//third_party/java/guava:escape",
        "//third_party/java/guava:htmlescapers",
        "//third_party/java/guava:reflect",
        "//third_party/java/guava:testing",
        "//third_party/java/jimfs",
        "//third_party/java/jsr305_annotations",
        "//third_party/java/junit",
        "//third_party/java/testparameterinjector",
        "//third_party/java/truth",
    ],
)
