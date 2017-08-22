# Description:
#   A devtools-common options parser.
#   Open-sourced as part of Bazel.
package(default_visibility = ["//src:__subpackages__"])

java_library(
    name = "invocation_policy",
    srcs = [
        "InvocationPolicyEnforcer.java",
        "InvocationPolicyParser.java",
    ],
    deps = [
        ":options",
        "//src/main/protobuf:invocation_policy_java_proto",
        "//third_party:guava",
        "//third_party:jsr305",
        "//third_party/protobuf:protobuf_java",
    ],
)

java_library(
    name = "options",
    exported_plugins = [":options_preprocessor"],
    exports = [":options_internal"],
)

java_plugin(
    name = "options_preprocessor",
    srcs = [
        "OptionProcessor.java",
    ],
    processor_class = "com.google.devtools.common.options.OptionProcessor",
    deps = [
        ":options_internal",
    ],
)

java_library(
    name = "options_internal",
    srcs = glob(
        ["*.java"],
        exclude = [
            "InvocationPolicyEnforcer.java",
            "InvocationPolicyParser.java",
        ],
    ),
    deps = [
        "//third_party:guava",
        "//third_party:jsr305",
    ],
)

filegroup(
    name = "srcs",
    testonly = 0,  # All srcs should be not test only, overwrite package default.
    srcs = glob(["**"]) + ["//src/main/java/com/google/devtools/common/options/testing:srcs"],
)
