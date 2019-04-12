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
    exported_plugins = ["//src/main/java/com/google/devtools/common/options/processor:options_preprocessor"],
    exports = [":options_internal"],
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
        "//third_party:auto_value",
        "//third_party:guava",
        "//third_party:jsr305",
    ],
)

filegroup(
    name = "srcs",
    srcs = glob(["**"]) + [
        "//src/main/java/com/google/devtools/common/options/testing:srcs",
        "//src/main/java/com/google/devtools/common/options/processor:srcs",
    ],
)
