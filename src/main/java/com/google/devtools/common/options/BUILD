load("@rules_java//java:defs.bzl", "java_library")

# Description:
#   A devtools-common options parser.
#   Open-sourced as part of Bazel.
package(
    default_applicable_licenses = ["//:license"],
    default_visibility = ["//src:__subpackages__"],
)

filegroup(
    name = "srcs",
    srcs = glob(["**"]) + [
        "//src/main/java/com/google/devtools/common/options/processor:srcs",
        "//src/main/java/com/google/devtools/common/options/testing:srcs",
    ],
    visibility = ["//src:__subpackages__"],
)

java_library(
    name = "invocation_policy",
    srcs = [
        "InvocationPolicyEnforcer.java",
        "InvocationPolicyParser.java",
    ],
    deps = [
        ":options_internal",
        "//src/main/protobuf:invocation_policy_java_proto",
        "//third_party/java/flogger",
        "//third_party/java/guava:base",
        "//third_party/java/guava:collect",
        "//third_party/java/guava:io",
        "//third_party/java/jsr305_annotations",
        "//third_party/java/protobuf",
    ],
)

java_library(
    name = "options",
    exported_plugins = [
        "//src/main/java/com/google/devtools/common/options/processor:options_preprocessor",
        "//src/main/java/com/google/devtools/common/options/processor:options_class_processor",
    ],
    exports = [":options_internal"],
)

java_library(
    name = "options_internal",
    srcs = glob(
        ["*.java"],
        exclude = [
            "HtmlUtils.java",
            "InvocationPolicyEnforcer.java",
            "InvocationPolicyParser.java",
        ],
    ),
    deps = [
        "//src/main/java/com/google/devtools/build/lib/skybridge:skybridge_interface",
        "//src/main/java/com/google/devtools/build/lib/util:pair",
        "//src/main/java/com/google/devtools/build/lib/util:string_encoding",
        "//src/main/java/com/google/devtools/build/lib/util/regex:regex_util",
        "//src/main/java/net/starlark/java/spelling",
        "//third_party/java/auto:auto_value",
        "//third_party/java/caffeine",
        "//third_party/java/guava:annotations",
        "//third_party/java/guava:base",
        "//third_party/java/guava:collect",
        "//third_party/java/guava:primitives",
        "//third_party/java/guava:reflect",
        "//third_party/java/jsr305_annotations",
    ],
)

java_library(
    name = "html_utils",
    srcs = ["HtmlUtils.java"],
    deps = [
        ":options_internal",
        "//src/main/java/com/google/devtools/build/lib/util:markdown",
        "//third_party/java/guava:base",
        "//third_party/java/guava:collect",
        "//third_party/java/guava:escape",
    ],
)
