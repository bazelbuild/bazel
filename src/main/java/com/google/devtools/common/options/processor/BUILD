load("@rules_java//java:defs.bzl", "java_library", "java_plugin")

# Description:
#   A preprocessor for the annotations used in the options parser.
package(
    default_applicable_licenses = ["//:license"],
    default_visibility = ["//src:__subpackages__"],
)

filegroup(
    name = "srcs",
    srcs = glob(["**"]),
    visibility = ["//src:__subpackages__"],
)

java_plugin(
    name = "options_preprocessor",
    processor_class = "com.google.devtools.common.options.processor.OptionProcessor",
    deps = [":options_preprocessor_lib"],
)

java_library(
    name = "options_preprocessor_lib",
    srcs = glob(["*.java"]),
    deps = [
        "//src/main/java/com/google/devtools/common/options:options_internal",
        "//third_party:guava",
    ],
)
