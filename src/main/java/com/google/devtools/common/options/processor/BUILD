load("@rules_java//java:defs.bzl", "java_plugin")

# Description:
#   A preprocessor for the annotations used in the options parser.
package(default_visibility = ["//src:__subpackages__"])

filegroup(
    name = "srcs",
    srcs = glob(["**"]),
    visibility = ["//src:__subpackages__"],
)

java_plugin(
    name = "options_preprocessor",
    srcs = glob(["*.java"]),
    processor_class = "com.google.devtools.common.options.processor.OptionProcessor",
    deps = [
        "//src/main/java/com/google/devtools/common/options:options_internal",
        "//third_party:guava",
    ],
)
