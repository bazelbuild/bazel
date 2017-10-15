# Description:
#   A preprocessor for the annotations used in the options parser.
package(default_visibility = ["//src:__subpackages__"])

licenses(["notice"])  # Apache 2.0

filegroup(
    name = "srcs",
    srcs = glob(
        ["**"],
    ),
    visibility = ["//src/main/java/com/google/devtools/common/options:__pkg__"],
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
