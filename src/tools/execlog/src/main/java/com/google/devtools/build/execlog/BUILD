load("@rules_java//java:defs.bzl", "java_library")

package(
    default_applicable_licenses = ["//:license"],
    default_visibility = ["//:__subpackages__"],
)

filegroup(
    name = "srcs",
    srcs = glob(["**"]),
    visibility = ["//src:__subpackages__"],
)

java_library(
    name = "parser",
    srcs = [
        "ExecLogParser.java",
        "ParserOptions.java",
    ],
    deps = [
        "//src/main/java/com/google/devtools/common/options",
        "//src/main/protobuf:spawn_java_proto",
        "//third_party:guava",
    ],
)

java_library(
    name = "converter",
    srcs = [
        "ConverterOptions.java",
        "ExecLogConverter.java",
    ],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/exec:spawn_log_context_utils",
        "//src/main/java/com/google/devtools/build/lib/util/io:io-proto",
        "//src/main/java/com/google/devtools/common/options",
        "//src/main/protobuf:spawn_java_proto",
        "//third_party:auto_value",
        "//third_party:guava",
    ],
)
