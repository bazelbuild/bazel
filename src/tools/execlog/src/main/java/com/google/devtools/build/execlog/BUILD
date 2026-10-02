load("@rules_java//java:defs.bzl", "java_library")

package(
    default_applicable_licenses = ["//:license"],
    default_visibility = ["//:__subpackages__"],
)

filegroup(
    name = "srcs",
    srcs = glob(["**"]),
    visibility = ["//src/tools/execlog:__pkg__"],
)

java_library(
    name = "parser",
    srcs = [
        "ExecLogParser.java",
        "ParserOptions.java",
    ],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/exec:spawn_log_reconstructor",
        "//src/main/java/com/google/devtools/build/lib/util/io:io-proto",
        "//src/main/java/com/google/devtools/common/options",
        "//src/main/protobuf:spawn_java_proto",
        "//third_party/java/guava:annotations",
        "//third_party/java/jsr305_annotations",
    ],
)

java_library(
    name = "converter",
    srcs = [
        "ConverterOptions.java",
        "ExecLogConverter.java",
    ],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/exec:spawn_log_reconstructor",
        "//src/main/java/com/google/devtools/build/lib/exec:stable_sort",
        "//src/main/java/com/google/devtools/build/lib/util/io:io-proto",
        "//src/main/java/com/google/devtools/common/options",
        "//src/main/protobuf:spawn_java_proto",
        "//third_party/java/guava:base",
        "//third_party/java/guava:collect",
    ],
)
