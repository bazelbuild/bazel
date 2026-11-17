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
        "ExecGraphParser.java",
        "ParserOptions.java",
    ],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/util/io:io-proto",
        "//src/main/java/com/google/devtools/common/options",
        "//src/main/protobuf:execution_graph_java_proto",
        "//third_party:guava",
        "//third_party:jsr305",
        "@zstd-jni",
    ],
)
