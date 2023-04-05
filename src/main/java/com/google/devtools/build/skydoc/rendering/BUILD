load("@rules_java//java:defs.bzl", "java_library")

package(
    default_applicable_licenses = ["//:license"],
    default_visibility = ["//src:__subpackages__"],
)

filegroup(
    name = "srcs",
    srcs = glob(["**"]),
    visibility = ["//src:__subpackages__"],
)

# TODO(arostovtsev): move function_util to rules/starlarkdocextract when we remove legacy Stardoc.
java_library(
    name = "function_util",
    srcs = [
        "DocstringParseException.java",
        "FunctionUtil.java",
    ],
    deps = [
        "//src/main/java/com/google/devtools/build/skydoc/rendering/proto:stardoc_output_java_proto",
        "//src/main/java/net/starlark/java/eval",
        "//src/main/java/net/starlark/java/syntax",
        "//src/tools/starlark/java/com/google/devtools/starlark/common",
        "//third_party:guava",
        "//third_party:jsr305",
    ],
)

java_library(
    name = "rendering",
    srcs = glob(
        ["*.java"],
        exclude = [
            "DocstringParseException.java",
            "FunctionUtil.java",
        ],
    ),
    deps = [
        ":function_util",
        "//src/main/java/com/google/devtools/build/skydoc/rendering/proto:stardoc_output_java_proto",
        "//src/main/java/net/starlark/java/eval",
        "//src/main/java/net/starlark/java/syntax",
        "//third_party:apache_velocity",
        "//third_party:guava",
    ],
)
