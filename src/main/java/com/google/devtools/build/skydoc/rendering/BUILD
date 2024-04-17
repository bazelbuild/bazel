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

# TODO(arostovtsev): move rendering_util to rules/starlarkdocextract when we remove legacy Stardoc.
java_library(
    name = "rendering_util",
    srcs = [
        "DocstringParseException.java",
        "LabelRenderer.java",
        "StarlarkFunctionInfoExtractor.java",
    ],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/cmdline",
        "//src/main/java/net/starlark/java/eval",
        "//src/main/java/net/starlark/java/syntax",
        "//src/main/protobuf:stardoc_output_java_proto",
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
            "LabelRenderer.java",
            "StarlarkFunctionInfoExtractor.java",
        ],
    ),
    deps = [
        ":rendering_util",
        "//src/main/java/net/starlark/java/eval",
        "//src/main/java/net/starlark/java/syntax",
        "//src/main/protobuf:stardoc_output_java_proto",
        "//third_party:error_prone_annotations",
    ],
)
