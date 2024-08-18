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

java_library(
    name = "labelrenderer",
    srcs = ["LabelRenderer.java"],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/cmdline",
        "//src/main/java/net/starlark/java/eval",
    ],
)

java_library(
    name = "starlarkfunctioninfoextractor",
    srcs = [
        "DocstringParseException.java",
        "StarlarkFunctionInfoExtractor.java",
    ],
    deps = [
        ":labelrenderer",
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
    name = "moduleinfoextractor",
    srcs = ["ModuleInfoExtractor.java"],
    deps = [
        ":labelrenderer",
        ":starlarkfunctioninfoextractor",
        "//src/main/java/com/google/devtools/build/lib/analysis:analysis_cluster",
        "//src/main/java/com/google/devtools/build/lib/bazel/bzlmod:module_extension",
        "//src/main/java/com/google/devtools/build/lib/bazel/repository/starlark",
        "//src/main/java/com/google/devtools/build/lib/cmdline",
        "//src/main/java/com/google/devtools/build/lib/packages",
        "//src/main/java/net/starlark/java/eval",
        "//src/main/protobuf:stardoc_output_java_proto",
        "//third_party:guava",
    ],
)
