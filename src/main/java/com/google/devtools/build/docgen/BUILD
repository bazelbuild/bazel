load("@rules_java//java:defs.bzl", "java_binary", "java_library")

# Description:
#   Documentation generator for Bazel
package(
    default_applicable_licenses = ["//:license"],
    default_visibility = ["//src:__subpackages__"],
)

filegroup(
    name = "srcs",
    srcs = glob(["**"]) + [
        "//src/main/java/com/google/devtools/build/docgen/annot:srcs",
    ],
)

java_library(
    name = "docgen_javalib",
    srcs = glob(["**/*.java"]),
    resources = [":template_files"],
    runtime_deps = [
        "//src/main/java/com/google/devtools/build/lib/bazel:main",
    ],
    deps = [
        "//src/main/java/com/google/devtools/build/docgen/annot",
        "//src/main/java/com/google/devtools/build/lib/analysis:analysis_cluster",
        "//src/main/java/com/google/devtools/build/lib/cmdline",
        "//src/main/java/com/google/devtools/build/lib/collect/nestedset",
        "//src/main/java/com/google/devtools/build/lib/concurrent:thread_safety",
        "//src/main/java/com/google/devtools/build/lib/packages",
        "//src/main/java/com/google/devtools/build/lib/starlarkdocextract:labelrenderer",
        "//src/main/java/com/google/devtools/build/lib/util:classpath",
        "//src/main/java/com/google/devtools/build/lib/util:filetype",
        "//src/main/java/com/google/devtools/common/options",
        "//src/main/java/net/starlark/java/annot",
        "//src/main/java/net/starlark/java/eval",
        "//src/main/java/net/starlark/java/syntax",
        "//src/main/protobuf:builtin_java_proto",
        "//src/main/protobuf:stardoc_output_java_proto",
        "//third_party/java/flogger",
        "//third_party/java/gson",
        "//third_party/java/guava:annotations",
        "//third_party/java/guava:base",
        "//third_party/java/guava:collect",
        "//third_party/java/guava:escape",
        "//third_party/java/guava:io",
        "//third_party/java/jakarta_velocity",
        "//third_party/java/jsr305_annotations",
        "//third_party/java/protobuf",
    ],
)

java_binary(
    name = "docgen_bin",
    srcs = ["BuildEncyclopediaGenerator.java"],
    main_class = "com.google.devtools.build.docgen.BuildEncyclopediaGenerator",
    deps = [
        ":docgen_javalib",
        "//src/main/java/com/google/devtools/build/lib/analysis:analysis_cluster",
        "//src/main/java/com/google/devtools/common/options",
    ],
)

java_binary(
    name = "skydoc_bin",
    srcs = ["StarlarkDocumentationGenerator.java"],
    main_class = "com.google.devtools.build.docgen.StarlarkDocumentationGenerator",
    deps = [
        ":docgen_javalib",
        "//src/main/java/com/google/devtools/common/options",
    ],
)

java_binary(
    name = "api_exporter",
    srcs = ["ApiExporter.java"],
    main_class = "com.google.devtools.build.docgen.ApiExporter",
    deps = [
        ":docgen_javalib",
        "//src/main/java/com/google/devtools/common/options",
        "//src/main/java/net/starlark/java/annot",
        "//src/main/java/net/starlark/java/eval",
        "//src/main/protobuf:builtin_java_proto",
        "//third_party/java/guava:collect",
    ],
)

filegroup(
    name = "template_files",
    srcs = glob([
        "templates/**/*.html",
        "templates/**/*.vm",
    ]),
    visibility = ["//visibility:private"],
)

filegroup(
    name = "bazel_link_map",
    srcs = ["bazel_link_map.json"],
    visibility = ["//src:__subpackages__"],
)
