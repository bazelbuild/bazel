# Description:
#   Skydoc is a documentation generator for Starlark files.
#
#   It is currently experimental and in development.
#
# Usage:
# skydoc <target_file> <output_file>

load("@rules_java//java:defs.bzl", "java_binary", "java_library")

package(
    default_applicable_licenses = ["//:license"],
    default_visibility = ["//src:__subpackages__"],
)

filegroup(
    name = "srcs",
    srcs = glob(["**"]) + [
        "//src/main/java/com/google/devtools/build/skydoc/fakebuildapi:srcs",
        "//src/main/java/com/google/devtools/build/skydoc/rendering:srcs",
        "//src/main/java/com/google/devtools/build/skydoc/rendering/proto:srcs",
    ],
    visibility = ["//src:__subpackages__"],
)

java_binary(
    name = "skydoc",
    main_class = "com.google.devtools.build.skydoc.SkydocMain",
    visibility = ["//visibility:public"],
    runtime_deps = [
        ":skydoc_lib",
    ],
)

java_library(
    name = "skydoc_lib",
    srcs = glob(["*.java"]),
    deps = [
        "//src/main/java/com/google/devtools/build/lib/cmdline",
        "//src/main/java/com/google/devtools/build/lib/collect/nestedset",
        "//src/main/java/com/google/devtools/build/lib/events",
        "//src/main/java/com/google/devtools/build/lib/packages/semantics",
        "//src/main/java/com/google/devtools/build/skydoc/fakebuildapi",
        "//src/main/java/com/google/devtools/build/skydoc/rendering",
        "//src/main/java/com/google/devtools/build/skydoc/rendering:rendering_util",
        "//src/main/java/com/google/devtools/build/skydoc/rendering/proto:stardoc_output_java_proto",
        "//src/main/java/com/google/devtools/common/options",
        "//src/main/java/net/starlark/java/annot",
        "//src/main/java/net/starlark/java/eval",
        "//src/main/java/net/starlark/java/lib/json",
        "//src/main/java/net/starlark/java/syntax",
        "//third_party:guava",
        "//tools/java/runfiles",
        "//tools/java/runfiles:runfiles_for_stardoc",
    ],
)
