load("@rules_java//java:defs.bzl", "java_library")

package(
    default_applicable_licenses = ["//:license"],
    default_visibility = ["//src:__subpackages__"],
)

filegroup(
    name = "srcs",
    srcs = glob(["*"]),
    visibility = ["//src:__subpackages__"],
)

java_library(
    name = "python",
    srcs = glob(["*.java"]),
    deps = [
        "//src/main/java/com/google/devtools/build/docgen/annot",
        "//src/main/java/com/google/devtools/build/lib/analysis:analysis_cluster",
        "//src/main/java/com/google/devtools/build/lib/analysis/config:build_options",
        "//src/main/java/com/google/devtools/build/lib/analysis/config:core_option_converters",
        "//src/main/java/com/google/devtools/build/lib/analysis/config:fragment",
        "//src/main/java/com/google/devtools/build/lib/analysis/config:fragment_options",
        "//src/main/java/com/google/devtools/build/lib/analysis/config:invalid_configuration_exception",
        "//src/main/java/com/google/devtools/build/lib/analysis/starlark/annotations",
        "//src/main/java/com/google/devtools/build/lib/cmdline",
        "//src/main/java/com/google/devtools/build/lib/concurrent:thread_safety",
        "//src/main/java/com/google/devtools/build/lib/events",
        "//src/main/java/com/google/devtools/build/lib/rules/python",
        "//src/main/java/com/google/devtools/build/lib/skyframe/serialization/autocodec:serialization-constant",
        "//src/main/java/com/google/devtools/build/lib/vfs:pathfragment",
        "//src/main/java/com/google/devtools/common/options",
        "//src/main/java/net/starlark/java/annot",
    ],
)
