load("@rules_java//java:defs.bzl", "java_library")

package(
    default_visibility = ["//src:__subpackages__"],
)

filegroup(
    name = "srcs",
    srcs = glob(["*"]),
    visibility = ["//src:__subpackages__"],
)

java_library(
    name = "pipeline",
    srcs = ["NinjaPipeline.java"],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/bazel/rules/ninja/file",
        "//src/main/java/com/google/devtools/build/lib/bazel/rules/ninja/parser:parser_impl",
        "//src/main/java/com/google/devtools/build/lib/vfs:pathfragment",
        "//third_party:guava",
    ],
)

java_library(
    name = "pipeline_impl",
    srcs = ["NinjaPipelineImpl.java"],
    deps = [
        ":pipeline",
        "//src/main/java/com/google/devtools/build/lib/bazel/rules/ninja/file",
        "//src/main/java/com/google/devtools/build/lib/bazel/rules/ninja/lexer",
        "//src/main/java/com/google/devtools/build/lib/bazel/rules/ninja/parser",
        "//src/main/java/com/google/devtools/build/lib/bazel/rules/ninja/parser:parser_impl",
        "//src/main/java/com/google/devtools/build/lib/concurrent",
        "//src/main/java/com/google/devtools/build/lib/util",
        "//src/main/java/com/google/devtools/build/lib/vfs",
        "//src/main/java/com/google/devtools/build/lib/vfs:pathfragment",
        "//third_party:guava",
    ],
)
