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
    name = "parser",
    srcs = ["NinjaParser.java"],
    deps = [
        ":parser_impl",
        "//src/main/java/com/google/devtools/build/lib/bazel/rules/ninja/file",
        "//src/main/java/com/google/devtools/build/lib/bazel/rules/ninja/lexer",
        "//src/main/java/com/google/devtools/build/lib/bazel/rules/ninja/pipeline",
        "//src/main/java/com/google/devtools/build/lib/util",
    ],
)

java_library(
    name = "parser_impl",
    srcs = glob(
        ["*.java"],
        exclude = ["NinjaParser.java"],
    ),
    deps = [
        "//src/main/java/com/google/devtools/build/lib/bazel/rules/ninja/file",
        "//src/main/java/com/google/devtools/build/lib/bazel/rules/ninja/lexer",
        "//src/main/java/com/google/devtools/build/lib/collect",
        "//src/main/java/com/google/devtools/build/lib/concurrent",
        "//src/main/java/com/google/devtools/build/lib/util",
        "//src/main/java/com/google/devtools/build/lib/vfs:pathfragment",
        "//third_party:error_prone_annotations",
        "//third_party:guava",
        "//third_party:jsr305",
    ],
)
