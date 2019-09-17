# Description: Bazel's Starlark interpreter
#
# This BUILD file defines two java_library rules that
# define a single Java package in two parts,
# the Starlark parser and evaluator.
# Eventually the Java package will be split in two,
# but splitting only the build rules helps enforce
# that only evaluator -> parser dependencies are permitted,
# without the churn to clients of renaming classes one at a time
# as they move between two different Java packages.
#
# Until the split, clients should depend only on
# //third_party/bazel/src/main/java/com/google/devtools/build/lib:syntax,
# which is a facade for the complete package.

load("@rules_java//java:defs.bzl", "java_library")

package(default_visibility = ["//src:__subpackages__"])

licenses(["notice"])  # Apache 2.0

filegroup(
    name = "srcs",
    srcs = glob(["**"]),
    visibility = ["//src/main/java/com/google/devtools/build/lib:__pkg__"],
)

# This should eventually include all AST nodes plus the lexer and parser.
FRONTEND_FILES = [
    "LineNumberTable.java",
    "ParserInputSource.java",
    "SkylarkImport.java",
    "StarlarkSemantics.java",
    "Token.java",
    "TokenKind.java",
]

# The Starlark frontend (syntax, scanner and parser)
java_library(
    name = "frontend",
    srcs = FRONTEND_FILES,
    deps = [
        "//src/main/java/com/google/devtools/build/lib:events",
        "//src/main/java/com/google/devtools/build/lib:skylarkinterface",
        "//src/main/java/com/google/devtools/build/lib:util",
        "//src/main/java/com/google/devtools/build/lib/cmdline",
        "//src/main/java/com/google/devtools/build/lib/collect/compacthashset",
        "//src/main/java/com/google/devtools/build/lib/concurrent",
        "//src/main/java/com/google/devtools/build/lib/skyframe/serialization/autocodec",
        "//src/main/java/com/google/devtools/build/lib/vfs:pathfragment",
        "//third_party:auto_value",
        "//third_party:guava",
        "//third_party:jsr305",
        "//third_party/protobuf:protobuf_java",
    ],
)

# The Starlark evaluator
#
# TODO(adonovan): migrate frontend classes to :syntax as we break dependencies.
# Node is the main blocker.
java_library(
    name = "evaluator",
    srcs = glob(
        ["*.java"],
        exclude = FRONTEND_FILES,
    ),
    deps = [
        ":frontend",
        "//src/main/java/com/google/devtools/build/lib:events",
        "//src/main/java/com/google/devtools/build/lib:skylarkinterface",
        "//src/main/java/com/google/devtools/build/lib:string_util",
        "//src/main/java/com/google/devtools/build/lib:util",
        "//src/main/java/com/google/devtools/build/lib/cmdline",
        "//src/main/java/com/google/devtools/build/lib/collect/compacthashset",
        "//src/main/java/com/google/devtools/build/lib/collect/nestedset",
        "//src/main/java/com/google/devtools/build/lib/concurrent",
        "//src/main/java/com/google/devtools/build/lib/profiler",
        "//src/main/java/com/google/devtools/build/lib/skyframe/serialization/autocodec",
        "//src/main/java/com/google/devtools/build/lib/vfs:pathfragment",
        "//third_party:auto_value",
        "//third_party:guava",
        "//third_party:jsr305",
        "//third_party/protobuf:protobuf_java",
    ],
)

# TODO(adonovan): merge this into :frontend and eliminate.
# Requires that lib.skylarkinterface depend on :frontend.
java_library(
    name = "semantics",
    srcs = ["StarlarkSemantics.java"],
    deps = [
        "//third_party:auto_value",
        "//third_party:guava",
    ],
)
