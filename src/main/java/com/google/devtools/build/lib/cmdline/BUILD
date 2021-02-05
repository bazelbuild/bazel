load("@rules_java//java:defs.bzl", "java_library")

package(default_visibility = ["//src:__subpackages__"])

filegroup(
    name = "srcs",
    srcs = glob(["**"]),
    visibility = ["//src:__subpackages__"],
)

# cmdline provides Label and target pattern parsing.
java_library(
    name = "cmdline",
    srcs = [
        "Label.java",
        "LabelConstants.java",
        "LabelSerializationProxy.java",
        "ResolvedTargets.java",
        "TargetParsingException.java",
        "TargetPattern.java",
        "TargetPatternResolver.java",
    ],
    exports = [":cmdline-primitives"],
    deps = [
        ":LabelValidator",
        ":cmdline-primitives",
        "//src/main/java/com/google/devtools/build/docgen/annot",
        "//src/main/java/com/google/devtools/build/lib/actions:commandline_item",
        "//src/main/java/com/google/devtools/build/lib/concurrent",
        "//src/main/java/com/google/devtools/build/lib/packages/semantics",
        "//src/main/java/com/google/devtools/build/lib/skyframe:detailed_exceptions",
        "//src/main/java/com/google/devtools/build/lib/skyframe/serialization/autocodec",
        "//src/main/java/com/google/devtools/build/lib/supplier",
        "//src/main/java/com/google/devtools/build/lib/util:detailed_exit_code",
        "//src/main/java/com/google/devtools/build/lib/util:string",
        "//src/main/java/com/google/devtools/build/lib/vfs:pathfragment",
        "//src/main/java/com/google/devtools/build/skyframe:skyframe-objects",
        "//src/main/java/net/starlark/java/annot",
        "//src/main/java/net/starlark/java/eval",
        "//src/main/protobuf:failure_details_java_proto",
        "//third_party:error_prone_annotations",
        "//third_party:guava",
        "//third_party:jsr305",
    ],
)

# Does not depend on Starlark. But this is no longer relevant because Starlark no longer depends on cmdline.
# TODO(adonovan): merge these packages.
java_library(
    name = "cmdline-primitives",
    srcs = [
        "LabelConstants.java",
        "LabelSyntaxException.java",
        "PackageIdentifier.java",
        "RepositoryName.java",
    ],
    deps = [
        ":LabelValidator",
        "//src/main/java/com/google/devtools/build/lib/concurrent",
        "//src/main/java/com/google/devtools/build/lib/skyframe/serialization/autocodec",
        "//src/main/java/com/google/devtools/build/lib/skyframe/serialization/autocodec:serialization-constant",
        "//src/main/java/com/google/devtools/build/lib/util",
        "//src/main/java/com/google/devtools/build/lib/util:string",
        "//src/main/java/com/google/devtools/build/lib/vfs:pathfragment",
        "//third_party:guava",
        "//third_party:jsr305",
    ],
)

# LabelValidator provides validation of Bazel build labels.
# This is a public API.
# TODO(adonovan): unsplit the lib.cmdline Java package by moving this logic to a subpackage.
java_library(
    name = "LabelValidator",
    srcs = ["LabelValidator.java"],
    deps = [
        "//third_party:guava",
        "//third_party:jsr305",
    ],
)
