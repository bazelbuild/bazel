load("@rules_java//java:defs.bzl", "java_library")

package(
    default_visibility = ["//src:__subpackages__"],
)

filegroup(
    name = "srcs",
    srcs = glob(["*"]),
    visibility = ["//src:__subpackages__"],
)

TYPE_SRCS = ["Type.java"]

BUILD_TYPE_SRCS = [
    "BuildType.java",
    "FilesetEntry.java",
    "License.java",
    "SelectorList.java",
    "SelectorValue.java",
    "TriState.java",
]

java_library(
    name = "packages",
    srcs = glob(
        ["*.java"],
        exclude = TYPE_SRCS + BUILD_TYPE_SRCS + [
            "BuilderFactoryForTesting.java",
            "StarlarkSemanticsOptions.java",
        ],
    ),
    deps = [
        ":build_type",
        ":type",
        "//src/main/java/com/google/devtools/build/lib/analysis:config/config_matching_provider",
        "//src/main/java/com/google/devtools/build/lib/analysis:config/transitions/configuration_transition",
        "//src/main/java/com/google/devtools/build/lib/analysis:config/transitions/no_transition",
        "//src/main/java/com/google/devtools/build/lib/analysis:config/transitions/patch_transition",
        "//src/main/java/com/google/devtools/build/lib/analysis:config/transitions/split_transition",
        "//src/main/java/com/google/devtools/build/lib/analysis:config/transitions/transition_factory",
        "//src/main/java/com/google/devtools/build/lib/analysis:rule_definition_context",
        "//src/main/java/com/google/devtools/build/lib/analysis:transitive_info_provider",
        "//src/main/java/com/google/devtools/build/lib/cmdline",
        "//src/main/java/com/google/devtools/build/lib/cmdline:LabelValidator",
        "//src/main/java/com/google/devtools/build/lib/collect",
        "//src/main/java/com/google/devtools/build/lib/collect/nestedset",
        "//src/main/java/com/google/devtools/build/lib/concurrent",
        "//src/main/java/com/google/devtools/build/lib/events",
        "//src/main/java/com/google/devtools/build/lib/profiler",
        "//src/main/java/com/google/devtools/build/lib/skyframe/serialization",
        "//src/main/java/com/google/devtools/build/lib/skyframe/serialization/autocodec",
        "//src/main/java/com/google/devtools/build/lib/skylarkbuildapi",
        "//src/main/java/com/google/devtools/build/lib/skylarkbuildapi/core",
        "//src/main/java/com/google/devtools/build/lib/syntax:evaluator",
        "//src/main/java/com/google/devtools/build/lib/syntax:frontend",
        "//src/main/java/com/google/devtools/build/lib/util",
        "//src/main/java/com/google/devtools/build/lib/util:filetype",
        "//src/main/java/com/google/devtools/build/lib/util:string",
        "//src/main/java/com/google/devtools/build/lib/vfs",
        "//src/main/java/com/google/devtools/build/lib/vfs:pathfragment",
        "//src/main/java/com/google/devtools/build/skyframe:skyframe-objects",
        "//src/main/java/com/google/devtools/common/options",
        "//src/main/java/net/starlark/java/annot",
        "//src/main/java/net/starlark/java/spelling",
        "//src/main/protobuf:build_java_proto",
        "//third_party:auto_value",
        "//third_party:flogger",
        "//third_party:guava",
        "//third_party:jsr305",
        "//third_party/protobuf:protobuf_java",
    ],
)

# TODO(kkress, adonovan) Eliminate this target, it is mostly unecessary. It
# depends on lib:build-base for BlazeDirectories, which it uses for a type
# parameter, but the param is unused.
java_library(
    name = "builder_factory_for_testing",
    srcs = ["BuilderFactoryForTesting.java"],
    deps = [
        ":packages",
        "//src/main/java/com/google/devtools/build/lib/analysis:blaze_directories",
        "//third_party:guava",
    ],
)

# This class is broken out of lib.packages because skydoc is not supposed to depend on lib.packages.
# TODO(adonovan): rethink skydoc's dependency mess.
java_library(
    name = "starlark_semantics_options",
    srcs = ["StarlarkSemanticsOptions.java"],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/concurrent",
        "//src/main/java/com/google/devtools/build/lib/syntax:evaluator",
        "//src/main/java/com/google/devtools/common/options",
        "//third_party:guava",
    ],
)

java_library(
    name = "type",
    srcs = TYPE_SRCS,
    deps = [
        "//src/main/java/com/google/devtools/build/lib/cmdline",
        "//src/main/java/com/google/devtools/build/lib/collect/nestedset",
        "//src/main/java/com/google/devtools/build/lib/skyframe/serialization/autocodec",
        "//src/main/java/com/google/devtools/build/lib/syntax:evaluator",
        "//src/main/java/com/google/devtools/build/lib/util:logging",
        "//src/main/java/com/google/devtools/build/lib/util:string",
        "//third_party:guava",
        "//third_party:jsr305",
    ],
)

java_library(
    name = "build_type",
    srcs = BUILD_TYPE_SRCS,
    deps = [
        ":type",
        "//src/main/java/com/google/devtools/build/lib/cmdline",
        "//src/main/java/com/google/devtools/build/lib/collect/nestedset",
        "//src/main/java/com/google/devtools/build/lib/concurrent",
        "//src/main/java/com/google/devtools/build/lib/skyframe/serialization/autocodec",
        "//src/main/java/com/google/devtools/build/lib/skylarkbuildapi",
        "//src/main/java/com/google/devtools/build/lib/syntax:evaluator",
        "//src/main/java/com/google/devtools/build/lib/syntax:frontend",
        "//src/main/java/com/google/devtools/build/lib/vfs:pathfragment",
        "//src/main/java/net/starlark/java/annot",
        "//third_party:guava",
        "//third_party:jsr305",
    ],
)
