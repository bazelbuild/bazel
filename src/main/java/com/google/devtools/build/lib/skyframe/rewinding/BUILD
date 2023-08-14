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
    name = "action_rewound_event",
    srcs = ["ActionRewoundEvent.java"],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/actions",
        "//src/main/java/com/google/devtools/build/lib/events",
    ],
)

java_library(
    name = "rewinding",
    srcs = [
        "ActionRewindStrategy.java",
        "ActionRewindingStats.java",
    ],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/actions",
        "//src/main/java/com/google/devtools/build/lib/actions:action_lookup_data",
        "//src/main/java/com/google/devtools/build/lib/actions:artifacts",
        "//src/main/java/com/google/devtools/build/lib/bugreport",
        "//src/main/java/com/google/devtools/build/lib/collect/nestedset:artifact_nested_set_key",
        "//src/main/java/com/google/devtools/build/lib/events",
        "//src/main/java/com/google/devtools/build/lib/skyframe:action_utils",
        "//src/main/java/com/google/devtools/build/lib/skyframe:artifact_function",
        "//src/main/java/com/google/devtools/build/lib/skyframe:skyframe_aware_action",
        "//src/main/java/com/google/devtools/build/lib/skyframe/proto:action_rewind_event_java_proto",
        "//src/main/java/com/google/devtools/build/lib/util:detailed_exit_code",
        "//src/main/java/com/google/devtools/build/skyframe",
        "//src/main/java/com/google/devtools/build/skyframe:skyframe-objects",
        "//src/main/protobuf:failure_details_java_proto",
        "//third_party:auto_value",
        "//third_party:flogger",
        "//third_party:guava",
        "//third_party:jsr305",
    ],
)

java_library(
    name = "rewindable_graph_inconsistency_receiver",
    srcs = ["RewindableGraphInconsistencyReceiver.java"],
    deps = [
        ":rewinding_inconsistency_utils",
        "//src/main/java/com/google/devtools/build/lib/skyframe:node_dropping_inconsistency_receiver",
        "//src/main/java/com/google/devtools/build/lib/util:string",
        "//src/main/java/com/google/devtools/build/skyframe",
        "//src/main/java/com/google/devtools/build/skyframe:graph_inconsistency_java_proto",
        "//src/main/java/com/google/devtools/build/skyframe:skyframe-objects",
        "//third_party:flogger",
        "//third_party:guava",
        "//third_party:jsr305",
    ],
)

java_library(
    name = "rewinding_inconsistency_utils",
    srcs = ["RewindingInconsistencyUtils.java"],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/actions:action_lookup_data",
        "//src/main/java/com/google/devtools/build/lib/actions:artifacts",
        "//src/main/java/com/google/devtools/build/lib/collect/nestedset:artifact_nested_set_key",
        "//src/main/java/com/google/devtools/build/lib/skyframe:action_template_expansion_value",
        "//src/main/java/com/google/devtools/build/lib/skyframe:aspect_completion_value",
        "//src/main/java/com/google/devtools/build/lib/skyframe:fileset_entry_key",
        "//src/main/java/com/google/devtools/build/lib/skyframe:recursive_filesystem_traversal",
        "//src/main/java/com/google/devtools/build/lib/skyframe:target_completion_value",
        "//src/main/java/com/google/devtools/build/lib/skyframe:test_completion_value",
        "//src/main/java/com/google/devtools/build/skyframe:skyframe-objects",
    ],
)
