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
    name = "lost_input_owners",
    srcs = ["LostInputOwners.java"],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/actions:artifacts",
        "//third_party:guava",
    ],
)

java_library(
    name = "rewinding",
    srcs = [
        "ActionRewindException.java",
        "ActionRewindStrategy.java",
        "PostableActionRewindingStats.java",
    ],
    deps = [
        ":action_rewound_event",
        ":lost_input_owners",
        "//src/main/java/com/google/devtools/build/lib/actions",
        "//src/main/java/com/google/devtools/build/lib/actions:action_lookup_data",
        "//src/main/java/com/google/devtools/build/lib/actions:artifacts",
        "//src/main/java/com/google/devtools/build/lib/actions:fileset_output_symlink",
        "//src/main/java/com/google/devtools/build/lib/actions:runfiles_metadata",
        "//src/main/java/com/google/devtools/build/lib/actions:runfiles_tree",
        "//src/main/java/com/google/devtools/build/lib/bugreport",
        "//src/main/java/com/google/devtools/build/lib/clock",
        "//src/main/java/com/google/devtools/build/lib/collect/nestedset:artifact_nested_set_key",
        "//src/main/java/com/google/devtools/build/lib/events",
        "//src/main/java/com/google/devtools/build/lib/profiler",
        "//src/main/java/com/google/devtools/build/lib/remote/common:lost_inputs_event",
        "//src/main/java/com/google/devtools/build/lib/skyframe:action_utils",
        "//src/main/java/com/google/devtools/build/lib/skyframe:artifact_function",
        "//src/main/java/com/google/devtools/build/lib/skyframe:detailed_exceptions",
        "//src/main/java/com/google/devtools/build/lib/skyframe:skyframe_action_executor",
        "//src/main/java/com/google/devtools/build/lib/skyframe:top_level_action_lookup_key_wrapper",
        "//src/main/java/com/google/devtools/build/lib/skyframe/proto:action_rewind_event_java_proto",
        "//src/main/java/com/google/devtools/build/lib/skyframe/serialization/analysis:dependencies_provider",
        "//src/main/java/com/google/devtools/build/lib/util:detailed_exit_code",
        "//src/main/java/com/google/devtools/build/skyframe:skyframe-objects",
        "//src/main/protobuf:failure_details_java_proto",
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
        "//src/main/java/com/google/devtools/build/lib/skyframe:target_completion_value",
        "//src/main/java/com/google/devtools/build/lib/skyframe:test_completion_value",
        "//src/main/java/com/google/devtools/build/lib/skyframe:top_level_action_lookup_key_wrapper",
        "//src/main/java/com/google/devtools/build/skyframe:skyframe-objects",
    ],
)

java_library(
    name = "lost_important_output_handler_module",
    testonly = True,
    srcs = ["LostImportantOutputHandlerModule.java"],
    deps = [
        "//src/main/java/com/google/devtools/build/lib:runtime",
        "//src/main/java/com/google/devtools/build/lib/actions",
        "//src/main/java/com/google/devtools/build/lib/actions:artifacts",
        "//src/main/java/com/google/devtools/build/lib/actions:file_metadata",
        "//src/main/java/com/google/devtools/build/lib/actions:fileset_output_symlink",
        "//src/main/java/com/google/devtools/build/lib/actions:important_output_handler",
        "//src/main/java/com/google/devtools/build/lib/exec:module_action_context_registry",
        "//src/main/java/com/google/devtools/build/lib/skyframe:tree_artifact_value",
        "//src/main/java/com/google/devtools/build/lib/skyframe/rewinding:lost_input_owners",
        "//src/main/java/com/google/devtools/build/lib/vfs",
        "//src/main/java/com/google/devtools/build/lib/vfs:pathfragment",
        "//third_party:error_prone_annotations",
        "//third_party:flogger",
        "//third_party:guava",
        "//third_party:jsr305",
        "//third_party:junit4",
        "//third_party:mockito",
        "//third_party:truth",
    ],
)
