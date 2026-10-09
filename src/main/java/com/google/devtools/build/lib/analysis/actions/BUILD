load("@rules_java//java:defs.bzl", "java_library")

package(
    default_applicable_licenses = ["//:license"],
    default_visibility = ["//src/main/java/com/google/devtools/build/lib:build_lib"],
)

filegroup(
    name = "srcs",
    srcs = glob(["**"]),
    visibility = ["//src/main/java/com/google/devtools/build/lib/analysis:__pkg__"],
)

java_library(
    name = "abstract_file_write_action",
    srcs = ["AbstractFileWriteAction.java"],
    deps = [
        ":file_write_action_context",
        "//src/main/java/com/google/devtools/build/lib/actions",
        "//src/main/java/com/google/devtools/build/lib/actions:artifacts",
        "//src/main/java/com/google/devtools/build/lib/collect/nestedset",
        "//src/main/java/com/google/devtools/build/lib/events",
        "//src/main/java/com/google/devtools/build/lib/profiler",
        "//src/main/java/com/google/devtools/build/lib/util:deterministic_writer",
        "//third_party/java/guava:collect",
        "//third_party/java/jsr305_annotations",
    ],
)

java_library(
    name = "action_construction_context",
    srcs = ["ActionConstructionContext.java"],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/actions",
        "//src/main/java/com/google/devtools/build/lib/actions:artifacts",
        "//src/main/java/com/google/devtools/build/lib/analysis:analysis_environment",
        "//src/main/java/com/google/devtools/build/lib/analysis:rule_error_consumer",
        "//src/main/java/com/google/devtools/build/lib/analysis/config:build_configuration",
        "//src/main/java/com/google/devtools/build/lib/analysis/platform",
        "//src/main/java/com/google/devtools/build/lib/packages",
        "//src/main/java/com/google/devtools/build/lib/vfs:pathfragment",
        "//third_party/java/jsr305_annotations",
    ],
)

java_library(
    name = "binary_file_write_action",
    srcs = ["BinaryFileWriteAction.java"],
    deps = [
        ":abstract_file_write_action",
        "//src/main/java/com/google/devtools/build/lib/actions",
        "//src/main/java/com/google/devtools/build/lib/actions:artifacts",
        "//src/main/java/com/google/devtools/build/lib/collect/nestedset",
        "//src/main/java/com/google/devtools/build/lib/concurrent:thread_safety",
        "//src/main/java/com/google/devtools/build/lib/util",
        "//third_party/java/guava:annotations",
        "//third_party/java/guava:base",
        "//third_party/java/guava:io",
        "//third_party/java/jsr305_annotations",
    ],
)

java_library(
    name = "buildinfo_file_write_action",
    srcs = ["BuildInfoFileWriteAction.java"],
    deps = [
        ":substitution",
        ":template",
        ":template_expansion_action",
        "//src/main/java/com/google/devtools/build/lib/actions",
        "//src/main/java/com/google/devtools/build/lib/actions:artifacts",
        "//src/main/java/com/google/devtools/build/lib/analysis:workspace_status_action",
        "//src/main/java/com/google/devtools/build/lib/cmdline",
        "//src/main/java/com/google/devtools/build/lib/collect/nestedset",
        "//src/main/java/com/google/devtools/build/lib/skyframe:workspace_status_value",
        "//src/main/java/com/google/devtools/build/lib/util",
        "//src/main/java/com/google/devtools/build/lib/util:detailed_exit_code",
        "//src/main/java/net/starlark/java/eval",
        "//src/main/protobuf:failure_details_java_proto",
        "//third_party/java/guava:base",
        "//third_party/java/guava:collect",
        "//third_party/java/jsr305_annotations",
    ],
)

java_library(
    name = "compression",
    srcs = ["Compression.java"],
)

java_library(
    name = "custom_command_line",
    srcs = ["CustomCommandLine.java"],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/actions",
        "//src/main/java/com/google/devtools/build/lib/actions:action_input",
        "//src/main/java/com/google/devtools/build/lib/actions:artifacts",
        "//src/main/java/com/google/devtools/build/lib/actions:commandline_item",
        "//src/main/java/com/google/devtools/build/lib/analysis/config:core_options",
        "//src/main/java/com/google/devtools/build/lib/cmdline",
        "//src/main/java/com/google/devtools/build/lib/collect/nestedset",
        "//src/main/java/com/google/devtools/build/lib/concurrent",
        "//src/main/java/com/google/devtools/build/lib/concurrent:thread_safety",
        "//src/main/java/com/google/devtools/build/lib/skyframe:tree_artifact_value",
        "//src/main/java/com/google/devtools/build/lib/skyframe/serialization:visible-for-serialization",
        "//src/main/java/com/google/devtools/build/lib/skyframe/serialization/autocodec",
        "//src/main/java/com/google/devtools/build/lib/skyframe/serialization/autocodec:serialization-constant",
        "//src/main/java/com/google/devtools/build/lib/util",
        "//src/main/java/com/google/devtools/build/lib/vfs:pathfragment",
        "//third_party/java/auto:auto_value",
        "//third_party/java/guava:annotations",
        "//third_party/java/guava:base",
        "//third_party/java/guava:collect",
        "//third_party/java/jsr305_annotations",
    ],
)

java_library(
    name = "file_write_action",
    srcs = ["FileWriteAction.java"],
    deps = [
        ":abstract_file_write_action",
        ":action_construction_context",
        ":compression",
        "//src/main/java/com/google/devtools/build/lib/actions",
        "//src/main/java/com/google/devtools/build/lib/actions:artifacts",
        "//src/main/java/com/google/devtools/build/lib/collect/nestedset",
        "//src/main/java/com/google/devtools/build/lib/concurrent:thread_safety",
        "//src/main/java/com/google/devtools/build/lib/events",
        "//src/main/java/com/google/devtools/build/lib/unsafe:string",
        "//src/main/java/com/google/devtools/build/lib/util",
        "//third_party/java/guava:base",
        "//third_party/java/guava:io",
        "//third_party/java/jsr305_annotations",
    ],
)

java_library(
    name = "file_write_action_context",
    srcs = ["FileWriteActionContext.java"],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/actions",
        "//src/main/java/com/google/devtools/build/lib/actions:artifacts",
        "//src/main/java/com/google/devtools/build/lib/util:deterministic_writer",
        "//third_party/java/guava:collect",
    ],
)

java_library(
    name = "lazy_write_nested_set_of_tuple_action",
    srcs = ["LazyWriteNestedSetOfTupleAction.java"],
    deps = [
        ":abstract_file_write_action",
        "//src/main/java/com/google/devtools/build/lib/actions",
        "//src/main/java/com/google/devtools/build/lib/actions:artifacts",
        "//src/main/java/com/google/devtools/build/lib/collect/nestedset",
        "//src/main/java/com/google/devtools/build/lib/unsafe:string",
        "//src/main/java/com/google/devtools/build/lib/util",
        "//src/main/java/com/google/devtools/build/lib/util:deterministic_writer",
        "//src/main/java/net/starlark/java/eval",
        "//third_party/java/jsr305_annotations",
    ],
)

java_library(
    name = "lazy_write_paths_file_action",
    srcs = ["LazyWritePathsFileAction.java"],
    deps = [
        ":abstract_file_write_action",
        "//src/main/java/com/google/devtools/build/lib/actions",
        "//src/main/java/com/google/devtools/build/lib/actions:artifacts",
        "//src/main/java/com/google/devtools/build/lib/collect/nestedset",
        "//src/main/java/com/google/devtools/build/lib/unsafe:string",
        "//src/main/java/com/google/devtools/build/lib/util",
        "//src/main/java/com/google/devtools/build/lib/util:deterministic_writer",
        "//third_party/java/guava:collect",
        "//third_party/java/jsr305_annotations",
    ],
)

java_library(
    name = "parameter_file_write_action",
    srcs = ["ParameterFileWriteAction.java"],
    deps = [
        ":abstract_file_write_action",
        ":path_mappers",
        "//src/main/java/com/google/devtools/build/lib/actions",
        "//src/main/java/com/google/devtools/build/lib/actions:artifacts",
        "//src/main/java/com/google/devtools/build/lib/actions:execution_requirements",
        "//src/main/java/com/google/devtools/build/lib/actions:parameter_file",
        "//src/main/java/com/google/devtools/build/lib/analysis/config:build_configuration",
        "//src/main/java/com/google/devtools/build/lib/analysis/config:core_options",
        "//src/main/java/com/google/devtools/build/lib/collect/nestedset",
        "//src/main/java/com/google/devtools/build/lib/concurrent:thread_safety",
        "//src/main/java/com/google/devtools/build/lib/util",
        "//src/main/java/com/google/devtools/build/lib/util:deterministic_writer",
        "//src/main/java/net/starlark/java/eval",
        "//src/main/protobuf:failure_details_java_proto",
        "//third_party/java/guava:annotations",
        "//third_party/java/guava:base",
        "//third_party/java/guava:collect",
        "//third_party/java/guava:io",
        "//third_party/java/jsr305_annotations",
    ],
)

java_library(
    name = "path_mappers",
    srcs = [
        "PathMappers.java",
        "StrippingPathMapper.java",
    ],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/actions",
        "//src/main/java/com/google/devtools/build/lib/actions:action_input",
        "//src/main/java/com/google/devtools/build/lib/actions:action_input_helper",
        "//src/main/java/com/google/devtools/build/lib/actions:artifacts",
        "//src/main/java/com/google/devtools/build/lib/actions:commandline_item",
        "//src/main/java/com/google/devtools/build/lib/actions:execution_requirements",
        "//src/main/java/com/google/devtools/build/lib/actions:file_metadata",
        "//src/main/java/com/google/devtools/build/lib/actions:virtual_action_input",
        "//src/main/java/com/google/devtools/build/lib/analysis/config:build_configuration",
        "//src/main/java/com/google/devtools/build/lib/analysis/config:core_options",
        "//src/main/java/com/google/devtools/build/lib/collect/nestedset",
        "//src/main/java/com/google/devtools/build/lib/starlarkbuildapi",
        "//src/main/java/com/google/devtools/build/lib/util",
        "//src/main/java/com/google/devtools/build/lib/vfs:pathfragment",
        "//third_party/java/guava:annotations",
        "//third_party/java/guava:collect",
        "//third_party/java/jsr305_annotations",
    ],
)

java_library(
    name = "shell_command",
    srcs = ["ShellCommand.java"],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/actions",
        "//src/main/java/com/google/devtools/build/lib/actions:artifacts",
        "//src/main/java/com/google/devtools/build/lib/vfs:pathfragment",
        "//third_party/java/guava:base",
        "//third_party/java/guava:collect",
    ],
)

java_library(
    name = "spawn_action",
    srcs = [
        "SpawnAction.java",
        "StarlarkAction.java",
    ],
    deps = [
        ":action_construction_context",
        ":custom_command_line",
        ":path_mappers",
        ":shell_command",
        "//src/main/java/com/google/devtools/build/lib/actions",
        "//src/main/java/com/google/devtools/build/lib/actions:action_input",
        "//src/main/java/com/google/devtools/build/lib/actions:action_input_helper",
        "//src/main/java/com/google/devtools/build/lib/actions:artifacts",
        "//src/main/java/com/google/devtools/build/lib/actions:commandline_item",
        "//src/main/java/com/google/devtools/build/lib/actions:execution_requirements",
        "//src/main/java/com/google/devtools/build/lib/actions:localhost_capacity",
        "//src/main/java/com/google/devtools/build/lib/actions:param_file_info",
        "//src/main/java/com/google/devtools/build/lib/analysis:starlark/args",
        "//src/main/java/com/google/devtools/build/lib/analysis/config:build_configuration",
        "//src/main/java/com/google/devtools/build/lib/analysis/config:core_options",
        "//src/main/java/com/google/devtools/build/lib/collect/nestedset",
        "//src/main/java/com/google/devtools/build/lib/concurrent",
        "//src/main/java/com/google/devtools/build/lib/exec:spawn_exec_exception",
        "//src/main/java/com/google/devtools/build/lib/exec:spawn_strategy_resolver",
        "//src/main/java/com/google/devtools/build/lib/packages:declared_exec_group",
        "//src/main/java/com/google/devtools/build/lib/skyframe/serialization:visible-for-serialization",
        "//src/main/java/com/google/devtools/build/lib/skyframe/serialization/autocodec",
        "//src/main/java/com/google/devtools/build/lib/starlarkbuildapi",
        "//src/main/java/com/google/devtools/build/lib/util",
        "//src/main/java/com/google/devtools/build/lib/util:detailed_exit_code",
        "//src/main/java/com/google/devtools/build/lib/util:shell_escaper",
        "//src/main/java/com/google/devtools/build/lib/vfs:pathfragment",
        "//src/main/java/net/starlark/java/eval",
        "//src/main/protobuf:extra_actions_base_java_proto",
        "//src/main/protobuf:failure_details_java_proto",
        "//third_party/java/error_prone:annotations",
        "//third_party/java/guava:annotations",
        "//third_party/java/guava:base",
        "//third_party/java/guava:collect",
        "//third_party/java/jsr305_annotations",
        "//third_party/java/protobuf",
    ],
)

java_library(
    name = "spawn_action_template",
    srcs = ["SpawnActionTemplate.java"],
    deps = [
        ":custom_command_line",
        ":spawn_action",
        "//src/main/java/com/google/devtools/build/lib/actions",
        "//src/main/java/com/google/devtools/build/lib/actions:action_lookup_key",
        "//src/main/java/com/google/devtools/build/lib/actions:artifacts",
        "//src/main/java/com/google/devtools/build/lib/actions:localhost_capacity",
        "//src/main/java/com/google/devtools/build/lib/analysis/config:build_configuration",
        "//src/main/java/com/google/devtools/build/lib/analysis/config:core_options",
        "//src/main/java/com/google/devtools/build/lib/collect/nestedset",
        "//src/main/java/com/google/devtools/build/lib/events",
        "//src/main/java/com/google/devtools/build/lib/skyframe:action_template_expansion_value",
        "//src/main/java/com/google/devtools/build/lib/util",
        "//src/main/java/com/google/devtools/build/lib/vfs:pathfragment",
        "//third_party/java/error_prone:annotations",
        "//third_party/java/guava:base",
        "//third_party/java/guava:collect",
        "//third_party/java/jsr305_annotations",
    ],
)

java_library(
    name = "substitution",
    srcs = ["Substitution.java"],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/concurrent:thread_safety",
        "//src/main/java/net/starlark/java/eval",
        "//third_party/java/guava:annotations",
        "//third_party/java/guava:base",
        "//third_party/java/jsr305_annotations",
    ],
)

java_library(
    name = "symlink_action",
    srcs = ["SymlinkAction.java"],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/actions",
        "//src/main/java/com/google/devtools/build/lib/actions:artifacts",
        "//src/main/java/com/google/devtools/build/lib/actions:file_metadata",
        "//src/main/java/com/google/devtools/build/lib/actions:fileset_output_tree",
        "//src/main/java/com/google/devtools/build/lib/analysis/platform",
        "//src/main/java/com/google/devtools/build/lib/collect/nestedset",
        "//src/main/java/com/google/devtools/build/lib/exec:spawn_log_context",
        "//src/main/java/com/google/devtools/build/lib/unix:unix_mode",
        "//src/main/java/com/google/devtools/build/lib/util",
        "//src/main/java/com/google/devtools/build/lib/util:detailed_exit_code",
        "//src/main/java/com/google/devtools/build/lib/vfs",
        "//src/main/java/com/google/devtools/build/lib/vfs:pathfragment",
        "//src/main/protobuf:failure_details_java_proto",
        "//third_party/java/guava:base",
        "//third_party/java/guava:collect",
        "//third_party/java/jsr305_annotations",
    ],
)

java_library(
    name = "symlink_tree_action",
    srcs = [
        "SymlinkTreeAction.java",
        "SymlinkTreeActionContext.java",
    ],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/actions",
        "//src/main/java/com/google/devtools/build/lib/actions:artifacts",
        "//src/main/java/com/google/devtools/build/lib/actions:file_metadata",
        "//src/main/java/com/google/devtools/build/lib/actions:fileset_output_tree",
        "//src/main/java/com/google/devtools/build/lib/analysis:runfiles",
        "//src/main/java/com/google/devtools/build/lib/analysis/config:build_configuration",
        "//src/main/java/com/google/devtools/build/lib/analysis/config:core_options",
        "//src/main/java/com/google/devtools/build/lib/collect/nestedset",
        "//src/main/java/com/google/devtools/build/lib/concurrent:thread_safety",
        "//src/main/java/com/google/devtools/build/lib/util",
        "//src/main/java/com/google/devtools/build/lib/util:os",
        "//third_party/java/guava:annotations",
        "//third_party/java/guava:base",
        "//third_party/java/guava:collect",
        "//third_party/java/jsr305_annotations",
    ],
)

java_library(
    name = "template",
    srcs = ["Template.java"],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/actions:artifacts",
        "//src/main/java/com/google/devtools/build/lib/concurrent:thread_safety",
        "//src/main/java/com/google/devtools/build/lib/util",
        "//src/main/java/com/google/devtools/build/lib/vfs",
        "//third_party/java/guava:annotations",
        "//third_party/java/jsr305_annotations",
    ],
)

java_library(
    name = "template_expansion_exception",
    srcs = ["TemplateExpansionException.java"],
    deps = [
        "//src/main/java/net/starlark/java/eval",
    ],
)

java_library(
    name = "template_expansion_action",
    srcs = [
        "LocalTemplateExpansionStrategy.java",
        "TemplateExpansionAction.java",
        "TemplateExpansionContext.java",
    ],
    exports = [
        ":template_expansion_exception",
    ],
    deps = [
        ":file_write_action_context",
        ":substitution",
        ":template",
        "//src/main/java/com/google/devtools/build/lib/actions",
        "//src/main/java/com/google/devtools/build/lib/actions:artifacts",
        "//src/main/java/com/google/devtools/build/lib/collect/nestedset",
        "//src/main/java/com/google/devtools/build/lib/concurrent:thread_safety",
        "//src/main/java/com/google/devtools/build/lib/util",
        "//src/main/java/com/google/devtools/build/lib/util:detailed_exit_code",
        "//src/main/java/com/google/devtools/build/lib/util:deterministic_writer",
        "//src/main/java/com/google/devtools/build/lib/util:string",
        "//src/main/java/net/starlark/java/eval",
        "//src/main/protobuf:failure_details_java_proto",
        "//third_party/java/auto:auto_value",
        "//third_party/java/guava:annotations",
        "//third_party/java/guava:collect",
        "//third_party/java/jsr305_annotations",
    ],
)
