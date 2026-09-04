load("@rules_java//java:defs.bzl", "java_library")

package(
    default_applicable_licenses = ["//:license"],
    default_visibility = ["//src:__subpackages__"],
)

filegroup(
    name = "srcs",
    srcs = glob(["*"]),
    visibility = ["//src/main/java/com/google/devtools/build/lib:__subpackages__"],
)

java_library(
    name = "local",
    srcs = [
        "LocalEnvProvider.java",
        "LocalSpawnRunner.java",
        "PosixLocalEnvProvider.java",
        "WindowsLocalEnvProvider.java",
        "XcodeLocalEnvProvider.java",
    ],
    data = [
        "//src/main/tools:process-wrapper",
    ],
    deps = [
        ":options",
        "//src/main/java/com/google/devtools/build/lib/actions",
        "//src/main/java/com/google/devtools/build/lib/actions:action_input",
        "//src/main/java/com/google/devtools/build/lib/actions:artifacts",
        "//src/main/java/com/google/devtools/build/lib/actions:resource_manager",
        "//src/main/java/com/google/devtools/build/lib/actions:runfiles_tree",
        "//src/main/java/com/google/devtools/build/lib/actions:virtual_action_input",
        "//src/main/java/com/google/devtools/build/lib/concurrent:thread_safety",
        "//src/main/java/com/google/devtools/build/lib/exec:bin_tools",
        "//src/main/java/com/google/devtools/build/lib/exec:runfiles_tree_updater",
        "//src/main/java/com/google/devtools/build/lib/exec:spawn_runner",
        "//src/main/java/com/google/devtools/build/lib/profiler",
        "//src/main/java/com/google/devtools/build/lib/rules/apple",
        "//src/main/java/com/google/devtools/build/lib/runtime:process_wrapper",
        "//src/main/java/com/google/devtools/build/lib/shell",
        "//src/main/java/com/google/devtools/build/lib/util",
        "//src/main/java/com/google/devtools/build/lib/util:os",
        "//src/main/java/com/google/devtools/build/lib/util:string_encoding",
        "//src/main/java/com/google/devtools/build/lib/util/io",
        "//src/main/java/com/google/devtools/build/lib/vfs",
        "//src/main/protobuf:failure_details_java_proto",
        "//third_party/java/error_prone:annotations",
        "//third_party/java/flogger",
        "//third_party/java/guava:base",
        "//third_party/java/guava:collect",
        "//third_party/java/jsr305_annotations",
    ],
)

java_library(
    name = "options",
    srcs = [
        "LocalExecutionOptions.java",
    ],
    deps = [
        "//src/main/java/com/google/devtools/common/options",
    ],
)
