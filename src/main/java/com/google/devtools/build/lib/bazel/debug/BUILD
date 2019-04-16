# Description:
#   Debugging helpers and modules
package(
    default_visibility = ["//src:__subpackages__"],
)

load("//tools/build_rules:utilities.bzl", "java_library_srcs")

filegroup(
    name = "srcs",
    srcs = glob(["**"]),
    visibility = [
        "//src:__pkg__",
        "//src/main/java/com/google/devtools/build/lib:__pkg__",
        "//src/test/shell/bazel:__pkg__",
    ],
)

proto_library(
    name = "workspace_log_proto",
    srcs = ["workspace_log.proto"],
    deps = [],
)

java_proto_library(
    name = "workspace_log_java_proto",
    deps = [":workspace_log_proto"],
)

java_library_srcs(
    name = "workspace_log_java_proto_srcs",
    deps = [":workspace_log_java_proto"],
)

filegroup(
    name = "dist_jars",
    srcs = [":workspace_log_java_proto_srcs"],
)

java_library(
    name = "workspace-rule-event",
    srcs = ["WorkspaceRuleEvent.java"],
    deps = [
        ":workspace_log_java_proto",
        "//src/main/java/com/google/devtools/build/lib:events",
    ],
)

java_library(
    name = "workspace-rule-module",
    srcs = ["WorkspaceRuleModule.java"],
    deps = [
        ":debugging-options",
        ":workspace-rule-event",
        "//src/main/java/com/google/devtools/build/lib:events",
        "//src/main/java/com/google/devtools/build/lib:exitcode-external",
        "//src/main/java/com/google/devtools/build/lib:io",
        "//src/main/java/com/google/devtools/build/lib:runtime",
        "//src/main/java/com/google/devtools/build/lib:util",
        "//src/main/java/com/google/devtools/build/lib/vfs:pathfragment",
        "//src/main/java/com/google/devtools/common/options",
        "//third_party:guava",
    ],
)

java_library(
    name = "debugging-options",
    srcs = ["DebuggingOptions.java"],
    deps = [
        "//src/main/java/com/google/devtools/build/lib:util",
        "//src/main/java/com/google/devtools/build/lib/vfs:pathfragment",
        "//src/main/java/com/google/devtools/common/options",
    ],
)
