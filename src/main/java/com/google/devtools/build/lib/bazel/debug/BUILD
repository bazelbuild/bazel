# Description:
#   Debugging helpers and modules
package(
    default_visibility = ["//src:__subpackages__"],
)

filegroup(
    name = "srcs",
    srcs = glob(["**"]),
    visibility = [
        "//src:__pkg__",
        "//src/main/java/com/google/devtools/build/lib:__pkg__",
        "//src/test/shell/bazel:__pkg__",
    ],
)

java_library(
    name = "workspace-rule-event",
    srcs = ["WorkspaceRuleEvent.java"],
    deps = [
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
        "//src/main/java/com/google/devtools/build/lib:runtime",
        "//src/main/java/com/google/devtools/common/options",
        "//third_party:guava",
    ],
)

java_library(
    name = "debugging-options",
    srcs = ["DebuggingOptions.java"],
    deps = [
        "//src/main/java/com/google/devtools/common/options",
    ],
)
