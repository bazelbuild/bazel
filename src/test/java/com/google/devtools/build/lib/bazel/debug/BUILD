load("@rules_java//java:defs.bzl", "java_library", "java_test")

package(
    default_applicable_licenses = ["//:license"],
    default_testonly = 1,
    default_visibility = ["//src:__subpackages__"],
)

filegroup(
    name = "srcs",
    testonly = 0,
    srcs = glob(["*"]),
    visibility = ["//src:__subpackages__"],
)

java_library(
    name = "WorkspaceRuleEventTest_lib",
    srcs = ["WorkspaceRuleEventTest.java"],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/bazel/debug:workspace-rule-event",
        "//src/main/java/com/google/devtools/build/lib/bazel/debug:workspace_log_java_proto",
        "//src/main/java/net/starlark/java/syntax",
        "//third_party:guava",
        "//third_party:junit4",
        "//third_party:truth",
    ],
)

java_test(
    name = "WorkspaceRuleEventTest",
    size = "small",
    runtime_deps = [
        ":WorkspaceRuleEventTest_lib",
    ],
)
