load("@rules_java//java:defs.bzl", "java_test")

package(
    default_applicable_licenses = ["//:license"],
    default_testonly = 1,
    default_visibility = [
        "//src:__subpackages__",
    ],
)

filegroup(
    name = "srcs",
    testonly = False,
    srcs = glob(["**"]),
    visibility = ["//src/tools/workspacelog:__pkg__"],
)

java_test(
    name = "WorkspaceLogParserTest",
    size = "small",
    srcs = ["WorkspaceLogParserTest.java"],
    test_class = "com.google.devtools.build.workspacelog.WorkspaceLogParserTest",
    deps = [
        "//src/main/java/com/google/devtools/build/lib/bazel/debug:workspace_log_java_proto",
        "//src/tools/workspacelog/src/main/java/com/google/devtools/build/workspacelog:parser",
        "//third_party/java/junit",
        "//third_party/java/truth",
    ],
)
