package(
    default_testonly = 1,
    default_visibility = ["//src:__subpackages__"],
)

filegroup(
    name = "srcs",
    testonly = 0,
    srcs = glob(
        ["**"],
        exclude = [
            "*~",
        ],
    ),
    visibility = ["//src/tools/workspacelog:__pkg__"],
)

java_test(
    name = "WorkspaceLogParserTest",
    size = "small",
    srcs = ["WorkspaceLogParserTest.java"],
    test_class = "com.google.devtools.build.workspacelog.WorkspaceLogParserTest",
    deps = [
        "//src/main/java/com/google/devtools/build/lib/bazel/debug:workspace_log_java_proto",
        "//src/main/java/com/google/devtools/common/options",
        "//src/tools/workspacelog/src/main/java/com/google/devtools/build/workspacelog:parser",
        "//third_party:junit4",
        "//third_party:truth",
    ],
)
