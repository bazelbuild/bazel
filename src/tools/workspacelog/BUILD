filegroup(
    name = "srcs",
    srcs = glob(["**"]) + [
        "//src/tools/workspacelog/src/main/java/com/google/devtools/build/workspacelog:srcs",
        "//src/tools/workspacelog/test/main/java/com/google/devtools/build/workspacelog:srcs",
    ],
    visibility = ["//src:__pkg__"],
)

java_binary(
    name = "parser",
    main_class = "com.google.devtools.build.workspacelog.WorkspaceLogParser",
    visibility = ["//visibility:public"],
    runtime_deps = ["//src/tools/workspacelog/src/main/java/com/google/devtools/build/workspacelog:parser"],
)
