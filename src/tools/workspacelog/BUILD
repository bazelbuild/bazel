load("@rules_java//java:defs.bzl", "java_binary")

package(
    default_applicable_licenses = ["//:license"],
    default_visibility = ["//:__pkg__"],
)

filegroup(
    name = "srcs",
    srcs = glob(["**"]) + [
        "//src/tools/workspacelog/src/main/java/com/google/devtools/build/workspacelog:srcs",
        "//src/tools/workspacelog/test/main/java/com/google/devtools/build/workspacelog:srcs",
    ],
    visibility = ["//src:__subpackages__"],
)

java_binary(
    name = "parser",
    main_class = "com.google.devtools.build.workspacelog.WorkspaceLogParser",
    visibility = ["//visibility:public"],
    runtime_deps = ["//src/tools/workspacelog/src/main/java/com/google/devtools/build/workspacelog:parser"],
)
