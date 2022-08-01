load("@rules_java//java:defs.bzl", "java_library", "java_test")
load("//:distdir_deps.bzl", "gen_workspace_stanza")

java_library(
    name = "deps",
    testonly = 1,
    exports = [
        "//src/test/java/com/google/devtools/build/lib/blackbox:base_deps",
        "//src/test/java/com/google/devtools/build/lib/blackbox:test_deps",
    ],
)

java_library(
    name = "framework",
    testonly = 1,
    srcs = glob(["*.java"]),
    data = [
        ":blackbox.WORKSPACE",
    ],
    visibility = ["//src/test/java/com/google/devtools/build/lib/blackbox:__subpackages__"],
    deps = [
        ":deps",
        "//third_party:flogger",
    ],
)

java_test(
    name = "framework_tests",
    data = [":blackbox.WORKSPACE"],
    tags = ["black_box_test"],
    test_class = "com.google.devtools.build.lib.AllTests",
    visibility = ["//src/test/java/com/google/devtools/build/lib/blackbox/tests:__pkg__"],
    runtime_deps = [":framework"],
)

filegroup(
    name = "srcs",
    testonly = 0,
    srcs = glob(["**"]),
    visibility = ["//src/test/java/com/google/devtools/build/lib/blackbox:__pkg__"],
)

gen_workspace_stanza(
    name = "blackbox_workspace",
    out = "blackbox.WORKSPACE",
    preamble = "load('@bazel_tools//tools/build_defs/repo:http.bzl', 'http_archive')",
    repos = [
        #"remote_java_tools",
        #"remote_java_tools_linux",
        #"remote_java_tools_windows",
        #"remote_java_tools_darwin",
        "rules_cc",
        #"rules_java",
        "rules_proto",
        #"remotejdk11_linux",
        #"remotejdk11_linux_aarch64",
        #"remotejdk11_linux_ppc64le",
        #"remotejdk11_linux_s390x",
        #"remotejdk11_macos",
        #"remotejdk11_macos_aarch64",
        #"remotejdk11_win",
        #"remotejdk11_win_arm64",
    ],
    visibility = ["//visibility:private"],
)
