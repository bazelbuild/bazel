load("@rules_java//java:defs.bzl", "java_library", "java_test")

package(
    default_testonly = 1,
    default_visibility = ["//src:__subpackages__"],
)

licenses(["notice"])

filegroup(
    name = "srcs",
    testonly = 0,
    srcs = glob(["**"]),
    visibility = ["//src:__subpackages__"],
)

java_library(
    name = "CmdLineTests_lib",
    srcs = glob(["*.java"]),
    deps = [
        "//src/main/java/com/google/devtools/build/lib/cmdline",
        "//src/main/java/com/google/devtools/build/lib/cmdline:LabelValidator",
        "//src/main/java/com/google/devtools/build/lib/cmdline:batch_callback",
        "//src/main/java/com/google/devtools/build/lib/cmdline:parallel_visitor",
        "//src/main/java/com/google/devtools/build/lib/cmdline:query_exception_marker_interface",
        "//src/main/java/com/google/devtools/build/lib/concurrent",
        "//src/main/java/com/google/devtools/build/lib/packages/semantics",
        "//src/main/java/com/google/devtools/build/lib/vfs:pathfragment",
        "//src/main/java/net/starlark/java/eval",
        "//src/test/java/com/google/devtools/build/lib/testutil:TestThread",
        "//third_party:guava",
        "//third_party:guava-testlib",
        "//third_party:junit4",
        "//third_party:truth",
    ],
)

java_test(
    name = "CmdLineTests",
    tags = ["foundations"],
    test_class = "com.google.devtools.build.lib.AllTests",
    runtime_deps = [
        ":CmdLineTests_lib",
        "//src/test/java/com/google/devtools/build/lib:test_runner",
    ],
)
