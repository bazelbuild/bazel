package_group(
    name = "tests",
    packages = [
        "//src/test/java/com/google/devtools/build/lib/blackbox/tests",
    ],
)

java_library(
    name = "context",
    testonly = 1,
    srcs = [
        "BlackBoxTestEnvironmentImpl.java",
    ],
    visibility = [
        ":tests",
        "//src/test/java/com/google/devtools/build/lib/blackbox/junit:__pkg__",
    ],
    deps = [
        ":common_tools_deps",
        ":default",
    ],
)

java_library(
    name = "common_tools_deps",
    testonly = 1,
    srcs = [
        "RunfilesUtil.java",
    ],
    exports = [
        "//src/test/java/com/google/devtools/build/lib/blackbox:base_deps",
        "//src/test/java/com/google/devtools/build/lib/blackbox/framework",
    ],
    deps = [
        "//src/test/java/com/google/devtools/build/lib/blackbox:base_deps",
        "//src/test/java/com/google/devtools/build/lib/blackbox/framework",
        "@bazel_tools//tools/java/runfiles",
    ],
)

java_library(
    name = "common_tools",
    testonly = 1,
    visibility = ["//src/test/java/com/google/devtools/build/lib/blackbox/junit:__pkg__"],
    exports = [
        ":cross_tools",
        ":cxx",
        ":default",
        ":java",
    ],
)

java_library(
    name = "default",
    testonly = 1,
    srcs = [
        "DefaultToolsSetup.java",
    ],
    data = [
        "//src:bazel",
        "//src/main/tools:linux-sandbox",
        "//src/main/tools:process-wrapper",
        # should be later split into groups by tools type
        "//third_party:srcs",
        # should be later split into groups by tools type
        "//tools:srcs",
    ],
    visibility = [":tests"],
    deps = [":common_tools_deps"],
)

java_library(
    name = "java",
    testonly = 1,
    srcs = [
        "JavaToolsSetup.java",
    ],
    data = [
        ":langtools-copy",
        "//src/java_tools/junitrunner/java/com/google/testing/junit/runner:Runner_deploy.jar",
        "//third_party/ijar",
        "//third_party/java/jdk/langtools:test-srcs",
        "@bazel_tools//tools/jdk:current_java_runtime",
    ] + select({
        # TODO(bazel-team): Once https://github.com/bazelbuild/bazel/issues/2241
        # is resolved, use cc implementation of singlejar on windows
        "//src/conditions:windows": ["//src/java_tools/singlejar:SingleJar_deploy.jar"],
        "//conditions:default": ["//src/tools/singlejar:singlejar"],
    }),
    visibility = [":tests"],
    deps = [
        ":common_tools_deps",
        "//third_party:truth",
    ],
)

# needed by java tools
genrule(
    name = "langtools-copy",
    testonly = 1,
    srcs = ["//third_party/java/jdk/langtools:javac_jar"],
    outs = ["langtools.jar"],
    cmd = "cp $< $@",
)

java_library(
    name = "cxx",
    testonly = 1,
    srcs = ["CxxToolsSetup.java"],
    visibility = [":tests"],
    deps = [":common_tools_deps"],
)

java_library(
    name = "cross_tools",
    testonly = 1,
    srcs = ["CrossToolsSetup.java"],
    visibility = [":tests"],
    deps = [":common_tools_deps"],
)

java_library(
    name = "python_tools",
    testonly = 1,
    srcs = ["PythonToolsSetup.java"],
    data = [
        "//tools/python:srcs",
    ],
    visibility = [":tests"],
    deps = [
        ":common_tools_deps",
    ],
)

filegroup(
    name = "srcs",
    testonly = 0,
    srcs = glob(["**"]),
    visibility = ["//src/test/java/com/google/devtools/build/lib/blackbox:__pkg__"],
)
