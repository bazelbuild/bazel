filegroup(
    name = "srcs",
    testonly = 0,
    srcs = glob(["**"]),
    visibility = ["//src/test/java/com/google/devtools/build/lib/blackbox:__subpackages__"],
)

java_library(
    name = "common_deps",
    testonly = 1,
    exports = [
        "//src/test/java/com/google/devtools/build/lib/blackbox:base_deps",
        "//src/test/java/com/google/devtools/build/lib/blackbox:test_deps",
        "//src/test/java/com/google/devtools/build/lib/blackbox/bazel:context",
        "//src/test/java/com/google/devtools/build/lib/blackbox/framework",
        "//src/test/java/com/google/devtools/build/lib/blackbox/junit:test_base",
    ],
)

java_test(
    name = "PythonBlackBoxTest",
    timeout = "moderate",
    srcs = ["PythonBlackBoxTest.java"],
    tags = ["black_box_test"],
    deps = [
        ":common_deps",
        "//src/test/java/com/google/devtools/build/lib/blackbox/bazel:python_tools",
    ],
)

java_test(
    name = "WorkspaceBlackBoxTest",
    timeout = "moderate",
    srcs = [
        "HelperStarlarkTexts.java",
        "WorkspaceBlackBoxTest.java",
    ],
    tags = ["black_box_test"],
    deps = [
        ":common_deps",
    ],
)

java_test(
    name = "BazelEmbeddedSkylarkBlackBoxTest",
    timeout = "moderate",
    srcs = [
        "BazelEmbeddedSkylarkBlackBoxTest.java",
        "HelperStarlarkTexts.java",
    ],
    tags = ["black_box_test"],
    deps = [
        ":common_deps",
        "//src/java_tools/singlejar/java/com/google/devtools/build/zip",
        "//src/main/java/com/google/devtools/build/lib:bazel-repository",
        "//src/main/java/com/google/devtools/build/lib/vfs",
        "//src/test/java/com/google/devtools/build/lib:foundations_testutil",
    ],
)

test_suite(
    name = "black_box_tests",
    tags = ["black_box_test"],
    tests = [
        "BazelEmbeddedSkylarkBlackBoxTest",
        "PythonBlackBoxTest",
        "WorkspaceBlackBoxTest",
        "//src/test/java/com/google/devtools/build/lib/blackbox/framework:framework_tests",
        "//src/test/java/com/google/devtools/build/lib/blackbox/junit:TimeoutTestWatcherTest",
    ],
)

test_suite(
    name = "blackbox_windows_tests",
    tests = [
        ":black_box_tests",
    ],
    visibility = ["//src/test/java/com/google/devtools/build/lib:__pkg__"],
)
