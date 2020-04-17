load("@rules_java//java:defs.bzl", "java_test")

package(
    default_testonly = 1,
    default_visibility = ["//src:__subpackages__"],
)

filegroup(
    name = "srcs",
    testonly = 0,
    srcs = glob(["*"]) + [
        "//src/test/java/com/google/devtools/build/lib/bazel/rules/android:srcs",
        "//src/test/java/com/google/devtools/build/lib/bazel/rules/genrule:srcs",
        "//src/test/java/com/google/devtools/build/lib/bazel/rules/java:srcs",
        "//src/test/java/com/google/devtools/build/lib/bazel/rules/ninja:srcs",
        "//src/test/java/com/google/devtools/build/lib/bazel/rules/python:srcs",
        "//src/test/java/com/google/devtools/build/lib/bazel/rules/sh:srcs",
    ],
    visibility = ["//src:__subpackages__"],
)

java_library(
    name = "BazelRuleTests_lib",
    srcs = glob(["*.java"]),
    deps = [
        "//src/main/java/com/google/devtools/build/lib:build-base",
        "//src/main/java/com/google/devtools/build/lib:core-rules",
        "//src/main/java/com/google/devtools/build/lib:core-workspace-rules",
        "//src/main/java/com/google/devtools/build/lib/actions",
        "//src/main/java/com/google/devtools/build/lib/bazel/rules",
        "//src/main/java/com/google/devtools/build/lib/packages",
        "//src/main/java/com/google/devtools/build/lib/rules/config",
        "//src/main/java/com/google/devtools/build/lib/util:os",
        "//src/main/java/com/google/devtools/build/lib/vfs:pathfragment",
        "//src/main/java/com/google/devtools/common/options",
        "//third_party:guava",
        "//third_party:junit4",
        "//third_party:truth",
    ],
)

java_test(
    name = "BazelRuleTests",
    test_class = "com.google.devtools.build.lib.AllTests",
    runtime_deps = [
        ":BazelRuleTests_lib",
        "//src/test/java/com/google/devtools/build/lib:test_runner",
    ],
)

test_suite(
    name = "windows_tests",
    tags = [
        "-no_windows",
        "-slow",
    ],
)

test_suite(
    name = "all_windows_tests",
    tests = [
        ":windows_tests",
        "//src/test/java/com/google/devtools/build/lib/bazel/rules/android:all_windows_tests",
        "//src/test/java/com/google/devtools/build/lib/bazel/rules/genrule:all_windows_tests",
        "//src/test/java/com/google/devtools/build/lib/bazel/rules/java:all_windows_tests",
        "//src/test/java/com/google/devtools/build/lib/bazel/rules/ninja:all_windows_tests",
        "//src/test/java/com/google/devtools/build/lib/bazel/rules/python:all_windows_tests",
        "//src/test/java/com/google/devtools/build/lib/bazel/rules/sh:all_windows_tests",
    ],
)
