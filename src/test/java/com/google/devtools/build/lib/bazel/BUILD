package(
    default_testonly = True,
    default_visibility = ["//src:__subpackages__"],
)

filegroup(
    name = "srcs",
    testonly = 0,
    srcs = glob(["**"]) + [
        "//src/test/java/com/google/devtools/build/lib/bazel/debug:srcs",
        "//src/test/java/com/google/devtools/build/lib/bazel/repository:srcs",
        "//src/test/java/com/google/devtools/build/lib/bazel/execlog:srcs",
    ],
    visibility = ["//src/test/java/com/google/devtools/build/lib:__pkg__"],
)

test_suite(
    name = "windows_tests",
    tags = [
        "-no_windows",
        "-slow",
    ],
    visibility = ["//visibility:private"],
)

test_suite(
    name = "all_windows_tests",
    tests = [
        ":windows_tests",
        "//src/test/java/com/google/devtools/build/lib/bazel/repository:all_windows_tests",
    ],
    visibility = ["//src/test/java/com/google/devtools/build/lib:__pkg__"],
)

java_test(
    name = "bazel-rules-tests",
    srcs = glob(["rules/**/*.java"]),
    resources = [
        "//tools/android:android_sdk_repository_template.bzl",
        "//tools/build_defs/cc:action_names.bzl",
        "//tools/cpp:cc_toolchain_config_lib.bzl",
    ] + glob(["rules/**/*.txt"]),
    tags = ["rules"],
    test_class = "com.google.devtools.build.lib.AllTests",
    deps = [
        "//src/main/java/com/google/devtools/build/lib:android-rules",
        "//src/main/java/com/google/devtools/build/lib:bazel-main",
        "//src/main/java/com/google/devtools/build/lib:bazel-rules",
        "//src/main/java/com/google/devtools/build/lib:build-base",
        "//src/main/java/com/google/devtools/build/lib:core-rules",
        "//src/main/java/com/google/devtools/build/lib:core-workspace-rules",
        "//src/main/java/com/google/devtools/build/lib:events",
        "//src/main/java/com/google/devtools/build/lib:packages-internal",
        "//src/main/java/com/google/devtools/build/lib:syntax",
        "//src/main/java/com/google/devtools/build/lib:util",
        "//src/main/java/com/google/devtools/build/lib/actions",
        "//src/main/java/com/google/devtools/build/lib/cmdline",
        "//src/main/java/com/google/devtools/build/lib/collect/nestedset",
        "//src/main/java/com/google/devtools/build/lib/rules/config",
        "//src/main/java/com/google/devtools/build/lib/rules/cpp",
        "//src/main/java/com/google/devtools/build/lib/rules/java:java-compilation",
        "//src/main/java/com/google/devtools/build/lib/vfs",
        "//src/main/java/com/google/devtools/common/options",
        "//src/main/protobuf:crosstool_config_java_proto",
        "//src/test/java/com/google/devtools/build/lib:actions_testutil",
        "//src/test/java/com/google/devtools/build/lib:analysis_testutil",
        "//src/test/java/com/google/devtools/build/lib:foundations_testutil",
        "//src/test/java/com/google/devtools/build/lib:packages_testutil",
        "//src/test/java/com/google/devtools/build/lib:test_runner",
        "//src/test/java/com/google/devtools/build/lib:testutil",
        "//src/test/java/com/google/devtools/build/lib/rules/android:AndroidLocalTestTest",
        "//third_party:guava",
        "//third_party:guava-testlib",
        "//third_party:junit4",
        "//third_party:truth",
    ],
)
