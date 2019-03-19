package(
    default_testonly = 1,
    default_visibility = ["//src:__subpackages__"],
)

filegroup(
    name = "srcs",
    testonly = 0,
    srcs = glob(["**"]),
    visibility = ["//src/test/java/com/google/devtools/build/lib:__pkg__"],
)

java_test(
    name = "cpp-rules-tests",
    srcs = glob(
        ["*.java"],
        exclude = [
            "CcImportBaseConfiguredTargetTest.java",
            "SkylarkCcCommonTestHelper.java",
        ],
    ) + ["proto/CcProtoLibraryTest.java"],
    resources = [
        "//tools/build_defs/cc:action_names.bzl",
        "//tools/cpp:cc_toolchain_config_lib.bzl",
        "//tools/cpp:lib_cc_configure",
    ],
    tags = ["rules"],
    test_class = "com.google.devtools.build.lib.AllTests",
    deps = [
        ":CcImportBaseConfiguredTargetTest",
        ":SkylarkCcCommonTestHelper",
        "//src/main/java/com/google/devtools/build/lib:bazel-main",
        "//src/main/java/com/google/devtools/build/lib:bazel-rules",
        "//src/main/java/com/google/devtools/build/lib:build-base",
        "//src/main/java/com/google/devtools/build/lib:core-rules",
        "//src/main/java/com/google/devtools/build/lib:core-workspace-rules",
        "//src/main/java/com/google/devtools/build/lib:io",
        "//src/main/java/com/google/devtools/build/lib:packages-internal",
        "//src/main/java/com/google/devtools/build/lib:skylark_semantics",
        "//src/main/java/com/google/devtools/build/lib:util",
        "//src/main/java/com/google/devtools/build/lib/actions",
        "//src/main/java/com/google/devtools/build/lib/actions:localhost_capacity",
        "//src/main/java/com/google/devtools/build/lib/analysis/platform",
        "//src/main/java/com/google/devtools/build/lib/cmdline",
        "//src/main/java/com/google/devtools/build/lib/collect/nestedset",
        "//src/main/java/com/google/devtools/build/lib/concurrent",
        "//src/main/java/com/google/devtools/build/lib/rules/cpp",
        "//src/main/java/com/google/devtools/build/lib/rules/platform",
        "//src/main/java/com/google/devtools/build/lib/skyframe/serialization",
        "//src/main/java/com/google/devtools/build/lib/skyframe/serialization/autocodec",
        "//src/main/java/com/google/devtools/build/lib/skyframe/serialization/testutils",
        "//src/main/java/com/google/devtools/build/lib/skylarkbuildapi/cpp",
        "//src/main/java/com/google/devtools/build/lib/vfs",
        "//src/main/java/com/google/devtools/build/lib/vfs/inmemoryfs",
        "//src/main/java/com/google/devtools/common/options",
        "//src/main/java/com/google/devtools/common/options:invocation_policy",
        "//src/main/protobuf:crosstool_config_java_proto",
        "//src/main/protobuf:extra_actions_base_java_proto",
        "//src/test/java/com/google/devtools/build/lib:actions_testutil",
        "//src/test/java/com/google/devtools/build/lib:analysis_testutil",
        "//src/test/java/com/google/devtools/build/lib:packages_testutil",
        "//src/test/java/com/google/devtools/build/lib:syntax_testutil",
        "//src/test/java/com/google/devtools/build/lib:testutil",
        "//third_party:guava",
        "//third_party:guava-testlib",
        "//third_party:jsr305",
        "//third_party:junit4",
        "//third_party:mockito",
        "//third_party:truth",
        "//third_party/protobuf:protobuf_java",
    ],
)

java_library(
    name = "util",
    srcs = [
        "LinkBuildVariablesTestCase.java",
    ],
    deps = [
        "//src/main/java/com/google/devtools/build/lib:build-base",
        "//src/main/java/com/google/devtools/build/lib:util",
        "//src/main/java/com/google/devtools/build/lib/actions",
        "//src/main/java/com/google/devtools/build/lib/rules/cpp",
        "//src/main/java/com/google/devtools/build/lib/vfs",
        "//src/main/protobuf:crosstool_config_java_proto",
        "//src/test/java/com/google/devtools/build/lib:analysis_testutil",
        "//third_party:guava",
        "//third_party:guava-testlib",
        "//third_party:jsr305",
        "//third_party:junit4",
        "//third_party:truth",
        "//third_party/java/dd_plist",
        "//third_party/protobuf:protobuf_java",
    ],
)

java_library(
    name = "CcImportBaseConfiguredTargetTest",
    srcs = ["CcImportBaseConfiguredTargetTest.java"],
    deps = [
        ":SkylarkCcCommonTestHelper",
        "//src/main/java/com/google/devtools/build/lib:build-base",
        "//src/main/java/com/google/devtools/build/lib/actions",
        "//src/main/java/com/google/devtools/build/lib/rules/cpp",
        "//src/test/java/com/google/devtools/build/lib:actions_testutil",
        "//src/test/java/com/google/devtools/build/lib:analysis_testutil",
        "//src/test/java/com/google/devtools/build/lib:packages_testutil",
        "//third_party:guava",
        "//third_party:junit4",
        "//third_party:truth",
    ],
)

java_library(
    name = "SkylarkCcCommonTestHelper",
    srcs = ["SkylarkCcCommonTestHelper.java"],
    deps = [
        "//src/test/java/com/google/devtools/build/lib:analysis_testutil",
        "//src/test/java/com/google/devtools/build/lib:testutil",
    ],
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
    ],
    visibility = ["//src/test/java/com/google/devtools/build/lib:__pkg__"],
)
