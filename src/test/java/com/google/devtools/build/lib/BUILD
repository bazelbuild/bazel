package(
    default_visibility = [
        ":__subpackages__",
        "//tools/test:__pkg__",
    ],
)

# Tests for Windows-specific functionality that can run cross-platform.
CROSS_PLATFORM_WINDOWS_TESTS = [
    "util/DependencySetWindowsTest.java",
    "vfs/PathFragmentWindowsTest.java",
    "vfs/WindowsPathTest.java",
]

# Tests for Windows-specific functionality that run on Windows.
WINDOWS_ON_WINDOWS_TESTS = glob(
    ["windows/*.java"],
    exclude = [
        "windows/MockSubprocess.java",
    ],
)

# All Windows-specific tests. Use this to exclude Windows tests from globs.
ALL_WINDOWS_TESTS = CROSS_PLATFORM_WINDOWS_TESTS + WINDOWS_ON_WINDOWS_TESTS

JNI_LIB = select({
    "//src/conditions:windows": ["//src/main/native/windows:windows_jni.dll"],
    "//conditions:default": [
        "//src/main/native:libunix.dylib",
        "//src/main/native:libunix.so",
    ],
})

# Generate list of all srcs via:
# bazel query 'filter("srcs", kind("filegroup rule", //src/test/java/com/google/devtools/build/lib/...))' | sort | sed -e "s/^/\"/" | sed -e "s/$/\",/" | fgrep -v "build/lib:srcs"
filegroup(
    name = "srcs",
    srcs = glob(["**"]) + [
        "//src/test/java/com/google/devtools/build/lib/analysis/platform:srcs",
        "//src/test/java/com/google/devtools/build/lib/analysis/skylark/annotations/processor:srcs",
        "//src/test/java/com/google/devtools/build/lib/analysis/whitelisting:srcs",
        "//src/test/java/com/google/devtools/build/lib/bazel/repository/cache:srcs",
        "//src/test/java/com/google/devtools/build/lib/bazel/debug:srcs",
        "//src/test/java/com/google/devtools/build/lib/bazel/repository/downloader:srcs",
        "//src/test/java/com/google/devtools/build/lib/bazel/repository:srcs",
        "//src/test/java/com/google/devtools/build/lib/buildeventservice:srcs",
        "//src/test/java/com/google/devtools/build/lib/buildeventstream:srcs",
        "//src/test/java/com/google/devtools/build/lib/buildeventstream/transports:srcs",
        "//src/test/java/com/google/devtools/build/lib/buildtool:srcs",
        "//src/test/java/com/google/devtools/build/lib/profiler/callcounts:srcs",
        "//src/test/java/com/google/devtools/build/lib/profiler/memory:srcs",
        "//src/test/java/com/google/devtools/build/lib/rules/android:srcs",
        "//src/test/java/com/google/devtools/build/lib/rules/apple:srcs",
        "//src/test/java/com/google/devtools/build/lib/rules/config:srcs",
        "//src/test/java/com/google/devtools/build/lib/rules/cpp:srcs",
        "//src/test/java/com/google/devtools/build/lib/rules/objc:srcs",
        "//src/test/java/com/google/devtools/build/lib/rules/platform:srcs",
        "//src/test/java/com/google/devtools/build/lib/rules/python:srcs",
        "//src/test/java/com/google/devtools/build/lib/rules/repository:srcs",
        "//src/test/java/com/google/devtools/build/lib/skyframe/packages:srcs",
        "//src/test/java/com/google/devtools/build/lib/skyframe/serialization:srcs",
        "//src/test/java/com/google/devtools/build/lib/skyframe:srcs",
        "//src/test/java/com/google/devtools/build/lib/skylark:srcs",
        "//src/test/java/com/google/devtools/build/lib/skylarkdebug/server:srcs",
        "//src/test/java/com/google/devtools/build/lib/skylarkinterface/processor:srcs",
        "//src/test/java/com/google/devtools/build/lib/unsafe:srcs",
        "//src/test/java/com/google/devtools/build/lib/blackbox:srcs",
    ],
    visibility = ["//src:__pkg__"],
)

# This should correspond to the list of "EMBEDDED_TOOLS" in TestConstants.java.bazel.
filegroup(
    name = "embedded_scripts",
    testonly = 1,
    data = [
        "//src/main/tools:build-runfiles",
        "//src/main/tools:linux-sandbox",
        "//src/main/tools:process-wrapper",
    ],
)

java_library(
    name = "testutil",
    testonly = 1,
    srcs = glob(
        ["testutil/*.java"],
        exclude = ["testutil/BazelPackageBuilderHelperForTesting.java"],
    ),
    visibility = ["//visibility:public"],
    deps = [
        ":guava_junit_truth",
        ":testutil/BazelPackageBuilderHelperForTesting",
        "//src/main/java/com/google/devtools/build/lib:bazel-main",
        "//src/main/java/com/google/devtools/build/lib:bazel-rules",
        "//src/main/java/com/google/devtools/build/lib:build-base",
        "//src/main/java/com/google/devtools/build/lib:events",
        "//src/main/java/com/google/devtools/build/lib:io",
        "//src/main/java/com/google/devtools/build/lib:os_util",
        "//src/main/java/com/google/devtools/build/lib:packages",
        "//src/main/java/com/google/devtools/build/lib:packages/BuilderFactoryForTesting",
        "//src/main/java/com/google/devtools/build/lib:skylark_semantics",
        "//src/main/java/com/google/devtools/build/lib:util",
        "//src/main/java/com/google/devtools/build/lib/actions",
        "//src/main/java/com/google/devtools/build/lib/clock",
        "//src/main/java/com/google/devtools/build/lib/collect",
        "//src/main/java/com/google/devtools/build/lib/collect/nestedset",
        "//src/main/java/com/google/devtools/build/lib/concurrent",
        "//src/main/java/com/google/devtools/build/lib/skyframe/packages",
        "//src/main/java/com/google/devtools/build/lib/vfs",
        "//src/main/java/com/google/devtools/build/lib/vfs/inmemoryfs",
        "//src/main/protobuf:invocation_policy_java_proto",
    ],
)

java_library(
    name = "testutil/BazelPackageBuilderHelperForTesting",
    testonly = 0,
    srcs = ["testutil/BazelPackageBuilderHelperForTesting.java"],
    visibility = ["//visibility:public"],
    deps = [
        "//src/main/java/com/google/devtools/build/lib:build-base",
        "//src/main/java/com/google/devtools/build/lib:packages",
        "//src/main/java/com/google/devtools/build/lib:skylark_semantics",
        "//src/main/java/com/google/devtools/build/lib/skyframe/packages",
        "//third_party:guava",
    ],
)

java_library(
    name = "foundations_testutil",
    testonly = 1,
    srcs = glob([
        "vfs/util/*.java",
        "events/util/*.java",
    ]),
    data = JNI_LIB,
    deps = [
        ":guava_junit_truth",
        ":testutil",
        "//src/main/java/com/google/devtools/build/lib:bazel-main",
        "//src/main/java/com/google/devtools/build/lib:events",
        "//src/main/java/com/google/devtools/build/lib:io",
        "//src/main/java/com/google/devtools/build/lib:packages",
        "//src/main/java/com/google/devtools/build/lib:unix",
        "//src/main/java/com/google/devtools/build/lib:util",
        "//src/main/java/com/google/devtools/build/lib/clock",
        "//src/main/java/com/google/devtools/build/lib/concurrent",
        "//src/main/java/com/google/devtools/build/lib/shell",
        "//src/main/java/com/google/devtools/build/lib/vfs",
        "//src/main/java/com/google/devtools/build/lib/vfs/inmemoryfs",
        "//src/main/java/com/google/devtools/build/lib/windows",
    ],
)

java_library(
    name = "test_runner",
    testonly = 1,
    srcs = ["AllTests.java"],
    visibility = ["//src/test/java/com/google/devtools/build:__subpackages__"],
    deps = [
        ":testutil",
        "//third_party:junit4",
    ],
)

java_test(
    name = "cmdline_test",
    size = "small",
    srcs = glob(["cmdline/*.java"]),
    tags = [
        "foundations",
    ],
    test_class = "com.google.devtools.build.lib.AllTests",
    deps = [
        ":guava_junit_truth",
        ":test_runner",
        ":testutil",
        "//src/main/java/com/google/devtools/build/lib/cmdline",
        "//src/main/java/com/google/devtools/build/lib/vfs:pathfragment",
    ],
)

java_test(
    name = "concurrent_test",
    size = "small",
    srcs = glob(["concurrent/*.java"]),
    flaky = True,
    tags = [
        "foundations",
    ],
    test_class = "com.google.devtools.build.lib.AllTests",
    deps = [
        ":guava_junit_truth",
        ":test_runner",
        ":testutil",
        "//src/main/java/com/google/devtools/build/lib/concurrent",
    ],
)

java_test(
    name = "collect_test",
    size = "small",
    srcs = glob(["collect/*.java"]),
    tags = [
        "foundations",
    ],
    test_class = "com.google.devtools.build.lib.AllTests",
    deps = [
        ":guava_junit_truth",
        ":test_runner",
        ":testutil",
        "//src/main/java/com/google/devtools/build/lib/collect",
        "//src/main/java/com/google/devtools/build/lib/collect/nestedset",
    ],
)

java_test(
    name = "collect_nestedset_test",
    size = "small",
    srcs = glob(["collect/nestedset/*.java"]),
    tags = [
        "foundations",
    ],
    test_class = "com.google.devtools.build.lib.AllTests",
    deps = [
        ":guava_junit_truth",
        ":test_runner",
        ":testutil",
        "//src/main/java/com/google/devtools/build/lib:util",
        "//src/main/java/com/google/devtools/build/lib/actions:commandline_item",
        "//src/main/java/com/google/devtools/build/lib/collect/nestedset",
        "//src/main/java/com/google/devtools/build/lib/collect/nestedset:fingerprint_cache",
        "//src/main/java/com/google/devtools/build/lib/collect/nestedset:testutils",
        "//src/main/java/com/google/devtools/build/lib/skyframe/serialization",
        "//src/main/java/com/google/devtools/build/lib/skyframe/serialization:constants",
        "//src/main/java/com/google/devtools/build/lib/skyframe/serialization/testutils",
        "//src/main/java/com/google/devtools/build/lib/vfs",
        "//third_party:mockito",
        "//third_party/protobuf:protobuf_java",
    ],
)

java_test(
    name = "dynamic_test",
    size = "small",
    srcs = glob(["dynamic/*.java"]),
    tags = [
        "no_windows",
    ],
    test_class = "com.google.devtools.build.lib.AllTests",
    deps = [
        ":actions_testutil",
        ":foundations_testutil",
        ":guava_junit_truth",
        ":testutil",
        "//src/main/java/com/google/devtools/build/lib:build-base",
        "//src/main/java/com/google/devtools/build/lib:io",
        "//src/main/java/com/google/devtools/build/lib/actions",
        "//src/main/java/com/google/devtools/build/lib/actions:localhost_capacity",
        "//src/main/java/com/google/devtools/build/lib/dynamic",
        "//src/main/java/com/google/devtools/build/lib/vfs",
        "//src/main/java/com/google/devtools/build/lib/vfs:pathfragment",
    ],
)

java_test(
    name = "events_test",
    size = "small",
    srcs = glob(["events/*.java"]),
    tags = [
        "foundations",
    ],
    test_class = "com.google.devtools.build.lib.AllTests",
    deps = [
        ":guava_junit_truth",
        ":test_runner",
        ":testutil",
        "//src/main/java/com/google/devtools/build/lib:events",
        "//src/main/java/com/google/devtools/build/lib:io",
        "//src/main/java/com/google/devtools/build/lib/skyframe/serialization/testutils",
        "//src/main/java/com/google/devtools/build/lib/vfs:pathfragment",
    ],
)

java_test(
    name = "testutil_test",
    size = "small",
    srcs = glob(["testutil/*Test.java"]),
    tags = [
        "foundations",
    ],
    test_class = "com.google.devtools.build.lib.AllTests",
    deps = [
        ":guava_junit_truth",
        ":test_runner",
        ":testutil",
    ],
)

java_test(
    name = "unix_test",
    size = "large",
    srcs = glob(["unix/*.java"]),
    tags = [
        "foundations",
        # This test cannot run on Windows, because it uses native Posix
        # functions via JNI.
        "no_windows",
    ],
    test_class = "com.google.devtools.build.lib.AllTests",
    deps = [
        ":foundations_testutil",
        ":guava_junit_truth",
        ":test_runner",
        ":testutil",
        ":vfs_symlink_aware_filesystem_test",
        "//src/main/java/com/google/devtools/build/lib:unix",
        "//src/main/java/com/google/devtools/build/lib:util",
        "//src/main/java/com/google/devtools/build/lib/vfs",
    ],
)

java_test(
    name = "util_test",
    size = "large",
    srcs = glob(
        [
            "util/*.java",
        ],
        exclude = [
            # java_rules_skylark doesn't support resource loading with
            # qualified paths.
            "util/ResourceFileLoaderTest.java",
        ] + ALL_WINDOWS_TESTS,
    ),
    tags = [
        "foundations",
        "no_windows",
    ],
    test_class = "com.google.devtools.build.lib.AllTests",
    deps = [
        ":guava_junit_truth",
        ":test_runner",
        ":testutil",
        "//src/main/java/com/google/devtools/build/lib:command-utils",
        "//src/main/java/com/google/devtools/build/lib:simple-log-handler",
        "//src/main/java/com/google/devtools/build/lib:single-line-formatter",
        "//src/main/java/com/google/devtools/build/lib:util",
        "//src/main/java/com/google/devtools/build/lib/analysis/platform",
        "//src/main/java/com/google/devtools/build/lib/cmdline",
        "//src/main/java/com/google/devtools/build/lib/shell",
        "//src/main/java/com/google/devtools/build/lib/skyframe/serialization/testutils",
        "//src/main/java/com/google/devtools/build/lib/vfs",
        "//src/main/java/com/google/devtools/build/lib/vfs/inmemoryfs",
        "//src/main/java/com/google/devtools/common/options",
        "//third_party:mockito",
        "//third_party:truth8",
    ],
)

java_test(
    name = "util_io_test",
    size = "large",
    srcs = glob(["util/io/*.java"]),
    tags = [
        "foundations",
        "no_windows",
    ],
    test_class = "com.google.devtools.build.lib.AllTests",
    deps = [
        ":guava_junit_truth",
        ":test_runner",
        ":testutil",
        "//src/main/java/com/google/devtools/build/lib:io",
        "//src/main/java/com/google/devtools/build/lib:util",
        "//src/main/protobuf:bazel_flags_java_proto",
        "//third_party:mockito",
    ],
)

java_test(
    name = "vfs_test",
    size = "large",
    srcs = glob(
        [
            "vfs/*.java",
        ],
        exclude = ALL_WINDOWS_TESTS,
    ),
    flaky = True,
    tags = [
        "foundations",
        "no_windows",
    ],
    test_class = "com.google.devtools.build.lib.AllTests",
    deps = [
        ":guava_junit_truth",
        ":foundations_testutil",
        ":test_runner",
        ":testutil",
        # TODO(laszlocsomor): remove the .../lib:unix this dependency, because
        # it prevents running the test on Windows.
        "//src/main/java/com/google/devtools/build/lib:unix",
        "//src/main/java/com/google/devtools/build/lib:util",
        "//src/main/java/com/google/devtools/build/lib/clock",
        "//src/main/java/com/google/devtools/build/lib/skyframe/serialization",
        "//src/main/java/com/google/devtools/build/lib/skyframe/serialization/testutils",
        "//src/main/java/com/google/devtools/build/lib/vfs",
        "//src/main/java/com/google/devtools/build/lib/vfs/inmemoryfs",
    ],
)

java_library(
    name = "vfs_filesystem_test",
    testonly = 1,
    srcs = ["vfs/FileSystemTest.java"],
    visibility = ["//visibility:private"],
    deps = [
        ":guava_junit_truth",
        ":testutil",
        "//src/main/java/com/google/devtools/build/lib:util",
        # TODO(laszlocsomor): remove the .../lib:unix this dependency, because
        # it prevents running the test on Windows.
        "//src/main/java/com/google/devtools/build/lib:unix",
        "//src/main/java/com/google/devtools/build/lib/vfs",
    ],
)

java_library(
    name = "vfs_symlink_aware_filesystem_test",
    testonly = 1,
    srcs = ["vfs/SymlinkAwareFileSystemTest.java"],
    visibility = ["//visibility:private"],
    deps = [
        ":guava_junit_truth",
        ":vfs_filesystem_test",
        "//src/main/java/com/google/devtools/build/lib:os_util",
        "//src/main/java/com/google/devtools/build/lib/vfs",
    ],
)

java_test(
    name = "vfs_inmemoryfs_test",
    size = "small",
    srcs = glob(["vfs/inmemoryfs/*.java"]),
    tags = [
        "foundations",
    ],
    test_class = "com.google.devtools.build.lib.AllTests",
    deps = [
        ":guava_junit_truth",
        ":test_runner",
        ":testutil",
        ":vfs_symlink_aware_filesystem_test",
        "//src/main/java/com/google/devtools/build/lib/clock",
        "//src/main/java/com/google/devtools/build/lib/vfs",
        "//src/main/java/com/google/devtools/build/lib/vfs/inmemoryfs",
    ],
)

# Tests that test Windows-specific functionality that run on other operating
# systems
java_test(
    name = "windows_test",
    srcs = CROSS_PLATFORM_WINDOWS_TESTS + ["vfs/PathAbstractTest.java"],
    jvm_flags = [
        "-Dblaze.os=Windows",
        "-Dbazel.windows_unix_root=C:/fake/msys",
    ],
    test_class = "com.google.devtools.build.lib.AllTests",
    deps = [
        ":foundations_testutil",
        ":guava_junit_truth",
        ":test_runner",
        ":testutil",
        "//src/main/java/com/google/devtools/build/lib:bazel-main",
        "//src/main/java/com/google/devtools/build/lib:util",
        "//src/main/java/com/google/devtools/build/lib/clock",
        "//src/main/java/com/google/devtools/build/lib/vfs",
        "//src/main/java/com/google/devtools/build/lib/vfs/inmemoryfs",
        "//src/main/java/com/google/devtools/build/lib/windows",
        "//src/main/java/com/google/devtools/common/options",
    ],
)

java_library(
    name = "windows_testutil",
    testonly = 1,
    srcs = [
        "windows/util/WindowsTestUtil.java",
    ],
    deps = [
        ":guava_junit_truth",
        "//src/main/java/com/google/devtools/build/lib/vfs",
        "//src/main/java/com/google/devtools/build/lib/windows",
        "//src/main/java/com/google/devtools/build/lib/windows/jni",
    ],
)

# Tests that need to run on Windows
java_test(
    name = "windows-tests",
    srcs = WINDOWS_ON_WINDOWS_TESTS,
    data = [
        ":MockSubprocess_deploy.jar",
    ] + JNI_LIB,
    jvm_flags = [
        "-Dbazel.windows_unix_root=C:/fake/msys",
    ],
    test_class = "com.google.devtools.build.lib.AllTests",
    deps = [
        ":guava_junit_truth",
        ":test_runner",
        ":testutil",
        ":windows_testutil",
        "//src/main/java/com/google/devtools/build/lib:os_util",
        "//src/main/java/com/google/devtools/build/lib/clock",
        "//src/main/java/com/google/devtools/build/lib/shell",
        "//src/main/java/com/google/devtools/build/lib/vfs",
        "//src/main/java/com/google/devtools/build/lib/vfs/inmemoryfs",
        "//src/main/java/com/google/devtools/build/lib/windows",
        "//src/main/java/com/google/devtools/build/lib/windows:windows_short_path",
        "//src/main/java/com/google/devtools/build/lib/windows/jni",
        "@bazel_tools//tools/java/runfiles",
    ],
)

java_library(
    name = "actions_testutil",
    testonly = 1,
    srcs = glob([
        "actions/util/*.java",
    ]),
    deps = [
        ":foundations_testutil",
        ":guava_junit_truth",
        ":test_runner",
        ":testutil",
        "//src/main/java/com/google/devtools/build/lib:bazel-main",
        "//src/main/java/com/google/devtools/build/lib:bazel-rules",
        "//src/main/java/com/google/devtools/build/lib:build-base",
        "//src/main/java/com/google/devtools/build/lib:events",
        "//src/main/java/com/google/devtools/build/lib:io",
        "//src/main/java/com/google/devtools/build/lib:packages",
        "//src/main/java/com/google/devtools/build/lib:util",
        "//src/main/java/com/google/devtools/build/lib/actions",
        "//src/main/java/com/google/devtools/build/lib/clock",
        "//src/main/java/com/google/devtools/build/lib/concurrent",
        "//src/main/java/com/google/devtools/build/lib/skyframe/serialization/autocodec",
        "//src/main/java/com/google/devtools/build/lib/vfs",
        "//src/main/java/com/google/devtools/build/lib/vfs/inmemoryfs",
        "//src/main/java/com/google/devtools/build/skyframe",
        "//src/main/java/com/google/devtools/build/skyframe:skyframe-objects",
        "//src/main/java/com/google/devtools/common/options",
        "//src/main/protobuf:action_cache_java_proto",
        "//third_party:jsr305",
        "//third_party:mockito",
    ],
)

java_test(
    name = "actions_test",
    srcs = glob([
        "actions/*.java",
    ]),
    test_class = "com.google.devtools.build.lib.AllTests",
    deps = [
        ":actions_testutil",
        ":foundations_testutil",
        ":guava_junit_truth",
        ":test_runner",
        ":testutil",
        "//src/main/java/com/google/devtools/build/lib:bazel-main",
        "//src/main/java/com/google/devtools/build/lib:bazel-rules",
        "//src/main/java/com/google/devtools/build/lib:build-base",
        "//src/main/java/com/google/devtools/build/lib:events",
        "//src/main/java/com/google/devtools/build/lib:packages",
        "//src/main/java/com/google/devtools/build/lib:util",
        "//src/main/java/com/google/devtools/build/lib/actions",
        "//src/main/java/com/google/devtools/build/lib/actions:localhost_capacity",
        "//src/main/java/com/google/devtools/build/lib/analysis/platform",
        "//src/main/java/com/google/devtools/build/lib/clock",
        "//src/main/java/com/google/devtools/build/lib/collect",
        "//src/main/java/com/google/devtools/build/lib/collect/nestedset",
        "//src/main/java/com/google/devtools/build/lib/concurrent",
        "//src/main/java/com/google/devtools/build/lib/rules/cpp",
        "//src/main/java/com/google/devtools/build/lib/rules/java:java-compilation",
        "//src/main/java/com/google/devtools/build/lib/rules/java:java-rules",
        "//src/main/java/com/google/devtools/build/lib/skyframe/serialization/autocodec",
        "//src/main/java/com/google/devtools/build/lib/skyframe/serialization/testutils",
        "//src/main/java/com/google/devtools/build/lib/vfs",
        "//src/main/java/com/google/devtools/build/lib/vfs/inmemoryfs",
        "//src/main/java/com/google/devtools/common/options",
        "//src/main/protobuf:action_cache_java_proto",
        "//third_party:jsr305",
        "//third_party:mockito",
        "//third_party:truth8",
    ],
)

java_library(
    name = "analysis_testutil",
    testonly = 1,
    srcs = glob(
        [
            "analysis/LicensingTests.java",
            "analysis/mock/BazelAnalysisMock.java",
            "analysis/util/*.java",
            "exec/util/*.java",
        ],
        exclude = ["analysis/util/DefaultBuildOptionsForTesting.java"],
    ),
    resources = [
        "analysis/mock/MOCK_CROSSTOOL",
    ],
    tags = ["analysis"],
    exports = [
        "//src/test/java/com/google/devtools/build/skyframe:testutil",
    ],
    deps = [
        ":actions_testutil",
        ":default_test_build_rules",
        ":foundations_testutil",
        ":guava_junit_truth",
        ":packages_testutil",
        ":testutil",
        "//src/main/java/com/google/devtools/build/lib:android-rules",
        "//src/main/java/com/google/devtools/build/lib:bazel-main",
        "//src/main/java/com/google/devtools/build/lib:bazel-repository",
        "//src/main/java/com/google/devtools/build/lib:bazel-rules",
        "//src/main/java/com/google/devtools/build/lib:build-base",
        "//src/main/java/com/google/devtools/build/lib:build-request-options",
        "//src/main/java/com/google/devtools/build/lib:core-rules",
        "//src/main/java/com/google/devtools/build/lib:events",
        "//src/main/java/com/google/devtools/build/lib:io",
        "//src/main/java/com/google/devtools/build/lib:keep-going-option",
        "//src/main/java/com/google/devtools/build/lib:loading-phase-threads-option",
        "//src/main/java/com/google/devtools/build/lib:packages",
        "//src/main/java/com/google/devtools/build/lib:proto-rules",
        "//src/main/java/com/google/devtools/build/lib:python-rules",
        "//src/main/java/com/google/devtools/build/lib:runtime",
        "//src/main/java/com/google/devtools/build/lib:skylark_semantics",
        "//src/main/java/com/google/devtools/build/lib:util",
        "//src/main/java/com/google/devtools/build/lib/actions",
        "//src/main/java/com/google/devtools/build/lib/actions:localhost_capacity",
        "//src/main/java/com/google/devtools/build/lib/analysis/platform",
        "//src/main/java/com/google/devtools/build/lib/causes",
        "//src/main/java/com/google/devtools/build/lib/clock",
        "//src/main/java/com/google/devtools/build/lib/collect",
        "//src/main/java/com/google/devtools/build/lib/collect/nestedset",
        "//src/main/java/com/google/devtools/build/lib/concurrent",
        "//src/main/java/com/google/devtools/build/lib/query2",
        "//src/main/java/com/google/devtools/build/lib/query2:query-output",
        "//src/main/java/com/google/devtools/build/lib/rules/apple",
        "//src/main/java/com/google/devtools/build/lib/rules/apple/swift",
        "//src/main/java/com/google/devtools/build/lib/rules/config",
        "//src/main/java/com/google/devtools/build/lib/rules/cpp",
        "//src/main/java/com/google/devtools/build/lib/rules/genquery",
        "//src/main/java/com/google/devtools/build/lib/rules/java:java-compilation",
        "//src/main/java/com/google/devtools/build/lib/rules/java:java-rules",
        "//src/main/java/com/google/devtools/build/lib/rules/objc",
        "//src/main/java/com/google/devtools/build/lib/rules/platform",
        "//src/main/java/com/google/devtools/build/lib/skyframe/packages",
        "//src/main/java/com/google/devtools/build/lib/vfs",
        "//src/main/java/com/google/devtools/build/skyframe",
        "//src/main/java/com/google/devtools/build/skyframe:skyframe-objects",
        "//src/main/java/com/google/devtools/common/options",
        "//src/main/java/com/google/devtools/common/options:invocation_policy",
        "//src/main/protobuf:extra_actions_base_java_proto",
        "//src/test/java/com/google/devtools/build/lib/skyframe:testutil",
        "//src/test/java/com/google/devtools/build/skyframe:testutil",
        "//third_party:jsr305",
        "//third_party:mockito",
        "//third_party/protobuf:protobuf_java",
    ],
)

java_library(
    name = "default_test_build_rules",
    testonly = 1,
    srcs = ["analysis/util/DefaultBuildOptionsForTesting.java"],
    deps = [
        "//src/main/java/com/google/devtools/build/lib:build-base",
        "//src/main/java/com/google/devtools/common/options",
    ],
)

java_library(
    name = "integration_testutil",
    testonly = 1,
    srcs = glob([
        "integration/util/*.java",
    ]),
    deps = [
        ":testutil",
        "//src/main/java/com/google/devtools/build/lib:build-base",
        "//src/main/java/com/google/devtools/build/lib/vfs",
    ],
)

java_test(
    name = "analysis_actions_test",
    srcs = glob([
        "analysis/actions/*.java",
    ]),
    tags = ["analysis"],
    test_class = "com.google.devtools.build.lib.AllTests",
    deps = [
        ":actions_testutil",
        ":analysis_testutil",
        ":foundations_testutil",
        ":guava_junit_truth",
        ":test_runner",
        ":testutil",
        "//src/main/java/com/google/devtools/build/lib:bazel-main",
        "//src/main/java/com/google/devtools/build/lib:bazel-rules",
        "//src/main/java/com/google/devtools/build/lib:build-base",
        "//src/main/java/com/google/devtools/build/lib:io",
        "//src/main/java/com/google/devtools/build/lib:util",
        "//src/main/java/com/google/devtools/build/lib/actions",
        "//src/main/java/com/google/devtools/build/lib/skyframe/serialization",
        "//src/main/java/com/google/devtools/build/lib/skyframe/serialization/testutils",
        "//src/main/java/com/google/devtools/build/lib/skyframe/serialization/testutils:depsutils",
        "//src/main/java/com/google/devtools/build/lib/vfs",
        "//src/main/protobuf:extra_actions_base_java_proto",
        "//third_party:jsr305",
        "//third_party:mockito",
    ],
)

java_test(
    name = "analysis_config_test",
    srcs = glob([
        "analysis/config/*.java",
    ]),
    tags = ["analysis"],
    test_class = "com.google.devtools.build.lib.AllTests",
    deps = [
        ":actions_testutil",
        ":analysis_testutil",
        ":foundations_testutil",
        ":guava_junit_truth",
        ":test_runner",
        ":testutil",
        "//src/main/java/com/google/devtools/build/lib:android-rules",
        "//src/main/java/com/google/devtools/build/lib:bazel-main",
        "//src/main/java/com/google/devtools/build/lib:bazel-rules",
        "//src/main/java/com/google/devtools/build/lib:build-base",
        "//src/main/java/com/google/devtools/build/lib:packages",
        "//src/main/java/com/google/devtools/build/lib:proto-rules",
        "//src/main/java/com/google/devtools/build/lib:python-rules",
        "//src/main/java/com/google/devtools/build/lib:util",
        "//src/main/java/com/google/devtools/build/lib/rules/cpp",
        "//src/main/java/com/google/devtools/build/lib/rules/java:java-compilation",
        "//src/main/java/com/google/devtools/build/lib/rules/java:java-rules",
        "//src/main/java/com/google/devtools/build/lib/rules/objc",
        "//src/main/java/com/google/devtools/build/lib/skyframe/serialization",
        "//src/main/java/com/google/devtools/build/lib/skyframe/serialization/testutils",
        "//src/main/java/com/google/devtools/build/lib/skyframe/trimming:trimmed_configuration_cache",
        "//src/main/java/com/google/devtools/build/lib/vfs",
        "//src/main/java/com/google/devtools/common/options",
        "//src/main/protobuf:extra_actions_base_java_proto",
        "//src/test/java/com/google/devtools/build/lib/skyframe/trimming:trimmable_test_fragments",
        "//third_party:auto_value",
        "//third_party:jsr305",
        "//third_party:mockito",
    ],
)

java_test(
    name = "analysis_constraints_test",
    srcs = glob([
        "analysis/constraints/*.java",
    ]),
    tags = ["analysis"],
    test_class = "com.google.devtools.build.lib.AllTests",
    deps = [
        ":actions_testutil",
        ":analysis_testutil",
        ":foundations_testutil",
        ":guava_junit_truth",
        ":test_runner",
        ":testutil",
        "//src/main/java/com/google/devtools/build/lib:bazel-main",
        "//src/main/java/com/google/devtools/build/lib:bazel-rules",
        "//src/main/java/com/google/devtools/build/lib:build-base",
        "//src/main/java/com/google/devtools/build/lib:packages",
        "//src/main/java/com/google/devtools/build/lib:util",
        "//src/main/java/com/google/devtools/build/lib/vfs",
    ],
)

java_test(
    name = "analysis_select_test",
    srcs = glob([
        "analysis/select/*.java",
    ]),
    tags = ["analysis"],
    test_class = "com.google.devtools.build.lib.AllTests",
    deps = [
        ":actions_testutil",
        ":analysis_testutil",
        ":foundations_testutil",
        ":guava_junit_truth",
        ":packages_testutil",
        ":test_runner",
        ":testutil",
        "//src/main/java/com/google/devtools/build/lib:bazel-main",
        "//src/main/java/com/google/devtools/build/lib:bazel-rules",
        "//src/main/java/com/google/devtools/build/lib:build-base",
        "//src/main/java/com/google/devtools/build/lib:packages",
        "//src/main/java/com/google/devtools/build/lib/actions",
        "//src/main/java/com/google/devtools/build/lib/vfs",
    ],
)

java_test(
    name = "analysis_test",
    srcs = glob([
        "analysis/*.java",
    ]),
    shard_count = 6,
    tags = [
        "analysis",
        "no_windows",
    ],
    test_class = "com.google.devtools.build.lib.AllTests",
    deps = [
        ":actions_testutil",
        ":analysis_testutil",
        ":foundations_testutil",
        ":guava_junit_truth",
        ":packages_testutil",
        ":test_runner",
        ":testutil",
        "//src/main/java/com/google/devtools/build/lib:bazel-main",
        "//src/main/java/com/google/devtools/build/lib:bazel-rules",
        "//src/main/java/com/google/devtools/build/lib:build-base",
        "//src/main/java/com/google/devtools/build/lib:events",
        "//src/main/java/com/google/devtools/build/lib:packages",
        "//src/main/java/com/google/devtools/build/lib:skylarkinterface",
        "//src/main/java/com/google/devtools/build/lib:util",
        "//src/main/java/com/google/devtools/build/lib/actions",
        "//src/main/java/com/google/devtools/build/lib/analysis/platform",
        "//src/main/java/com/google/devtools/build/lib/buildeventstream",
        "//src/main/java/com/google/devtools/build/lib/buildeventstream/proto:build_event_stream_java_proto",
        "//src/main/java/com/google/devtools/build/lib/causes",
        "//src/main/java/com/google/devtools/build/lib/collect",
        "//src/main/java/com/google/devtools/build/lib/collect/nestedset",
        "//src/main/java/com/google/devtools/build/lib/rules/cpp",
        "//src/main/java/com/google/devtools/build/lib/rules/java:java-compilation",
        "//src/main/java/com/google/devtools/build/lib/rules/java:java-rules",
        "//src/main/java/com/google/devtools/build/lib/skyframe/serialization/autocodec",
        "//src/main/java/com/google/devtools/build/lib/skyframe/serialization/testutils",
        "//src/main/java/com/google/devtools/build/lib/skyframe/serialization/testutils:depsutils",
        "//src/main/java/com/google/devtools/build/lib/vfs",
        "//src/main/java/com/google/devtools/build/lib/vfs/inmemoryfs",
        "//src/main/java/com/google/devtools/build/skyframe",
        "//src/main/java/com/google/devtools/build/skyframe:skyframe-objects",
        "//src/test/java/com/google/devtools/build/lib/rules/platform:testutil",
        "//src/test/java/com/google/devtools/build/lib/skyframe:testutil",
        "//third_party:auto_value",
        "//third_party:jsr305",
        "//third_party:mockito",
    ],
)

java_test(
    name = "graph_test",
    srcs = glob([
        "graph/*.java",
    ]),
    test_class = "com.google.devtools.build.lib.AllTests",
    deps = [
        ":foundations_testutil",
        ":guava_junit_truth",
        ":test_runner",
        ":testutil",
        "//src/main/java/com/google/devtools/build/lib:events",
        "//src/main/java/com/google/devtools/build/lib:packages-internal",
        "//src/main/java/com/google/devtools/build/lib/cmdline",
        "//src/main/java/com/google/devtools/build/lib/collect",
        "//src/main/java/com/google/devtools/build/lib/graph",
    ],
)

java_library(
    name = "packages_testutil",
    testonly = 1,
    srcs = glob([
        "packages/util/*.java",
    ]),
    resources = [
        "packages/util/MOCK_ANDROID_CROSSTOOL",
        "packages/util/MOCK_OSX_CROSSTOOL",
    ],
    deps = [
        ":default_test_build_rules",
        ":foundations_testutil",
        ":guava_junit_truth",
        ":testutil",
        "//src/main/java/com/google/devtools/build/docgen:docgen_javalib",
        "//src/main/java/com/google/devtools/build/lib:bazel-main",
        "//src/main/java/com/google/devtools/build/lib:bazel-rules",
        "//src/main/java/com/google/devtools/build/lib:build-base",
        "//src/main/java/com/google/devtools/build/lib:events",
        "//src/main/java/com/google/devtools/build/lib:io",
        "//src/main/java/com/google/devtools/build/lib:packages",
        "//src/main/java/com/google/devtools/build/lib:runtime",
        "//src/main/java/com/google/devtools/build/lib:skylark_semantics",
        "//src/main/java/com/google/devtools/build/lib:util",
        "//src/main/java/com/google/devtools/build/lib/actions",
        "//src/main/java/com/google/devtools/build/lib/clock",
        "//src/main/java/com/google/devtools/build/lib/rules/cpp",
        "//src/main/java/com/google/devtools/build/lib/skyframe/packages",
        "//src/main/java/com/google/devtools/build/lib/vfs",
        "//src/main/java/com/google/devtools/build/lib/vfs/inmemoryfs",
        "//src/main/java/com/google/devtools/build/skyframe",
        "//src/main/java/com/google/devtools/build/skyframe:skyframe-objects",
        "//src/main/java/com/google/devtools/common/options",
        "//src/main/java/com/google/devtools/common/options:invocation_policy",
        "//src/main/protobuf:build_java_proto",
        "//src/main/protobuf:crosstool_config_java_proto",
        "//src/main/protobuf:extra_actions_base_java_proto",
        "//third_party:jsr305",
        "//third_party:mockito",
        "//third_party/protobuf:protobuf_java",
    ],
)

java_test(
    name = "packages_test",
    srcs = glob(
        [
            "packages/*.java",
        ],
        exclude = [
            "packages/BazelDocumentationTest.java",
        ],
    ),
    test_class = "com.google.devtools.build.lib.AllTests",
    deps = [
        ":actions_testutil",
        ":analysis_testutil",
        ":foundations_testutil",
        ":guava_junit_truth",
        ":packages_testutil",
        ":test_runner",
        ":testutil",
        "//src/main/java/com/google/devtools/build/lib:bazel-main",
        "//src/main/java/com/google/devtools/build/lib:bazel-rules",
        "//src/main/java/com/google/devtools/build/lib:build-base",
        "//src/main/java/com/google/devtools/build/lib:events",
        "//src/main/java/com/google/devtools/build/lib:packages",
        "//src/main/java/com/google/devtools/build/lib:skylark_semantics",
        "//src/main/java/com/google/devtools/build/lib:skylarkinterface",
        "//src/main/java/com/google/devtools/build/lib:util",
        "//src/main/java/com/google/devtools/build/lib/actions",
        "//src/main/java/com/google/devtools/build/lib/skyframe/serialization",
        "//src/main/java/com/google/devtools/build/lib/skyframe/serialization/testutils",
        "//src/main/java/com/google/devtools/build/lib/vfs",
        "//src/main/java/com/google/devtools/build/skyframe",
        "//src/main/java/com/google/devtools/build/skyframe:skyframe-objects",
        "//src/main/protobuf:build_java_proto",
        "//third_party:jsr305",
        "//third_party:mockito",
        "//third_party/protobuf:protobuf_java",
    ],
)

java_test(
    name = "pkgcache_test",
    srcs = glob(
        [
            "pkgcache/*.java",
        ],
    ),
    test_class = "com.google.devtools.build.lib.AllTests",
    deps = [
        ":analysis_testutil",
        ":default_test_build_rules",
        ":guava_junit_truth",
        ":packages_testutil",
        ":test_runner",
        ":testutil",
        "//src/main/java/com/google/devtools/build/lib:bazel-rules",
        "//src/main/java/com/google/devtools/build/lib:build-base",
        "//src/main/java/com/google/devtools/build/lib:io",
        "//src/main/java/com/google/devtools/build/lib:packages",
        "//src/main/java/com/google/devtools/build/lib/actions",
        "//src/main/java/com/google/devtools/build/lib/vfs/inmemoryfs",
        "//src/main/java/com/google/devtools/build/skyframe",
        "//src/main/java/com/google/devtools/build/skyframe:skyframe-objects",
        "//src/main/java/com/google/devtools/common/options",
        "//src/main/java/com/google/devtools/common/options:invocation_policy",
        "//third_party:jsr305",
    ],
)

java_test(
    name = "server_test",
    srcs = glob([
        "server/*.java",
    ]),
    tags = [
        "no_windows",
        "server",
    ],
    test_class = "com.google.devtools.build.lib.AllTests",
    deps = [
        ":foundations_testutil",
        ":guava_junit_truth",
        ":test_runner",
        ":testutil",
        "//src/main/java/com/google/devtools/build/lib:bazel-main",
        "//src/main/java/com/google/devtools/build/lib:io",
        "//src/main/java/com/google/devtools/build/lib:runtime",
        "//src/main/java/com/google/devtools/build/lib:server",
        "//src/main/java/com/google/devtools/build/lib:unix",
        "//src/main/java/com/google/devtools/build/lib:util",
        "//src/main/java/com/google/devtools/build/lib/collect",
        "//src/main/java/com/google/devtools/build/lib/vfs",
        "//src/main/protobuf:command_server_java_proto",
        "//third_party:jsr305",
        "//third_party:mockito",
        "//third_party/grpc:grpc-jar",
        "//third_party/protobuf:protobuf_java",
    ],
)

java_test(
    name = "syntax_test",
    srcs = glob([
        "syntax/*.java",
    ]),
    test_class = "com.google.devtools.build.lib.AllTests",
    deps = [
        ":foundations_testutil",
        ":guava_junit_truth",
        ":syntax_testutil",
        ":test_runner",
        ":testutil",
        "//src/main/java/com/google/devtools/build/lib:bazel-main",
        "//src/main/java/com/google/devtools/build/lib:bazel-rules",
        "//src/main/java/com/google/devtools/build/lib:build-base",
        "//src/main/java/com/google/devtools/build/lib:events",
        "//src/main/java/com/google/devtools/build/lib:packages",
        "//src/main/java/com/google/devtools/build/lib:skylark_semantics",
        "//src/main/java/com/google/devtools/build/lib:skylarkinterface",
        "//src/main/java/com/google/devtools/build/lib:util",
        "//src/main/java/com/google/devtools/build/lib/actions",
        "//src/main/java/com/google/devtools/build/lib/collect",
        "//src/main/java/com/google/devtools/build/lib/collect/nestedset",
        "//src/main/java/com/google/devtools/build/lib/concurrent",
        "//src/main/java/com/google/devtools/build/lib/skyframe/serialization/testutils",
        "//src/main/java/com/google/devtools/build/lib/vfs",
        "//third_party:jsr305",
    ],
)

java_test(
    name = "profiler-tests",
    srcs = glob(
        [
            "profiler/*.java",
        ],
        exclude = ["profiler/*Benchmark.java"],
    ),
    test_class = "com.google.devtools.build.lib.AllTests",
    deps = [
        ":foundations_testutil",
        ":guava_junit_truth",
        ":syntax_testutil",
        ":test_runner",
        ":testutil",
        "//src/main/java/com/google/devtools/build/lib/clock",
        "//src/main/java/com/google/devtools/build/lib/profiler",
        "//src/main/java/com/google/devtools/build/lib/profiler:profiler-output",
        "//src/main/java/com/google/devtools/build/lib/vfs",
        "//third_party:guava",
        "//third_party:jsr305",
        "//third_party:mockito",
    ],
)

java_test(
    name = "BazelDocumentationTests",
    size = "medium",
    srcs = ["packages/BazelDocumentationTest.java"],
    data = [
        "//site:docs/user-manual.html",
    ],
    test_class = "com.google.devtools.build.lib.packages.BazelDocumentationTest",
    deps = [
        ":foundations_testutil",
        ":guava_junit_truth",
        ":packages_testutil",
        ":windows_testutil",
        "//src/main/java/com/google/devtools/build/lib:bazel-main",
        "//src/main/java/com/google/devtools/build/lib:bazel-rules",
        "//src/main/java/com/google/devtools/build/lib:os_util",
        "//src/main/java/com/google/devtools/build/lib:packages",
        "//third_party:jsr305",
        "@bazel_tools//tools/java/runfiles",
    ],
)

java_library(
    name = "syntax_testutil",
    testonly = 1,
    srcs = glob([
        "syntax/util/*.java",
    ]),
    visibility = ["//visibility:public"],
    deps = [
        ":foundations_testutil",
        ":guava_junit_truth",
        ":test_runner",
        ":testutil",
        "//src/main/java/com/google/devtools/build/lib:bazel-main",
        "//src/main/java/com/google/devtools/build/lib:bazel-rules",
        "//src/main/java/com/google/devtools/build/lib:build-base",
        "//src/main/java/com/google/devtools/build/lib:events",
        "//src/main/java/com/google/devtools/build/lib:packages",
        "//src/main/java/com/google/devtools/build/lib:util",
        "//src/main/java/com/google/devtools/build/lib/actions",
        "//src/main/java/com/google/devtools/build/lib/collect",
        "//src/main/java/com/google/devtools/build/lib/concurrent",
        "//src/main/java/com/google/devtools/build/lib/vfs",
        "//third_party:jsr305",
    ],
)

java_test(
    name = "filegroup-rules-tests",
    srcs = glob(["rules/filegroup/*.java"]),
    tags = ["rules"],
    test_class = "com.google.devtools.build.lib.AllTests",
    deps = [
        ":actions_testutil",
        ":analysis_testutil",
        ":guava_junit_truth",
        ":testutil",
        "//src/main/java/com/google/devtools/build/lib:bazel-main",
        "//src/main/java/com/google/devtools/build/lib:bazel-rules",
        "//src/main/java/com/google/devtools/build/lib:build-base",
        "//src/main/java/com/google/devtools/build/lib:util",
        "//src/main/java/com/google/devtools/build/lib/rules/java:java-compilation",
        "//src/main/java/com/google/devtools/build/lib/rules/java:java-rules",
        "//src/main/java/com/google/devtools/build/lib/vfs",
        "//third_party:jsr305",
    ],
)

JAVA_COMPILE_ACTION_TEST_HELPER_SRCS = ["rules/java/JavaCompileActionTestHelper.java"]

java_library(
    name = "java_compile_action_test_helper",
    testonly = 1,
    srcs = JAVA_COMPILE_ACTION_TEST_HELPER_SRCS,
    deps = [
        "//src/java_tools/buildjar/java/com/google/devtools/build/buildjar:optionsparser",
        "//src/main/java/com/google/devtools/build/lib:build-base",
        "//third_party:guava",
    ],
)

java_test(
    name = "java-rules-tests",
    srcs = glob(
        ["rules/java/*.java"],
        exclude = JAVA_COMPILE_ACTION_TEST_HELPER_SRCS,
    ),
    tags = ["rules"],
    test_class = "com.google.devtools.build.lib.AllTests",
    deps = [
        ":actions_testutil",
        ":analysis_testutil",
        ":guava_junit_truth",
        ":java_compile_action_test_helper",
        ":testutil",
        "//src/main/java/com/google/devtools/build/lib:bazel-main",
        "//src/main/java/com/google/devtools/build/lib:bazel-rules",
        "//src/main/java/com/google/devtools/build/lib:build-base",
        "//src/main/java/com/google/devtools/build/lib:filetype",
        "//src/main/java/com/google/devtools/build/lib:packages-internal",
        "//src/main/java/com/google/devtools/build/lib/actions",
        "//src/main/java/com/google/devtools/build/lib/cmdline",
        "//src/main/java/com/google/devtools/build/lib/collect",
        "//src/main/java/com/google/devtools/build/lib/collect/nestedset",
        "//src/main/java/com/google/devtools/build/lib/rules/java:java-compilation",
        "//src/main/java/com/google/devtools/build/lib/rules/java:java-rules",
        "//src/main/java/com/google/devtools/build/lib/vfs",
        "//third_party:jsr305",
    ],
)

java_test(
    name = "StrictDepsUtilsTest",
    srcs = ["rules/java/proto/StrictDepsUtilsTest.java"],
    deps = [
        ":analysis_testutil",
        ":guava_junit_truth",
        "//src/main/java/com/google/devtools/build/lib:build-base",
        "//src/main/java/com/google/devtools/build/lib/rules/java:java-rules",
    ],
)

java_test(
    name = "test-rules-tests",
    srcs = ["rules/test/SkylarkTestingModuleTest.java"],
    tags = ["rules"],
    test_class = "com.google.devtools.build.lib.rules.test.SkylarkTestingModuleTest",
    deps = [
        ":analysis_testutil",
        ":guava_junit_truth",
        "//src/main/java/com/google/devtools/build/lib:build-base",
        "//third_party/protobuf:protobuf_java",
    ],
)

java_test(
    name = "bazel-rules-tests",
    srcs = glob([
        "bazel/rules/**/*.java",
    ]),
    resources = [
        "//tools/android:android_sdk_repository_template.bzl",
    ] + glob(["bazel/rules/**/*.txt"]),
    tags = ["rules"],
    test_class = "com.google.devtools.build.lib.AllTests",
    deps = [
        ":actions_testutil",
        ":analysis_testutil",
        ":foundations_testutil",
        ":guava_junit_truth",
        ":packages_testutil",
        ":test_runner",
        ":testutil",
        "//src/main/java/com/google/devtools/build/lib:android-rules",
        "//src/main/java/com/google/devtools/build/lib:bazel-main",
        "//src/main/java/com/google/devtools/build/lib:bazel-rules",
        "//src/main/java/com/google/devtools/build/lib:build-base",
        "//src/main/java/com/google/devtools/build/lib:core-rules",
        "//src/main/java/com/google/devtools/build/lib:core-workspace-rules",
        "//src/main/java/com/google/devtools/build/lib:events",
        "//src/main/java/com/google/devtools/build/lib:packages-internal",
        "//src/main/java/com/google/devtools/build/lib:python-rules",
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
        "//src/test/java/com/google/devtools/build/lib/rules/android:AndroidLocalTestTest",
    ],
)

java_test(
    name = "runtime-tests",
    srcs = glob([
        "runtime/*.java",
        "runtime/commands/*.java",
    ]),
    test_class = "com.google.devtools.build.lib.AllTests",
    deps = [
        ":actions_testutil",
        ":foundations_testutil",
        ":guava_junit_truth",
        ":test_runner",
        ":testutil",
        "//src/main/java/com/google/devtools/build/lib:bazel-modules",
        "//src/main/java/com/google/devtools/build/lib:bazel-rules",
        "//src/main/java/com/google/devtools/build/lib:build-base",
        "//src/main/java/com/google/devtools/build/lib:io",
        "//src/main/java/com/google/devtools/build/lib:packages",
        "//src/main/java/com/google/devtools/build/lib:runtime",
        "//src/main/java/com/google/devtools/build/lib:unix",
        "//src/main/java/com/google/devtools/build/lib:util",
        "//src/main/java/com/google/devtools/build/lib/actions",
        "//src/main/java/com/google/devtools/build/lib/bazel/repository/downloader",
        "//src/main/java/com/google/devtools/build/lib/buildeventstream",
        "//src/main/java/com/google/devtools/build/lib/buildeventstream/proto:build_event_stream_java_proto",
        "//src/main/java/com/google/devtools/build/lib/buildeventstream/transports",
        "//src/main/java/com/google/devtools/build/lib/clock",
        "//src/main/java/com/google/devtools/build/lib/collect/nestedset",
        "//src/main/java/com/google/devtools/build/lib/sandbox",
        "//src/main/java/com/google/devtools/build/lib/vfs",
        "//src/main/java/com/google/devtools/build/lib/vfs/inmemoryfs",
        "//src/main/java/com/google/devtools/common/options",
        "//src/main/java/com/google/devtools/common/options:invocation_policy",
        "//src/main/protobuf:command_line_java_proto",
        "//src/main/protobuf:invocation_policy_java_proto",
        "//src/main/protobuf:test_status_java_proto",
        "//src/test/java/com/google/devtools/common/options:testutils",
        "//third_party:mockito",
    ],
)

java_test(
    name = "remote-tests",
    srcs = glob(["remote/**/*.java"]),
    test_class = "com.google.devtools.build.lib.AllTests",
    deps = [
        ":analysis_testutil",
        ":foundations_testutil",
        ":guava_junit_truth",
        ":test_runner",
        ":testutil",
        "//src/main/java/com/google/devtools/build/lib:build-base",
        "//src/main/java/com/google/devtools/build/lib:events",
        "//src/main/java/com/google/devtools/build/lib:exitcode-external",
        "//src/main/java/com/google/devtools/build/lib:io",
        "//src/main/java/com/google/devtools/build/lib:util",
        "//src/main/java/com/google/devtools/build/lib/actions",
        "//src/main/java/com/google/devtools/build/lib/actions:localhost_capacity",
        "//src/main/java/com/google/devtools/build/lib/authandtls",
        "//src/main/java/com/google/devtools/build/lib/buildeventstream",
        "//src/main/java/com/google/devtools/build/lib/clock",
        "//src/main/java/com/google/devtools/build/lib/remote",
        "//src/main/java/com/google/devtools/build/lib/remote/blobstore",
        "//src/main/java/com/google/devtools/build/lib/remote/blobstore/http",
        "//src/main/java/com/google/devtools/build/lib/remote/logging",
        "//src/main/java/com/google/devtools/build/lib/remote/util",
        "//src/main/java/com/google/devtools/build/lib/vfs",
        "//src/main/java/com/google/devtools/build/lib/vfs/inmemoryfs",
        "//src/main/java/com/google/devtools/common/options",
        "//src/main/protobuf:remote_execution_log_java_proto",
        "//third_party:api_client",
        "//third_party:auth",
        "//third_party:mockito",
        "//third_party:netty",
        "//third_party/grpc:grpc-jar",
        "//third_party/protobuf:protobuf_java",
        "@googleapis//:google_bytestream_bytestream_java_grpc",
        "@googleapis//:google_bytestream_bytestream_java_proto",
        "@googleapis//:google_longrunning_operations_java_proto",
        "@googleapis//:google_rpc_code_java_proto",
        "@googleapis//:google_rpc_error_details_java_proto",
        "@googleapis//:google_rpc_status_java_proto",
        "@remoteapis//:build_bazel_remote_execution_v2_remote_execution_java_grpc",
        "@remoteapis//:build_bazel_remote_execution_v2_remote_execution_java_proto",
        "@remoteapis//:build_bazel_semver_semver_java_proto",
    ],
)

java_library(
    name = "sandboxfs-base-tests",
    testonly = 1,
    srcs = ["sandbox/BaseSandboxfsProcessTest.java"],
    deps = [
        ":testutil",
        "//src/main/java/com/google/devtools/build/lib/sandbox",
        "//src/main/java/com/google/devtools/build/lib/vfs",
        "//third_party:guava",
        "//third_party:junit4",
        "//third_party:truth",
    ],
)

java_test(
    name = "sandbox-tests",
    srcs = glob(
        ["sandbox/*.java"],
        exclude = [
            "sandbox/BaseSandboxfsProcessTest.java",
            "sandbox/RealSandboxfsProcessTest.java",
        ],
    ),
    data = [":embedded_scripts"],
    local = 1,
    tags = ["no_windows"],
    test_class = "com.google.devtools.build.lib.AllTests",
    deps = [
        ":analysis_testutil",
        ":foundations_testutil",
        ":guava_junit_truth",
        ":sandboxfs-base-tests",
        ":testutil",
        "//src/main/java/com/google/devtools/build/lib:bazel-rules",
        "//src/main/java/com/google/devtools/build/lib:build-base",
        "//src/main/java/com/google/devtools/build/lib:os_util",
        "//src/main/java/com/google/devtools/build/lib:util",
        "//src/main/java/com/google/devtools/build/lib/actions",
        "//src/main/java/com/google/devtools/build/lib/sandbox",
        "//src/main/java/com/google/devtools/build/lib/vfs",
        "//src/main/java/com/google/devtools/build/lib/vfs/inmemoryfs",
        "//src/main/java/com/google/devtools/common/options",
    ],
)

java_test(
    name = "sandboxfs-integration-tests",
    srcs = ["sandbox/RealSandboxfsProcessTest.java"],
    data = [":embedded_scripts"],
    local = 1,
    tags = [
        # On macOS:
        #   sudo sysctl -w vfs.generic.osxfuse.tunables.allow_other=1
        # Test requires:
        #   --test_env=SANDBOXFS=/path/to/sandboxfs
        "manual",
        "no-sandbox",
        "no_windows",
    ],
    test_class = "com.google.devtools.build.lib.AllTests",
    deps = [
        ":sandboxfs-base-tests",
        ":test_runner",
        "//src/main/java/com/google/devtools/build/lib/sandbox",
        "//src/main/java/com/google/devtools/build/lib/vfs",
        "//third_party:junit4",
    ],
)

java_test(
    name = "standalone-tests",
    srcs = glob(["standalone/*.java"]),
    data = [":embedded_scripts"],
    tags = ["no_windows"],
    test_class = "com.google.devtools.build.lib.AllTests",
    deps = [
        ":actions_testutil",
        ":analysis_testutil",
        ":foundations_testutil",
        ":guava_junit_truth",
        ":integration_testutil",
        ":testutil",
        "//src/main/java/com/google/devtools/build/lib:bazel-rules",
        "//src/main/java/com/google/devtools/build/lib:build-base",
        "//src/main/java/com/google/devtools/build/lib:events",
        "//src/main/java/com/google/devtools/build/lib:io",
        "//src/main/java/com/google/devtools/build/lib:os_util",
        "//src/main/java/com/google/devtools/build/lib:util",
        "//src/main/java/com/google/devtools/build/lib/actions",
        "//src/main/java/com/google/devtools/build/lib/actions:localhost_capacity",
        "//src/main/java/com/google/devtools/build/lib/clock",
        "//src/main/java/com/google/devtools/build/lib/exec/local",
        "//src/main/java/com/google/devtools/build/lib/exec/local:options",
        "//src/main/java/com/google/devtools/build/lib/rules/apple",
        "//src/main/java/com/google/devtools/build/lib/shell",
        "//src/main/java/com/google/devtools/build/lib/standalone",
        "//src/main/java/com/google/devtools/build/lib/vfs",
        "//src/main/java/com/google/devtools/common/options",
    ],
)

java_test(
    name = "exec-tests",
    srcs = glob(["exec/*.java"]),
    tags = ["exec"],
    test_class = "com.google.devtools.build.lib.AllTests",
    deps = [
        ":actions_testutil",
        ":analysis_testutil",
        ":foundations_testutil",
        ":guava_junit_truth",
        ":test_runner",
        ":testutil",
        "//src/main/java/com/google/devtools/build/lib:bazel-rules",
        "//src/main/java/com/google/devtools/build/lib:build-base",
        "//src/main/java/com/google/devtools/build/lib:events",
        "//src/main/java/com/google/devtools/build/lib:io",
        "//src/main/java/com/google/devtools/build/lib:util",
        "//src/main/java/com/google/devtools/build/lib/actions",
        "//src/main/java/com/google/devtools/build/lib/buildeventstream/proto:build_event_stream_java_proto",
        "//src/main/java/com/google/devtools/build/lib/cmdline",
        "//src/main/java/com/google/devtools/build/lib/skyframe/serialization/testutils",
        "//src/main/java/com/google/devtools/build/lib/vfs",
        "//src/main/java/com/google/devtools/build/lib/vfs/inmemoryfs",
        "//src/main/java/com/google/devtools/common/options",
        "//src/main/protobuf:spawn_java_proto",
        "//src/main/protobuf:test_status_java_proto",
        "//third_party:mockito",
        "//third_party/protobuf:protobuf_java",
    ],
)

java_test(
    name = "exec-local-tests",
    srcs = glob(["exec/local/*.java"]),
    tags = ["exec"],
    test_class = "com.google.devtools.build.lib.AllTests",
    deps = [
        ":analysis_testutil",
        ":guava_junit_truth",
        ":testutil",
        "//src/main/java/com/google/devtools/build/lib:build-base",
        "//src/main/java/com/google/devtools/build/lib:io",
        "//src/main/java/com/google/devtools/build/lib:unix",
        "//src/main/java/com/google/devtools/build/lib:util",
        "//src/main/java/com/google/devtools/build/lib/actions",
        "//src/main/java/com/google/devtools/build/lib/actions:localhost_capacity",
        "//src/main/java/com/google/devtools/build/lib/exec/local",
        "//src/main/java/com/google/devtools/build/lib/exec/local:options",
        "//src/main/java/com/google/devtools/build/lib/shell",
        "//src/main/java/com/google/devtools/build/lib/vfs",
        "//src/main/java/com/google/devtools/build/lib/vfs/inmemoryfs",
        "//src/main/java/com/google/devtools/common/options",
        "//src/main/tools:process-wrapper",
        "//src/test/shell/integration:spend_cpu_time",
        "//third_party:mockito",
        "//third_party:truth8",
    ],
)

java_test(
    name = "ProtoCompileActionBuilderTest",
    srcs = ["rules/proto/ProtoCompileActionBuilderTest.java"],
    deps = [
        ":actions_testutil",
        ":guava_junit_truth",
        "//src/main/java/com/google/devtools/build/lib:build-base",
        "//src/main/java/com/google/devtools/build/lib:proto-rules",
        "//src/main/java/com/google/devtools/build/lib:util",
        "//src/main/java/com/google/devtools/build/lib/actions",
        "//src/main/java/com/google/devtools/build/lib/cmdline",
        "//src/main/java/com/google/devtools/build/lib/collect",
        "//src/main/java/com/google/devtools/build/lib/collect/nestedset",
        "//src/main/java/com/google/devtools/build/lib/vfs",
        "//src/main/java/com/google/devtools/build/lib/vfs/inmemoryfs",
        "//third_party:mockito",
    ],
)

java_test(
    name = "ProtoLangToolchainTest",
    srcs = ["rules/proto/ProtoLangToolchainTest.java"],
    deps = [
        ":actions_testutil",
        ":analysis_testutil",
        ":guava_junit_truth",
        "//src/main/java/com/google/devtools/build/lib:build-base",
        "//src/main/java/com/google/devtools/build/lib:proto-rules",
        "//src/main/java/com/google/devtools/build/lib:util",
        "//src/main/java/com/google/devtools/build/lib/cmdline",
    ],
)

java_test(
    name = "BazelProtoLibraryTest",
    srcs = ["rules/proto/BazelProtoLibraryTest.java"],
    deps = [
        ":actions_testutil",
        ":analysis_testutil",
        ":guava_junit_truth",
        "//src/main/java/com/google/devtools/build/lib:build-base",
        "//src/main/java/com/google/devtools/build/lib:proto-rules",
        "//src/main/java/com/google/devtools/build/lib/actions",
        "//src/main/java/com/google/devtools/build/lib/vfs",
    ],
)

java_test(
    name = "repository-tests",
    srcs = glob(["repository/*.java"]),
    test_class = "com.google.devtools.build.lib.AllTests",
    deps = [
        ":analysis_testutil",
        "//src/main/java/com/google/devtools/build/lib:build-base",
        "//src/main/java/com/google/devtools/build/lib:events",
        "//src/main/java/com/google/devtools/build/lib:io",
        "//src/main/java/com/google/devtools/build/lib:packages-internal",
        "//src/main/java/com/google/devtools/build/lib:skylark_semantics",
        "//src/main/java/com/google/devtools/build/lib/actions",
        "//src/main/java/com/google/devtools/build/lib/vfs",
        "//src/main/java/com/google/devtools/build/skyframe",
        "//src/main/java/com/google/devtools/build/skyframe:skyframe-objects",
        "//third_party:auto_value",
        "//third_party:guava",
        "//third_party:truth",
    ],
)

java_test(
    name = "rules-tests",
    srcs = glob(["rules/*.java"]),
    test_class = "com.google.devtools.build.lib.AllTests",
    deps = [
        ":actions_testutil",
        ":analysis_testutil",
        ":guava_junit_truth",
        ":packages_testutil",
        ":testutil",
        "//src/main/java/com/google/devtools/build/lib:build-base",
        "//src/main/java/com/google/devtools/build/lib:events",
        "//src/main/java/com/google/devtools/build/lib:io",
        "//src/main/java/com/google/devtools/build/lib:packages-internal",
        "//src/main/java/com/google/devtools/build/lib:proto-rules",
        "//src/main/java/com/google/devtools/build/lib/actions",
        "//src/main/java/com/google/devtools/build/lib/collect",
        "//src/main/java/com/google/devtools/build/lib/collect/nestedset",
        "//src/main/java/com/google/devtools/build/lib/rules/cpp",
        "//src/main/java/com/google/devtools/build/skyframe",
        "//src/main/java/com/google/devtools/build/skyframe:skyframe-objects",
        "//third_party:auto_value",
    ],
)

java_test(
    name = "worker-tests",
    srcs = glob(["worker/*Test.java"]),
    test_class = "com.google.devtools.build.lib.AllTests",
    deps = [
        ":foundations_testutil",
        ":guava_junit_truth",
        ":test_runner",
        ":testutil",
        "//src/main/java/com/google/devtools/build/lib:os_util",
        "//src/main/java/com/google/devtools/build/lib/sandbox",
        "//src/main/java/com/google/devtools/build/lib/vfs",
        "//src/main/java/com/google/devtools/build/lib/vfs/inmemoryfs",
        "//src/main/java/com/google/devtools/build/lib/worker",
        "//src/main/protobuf:worker_protocol_java_proto",
    ],
)

java_library(
    name = "guava_junit_truth",
    testonly = 1,
    visibility = ["//visibility:private"],
    exports = [
        "//third_party:guava",
        "//third_party:guava-testlib",
        "//third_party:junit4",
        "//third_party:truth",
    ],
)

java_binary(
    name = "MockSubprocess",
    srcs = ["windows/MockSubprocess.java"],
)

java_library(
    name = "ExampleWorker-lib",
    srcs = glob(["worker/ExampleWorker*.java"]),
    visibility = [
        "//src/test/shell/integration:__pkg__",
    ],
    deps = [
        "//src/main/java/com/google/devtools/common/options",
        "//src/main/protobuf:worker_protocol_java_proto",
        "//third_party:guava",
    ],
)

java_binary(
    name = "ExampleWorker",
    main_class = "com.google.devtools.build.lib.worker.ExampleWorker",
    visibility = [
        "//src/test/shell/integration:__pkg__",
    ],
    runtime_deps = [
        ":ExampleWorker-lib",
    ],
)

TEST_SUITES = [
    "ziputils",
    "rules",
    "analysis",
    "foundations",
    "shell",
    "server",
    "skyframe",
    "exec",
]

[
    test_suite(
        name = n,
        tags = [n],
    )
    for n in TEST_SUITES
]

test_suite(
    name = "others",
    tags = ["-" + n for n in TEST_SUITES],
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
        "//src/test/java/com/google/devtools/build/lib/analysis/platform:all_windows_tests",
        "//src/test/java/com/google/devtools/build/lib/analysis/skylark/annotations/processor:all_windows_tests",
        "//src/test/java/com/google/devtools/build/lib/analysis/whitelisting:all_windows_tests",
        "//src/test/java/com/google/devtools/build/lib/bazel/repository:all_windows_tests",
        "//src/test/java/com/google/devtools/build/lib/blackbox/tests:blackbox_windows_tests",
        "//src/test/java/com/google/devtools/build/lib/buildeventservice:all_windows_tests",
        "//src/test/java/com/google/devtools/build/lib/buildeventstream/transports:all_windows_tests",
        "//src/test/java/com/google/devtools/build/lib/buildtool:all_windows_tests",
        "//src/test/java/com/google/devtools/build/lib/profiler/callcounts:all_windows_tests",
        "//src/test/java/com/google/devtools/build/lib/profiler/memory:all_windows_tests",
        "//src/test/java/com/google/devtools/build/lib/rules/android:all_windows_tests",
        "//src/test/java/com/google/devtools/build/lib/rules/apple:all_windows_tests",
        "//src/test/java/com/google/devtools/build/lib/rules/config:all_windows_tests",
        "//src/test/java/com/google/devtools/build/lib/rules/cpp:all_windows_tests",
        "//src/test/java/com/google/devtools/build/lib/rules/platform:all_windows_tests",
        "//src/test/java/com/google/devtools/build/lib/rules/repository:all_windows_tests",
        "//src/test/java/com/google/devtools/build/lib/shell:all_windows_tests",
        "//src/test/java/com/google/devtools/build/lib/skyframe:all_windows_tests",
        "//src/test/java/com/google/devtools/build/lib/skylark:all_windows_tests",
        "//src/test/java/com/google/devtools/build/lib/skylarkinterface/processor:all_windows_tests",
    ],
    visibility = ["//src:__pkg__"],
)
