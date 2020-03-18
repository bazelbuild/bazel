load("@rules_java//java:defs.bzl", "java_binary", "java_library", "java_test")
load("@rules_cc//cc:defs.bzl", "cc_binary")

package(
    default_visibility = [
        ":__subpackages__",
        "//tools/test:__pkg__",
    ],
)

# Tests for Windows-specific functionality that can run cross-platform.
# These don't need to run on Windows, they merely use Windows- and case-insensitive path semantics.
CROSS_PLATFORM_WINDOWS_TESTS = [
    "util/DependencySetWindowsTest.java",
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

# Generate list of all srcs via:
# bazel query 'filter("srcs", kind("filegroup rule", //src/test/java/com/google/devtools/build/lib/...))' | sort | sed -e "s/^/\"/" | sed -e "s/$/\",/" | fgrep -v "build/lib:srcs"
filegroup(
    name = "srcs",
    srcs = glob(["**"]) + [
        "//src/test/java/com/google/devtools/build/lib/actions:srcs",
        "//src/test/java/com/google/devtools/build/lib/analysis:srcs",
        "//src/test/java/com/google/devtools/build/lib/bazel:srcs",
        "//src/test/java/com/google/devtools/build/lib/blackbox:srcs",
        "//src/test/java/com/google/devtools/build/lib/buildeventservice:srcs",
        "//src/test/java/com/google/devtools/build/lib/buildeventstream:srcs",
        "//src/test/java/com/google/devtools/build/lib/buildeventstream/transports:srcs",
        "//src/test/java/com/google/devtools/build/lib/buildtool:srcs",
        "//src/test/java/com/google/devtools/build/lib/events:srcs",
        "//src/test/java/com/google/devtools/build/lib/exec:srcs",
        "//src/test/java/com/google/devtools/build/lib/integration/util:srcs",
        "//src/test/java/com/google/devtools/build/lib/metrics:srcs",
        "//src/test/java/com/google/devtools/build/lib/outputfilter:srcs",
        "//src/test/java/com/google/devtools/build/lib/packages:srcs",
        "//src/test/java/com/google/devtools/build/lib/profiler:srcs",
        "//src/test/java/com/google/devtools/build/lib/profiler/callcounts:srcs",
        "//src/test/java/com/google/devtools/build/lib/profiler/memory:srcs",
        "//src/test/java/com/google/devtools/build/lib/query2:srcs",
        "//src/test/java/com/google/devtools/build/lib/query2/engine:srcs",
        "//src/test/java/com/google/devtools/build/lib/query2/testutil:srcs",
        "//src/test/java/com/google/devtools/build/lib/remote:srcs",
        "//src/test/java/com/google/devtools/build/lib/rules/android:srcs",
        "//src/test/java/com/google/devtools/build/lib/rules/apple:srcs",
        "//src/test/java/com/google/devtools/build/lib/rules/config:srcs",
        "//src/test/java/com/google/devtools/build/lib/rules/cpp:srcs",
        "//src/test/java/com/google/devtools/build/lib/rules/objc:srcs",
        "//src/test/java/com/google/devtools/build/lib/rules/platform:srcs",
        "//src/test/java/com/google/devtools/build/lib/rules/python:srcs",
        "//src/test/java/com/google/devtools/build/lib/rules/repository:srcs",
        "//src/test/java/com/google/devtools/build/lib/sandbox:srcs",
        "//src/test/java/com/google/devtools/build/lib/skyframe/packages:srcs",
        "//src/test/java/com/google/devtools/build/lib/skyframe/serialization:srcs",
        "//src/test/java/com/google/devtools/build/lib/skyframe:srcs",
        "//src/test/java/com/google/devtools/build/lib/skylark:srcs",
        "//src/test/java/com/google/devtools/build/lib/skylarkdebug/server:srcs",
        "//src/test/java/com/google/devtools/build/lib/skylarkinterface/processor:srcs",
        "//src/test/java/com/google/devtools/build/lib/supplier:srcs",
        "//src/test/java/com/google/devtools/build/lib/versioning:srcs",
        "//src/test/java/com/google/devtools/build/lib/vfs:srcs",
        "//src/test/java/com/google/devtools/build/lib/unsafe:srcs",
        "//src/test/java/com/google/devtools/build/lib/testutil:srcs",
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

alias(
    name = "test_runner",
    actual = ":AllTests",
    visibility = ["//src/test:__subpackages__"],
)

java_library(
    name = "AllTests",
    testonly = 1,
    srcs = ["AllTests.java"],
    visibility = ["//src/test:__subpackages__"],
    deps = [
        "//src/test/java/com/google/devtools/build/lib/testutil",
        "//src/test/java/com/google/devtools/build/lib/testutil:TestSuite",
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
        ":AllTests",
        "//src/main/java/com/google/devtools/build/lib/cmdline",
        "//src/main/java/com/google/devtools/build/lib/cmdline:LabelValidator",
        "//src/main/java/com/google/devtools/build/lib/syntax:frontend",
        "//src/main/java/com/google/devtools/build/lib/vfs:pathfragment",
        "//src/test/java/com/google/devtools/build/lib/testutil",
        "//src/test/java/com/google/devtools/build/lib/testutil:JunitUtils",
        "//src/test/java/com/google/devtools/build/lib/testutil:TestUtils",
        "//third_party:guava",
        "//third_party:guava-testlib",
        "//third_party:junit4",
        "//third_party:truth",
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
        ":AllTests",
        "//src/main/java/com/google/devtools/build/lib/concurrent",
        "//src/test/java/com/google/devtools/build/lib/testutil",
        "//src/test/java/com/google/devtools/build/lib/testutil:JunitUtils",
        "//src/test/java/com/google/devtools/build/lib/testutil:TestThread",
        "//src/test/java/com/google/devtools/build/lib/testutil:TestUtils",
        "//third_party:guava",
        "//third_party:guava-testlib",
        "//third_party:junit4",
        "//third_party:truth",
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
        ":AllTests",
        "//src/main/java/com/google/devtools/build/lib/collect",
        "//src/main/java/com/google/devtools/build/lib/collect/nestedset",
        "//src/test/java/com/google/devtools/build/lib/testutil",
        "//src/test/java/com/google/devtools/build/lib/testutil:JunitUtils",
        "//third_party:guava",
        "//third_party:guava-testlib",
        "//third_party:junit4",
        "//third_party:truth",
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
        ":AllTests",
        "//src/main/java/com/google/devtools/build/lib:util",
        "//src/main/java/com/google/devtools/build/lib/actions:commandline_item",
        "//src/main/java/com/google/devtools/build/lib/collect/nestedset",
        "//src/main/java/com/google/devtools/build/lib/collect/nestedset:fingerprint_cache",
        "//src/main/java/com/google/devtools/build/lib/collect/nestedset:testutils",
        "//src/main/java/com/google/devtools/build/lib/skyframe/serialization",
        "//src/main/java/com/google/devtools/build/lib/skyframe/serialization:constants",
        "//src/main/java/com/google/devtools/build/lib/skyframe/serialization/testutils",
        "//src/main/java/com/google/devtools/build/lib/vfs",
        "//src/main/java/com/google/devtools/build/lib/vfs:pathfragment",
        "//src/test/java/com/google/devtools/build/lib/testutil",
        "//src/test/java/com/google/devtools/build/lib/testutil:JunitUtils",
        "//third_party:guava",
        "//third_party:guava-testlib",
        "//third_party:junit4",
        "//third_party:mockito",
        "//third_party:truth",
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
        ":AllTests",
        "//src/main/java/com/google/devtools/build/lib:build-base",
        "//src/main/java/com/google/devtools/build/lib/actions",
        "//src/main/java/com/google/devtools/build/lib/actions:localhost_capacity",
        "//src/main/java/com/google/devtools/build/lib/dynamic",
        "//src/main/java/com/google/devtools/build/lib/util/io",
        "//src/main/java/com/google/devtools/build/lib/vfs",
        "//src/main/java/com/google/devtools/build/lib/vfs:pathfragment",
        "//src/main/java/com/google/devtools/common/options:options_internal",
        "//src/main/protobuf:test_status_java_proto",
        "//src/test/java/com/google/devtools/build/lib/actions/util",
        "//src/test/java/com/google/devtools/build/lib/events:testutil",
        "//src/test/java/com/google/devtools/build/lib/testutil",
        "//src/test/java/com/google/devtools/build/lib/testutil:JunitUtils",
        "//src/test/java/com/google/devtools/build/lib/testutil:TestThread",
        "//src/test/java/com/google/devtools/build/lib/testutil:TestUtils",
        "//src/test/java/com/google/devtools/build/lib/vfs/util",
        "//third_party:auto_value",
        "//third_party:guava",
        "//third_party:guava-testlib",
        "//third_party:junit4",
        "//third_party:truth",
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
        ":AllTests",
        "//src/main/java/com/google/devtools/build/lib:string_util",
        "//src/main/java/com/google/devtools/build/lib:util",
        "//src/main/java/com/google/devtools/build/lib/unix",
        "//src/main/java/com/google/devtools/build/lib/vfs",
        "//src/main/java/com/google/devtools/build/lib/vfs:pathfragment",
        "//src/test/java/com/google/devtools/build/lib/events:testutil",
        "//src/test/java/com/google/devtools/build/lib/testutil",
        "//src/test/java/com/google/devtools/build/lib/testutil:JunitUtils",
        "//src/test/java/com/google/devtools/build/lib/testutil:TestUtils",
        "//src/test/java/com/google/devtools/build/lib/vfs:SymlinkAwareFileSystemTest",
        "//src/test/java/com/google/devtools/build/lib/vfs/util",
        "//third_party:guava",
        "//third_party:guava-testlib",
        "//third_party:junit4",
        "//third_party:truth",
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
        ":AllTests",
        "//src/main/java/com/google/devtools/build/lib:command-utils",
        "//src/main/java/com/google/devtools/build/lib:exitcode-external",
        "//src/main/java/com/google/devtools/build/lib:failure_detail_util",
        "//src/main/java/com/google/devtools/build/lib:resource-converter",
        "//src/main/java/com/google/devtools/build/lib:shell_escaper",
        "//src/main/java/com/google/devtools/build/lib:simple-log-handler",
        "//src/main/java/com/google/devtools/build/lib:single-line-formatter",
        "//src/main/java/com/google/devtools/build/lib:util",
        "//src/main/java/com/google/devtools/build/lib/actions:localhost_capacity",
        "//src/main/java/com/google/devtools/build/lib/analysis/platform",
        "//src/main/java/com/google/devtools/build/lib/cmdline",
        "//src/main/java/com/google/devtools/build/lib/shell",
        "//src/main/java/com/google/devtools/build/lib/skyframe/serialization/testutils",
        "//src/main/java/com/google/devtools/build/lib/vfs",
        "//src/main/java/com/google/devtools/build/lib/vfs:pathfragment",
        "//src/main/java/com/google/devtools/build/lib/vfs/inmemoryfs",
        "//src/main/java/com/google/devtools/common/options",
        "//src/main/protobuf:failure_details_java_proto",
        "//src/test/java/com/google/devtools/build/lib/testutil",
        "//src/test/java/com/google/devtools/build/lib/testutil:JunitUtils",
        "//third_party:guava",
        "//third_party:guava-testlib",
        "//third_party:junit4",
        "//third_party:mockito",
        "//third_party:truth",
        "//third_party/protobuf:protobuf_java",
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
        ":AllTests",
        "//src/main/java/com/google/devtools/build/lib:util",
        "//src/main/java/com/google/devtools/build/lib/util/io",
        "//src/main/java/com/google/devtools/build/lib/util/io:out-err",
        "//src/main/java/com/google/devtools/build/lib/vfs",
        "//src/main/java/com/google/devtools/build/lib/vfs:pathfragment",
        "//src/main/java/com/google/devtools/build/lib/vfs/inmemoryfs",
        "//src/main/protobuf:bazel_flags_java_proto",
        "//src/test/java/com/google/devtools/build/lib/testutil",
        "//src/test/java/com/google/devtools/build/lib/testutil:JunitUtils",
        "//third_party:guava",
        "//third_party:guava-testlib",
        "//third_party:junit4",
        "//third_party:mockito",
        "//third_party:truth",
    ],
)

java_library(
    name = "windows_testutil",
    testonly = 1,
    srcs = [
        "windows/util/WindowsTestUtil.java",
    ],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/vfs",
        "//src/main/java/com/google/devtools/build/lib/vfs:pathfragment",
        "//src/main/java/com/google/devtools/build/lib/windows",
        "//src/main/java/com/google/devtools/build/lib/windows/jni",
        "//third_party:guava",
        "//third_party:guava-testlib",
        "//third_party:junit4",
        "//third_party:truth",
    ],
)

# Tests that need to run on Windows
java_test(
    name = "windows-tests",
    srcs = WINDOWS_ON_WINDOWS_TESTS,
    data = [
        ":MockSubprocess_deploy.jar",
        ":printarg",
    ],
    jvm_flags = [
        "-Dbazel.windows_unix_root=C:/fake/msys",
    ],
    test_class = "com.google.devtools.build.lib.AllTests",
    deps = [
        ":AllTests",
        ":windows_testutil",
        "//src/main/java/com/google/devtools/build/lib:os_util",
        "//src/main/java/com/google/devtools/build/lib/clock",
        "//src/main/java/com/google/devtools/build/lib/shell",
        "//src/main/java/com/google/devtools/build/lib/vfs",
        "//src/main/java/com/google/devtools/build/lib/vfs:pathfragment",
        "//src/main/java/com/google/devtools/build/lib/vfs/inmemoryfs",
        "//src/main/java/com/google/devtools/build/lib/windows",
        "//src/main/java/com/google/devtools/build/lib/windows:windows_short_path",
        "//src/main/java/com/google/devtools/build/lib/windows/jni",
        "//src/test/java/com/google/devtools/build/lib/testutil",
        "//third_party:guava",
        "//third_party:guava-testlib",
        "//third_party:junit4",
        "//third_party:truth",
        "@bazel_tools//tools/java/runfiles",
    ],
)

cc_binary(
    name = "printarg",
    testonly = 1,
    srcs = ["windows/printarg.cc"],
)

java_test(
    name = "graph_test",
    srcs = glob([
        "graph/*.java",
    ]),
    test_class = "com.google.devtools.build.lib.AllTests",
    deps = [
        ":AllTests",
        "//src/main/java/com/google/devtools/build/lib:events",
        "//src/main/java/com/google/devtools/build/lib/cmdline",
        "//src/main/java/com/google/devtools/build/lib/collect",
        "//src/main/java/com/google/devtools/build/lib/graph",
        "//src/main/java/com/google/devtools/build/lib/packages",
        "//src/test/java/com/google/devtools/build/lib/events:testutil",
        "//src/test/java/com/google/devtools/build/lib/testutil",
        "//src/test/java/com/google/devtools/build/lib/vfs/util",
        "//third_party:guava",
        "//third_party:guava-testlib",
        "//third_party:junit4",
        "//third_party:truth",
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
        ":AllTests",
        "//src/main/java/com/google/devtools/build/lib:bazel-rules",
        "//src/main/java/com/google/devtools/build/lib:build-base",
        "//src/main/java/com/google/devtools/build/lib:events",
        "//src/main/java/com/google/devtools/build/lib:util",
        "//src/main/java/com/google/devtools/build/lib/actions",
        "//src/main/java/com/google/devtools/build/lib/clock",
        "//src/main/java/com/google/devtools/build/lib/cmdline",
        "//src/main/java/com/google/devtools/build/lib/packages",
        "//src/main/java/com/google/devtools/build/lib/packages:starlark_semantics_options",
        "//src/main/java/com/google/devtools/build/lib/pkgcache",
        "//src/main/java/com/google/devtools/build/lib/syntax:frontend",
        "//src/main/java/com/google/devtools/build/lib/util/io",
        "//src/main/java/com/google/devtools/build/lib/vfs",
        "//src/main/java/com/google/devtools/build/lib/vfs:pathfragment",
        "//src/main/java/com/google/devtools/build/lib/vfs/inmemoryfs",
        "//src/main/java/com/google/devtools/build/skyframe",
        "//src/main/java/com/google/devtools/build/skyframe:skyframe-objects",
        "//src/main/java/com/google/devtools/common/options",
        "//src/main/java/com/google/devtools/common/options:invocation_policy",
        "//src/test/java/com/google/devtools/build/lib/analysis/util",
        "//src/test/java/com/google/devtools/build/lib/analysis/util:test-build-options",
        "//src/test/java/com/google/devtools/build/lib/packages:testutil",
        "//src/test/java/com/google/devtools/build/lib/testutil",
        "//src/test/java/com/google/devtools/build/lib/testutil:JunitUtils",
        "//src/test/java/com/google/devtools/build/lib/testutil:TestConstants",
        "//third_party:guava",
        "//third_party:guava-testlib",
        "//third_party:jsr305",
        "//third_party:junit4",
        "//third_party:truth",
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
        ":AllTests",
        "//src/main/java/com/google/devtools/build/lib:bazel-main",
        "//src/main/java/com/google/devtools/build/lib:runtime",
        "//src/main/java/com/google/devtools/build/lib:server",
        "//src/main/java/com/google/devtools/build/lib:util",
        "//src/main/java/com/google/devtools/build/lib/collect",
        "//src/main/java/com/google/devtools/build/lib/unix",
        "//src/main/java/com/google/devtools/build/lib/util/io",
        "//src/main/java/com/google/devtools/build/lib/util/io:out-err",
        "//src/main/java/com/google/devtools/build/lib/vfs",
        "//src/main/java/com/google/devtools/build/lib/vfs:pathfragment",
        "//src/main/java/com/google/devtools/build/lib/vfs/inmemoryfs",
        "//src/main/protobuf:command_server_java_grpc",
        "//src/main/protobuf:command_server_java_proto",
        "//src/main/protobuf:failure_details_java_proto",
        "//src/main/protobuf:invocation_policy_java_proto",
        "//src/test/java/com/google/devtools/build/lib/events:testutil",
        "//src/test/java/com/google/devtools/build/lib/testutil",
        "//src/test/java/com/google/devtools/build/lib/testutil:JunitUtils",
        "//src/test/java/com/google/devtools/build/lib/testutil:TestThread",
        "//src/test/java/com/google/devtools/build/lib/testutil:TestUtils",
        "//src/test/java/com/google/devtools/build/lib/vfs/util",
        "//third_party:guava",
        "//third_party:guava-testlib",
        "//third_party:jsr305",
        "//third_party:junit4",
        "//third_party:mockito",
        "//third_party:truth",
        "//third_party/grpc:grpc-jar",
        "//third_party/protobuf:protobuf_java",
    ],
)

java_test(
    name = "syntax_test",
    srcs = glob(
        include = ["syntax/*.java"],
        exclude = ["syntax/CpuProfilerTest.java"],  # open source, but runs under Blaze only
    ),
    test_class = "com.google.devtools.build.lib.AllTests",
    deps = [
        ":AllTests",
        ":syntax_testutil",
        "//src/main/java/com/google/devtools/build/lib:bazel-main",
        "//src/main/java/com/google/devtools/build/lib:bazel-rules",
        "//src/main/java/com/google/devtools/build/lib:build-base",
        "//src/main/java/com/google/devtools/build/lib:events",
        "//src/main/java/com/google/devtools/build/lib:string_util",
        "//src/main/java/com/google/devtools/build/lib:syntax",
        "//src/main/java/com/google/devtools/build/lib:util",
        "//src/main/java/com/google/devtools/build/lib/actions",
        "//src/main/java/com/google/devtools/build/lib/cmdline",
        "//src/main/java/com/google/devtools/build/lib/collect",
        "//src/main/java/com/google/devtools/build/lib/collect/nestedset",
        "//src/main/java/com/google/devtools/build/lib/concurrent",
        "//src/main/java/com/google/devtools/build/lib/packages",
        "//src/main/java/com/google/devtools/build/lib/packages:starlark_semantics_options",
        "//src/main/java/com/google/devtools/build/lib/skyframe/serialization/testutils",
        "//src/main/java/com/google/devtools/build/lib/skylarkinterface",
        "//src/main/java/com/google/devtools/build/lib/vfs",
        "//src/main/java/com/google/devtools/build/lib/vfs:pathfragment",
        "//src/main/java/com/google/devtools/common/options:options_internal",
        "//src/test/java/com/google/devtools/build/lib/events:testutil",
        "//src/test/java/com/google/devtools/build/lib/testutil",
        "//src/test/java/com/google/devtools/build/lib/testutil:JunitUtils",
        "//src/test/java/com/google/devtools/build/lib/vfs/util",
        "//third_party:guava",
        "//third_party:guava-testlib",
        "//third_party:jsr305",
        "//third_party:junit4",
        "//third_party:truth",
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
        ":AllTests",
        "//src/main/java/com/google/devtools/build/lib:bazel-main",
        "//src/main/java/com/google/devtools/build/lib:bazel-rules",
        "//src/main/java/com/google/devtools/build/lib:build-base",
        "//src/main/java/com/google/devtools/build/lib:events",
        "//src/main/java/com/google/devtools/build/lib:util",
        "//src/main/java/com/google/devtools/build/lib/actions",
        "//src/main/java/com/google/devtools/build/lib/collect",
        "//src/main/java/com/google/devtools/build/lib/concurrent",
        "//src/main/java/com/google/devtools/build/lib/packages",
        "//src/main/java/com/google/devtools/build/lib/packages:starlark_semantics_options",
        "//src/main/java/com/google/devtools/build/lib/syntax:evaluator",
        "//src/main/java/com/google/devtools/build/lib/syntax:frontend",
        "//src/main/java/com/google/devtools/build/lib/vfs:pathfragment",
        "//src/main/java/com/google/devtools/common/options:options_internal",
        "//src/test/java/com/google/devtools/build/lib/events:testutil",
        "//src/test/java/com/google/devtools/build/lib/testutil",
        "//src/test/java/com/google/devtools/build/lib/vfs/util",
        "//third_party:guava",
        "//third_party:guava-testlib",
        "//third_party:jsr305",
        "//third_party:junit4",
        "//third_party:truth",
    ],
)

java_test(
    name = "filegroup-rules-tests",
    srcs = glob(["rules/filegroup/*.java"]),
    tags = ["rules"],
    test_class = "com.google.devtools.build.lib.AllTests",
    deps = [
        ":AllTests",
        "//src/main/java/com/google/devtools/build/lib:bazel-main",
        "//src/main/java/com/google/devtools/build/lib:bazel-rules",
        "//src/main/java/com/google/devtools/build/lib:build-base",
        "//src/main/java/com/google/devtools/build/lib:util",
        "//src/main/java/com/google/devtools/build/lib/rules/java:java-compilation",
        "//src/main/java/com/google/devtools/build/lib/rules/java:java-rules",
        "//src/main/java/com/google/devtools/build/lib/vfs:pathfragment",
        "//src/test/java/com/google/devtools/build/lib/actions/util",
        "//src/test/java/com/google/devtools/build/lib/analysis/util",
        "//src/test/java/com/google/devtools/build/lib/testutil",
        "//src/test/java/com/google/devtools/build/lib/testutil:JunitUtils",
        "//third_party:guava",
        "//third_party:guava-testlib",
        "//third_party:jsr305",
        "//third_party:junit4",
        "//third_party:truth",
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
        "//src/main/java/com/google/devtools/build/lib/rules/java:java-compilation",
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
        ":AllTests",
        ":java_compile_action_test_helper",
        "//src/main/java/com/google/devtools/build/lib:bazel-main",
        "//src/main/java/com/google/devtools/build/lib:bazel-rules",
        "//src/main/java/com/google/devtools/build/lib:build-base",
        "//src/main/java/com/google/devtools/build/lib:filetype",
        "//src/main/java/com/google/devtools/build/lib/actions",
        "//src/main/java/com/google/devtools/build/lib/analysis/platform",
        "//src/main/java/com/google/devtools/build/lib/cmdline",
        "//src/main/java/com/google/devtools/build/lib/collect",
        "//src/main/java/com/google/devtools/build/lib/collect/nestedset",
        "//src/main/java/com/google/devtools/build/lib/packages",
        "//src/main/java/com/google/devtools/build/lib/rules/java:java-compilation",
        "//src/main/java/com/google/devtools/build/lib/rules/java:java-rules",
        "//src/main/java/com/google/devtools/build/lib/syntax:evaluator",
        "//src/main/java/com/google/devtools/build/lib/vfs:pathfragment",
        "//src/test/java/com/google/devtools/build/lib/actions/util",
        "//src/test/java/com/google/devtools/build/lib/analysis/util",
        "//src/test/java/com/google/devtools/build/lib/testutil",
        "//src/test/java/com/google/devtools/build/lib/testutil:TestConstants",
        "//third_party:guava",
        "//third_party:guava-testlib",
        "//third_party:jsr305",
        "//third_party:junit4",
        "//third_party:truth",
    ],
)

java_test(
    name = "test-rules-tests",
    srcs = ["rules/test/SkylarkTestingModuleTest.java"],
    tags = ["rules"],
    test_class = "com.google.devtools.build.lib.rules.test.SkylarkTestingModuleTest",
    deps = [
        "//src/main/java/com/google/devtools/build/lib:build-base",
        "//src/test/java/com/google/devtools/build/lib/analysis/util",
        "//third_party:guava",
        "//third_party:guava-testlib",
        "//third_party:junit4",
        "//third_party:truth",
        "//third_party/protobuf:protobuf_java",
    ],
)

java_test(
    name = "runtime-tests",
    srcs = glob(
        [
            "runtime/*.java",
            "runtime/commands/*.java",
        ],
        exclude = ["runtime/commands/ConfigCommandTest.java"],
    ),
    test_class = "com.google.devtools.build.lib.AllTests",
    deps = [
        ":AllTests",
        "//src/main/java/com/google/devtools/build/lib:bazel-modules",
        "//src/main/java/com/google/devtools/build/lib:bazel-rules",
        "//src/main/java/com/google/devtools/build/lib:build-base",
        "//src/main/java/com/google/devtools/build/lib:build-request-options",
        "//src/main/java/com/google/devtools/build/lib:detailed_exit_code",
        "//src/main/java/com/google/devtools/build/lib:events",
        "//src/main/java/com/google/devtools/build/lib:loading-phase-threads-option",
        "//src/main/java/com/google/devtools/build/lib:runtime",
        "//src/main/java/com/google/devtools/build/lib:string_util",
        "//src/main/java/com/google/devtools/build/lib:util",
        "//src/main/java/com/google/devtools/build/lib/actions",
        "//src/main/java/com/google/devtools/build/lib/actions:localhost_capacity",
        "//src/main/java/com/google/devtools/build/lib/bazel/repository/downloader",
        "//src/main/java/com/google/devtools/build/lib/buildeventstream",
        "//src/main/java/com/google/devtools/build/lib/buildeventstream/proto:build_event_stream_java_proto",
        "//src/main/java/com/google/devtools/build/lib/buildeventstream/transports",
        "//src/main/java/com/google/devtools/build/lib/clock",
        "//src/main/java/com/google/devtools/build/lib/cmdline",
        "//src/main/java/com/google/devtools/build/lib/collect/nestedset",
        "//src/main/java/com/google/devtools/build/lib/packages",
        "//src/main/java/com/google/devtools/build/lib/packages:starlark_semantics_options",
        "//src/main/java/com/google/devtools/build/lib/query2",
        "//src/main/java/com/google/devtools/build/lib/query2/engine",
        "//src/main/java/com/google/devtools/build/lib/sandbox",
        "//src/main/java/com/google/devtools/build/lib/unix",
        "//src/main/java/com/google/devtools/build/lib/util/io",
        "//src/main/java/com/google/devtools/build/lib/util/io:out-err",
        "//src/main/java/com/google/devtools/build/lib/vfs",
        "//src/main/java/com/google/devtools/build/lib/vfs:pathfragment",
        "//src/main/java/com/google/devtools/build/lib/vfs/inmemoryfs",
        "//src/main/java/com/google/devtools/common/options",
        "//src/main/java/com/google/devtools/common/options:invocation_policy",
        "//src/main/protobuf:command_line_java_proto",
        "//src/main/protobuf:invocation_policy_java_proto",
        "//src/main/protobuf:test_status_java_proto",
        "//src/test/java/com/google/devtools/build/lib/actions/util",
        "//src/test/java/com/google/devtools/build/lib/events:testutil",
        "//src/test/java/com/google/devtools/build/lib/skylark:testutil",
        "//src/test/java/com/google/devtools/build/lib/testutil",
        "//src/test/java/com/google/devtools/build/lib/testutil:JunitUtils",
        "//src/test/java/com/google/devtools/build/lib/testutil:TestConstants",
        "//src/test/java/com/google/devtools/build/lib/testutil:TestUtils",
        "//src/test/java/com/google/devtools/build/lib/vfs/util",
        "//src/test/java/com/google/devtools/common/options:testutils",
        "//third_party:apache_commons_lang",
        "//third_party:guava",
        "//third_party:guava-testlib",
        "//third_party:junit4",
        "//third_party:mockito",
        "//third_party:truth",
    ],
)

java_test(
    name = "ConfigCommandTest",
    srcs = ["runtime/commands/ConfigCommandTest.java"],
    # TODO(bazel-team) The test is currently broken due to BazelAnalysisMock not supporting integration tests.
    tags = ["manual"],
)

java_test(
    name = "standalone-tests",
    srcs = glob(["standalone/*.java"]),
    data = [":embedded_scripts"],
    test_class = "com.google.devtools.build.lib.AllTests",
    deps = [
        ":AllTests",
        "//src/main/java/com/google/devtools/build/lib:bazel-rules",
        "//src/main/java/com/google/devtools/build/lib:build-base",
        "//src/main/java/com/google/devtools/build/lib:events",
        "//src/main/java/com/google/devtools/build/lib:os_util",
        "//src/main/java/com/google/devtools/build/lib:util",
        "//src/main/java/com/google/devtools/build/lib/actions",
        "//src/main/java/com/google/devtools/build/lib/actions:localhost_capacity",
        "//src/main/java/com/google/devtools/build/lib/clock",
        "//src/main/java/com/google/devtools/build/lib/collect/nestedset",
        "//src/main/java/com/google/devtools/build/lib/exec/local",
        "//src/main/java/com/google/devtools/build/lib/exec/local:options",
        "//src/main/java/com/google/devtools/build/lib/rules/apple",
        "//src/main/java/com/google/devtools/build/lib/shell",
        "//src/main/java/com/google/devtools/build/lib/standalone",
        "//src/main/java/com/google/devtools/build/lib/util/io",
        "//src/main/java/com/google/devtools/build/lib/vfs",
        "//src/main/java/com/google/devtools/build/lib/vfs:pathfragment",
        "//src/main/java/com/google/devtools/common/options",
        "//src/test/java/com/google/devtools/build/lib/actions/util",
        "//src/test/java/com/google/devtools/build/lib/analysis/util",
        "//src/test/java/com/google/devtools/build/lib/events:testutil",
        "//src/test/java/com/google/devtools/build/lib/exec/util",
        "//src/test/java/com/google/devtools/build/lib/integration/util",
        "//src/test/java/com/google/devtools/build/lib/testutil",
        "//src/test/java/com/google/devtools/build/lib/testutil:JunitUtils",
        "//src/test/java/com/google/devtools/build/lib/testutil:TestConstants",
        "//src/test/java/com/google/devtools/build/lib/testutil:TestUtils",
        "//src/test/java/com/google/devtools/build/lib/vfs/util",
        "//third_party:guava",
        "//third_party:guava-testlib",
        "//third_party:junit4",
        "//third_party:mockito",
        "//third_party:truth",
    ],
)

java_test(
    name = "ProtoCompileActionBuilderTest",
    srcs = ["rules/proto/ProtoCompileActionBuilderTest.java"],
    deps = [
        "//src/main/java/com/google/devtools/build/lib:build-base",
        "//src/main/java/com/google/devtools/build/lib:events",
        "//src/main/java/com/google/devtools/build/lib:proto-rules",
        "//src/main/java/com/google/devtools/build/lib:util",
        "//src/main/java/com/google/devtools/build/lib/actions",
        "//src/main/java/com/google/devtools/build/lib/cmdline",
        "//src/main/java/com/google/devtools/build/lib/collect",
        "//src/main/java/com/google/devtools/build/lib/collect/nestedset",
        "//src/main/java/com/google/devtools/build/lib/vfs",
        "//src/main/java/com/google/devtools/build/lib/vfs:pathfragment",
        "//src/main/java/com/google/devtools/build/lib/vfs/inmemoryfs",
        "//src/test/java/com/google/devtools/build/lib/actions/util",
        "//src/test/java/com/google/devtools/build/lib/testutil",
        "//src/test/java/com/google/devtools/build/lib/testutil:JunitUtils",
        "//third_party:guava",
        "//third_party:guava-testlib",
        "//third_party:junit4",
        "//third_party:mockito",
        "//third_party:truth",
    ],
)

java_test(
    name = "ProtoLangToolchainTest",
    srcs = ["rules/proto/ProtoLangToolchainTest.java"],
    deps = [
        "//src/main/java/com/google/devtools/build/lib:build-base",
        "//src/main/java/com/google/devtools/build/lib:proto-rules",
        "//src/main/java/com/google/devtools/build/lib:util",
        "//src/main/java/com/google/devtools/build/lib/cmdline",
        "//src/test/java/com/google/devtools/build/lib/actions/util",
        "//src/test/java/com/google/devtools/build/lib/analysis/util",
        "//src/test/java/com/google/devtools/build/lib/packages:testutil",
        "//src/test/java/com/google/devtools/build/lib/testutil",
        "//src/test/java/com/google/devtools/build/lib/testutil:TestConstants",
        "//third_party:guava",
        "//third_party:guava-testlib",
        "//third_party:junit4",
        "//third_party:truth",
    ],
)

java_test(
    name = "BazelProtoLibraryTest",
    srcs = ["rules/proto/BazelProtoLibraryTest.java"],
    deps = [
        "//src/main/java/com/google/devtools/build/lib:build-base",
        "//src/main/java/com/google/devtools/build/lib:proto-rules",
        "//src/main/java/com/google/devtools/build/lib:util",
        "//src/main/java/com/google/devtools/build/lib/actions",
        "//src/main/java/com/google/devtools/build/lib/vfs",
        "//src/main/java/com/google/devtools/build/lib/vfs:pathfragment",
        "//src/test/java/com/google/devtools/build/lib/actions/util",
        "//src/test/java/com/google/devtools/build/lib/analysis/util",
        "//src/test/java/com/google/devtools/build/lib/packages:testutil",
        "//src/test/java/com/google/devtools/build/lib/testutil",
        "//src/test/java/com/google/devtools/build/lib/testutil:JunitUtils",
        "//src/test/java/com/google/devtools/build/lib/testutil:TestConstants",
        "//third_party:guava",
        "//third_party:guava-testlib",
        "//third_party:junit4",
        "//third_party:truth",
    ],
)

java_test(
    name = "BazelProtoInfoStarlarkTest",
    srcs = ["rules/proto/BazelProtoInfoStarlarkTest.java"],
    deps = [
        "//src/main/java/com/google/devtools/build/lib:build-base",
        "//src/main/java/com/google/devtools/build/lib:proto-rules",
        "//src/main/java/com/google/devtools/build/lib/actions",
        "//src/main/java/com/google/devtools/build/lib/cmdline",
        "//src/main/java/com/google/devtools/build/lib/packages",
        "//src/main/java/com/google/devtools/build/lib/skylarkbuildapi/proto",
        "//src/main/java/com/google/devtools/build/lib/vfs:pathfragment",
        "//src/test/java/com/google/devtools/build/lib/actions/util",
        "//src/test/java/com/google/devtools/build/lib/analysis/util",
        "//src/test/java/com/google/devtools/build/lib/packages:testutil",
        "//src/test/java/com/google/devtools/build/lib/testutil",
        "//src/test/java/com/google/devtools/build/lib/testutil:JunitUtils",
        "//src/test/java/com/google/devtools/build/lib/testutil:TestConstants",
        "//third_party:guava",
        "//third_party:guava-testlib",
        "//third_party:junit4",
        "//third_party:truth",
    ],
)

java_test(
    name = "repository-tests",
    srcs = glob(["repository/*.java"]),
    test_class = "com.google.devtools.build.lib.AllTests",
    deps = [
        ":AllTests",
        "//src/main/java/com/google/devtools/build/lib:build-base",
        "//src/main/java/com/google/devtools/build/lib:events",
        "//src/main/java/com/google/devtools/build/lib:syntax",
        "//src/main/java/com/google/devtools/build/lib/actions",
        "//src/main/java/com/google/devtools/build/lib/cmdline",
        "//src/main/java/com/google/devtools/build/lib/packages",
        "//src/main/java/com/google/devtools/build/lib/pkgcache",
        "//src/main/java/com/google/devtools/build/lib/util/io",
        "//src/main/java/com/google/devtools/build/lib/vfs",
        "//src/main/java/com/google/devtools/build/lib/vfs:pathfragment",
        "//src/main/java/com/google/devtools/build/skyframe",
        "//src/main/java/com/google/devtools/build/skyframe:skyframe-objects",
        "//src/test/java/com/google/devtools/build/lib/analysis/util",
        "//src/test/java/com/google/devtools/build/skyframe:testutil",
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
        ":AllTests",
        "//src/main/java/com/google/devtools/build/lib:build-base",
        "//src/main/java/com/google/devtools/build/lib:events",
        "//src/main/java/com/google/devtools/build/lib:proto-rules",
        "//src/main/java/com/google/devtools/build/lib/actions",
        "//src/main/java/com/google/devtools/build/lib/cmdline",
        "//src/main/java/com/google/devtools/build/lib/collect",
        "//src/main/java/com/google/devtools/build/lib/collect/nestedset",
        "//src/main/java/com/google/devtools/build/lib/packages",
        "//src/main/java/com/google/devtools/build/lib/rules/cpp",
        "//src/main/java/com/google/devtools/build/lib/skylarkbuildapi/proto",
        "//src/main/java/com/google/devtools/build/lib/util/io",
        "//src/main/java/com/google/devtools/build/skyframe",
        "//src/main/java/com/google/devtools/build/skyframe:skyframe-objects",
        "//src/test/java/com/google/devtools/build/lib/actions/util",
        "//src/test/java/com/google/devtools/build/lib/analysis/util",
        "//src/test/java/com/google/devtools/build/lib/packages:testutil",
        "//src/test/java/com/google/devtools/build/lib/testutil",
        "//src/test/java/com/google/devtools/build/lib/testutil:TestConstants",
        "//third_party:auto_value",
        "//third_party:guava",
        "//third_party:guava-testlib",
        "//third_party:junit4",
        "//third_party:truth",
    ],
)

java_test(
    name = "worker-tests",
    srcs = glob(["worker/*Test.java"]),
    test_class = "com.google.devtools.build.lib.AllTests",
    deps = [
        ":AllTests",
        "//src/main/java/com/google/devtools/build/lib:os_util",
        "//src/main/java/com/google/devtools/build/lib:resource-converter",
        "//src/main/java/com/google/devtools/build/lib:util",
        "//src/main/java/com/google/devtools/build/lib/actions",
        "//src/main/java/com/google/devtools/build/lib/sandbox",
        "//src/main/java/com/google/devtools/build/lib/vfs",
        "//src/main/java/com/google/devtools/build/lib/vfs:pathfragment",
        "//src/main/java/com/google/devtools/build/lib/vfs/inmemoryfs",
        "//src/main/java/com/google/devtools/build/lib/worker",
        "//src/main/java/com/google/devtools/common/options:options_internal",
        "//src/main/protobuf:worker_protocol_java_proto",
        "//src/test/java/com/google/devtools/build/lib/events:testutil",
        "//src/test/java/com/google/devtools/build/lib/testutil",
        "//src/test/java/com/google/devtools/build/lib/testutil:JunitUtils",
        "//src/test/java/com/google/devtools/build/lib/testutil:TestUtils",
        "//src/test/java/com/google/devtools/build/lib/vfs/util",
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

java_binary(
    name = "ExampleWorkerMultiplexer",
    main_class = "com.google.devtools.build.lib.worker.ExampleWorkerMultiplexer",
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
    "shell",
    "server",
    "skyframe",
]

[
    test_suite(
        name = n,
        tags = [n],
    )
    for n in TEST_SUITES
]

test_suite(
    name = "exec",
    tests = [
        "//src/test/java/com/google/devtools/build/lib/exec:ExecTests",
        "//src/test/java/com/google/devtools/build/lib/exec/local:ExecLocalTests",
    ],
)

test_suite(
    name = "foundations",
    tests = [
        ":cmdline_test",
        ":collect_nestedset_test",
        ":collect_test",
        ":concurrent_test",
        "//src/test/java/com/google/devtools/build/lib/events:EventsTests",
        "//src/test/java/com/google/devtools/build/lib/testutil:TestUtilTests",
    ],
)

test_suite(
    name = "analysis",
    tests = [
        "//src/test/java/com/google/devtools/build/lib/analysis:AllAnalysisTests",
    ],
)

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
        "//src/test/java/com/google/devtools/build/lib/bazel:all_windows_tests",
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
