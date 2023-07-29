load("@rules_java//java:defs.bzl", "java_library", "java_test")

package(
    default_visibility = [
        ":__subpackages__",
        "//tools/test:__pkg__",
    ],
)

# Generate list of all srcs via:
# bazel query 'filter("srcs", kind("filegroup rule", //src/test/java/com/google/devtools/build/lib/...))' | sort | sed -e "s/^/\"/" | sed -e "s/$/\",/" | fgrep -v "build/lib:srcs"
filegroup(
    name = "srcs",
    srcs = glob(["**"]) + [
        "//src/test/java/com/google/devtools/build/lib/actions:srcs",
        "//src/test/java/com/google/devtools/build/lib/analysis:srcs",
        "//src/test/java/com/google/devtools/build/lib/authandtls:srcs",
        "//src/test/java/com/google/devtools/build/lib/bazel:srcs",
        "//src/test/java/com/google/devtools/build/lib/bazel/google:srcs",
        "//src/test/java/com/google/devtools/build/lib/blackbox:srcs",
        "//src/test/java/com/google/devtools/build/lib/bugreport:srcs",
        "//src/test/java/com/google/devtools/build/lib/buildeventservice:srcs",
        "//src/test/java/com/google/devtools/build/lib/buildeventstream:srcs",
        "//src/test/java/com/google/devtools/build/lib/buildtool:srcs",
        "//src/test/java/com/google/devtools/build/lib/cmdline:srcs",
        "//src/test/java/com/google/devtools/build/lib/collect:srcs",
        "//src/test/java/com/google/devtools/build/lib/concurrent:srcs",
        "//src/test/java/com/google/devtools/build/lib/dynamic:srcs",
        "//src/test/java/com/google/devtools/build/lib/events:srcs",
        "//src/test/java/com/google/devtools/build/lib/exec:srcs",
        "//src/test/java/com/google/devtools/build/lib/generatedprojecttest:srcs",
        "//src/test/java/com/google/devtools/build/lib/generatedprojecttest/util:srcs",
        "//src/test/java/com/google/devtools/build/lib/graph:srcs",
        "//src/test/java/com/google/devtools/build/lib/integration/util:srcs",
        "//src/test/java/com/google/devtools/build/lib/io:srcs",
        "//src/test/java/com/google/devtools/build/lib/metrics:srcs",
        "//src/test/java/com/google/devtools/build/lib/outputfilter:srcs",
        "//src/test/java/com/google/devtools/build/lib/packages:srcs",
        "//src/test/java/com/google/devtools/build/lib/packages/metrics:srcs",
        "//src/test/java/com/google/devtools/build/lib/packages/semantics:srcs",
        "//src/test/java/com/google/devtools/build/lib/pkgcache:srcs",
        "//src/test/java/com/google/devtools/build/lib/platform:srcs",
        "//src/test/java/com/google/devtools/build/lib/platform/darwin:srcs",
        "//src/test/java/com/google/devtools/build/lib/profiler:srcs",
        "//src/test/java/com/google/devtools/build/lib/profiler/callcounts:srcs",
        "//src/test/java/com/google/devtools/build/lib/profiler/memory:srcs",
        "//src/test/java/com/google/devtools/build/lib/query2:srcs",
        "//src/test/java/com/google/devtools/build/lib/remote:srcs",
        "//src/test/java/com/google/devtools/build/lib/repository:srcs",
        "//src/test/java/com/google/devtools/build/lib/rules:srcs",
        "//src/test/java/com/google/devtools/build/lib/sandbox:srcs",
        "//src/test/java/com/google/devtools/build/lib/server:srcs",
        "//src/test/java/com/google/devtools/build/lib/skyframe:srcs",
        "//src/test/java/com/google/devtools/build/lib/skyframe/packages:srcs",
        "//src/test/java/com/google/devtools/build/lib/skyframe/serialization:srcs",
        "//src/test/java/com/google/devtools/build/lib/standalone:srcs",
        "//src/test/java/com/google/devtools/build/lib/starlark:srcs",
        "//src/test/java/com/google/devtools/build/lib/starlarkdebug/server:srcs",
        "//src/test/java/com/google/devtools/build/lib/supplier:srcs",
        "//src/test/java/com/google/devtools/build/lib/testing/common:srcs",
        "//src/test/java/com/google/devtools/build/lib/testutil:srcs",
        "//src/test/java/com/google/devtools/build/lib/unix:srcs",
        "//src/test/java/com/google/devtools/build/lib/unsafe:srcs",
        "//src/test/java/com/google/devtools/build/lib/util:srcs",
        "//src/test/java/com/google/devtools/build/lib/versioning:srcs",
        "//src/test/java/com/google/devtools/build/lib/vfs:srcs",
        "//src/test/java/com/google/devtools/build/lib/view/cpp:srcs",
        "//src/test/java/com/google/devtools/build/lib/view/java:srcs",
        "//src/test/java/com/google/devtools/build/lib/view/util:srcs",
        "//src/test/java/com/google/devtools/build/lib/windows:srcs",
        "//src/test/java/com/google/devtools/build/lib/worker:srcs",
        "//src/test/java/net/starlark/java/annot/processor:srcs",
        "//src/test/java/net/starlark/java/eval:srcs",
        "//src/test/java/net/starlark/java/syntax:srcs",
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
        "//third_party:mockito",
    ],
)

java_test(
    name = "runtime-tests",
    srcs = glob(
        [
            "runtime/*.java",
            "runtime/commands/*.java",
        ],
        exclude = [
            "runtime/commands/ConfigCommandTest.java",
            "runtime/commands/DumpCommandTest.java",
        ],
    ),
    test_class = "com.google.devtools.build.lib.AllTests",
    deps = [
        ":AllTests",
        "//src/main/java/com/google/devtools/build/lib:build-request-options",
        "//src/main/java/com/google/devtools/build/lib:loading-phase-threads-option",
        "//src/main/java/com/google/devtools/build/lib:runtime",
        "//src/main/java/com/google/devtools/build/lib:runtime/blaze_command_result",
        "//src/main/java/com/google/devtools/build/lib:runtime/command_dispatcher",
        "//src/main/java/com/google/devtools/build/lib:runtime/command_line_path_factory",
        "//src/main/java/com/google/devtools/build/lib:runtime/memory_pressure",
        "//src/main/java/com/google/devtools/build/lib:runtime/safe_request_logging",
        "//src/main/java/com/google/devtools/build/lib/actions",
        "//src/main/java/com/google/devtools/build/lib/actions:action_input_helper",
        "//src/main/java/com/google/devtools/build/lib/actions:action_lookup_data",
        "//src/main/java/com/google/devtools/build/lib/actions:action_lookup_key",
        "//src/main/java/com/google/devtools/build/lib/actions:artifacts",
        "//src/main/java/com/google/devtools/build/lib/actions:execution_requirements",
        "//src/main/java/com/google/devtools/build/lib/actions:file_metadata",
        "//src/main/java/com/google/devtools/build/lib/actions:localhost_capacity",
        "//src/main/java/com/google/devtools/build/lib/analysis:analysis_cluster",
        "//src/main/java/com/google/devtools/build/lib/analysis:blaze_directories",
        "//src/main/java/com/google/devtools/build/lib/analysis:blaze_version_info",
        "//src/main/java/com/google/devtools/build/lib/analysis:config/build_configuration",
        "//src/main/java/com/google/devtools/build/lib/analysis:config/build_options",
        "//src/main/java/com/google/devtools/build/lib/analysis:config/core_options",
        "//src/main/java/com/google/devtools/build/lib/analysis:config/fragment",
        "//src/main/java/com/google/devtools/build/lib/analysis:config/fragment_factory",
        "//src/main/java/com/google/devtools/build/lib/analysis:config/fragment_options",
        "//src/main/java/com/google/devtools/build/lib/analysis:config/fragment_registry",
        "//src/main/java/com/google/devtools/build/lib/analysis:configured_target",
        "//src/main/java/com/google/devtools/build/lib/analysis:server_directories",
        "//src/main/java/com/google/devtools/build/lib/analysis:test/test_configuration",
        "//src/main/java/com/google/devtools/build/lib/bazel:modules",
        "//src/main/java/com/google/devtools/build/lib/bazel/repository/downloader",
        "//src/main/java/com/google/devtools/build/lib/bazel/rules",
        "//src/main/java/com/google/devtools/build/lib/bugreport",
        "//src/main/java/com/google/devtools/build/lib/buildeventstream",
        "//src/main/java/com/google/devtools/build/lib/buildeventstream/proto:build_event_stream_java_proto",
        "//src/main/java/com/google/devtools/build/lib/buildeventstream/transports",
        "//src/main/java/com/google/devtools/build/lib/clock",
        "//src/main/java/com/google/devtools/build/lib/cmdline",
        "//src/main/java/com/google/devtools/build/lib/collect/nestedset",
        "//src/main/java/com/google/devtools/build/lib/events",
        "//src/main/java/com/google/devtools/build/lib/exec:bin_tools",
        "//src/main/java/com/google/devtools/build/lib/exec:execution_options",
        "//src/main/java/com/google/devtools/build/lib/packages",
        "//src/main/java/com/google/devtools/build/lib/packages/semantics",
        "//src/main/java/com/google/devtools/build/lib/pkgcache",
        "//src/main/java/com/google/devtools/build/lib/profiler",
        "//src/main/java/com/google/devtools/build/lib/query2",
        "//src/main/java/com/google/devtools/build/lib/query2/common:abstract-blaze-query-env",
        "//src/main/java/com/google/devtools/build/lib/query2/engine",
        "//src/main/java/com/google/devtools/build/lib/query2/query/output",
        "//src/main/java/com/google/devtools/build/lib/runtime/commands",
        "//src/main/java/com/google/devtools/build/lib/skyframe:build_configuration",
        "//src/main/java/com/google/devtools/build/lib/skyframe:configuration_phase_started_event",
        "//src/main/java/com/google/devtools/build/lib/skyframe:configured_target_key",
        "//src/main/java/com/google/devtools/build/lib/skyframe:configured_target_progress_receiver",
        "//src/main/java/com/google/devtools/build/lib/skyframe:loading_phase_started_event",
        "//src/main/java/com/google/devtools/build/lib/skyframe:package_progress_receiver",
        "//src/main/java/com/google/devtools/build/lib/skyframe:top_level_status_events",
        "//src/main/java/com/google/devtools/build/lib/unix",
        "//src/main/java/com/google/devtools/build/lib/util",
        "//src/main/java/com/google/devtools/build/lib/util:abrupt_exit_exception",
        "//src/main/java/com/google/devtools/build/lib/util:detailed_exit_code",
        "//src/main/java/com/google/devtools/build/lib/util:exit_code",
        "//src/main/java/com/google/devtools/build/lib/util:os",
        "//src/main/java/com/google/devtools/build/lib/util:string",
        "//src/main/java/com/google/devtools/build/lib/util/io",
        "//src/main/java/com/google/devtools/build/lib/util/io:out-err",
        "//src/main/java/com/google/devtools/build/lib/vfs",
        "//src/main/java/com/google/devtools/build/lib/vfs:pathfragment",
        "//src/main/java/com/google/devtools/build/lib/vfs/inmemoryfs",
        "//src/main/java/com/google/devtools/common/options",
        "//src/main/java/com/google/devtools/common/options:invocation_policy",
        "//src/main/java/net/starlark/java/syntax",
        "//src/main/protobuf:any_java_proto",
        "//src/main/protobuf:command_line_java_proto",
        "//src/main/protobuf:execution_graph_java_proto",
        "//src/main/protobuf:failure_details_java_proto",
        "//src/main/protobuf:invocation_policy_java_proto",
        "//src/main/protobuf:memory_pressure_java_proto",
        "//src/main/protobuf:test_status_java_proto",
        "//src/main/protobuf:wrappers_java_proto",
        "//src/test/java/com/google/devtools/build/lib/actions/util",
        "//src/test/java/com/google/devtools/build/lib/analysis/util",
        "//src/test/java/com/google/devtools/build/lib/buildtool/util",
        "//src/test/java/com/google/devtools/build/lib/events:testutil",
        "//src/test/java/com/google/devtools/build/lib/exec/util",
        "//src/test/java/com/google/devtools/build/lib/starlark/util",
        "//src/test/java/com/google/devtools/build/lib/testutil",
        "//src/test/java/com/google/devtools/build/lib/testutil:JunitUtils",
        "//src/test/java/com/google/devtools/build/lib/testutil:TestConstants",
        "//src/test/java/com/google/devtools/build/lib/testutil:TestThread",
        "//src/test/java/com/google/devtools/build/lib/testutil:TestUtils",
        "//src/test/java/com/google/devtools/build/lib/vfs/util",
        "//src/test/java/com/google/devtools/common/options:testutils",
        "//third_party:apache_commons_lang",
        "//third_party:guava",
        "//third_party:guava-testlib",
        "//third_party:junit4",
        "//third_party:mockito",
        "//third_party:truth",
        "//third_party/protobuf:protobuf_java",
        "@com_google_protobuf//java/util",
        "@maven//:com_google_testparameterinjector_test_parameter_injector",
        "@zstd-jni//:zstd-jni",
    ],
)

java_test(
    name = "ConfigCommandTest",
    srcs = ["runtime/commands/ConfigCommandTest.java"],
    # TODO(bazel-team) The test is currently broken due to BazelAnalysisMock not supporting integration tests.
    tags = ["manual"],
)

java_test(
    name = "DumpCommandTest",
    srcs = ["runtime/commands/DumpCommandTest.java"],
    deps = [
        "//src/main/java/com/google/devtools/build/lib:runtime",
        "//src/main/java/com/google/devtools/build/lib:runtime/blaze_command_result",
        "//src/main/java/com/google/devtools/build/lib/analysis:view_creation_failed_exception",
        "//src/main/java/com/google/devtools/build/lib/runtime/commands",
        "//src/main/java/com/google/devtools/build/lib/util:exit_code",
        "//src/main/java/com/google/devtools/build/lib/util/io",
        "//src/test/java/com/google/devtools/build/lib/buildtool/util",
        "//third_party:guava",
        "//third_party:junit4",
        "//third_party:truth",
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
        "//src/test/java/com/google/devtools/build/lib/cmdline:CmdLineTests",
        "//src/test/java/com/google/devtools/build/lib/collect:CollectTests",
        "//src/test/java/com/google/devtools/build/lib/concurrent:ConcurrentTests",
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
