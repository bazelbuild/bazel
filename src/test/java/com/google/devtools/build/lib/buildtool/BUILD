load("@rules_java//java:defs.bzl", "java_library", "java_test")
load("@bazel_skylib//rules:build_test.bzl", "build_test")

package(
    default_testonly = 1,
    default_visibility = ["//src:__subpackages__"],
)

filegroup(
    name = "srcs",
    testonly = 0,
    srcs = glob(["*"]) + [
        "//src/test/java/com/google/devtools/build/lib/buildtool/util:srcs",
    ],
    visibility = ["//src:__subpackages__"],
)

java_test(
    name = "JobsConverterTest",
    srcs = [
        "JobsConverterTest.java",
    ],
    deps = [
        "//src/main/java/com/google/devtools/build/lib:build-request-options",
        "//src/main/java/com/google/devtools/build/lib/actions:localhost_capacity",
        "//src/main/java/com/google/devtools/common/options",
        "//third_party:junit4",
        "//third_party:truth",
    ],
)

java_test(
    name = "SymlinkForestTest",
    srcs = [
        "SymlinkForestTest.java",
    ],
    deps = [
        "//src/main/java/com/google/devtools/build/lib:runtime",
        "//src/main/java/com/google/devtools/build/lib/cmdline",
        "//src/main/java/com/google/devtools/build/lib/util:abrupt_exit_exception",
        "//src/main/java/com/google/devtools/build/lib/vfs",
        "//src/main/java/com/google/devtools/build/lib/vfs:pathfragment",
        "//src/main/java/com/google/devtools/build/lib/vfs/inmemoryfs",
        "//src/test/java/com/google/devtools/build/lib/testutil",
        "//src/test/java/com/google/devtools/build/lib/testutil:TestConstants",
        "//third_party:guava",
        "//third_party:junit4",
        "//third_party:truth",
    ],
)

test_suite(
    name = "BuildtoolTests",
    tests = [
        ":JobsConverterTest",
        ":SymlinkForestTest",
    ],
)

java_test(
    name = "ActionListenerIntegrationTest",
    srcs = ["ActionListenerIntegrationTest.java"],
    tags = [
        "manual",
        "no_windows",
    ],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/actions",
        "//src/main/java/com/google/devtools/build/lib/actions:artifacts",
        "//src/main/java/com/google/devtools/build/lib/analysis:analysis_cluster",
        "//src/main/java/com/google/devtools/build/lib/analysis:configured_target",
        "//src/main/java/com/google/devtools/build/lib/analysis:extra_action_artifacts_provider",
        "//src/main/java/com/google/devtools/build/lib/analysis:view_creation_failed_exception",
        "//src/main/java/com/google/devtools/build/lib/collect/nestedset",
        "//src/main/java/com/google/devtools/build/lib/rules/java:java-compilation",
        "//src/main/java/com/google/devtools/build/lib/util",
        "//src/main/java/com/google/devtools/build/lib/vfs",
        "//src/main/java/com/google/devtools/common/options",
        "//src/main/protobuf:extra_actions_base_java_proto",
        "//src/test/java/com/google/devtools/build/lib/buildtool/util",
        "//src/test/java/com/google/devtools/build/lib/testutil",
        "//third_party:guava",
        "//third_party:junit4",
        "//third_party:truth",
    ],
)

java_test(
    name = "BogusExecutionTimeTest",
    srcs = ["BogusExecutionTimeTest.java"],
    tags = [
        "manual",
        "no_windows",
    ],
    deps = [
        "//src/main/java/com/google/devtools/build/lib:runtime",
        "//src/test/java/com/google/devtools/build/lib/buildtool/util",
        "//src/test/java/com/google/devtools/build/lib/testutil",
        "//third_party:junit4",
    ],
)

java_test(
    name = "CompileAfterOptionChangeTest",
    srcs = ["CompileAfterOptionChangeTest.java"],
    tags = [
        "manual",
        "no_windows",
    ],
    deps = [
        "//src/main/java/com/google/devtools/build/lib:runtime",
        "//src/main/java/com/google/devtools/build/lib/vfs",
        "//src/test/java/com/google/devtools/build/lib/buildtool/util",
        "//src/test/java/com/google/devtools/build/lib/testutil",
        "//third_party:junit4",
        "//third_party:truth",
    ],
)

java_test(
    name = "CompileOneDependencyIntegrationTest",
    srcs = ["CompileOneDependencyIntegrationTest.java"],
    tags = [
        "manual",
        "no_windows",
    ],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/actions",
        "//src/main/java/com/google/devtools/build/lib/events",
        "//src/test/java/com/google/devtools/build/lib/buildtool/util",
        "//src/test/java/com/google/devtools/build/lib/testutil",
        "//src/test/java/com/google/devtools/build/lib/testutil:JunitUtils",
        "//third_party:junit4",
        "//third_party:truth",
    ],
)

java_test(
    name = "ContextProviderInitializationTest",
    srcs = ["ContextProviderInitializationTest.java"],
    tags = [
        "manual",
        "no_windows",
    ],
    deps = [
        "//src/main/java/com/google/devtools/build/lib:runtime",
        "//src/main/java/com/google/devtools/build/lib/actions",
        "//src/main/java/com/google/devtools/build/lib/analysis:artifacts_to_owner_labels",
        "//src/main/java/com/google/devtools/build/lib/exec:executor_builder",
        "//src/main/java/com/google/devtools/build/lib/exec:executor_lifecycle_listener",
        "//src/main/java/com/google/devtools/build/lib/util:abrupt_exit_exception",
        "//src/main/java/com/google/devtools/build/lib/util:detailed_exit_code",
        "//src/main/protobuf:failure_details_java_proto",
        "//src/test/java/com/google/devtools/build/lib/buildtool/util",
        "//third_party:guava",
        "//third_party:junit4",
    ],
)

java_test(
    name = "CorruptedActionCacheTest",
    srcs = ["CorruptedActionCacheTest.java"],
    tags = [
        "manual",
        "no_windows",
    ],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/util",
        "//src/main/java/com/google/devtools/build/lib/vfs",
        "//src/test/java/com/google/devtools/build/lib/buildtool/util",
        "//src/test/java/com/google/devtools/build/lib/packages:testutil",
        "//src/test/java/com/google/devtools/build/lib/testutil",
        "//third_party:junit4",
        "//third_party:truth",
    ],
)

java_test(
    name = "DanglingSymlinkTest",
    srcs = ["DanglingSymlinkTest.java"],
    tags = [
        "manual",
        "no_windows",
    ],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/actions",
        "//src/main/java/com/google/devtools/build/lib/vfs",
        "//src/main/java/com/google/devtools/build/lib/vfs:pathfragment",
        "//src/test/java/com/google/devtools/build/lib/buildtool/util",
        "//src/test/java/com/google/devtools/build/lib/packages:testutil",
        "//src/test/java/com/google/devtools/build/lib/testutil",
        "//third_party:junit4",
        "//third_party:truth",
    ],
)

java_test(
    name = "UnusedInputsFailureIntegrationTest",
    srcs = ["UnusedInputsFailureIntegrationTest.java"],
    tags = ["manual"],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/actions",
        "//src/main/java/com/google/devtools/build/lib/vfs",
        "//src/main/java/com/google/devtools/build/lib/vfs:pathfragment",
        "//src/test/java/com/google/devtools/build/lib/buildtool/util",
        "//third_party:junit4",
        "//third_party:truth",
        "@com_google_testparameterinjector//:testparameterinjector",
    ],
)

java_test(
    name = "DirectoryArtifactWarningTest",
    srcs = ["DirectoryArtifactWarningTest.java"],
    tags = [
        "manual",
        "no_windows",
    ],
    deps = [
        "//src/test/java/com/google/devtools/build/lib/buildtool/util",
        "//src/test/java/com/google/devtools/build/lib/packages:testutil",
        "//src/test/java/com/google/devtools/build/lib/testutil",
        "//third_party:junit4",
    ],
)

java_test(
    name = "EditDuringBuildTest",
    srcs = ["EditDuringBuildTest.java"],
    tags = [
        "manual",
        "no_windows",
    ],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/vfs",
        "//src/test/java/com/google/devtools/build/lib/buildtool/util",
        "//src/test/java/com/google/devtools/build/lib/packages:testutil",
        "//src/test/java/com/google/devtools/build/lib/testutil",
        "//third_party:guava",
        "//third_party:junit4",
        "//third_party:truth",
    ],
)

java_test(
    name = "EnvironmentRestrictedBuildTest",
    srcs = ["EnvironmentRestrictedBuildTest.java"],
    tags = [
        "manual",
        "no_windows",
    ],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/analysis:configured_target",
        "//src/main/java/com/google/devtools/build/lib/analysis:view_creation_failed_exception",
        "//src/test/java/com/google/devtools/build/lib/buildtool/util",
        "//third_party:guava",
        "//third_party:junit4",
        "//third_party:truth",
    ],
)

java_test(
    name = "ExecutionPhaseContainingPackageLookupTest",
    srcs = ["ExecutionPhaseContainingPackageLookupTest.java"],
    tags = [
        "manual",
        "no_windows",
    ],
    deps = [
        ":IoHookTestCase",
        "//src/main/java/com/google/devtools/build/lib/actions",
        "//src/main/java/com/google/devtools/build/lib/util",
        "//src/test/java/com/google/devtools/build/lib/testutil",
        "//src/test/java/com/google/devtools/build/lib/testutil:TestUtils",
        "//third_party:guava",
        "//third_party:junit4",
        "//third_party:truth",
    ],
)

java_test(
    name = "GenQueryIntegrationTest",
    srcs = ["GenQueryIntegrationTest.java"],
    tags = [
        "manual",
        "no_windows",
    ],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/actions",
        "//src/main/java/com/google/devtools/build/lib/actions:artifacts",
        "//src/main/java/com/google/devtools/build/lib/analysis:view_creation_failed_exception",
        "//src/test/java/com/google/devtools/build/lib/buildtool/util",
        "//src/test/java/com/google/devtools/build/lib/testutil:JunitUtils",
        "//third_party:guava",
        "//third_party:junit4",
        "//third_party:truth",
    ],
)

java_test(
    name = "IncludeValidationTest",
    srcs = ["IncludeValidationTest.java"],
    tags = [
        "manual",
        "no_windows",
    ],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/actions",
        "//src/test/java/com/google/devtools/build/lib/buildtool/util",
        "//src/test/java/com/google/devtools/build/lib/testutil",
        "//third_party:junit4",
    ],
)

java_test(
    name = "InconsistentFilesystemTest",
    srcs = ["InconsistentFilesystemTest.java"],
    tags = [
        "manual",
        "no_windows",
    ],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/unix",
        "//src/main/java/com/google/devtools/build/lib/vfs",
        "//src/main/java/com/google/devtools/build/lib/vfs:pathfragment",
        "//src/test/java/com/google/devtools/build/lib/buildtool/util",
        "//third_party:junit4",
    ],
)

java_test(
    name = "InstrumentationFilterSupportTest",
    srcs = [
        "InstrumentationFilterSupportTest.java",
    ],
    deps = [
        "//src/main/java/com/google/devtools/build/lib:runtime",
        "//src/main/java/com/google/devtools/build/lib/events",
        "//src/main/java/com/google/devtools/build/lib/packages",
        "//src/test/java/com/google/devtools/build/lib/analysis/util",
        "//third_party:junit4",
        "//third_party:truth",
    ],
)

java_test(
    name = "LabelCrossesPackageBoundaryTest",
    srcs = ["LabelCrossesPackageBoundaryTest.java"],
    tags = [
        "manual",
        "no_windows",
    ],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/cmdline",
        "//src/test/java/com/google/devtools/build/lib/buildtool/util",
        "//src/test/java/com/google/devtools/build/lib/testutil",
        "//third_party:junit4",
    ],
)

java_test(
    name = "MiscAnalysisTest",
    srcs = ["MiscAnalysisTest.java"],
    tags = [
        "manual",
        "no_windows",
    ],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/actions",
        "//src/main/java/com/google/devtools/build/lib/analysis:configured_target",
        "//src/main/java/com/google/devtools/build/lib/analysis:view_creation_failed_exception",
        "//src/main/java/com/google/devtools/build/lib/events",
        "//src/test/java/com/google/devtools/build/lib/analysis/util",
        "//src/test/java/com/google/devtools/build/lib/buildtool/util",
        "//src/test/java/com/google/devtools/build/lib/packages:testutil",
        "//src/test/java/com/google/devtools/build/lib/testutil:JunitUtils",
        "//third_party:guava",
        "//third_party:guava-testlib",
        "//third_party:junit4",
        "//third_party:truth",
    ],
)

java_test(
    name = "MissingInputActionTest",
    srcs = ["MissingInputActionTest.java"],
    tags = [
        "manual",
        "no_windows",
    ],
    deps = [
        "//src/main/java/com/google/devtools/build/lib:runtime",
        "//src/main/java/com/google/devtools/build/lib/actions",
        "//src/main/java/com/google/devtools/build/lib/bazel:modules",
        "//src/main/java/com/google/devtools/build/lib/vfs",
        "//src/main/protobuf:failure_details_java_proto",
        "//src/test/java/com/google/devtools/build/lib/buildtool/util",
        "//src/test/java/com/google/devtools/build/lib/packages:testutil",
        "//src/test/java/com/google/devtools/build/lib/testutil",
        "//third_party:junit4",
        "//third_party:truth",
    ],
)

java_test(
    name = "ModifyBuildFileTest",
    srcs = ["ModifyBuildFileTest.java"],
    tags = [
        "manual",
        "no_windows",
    ],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/actions",
        "//src/main/java/com/google/devtools/build/lib/cmdline",
        "//src/main/java/com/google/devtools/build/lib/vfs",
        "//src/test/java/com/google/devtools/build/lib/buildtool/util",
        "//third_party:junit4",
        "//third_party:truth",
    ],
)

java_test(
    name = "NoOutputActionTest",
    srcs = ["NoOutputActionTest.java"],
    tags = [
        "manual",
        "no_windows",
    ],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/actions",
        "//src/test/java/com/google/devtools/build/lib/buildtool/util",
        "//src/test/java/com/google/devtools/build/lib/packages:testutil",
        "//src/test/java/com/google/devtools/build/lib/testutil",
        "//third_party:junit4",
        "//third_party:truth",
    ],
)

java_test(
    name = "OutputArtifactConflictTest",
    srcs = ["OutputArtifactConflictTest.java"],
    tags = [
        "manual",
        "no_windows",
    ],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/actions",
        "//src/main/java/com/google/devtools/build/lib/analysis:view_creation_failed_exception",
        "//src/test/java/com/google/devtools/build/lib/buildtool/util",
        "//src/test/java/com/google/devtools/build/lib/testutil:JunitUtils",
        "//third_party:guava",
        "//third_party:junit4",
        "//third_party:truth",
    ],
)

java_test(
    name = "PackageGroupIntegrationTest",
    srcs = ["PackageGroupIntegrationTest.java"],
    tags = [
        "manual",
        "no_windows",
    ],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/analysis:view_creation_failed_exception",
        "//src/test/java/com/google/devtools/build/lib/analysis/util",
        "//src/test/java/com/google/devtools/build/lib/buildtool/util",
        "//src/test/java/com/google/devtools/build/lib/testutil",
        "//third_party:junit4",
    ],
)

java_test(
    name = "ProgressReportingTest",
    srcs = ["ProgressReportingTest.java"],
    tags = [
        "manual",
        "no_windows",
    ],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/events",
        "//src/main/java/com/google/devtools/build/lib/unix",
        "//src/main/java/com/google/devtools/build/lib/vfs",
        "//src/main/java/com/google/devtools/build/lib/vfs:pathfragment",
        "//src/test/java/com/google/devtools/build/lib/analysis/util",
        "//src/test/java/com/google/devtools/build/lib/buildtool/util",
        "//src/test/java/com/google/devtools/build/lib/testutil",
        "//src/test/java/com/google/devtools/build/lib/testutil:JunitUtils",
        "//third_party:guava",
        "//third_party:junit4",
        "//third_party:truth",
    ],
)

java_test(
    name = "QueryIntegrationTest",
    srcs = ["QueryIntegrationTest.java"],
    tags = [
        "manual",
        "no_windows",
    ],
    deps = [
        "//src/main/java/com/google/devtools/build/lib:runtime",
        "//src/main/java/com/google/devtools/build/lib:runtime/blaze_command_result",
        "//src/main/java/com/google/devtools/build/lib/events",
        "//src/main/java/com/google/devtools/build/lib/query2/query/output",
        "//src/main/java/com/google/devtools/build/lib/runtime/commands",
        "//src/main/java/com/google/devtools/build/lib/unix",
        "//src/main/java/com/google/devtools/build/lib/util:exit_code",
        "//src/main/java/com/google/devtools/build/lib/vfs",
        "//src/main/java/com/google/devtools/build/lib/vfs:pathfragment",
        "//src/main/java/com/google/devtools/common/options",
        "//src/main/protobuf:build_java_proto",
        "//src/main/protobuf:invocation_policy_java_proto",
        "//src/test/java/com/google/devtools/build/lib/buildtool/util",
        "//src/test/java/com/google/devtools/build/lib/events:testutil",
        "//third_party:guava",
        "//third_party:jsr305",
        "//third_party:junit4",
        "//third_party:truth",
        "//third_party/protobuf:protobuf_java",
    ],
)

java_test(
    name = "SymlinkDependencyAnalysisTest",
    srcs = ["SymlinkDependencyAnalysisTest.java"],
    tags = [
        "manual",
        "no_windows",
    ],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/vfs",
        "//src/test/java/com/google/devtools/build/lib/buildtool/util",
        "//src/test/java/com/google/devtools/build/lib/packages:testutil",
        "//src/test/java/com/google/devtools/build/lib/testutil",
        "//third_party:guava",
        "//third_party:junit4",
        "//third_party:truth",
    ],
)

java_library(
    name = "TransitiveDataDependencyTest",
    srcs = ["TransitiveDataDependencyTest.java"],
    tags = [
        "no_windows",
    ],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/actions",
        "//src/main/java/com/google/devtools/build/lib/actions:artifacts",
        "//src/main/java/com/google/devtools/build/lib/analysis:configured_target",
        "//src/main/java/com/google/devtools/build/lib/util/io",
        "//src/main/java/com/google/devtools/build/lib/util/io:out-err",
        "//src/test/java/com/google/devtools/build/lib/buildtool/util",
        "//src/test/java/com/google/devtools/build/lib/testutil",
        "//third_party:guava",
        "//third_party:junit4",
        "//third_party:truth",
    ],
)

java_test(
    name = "BuildResultTestCase",
    srcs = ["BuildResultTestCase.java"],
    shard_count = 4,
    tags = [
        "manual",
        "no_windows",
    ],
    test_class = "com.google.devtools.build.lib.AllTests",
    runtime_deps = ["//src/test/java/com/google/devtools/build/lib:test_runner"],
    deps = [
        "//src/main/java/com/google/devtools/build/lib:runtime",
        "//src/main/java/com/google/devtools/build/lib/actions",
        "//src/main/java/com/google/devtools/build/lib/util/io",
        "//src/main/java/com/google/devtools/build/lib/util/io:out-err",
        "//src/test/java/com/google/devtools/build/lib/buildtool/util",
        "//src/test/java/com/google/devtools/build/lib/testutil",
        "//third_party:junit4",
        "//third_party:truth",
    ],
)

java_test(
    name = "CustomRealFilesystemBuildIntegrationTest",
    srcs = ["CustomRealFilesystemBuildIntegrationTest.java"],
    shard_count = 4,
    tags = ["manual"],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/actions",
        "//src/main/java/com/google/devtools/build/lib/actions:file_metadata",
        "//src/main/java/com/google/devtools/build/lib/cmdline",
        "//src/main/java/com/google/devtools/build/lib/unix",
        "//src/main/java/com/google/devtools/build/lib/util:exit_code",
        "//src/main/java/com/google/devtools/build/lib/vfs",
        "//src/main/java/com/google/devtools/build/lib/vfs:pathfragment",
        "//src/main/protobuf:failure_details_java_proto",
        "//src/test/java/com/google/devtools/build/lib/buildtool/util",
        "//src/test/java/com/google/devtools/build/lib/testutil",
        "//src/test/java/com/google/devtools/build/lib/testutil:JunitUtils",
        "//src/test/java/com/google/devtools/build/lib/testutil:TestUtils",
        "//src/test/java/com/google/devtools/build/skyframe:testutil",
        "//third_party:guava",
        "//third_party:junit4",
        "//third_party:truth",
        "@com_google_testparameterinjector//:testparameterinjector",
    ],
)

java_library(
    name = "IoHookTestCase",
    srcs = ["IoHookTestCase.java"],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/skyframe:mutable_supplier",
        "//src/main/java/com/google/devtools/build/lib/unix",
        "//src/main/java/com/google/devtools/build/lib/vfs",
        "//src/main/java/com/google/devtools/build/lib/vfs:pathfragment",
        "//src/test/java/com/google/devtools/build/lib/buildtool/util",
        "//src/test/java/com/google/devtools/build/lib/testutil",
    ],
)

java_test(
    name = "AqueryBuildToolTest",
    srcs = ["AqueryBuildToolTest.java"],
    tags = ["manual"],
    deps = [
        "//src/main/java/com/google/devtools/build/lib:runtime",
        "//src/main/java/com/google/devtools/build/lib:runtime/blaze_command_result",
        "//src/main/java/com/google/devtools/build/lib/query2",
        "//src/main/java/com/google/devtools/build/lib/query2/engine",
        "//src/main/java/com/google/devtools/build/lib/runtime/commands",
        "//src/main/java/com/google/devtools/common/options",
        "//src/main/protobuf:failure_details_java_proto",
        "//src/test/java/com/google/devtools/build/lib/buildtool/util",
        "//third_party:guava",
        "//third_party:junit4",
        "//third_party:truth",
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
    ],
)

# TODO(b/19011399): Remove any manual tags used to disable the tests on bazel,
#  and enable manual tests inside Google if appropriate.
# Keep in sync with Copybara transformation that makes these tests manual.

build_test(
    name = "all_manual_tests",
    targets = [
        ":ActionListenerIntegrationTest",
        ":AqueryBuildToolTest",
        ":BogusExecutionTimeTest",
        ":BuildResultTestCase",
        ":CompileAfterOptionChangeTest",
        ":CompileOneDependencyIntegrationTest",
        ":ContextProviderInitializationTest",
        ":CorruptedActionCacheTest",
        ":CustomRealFilesystemBuildIntegrationTest",
        ":DanglingSymlinkTest",
        ":DirectoryArtifactWarningTest",
        ":EditDuringBuildTest",
        ":EnvironmentRestrictedBuildTest",
        ":ExecutionPhaseContainingPackageLookupTest",
        ":GenQueryIntegrationTest",
        ":IncludeValidationTest",
        ":InconsistentFilesystemTest",
        ":LabelCrossesPackageBoundaryTest",
        ":MiscAnalysisTest",
        ":MissingInputActionTest",
        ":ModifyBuildFileTest",
        ":NoOutputActionTest",
        ":OutputArtifactConflictTest",
        ":PackageGroupIntegrationTest",
        ":ProgressReportingTest",
        ":QueryIntegrationTest",
        ":SymlinkDependencyAnalysisTest",
        # Disabled in Google.
        "//src/test/java/com/google/devtools/build/lib/bazel/rules/android:BazelAndroidTests",
        # Disabled in Google.
        "//src/test/java/com/google/devtools/build/lib/bazel/rules/python:PythonTests",
        # Disabled in Google.
        "//src/test/java/com/google/devtools/build/lib/bazel/rules/sh:ShTests",
        # Disabled in Google.
        "//src/test/java/com/google/devtools/build/lib/packages:BazelDocumentationTests",
        "//src/test/java/com/google/devtools/build/lib/rules/android:AndroidDataBindingV2Test",
    ],
)
