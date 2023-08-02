load("@rules_java//java:defs.bzl", "java_library", "java_test")

package(
    default_applicable_licenses = ["//:license"],
    default_testonly = True,
    default_visibility = ["//src:__subpackages__"],
)

filegroup(
    name = "srcs",
    testonly = False,
    srcs = glob(["**"]),
    visibility = ["//src:__subpackages__"],
)

java_library(
    name = "rewinding_tests_helper",
    srcs = ["RewindingTestsHelper.java"],
    data = ["//src/test/java/com/google/devtools/build/lib:embedded_scripts"],
    deps = [
        "//src/main/java/com/google/devtools/build/lib:keep-going-option",
        "//src/main/java/com/google/devtools/build/lib/actions",
        "//src/main/java/com/google/devtools/build/lib/actions:action_lookup_data",
        "//src/main/java/com/google/devtools/build/lib/actions:artifacts",
        "//src/main/java/com/google/devtools/build/lib/actions:middleman_type",
        "//src/main/java/com/google/devtools/build/lib/analysis:config/core_options",
        "//src/main/java/com/google/devtools/build/lib/bugreport",
        "//src/main/java/com/google/devtools/build/lib/cmdline",
        "//src/main/java/com/google/devtools/build/lib/collect/nestedset:artifact_nested_set_key",
        "//src/main/java/com/google/devtools/build/lib/exec:spawn_exec_exception",
        "//src/main/java/com/google/devtools/build/lib/skyframe/rewinding",
        "//src/main/java/com/google/devtools/build/lib/vfs",
        "//src/main/java/com/google/devtools/build/skyframe",
        "//src/main/java/com/google/devtools/build/skyframe:graph_inconsistency_java_proto",
        "//src/main/java/com/google/devtools/build/skyframe:skyframe-objects",
        "//src/main/protobuf:failure_details_java_proto",
        "//src/test/java/com/google/devtools/build/lib/buildtool/util",
        "//src/test/java/com/google/devtools/build/lib/testutil:TestConstants",
        "//src/test/java/com/google/devtools/build/lib/testutil:TestUtils",
        "//src/test/java/com/google/devtools/build/lib/testutil:action_event_recorder",
        "//src/test/java/com/google/devtools/build/lib/testutil:controllable_action_strategy_module",
        "//src/test/java/com/google/devtools/build/lib/testutil:spawn_controller",
        "//src/test/java/com/google/devtools/build/lib/testutil:spawn_input_utils",
        "//src/test/java/com/google/devtools/build/skyframe:testutil",
        "//third_party:error_prone_annotations",
        "//third_party:guava",
        "//third_party:junit4",
        "//third_party:mockito",
        "//third_party:truth",
    ],
)

java_test(
    name = "RewindingTest",
    srcs = ["RewindingTest.java"],
    shard_count = 6,
    tags = ["no_windows"],  # BuildIntegrationTestCase isn't fully compatible with Windows.
    deps = [
        ":rewinding_tests_helper",
        "//src/main/java/com/google/devtools/build/lib:runtime",
        "//src/main/java/com/google/devtools/build/lib/actions:artifacts",
        "//src/main/java/com/google/devtools/build/lib/includescanning",
        "//src/test/java/com/google/devtools/build/lib/analysis/util",
        "//src/test/java/com/google/devtools/build/lib/buildtool/util",
        "//src/test/java/com/google/devtools/build/lib/testutil:action_event_recorder",
        "//third_party:guava",
        "//third_party:junit4",
        "//third_party:truth",
        "@maven//:com_google_testparameterinjector_test_parameter_injector",
    ],
)
