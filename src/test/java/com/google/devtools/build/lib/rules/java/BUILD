load("@rules_java//java:defs.bzl", "java_library", "java_test")

package(
    default_applicable_licenses = ["//:license"],
    default_testonly = 1,
    default_visibility = ["//src:__subpackages__"],
)

filegroup(
    name = "srcs",
    testonly = 0,
    srcs = glob(["**"]) + [
        "//src/test/java/com/google/devtools/build/lib/rules/java/proto:srcs",
    ],
    visibility = ["//src/test/java/com/google/devtools/build/lib/rules:__pkg__"],
)

java_library(
    name = "java-info",
    srcs = [
        "JavaCcInfoProvider.java",
        "JavaCompilationArgsProvider.java",
        "JavaCompilationInfoProvider.java",
        "JavaGenJarsProvider.java",
        "JavaInfo.java",
        "JavaModuleFlagsProvider.java",
        "JavaRuleOutputJarsProvider.java",
        "JavaSourceJarsProvider.java",
    ],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/actions:artifacts",
        "//src/main/java/com/google/devtools/build/lib/analysis:analysis_cluster",
        "//src/main/java/com/google/devtools/build/lib/analysis:transitive_info_collection",
        "//src/main/java/com/google/devtools/build/lib/cmdline",
        "//src/main/java/com/google/devtools/build/lib/collect/compacthashset",
        "//src/main/java/com/google/devtools/build/lib/collect/nestedset",
        "//src/main/java/com/google/devtools/build/lib/concurrent:thread_safety",
        "//src/main/java/com/google/devtools/build/lib/packages",
        "//src/main/java/com/google/devtools/build/lib/packages:provider",
        "//src/main/java/com/google/devtools/build/lib/rules/java:java-compilation",
        "//src/main/java/com/google/devtools/build/lib/skyframe:bzl_load_value",
        "//src/main/java/com/google/devtools/build/lib/skyframe/serialization/autocodec",
        "//src/main/java/com/google/devtools/build/lib/skyframe/serialization/autocodec:serialization-constant",
        "//src/main/java/com/google/devtools/build/lib/starlarkbuildapi/java",
        "//src/main/java/net/starlark/java/eval",
        "//src/main/java/net/starlark/java/syntax",
        "//src/test/java/com/google/devtools/build/lib/rules/cpp:cc-info",
        "//third_party/java/auto:auto_value",
        "//third_party/java/error_prone:annotations",
        "//third_party/java/guava:annotations",
        "//third_party/java/guava:base",
        "//third_party/java/guava:collect",
        "//third_party/java/jsr305_annotations",
        "//third_party/java/protobuf",
        "//third_party/java/protobuf:protobuf_lite",
    ],
)

java_test(
    name = "JavaCompileActionBuilderTest",
    srcs = ["JavaCompileActionBuilderTest.java"],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/actions",
        "//src/main/java/com/google/devtools/build/lib/actions:action_input_prefetcher",
        "//src/main/java/com/google/devtools/build/lib/actions:artifacts",
        "//src/main/java/com/google/devtools/build/lib/actions:file_metadata",
        "//src/main/java/com/google/devtools/build/lib/cmdline",
        "//src/main/java/com/google/devtools/build/lib/remote:remote_action_file_system",
        "//src/main/java/com/google/devtools/build/lib/remote:remote_action_input_fetcher",
        "//src/main/java/com/google/devtools/build/lib/remote/common:bulk_transfer_exception",
        "//src/main/java/com/google/devtools/build/lib/remote/common:cache_not_found_exception",
        "//src/main/java/com/google/devtools/build/lib/rules/java:java-compilation",
        "//src/main/java/com/google/devtools/build/lib/vfs",
        "//src/main/java/com/google/devtools/build/lib/vfs/inmemoryfs",
        "//src/main/protobuf:deps_java_proto",
        "//src/test/java/com/google/devtools/build/lib/actions/util",
        "//src/test/java/com/google/devtools/build/lib/analysis/util",
        "//src/test/java/com/google/devtools/build/lib/rules/java:java_compile_action_test_helper",
        "//src/test/java/com/google/devtools/build/lib/testutil:JunitUtils",
        "//src/test/java/com/google/devtools/build/lib/testutil:TestConstants",
        "//third_party/java/guava:collect",
        "//third_party/java/guava:concurrent",
        "//third_party/java/junit",
        "//third_party/java/mockito",
        "//third_party/java/truth",
        "@remoteapis//:build_bazel_remote_execution_v2_remote_execution_java_proto",
    ],
)

java_test(
    name = "JavaConfigurationTest",
    srcs = ["JavaConfigurationTest.java"],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/analysis/config:invalid_configuration_exception",
        "//src/test/java/com/google/devtools/build/lib/analysis/util",
        "//third_party/java/junit",
        "//third_party/java/truth",
    ],
)

java_test(
    name = "JavaInfoRoundtripTest",
    srcs = ["JavaInfoRoundtripTest.java"],
    data = ["//tools/build_defs/inspect:defs"],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/analysis:configured_target",
        "//src/main/java/com/google/devtools/build/lib/packages:provider",
        "//src/main/java/com/google/devtools/build/lib/vfs",
        "//src/main/java/com/google/devtools/build/lib/vfs:pathfragment",
        "//src/main/java/net/starlark/java/eval",
        "//src/test/java/com/google/devtools/build/lib/analysis/util",
        "//src/test/java/com/google/devtools/build/lib/testutil:TestConstants",
        "//third_party/java/guava:collect",
        "//third_party/java/junit",
        "//third_party/java/truth",
    ],
)

java_test(
    name = "JavaInfoStarlarkApiTest",
    srcs = ["JavaInfoStarlarkApiTest.java"],
    shard_count = 5,
    deps = [
        "//src/main/java/com/google/devtools/build/lib/actions:artifacts",
        "//src/main/java/com/google/devtools/build/lib/analysis:configured_target",
        "//src/main/java/com/google/devtools/build/lib/cmdline",
        "//src/main/java/com/google/devtools/build/lib/collect/nestedset",
        "//src/main/java/com/google/devtools/build/lib/packages:provider",
        "//src/main/java/com/google/devtools/build/lib/rules/java:java-compilation",
        "//src/main/java/com/google/devtools/build/lib/skyframe:bzl_load_value",
        "//src/main/java/com/google/devtools/build/lib/vfs",
        "//src/main/java/net/starlark/java/eval",
        "//src/test/java/com/google/devtools/build/lib/actions/util",
        "//src/test/java/com/google/devtools/build/lib/analysis/util",
        "//src/test/java/com/google/devtools/build/lib/rules/java:java-info",
        "//src/test/java/com/google/devtools/build/lib/testutil:TestConstants",
        "//third_party/java/guava:collect",
        "//third_party/java/junit",
        "//third_party/java/truth",
    ],
)

java_test(
    name = "JavaInfoTest",
    srcs = ["JavaInfoTest.java"],
    deps = [
        "//src/test/java/com/google/devtools/build/lib/rules/java:java-info",
        "//third_party/java/junit",
        "//third_party/java/truth",
    ],
)

java_test(
    name = "JavaCompilationInfoProviderTest",
    srcs = ["JavaCompilationInfoProviderTest.java"],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/actions:artifacts",
        "//src/main/java/com/google/devtools/build/lib/collect/nestedset",
        "//src/main/java/com/google/devtools/build/lib/vfs",
        "//src/test/java/com/google/devtools/build/lib/actions/util",
        "//src/test/java/com/google/devtools/build/lib/rules/java:java-info",
        "//src/test/java/com/google/devtools/build/lib/testutil",
        "//third_party/java/guava:testing",
        "//third_party/java/junit",
    ],
)

java_test(
    name = "JavaOptionsTest",
    srcs = ["JavaOptionsTest.java"],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/analysis/config:build_options",
        "//src/main/java/com/google/devtools/build/lib/rules/java:java-compilation",
        "//src/test/java/com/google/devtools/build/lib/analysis/util",
        "//third_party/java/guava:collect",
        "//third_party/java/junit",
        "//third_party/java/truth",
    ],
)

java_test(
    name = "JavaRuntimeTest",
    srcs = ["JavaRuntimeTest.java"],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/analysis:configured_target",
        "//src/main/java/com/google/devtools/build/lib/analysis:provider_collection",
        "//src/main/java/com/google/devtools/build/lib/analysis/platform",
        "//src/main/java/com/google/devtools/build/lib/packages",
        "//src/main/java/com/google/devtools/build/lib/packages:provider",
        "//src/main/java/com/google/devtools/build/lib/rules/java:java-compilation",
        "//src/main/java/net/starlark/java/eval",
        "//src/test/java/com/google/devtools/build/lib/analysis/util",
        "//third_party/java/junit",
        "//third_party/java/truth",
    ],
)

java_test(
    name = "JavaStarlarkApiTest",
    srcs = ["JavaStarlarkApiTest.java"],
    deps = [
        ":java-info",
        "//src/main/java/com/google/devtools/build/lib/rules/java:java-compilation",
        "//src/main/java/com/google/devtools/build/lib/starlarkbuildapi/java",
        "//src/main/java/net/starlark/java/annot",
        "//src/test/java/com/google/devtools/build/lib/analysis/util",
        "//src/test/java/com/google/devtools/build/lib/testutil:TestConstants",
        "//third_party/java/junit",
        "//third_party/java/testparameterinjector",
        "//third_party/java/truth",
    ],
)

java_test(
    name = "JavaPluginsFlagAliasTest",
    srcs = ["JavaPluginsFlagAliasTest.java"],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/analysis:configured_target",
        "//src/main/java/com/google/devtools/build/lib/rules/java:java-compilation",
        "//src/test/java/com/google/devtools/build/lib/analysis/util",
        "//src/test/java/com/google/devtools/build/lib/testutil:TestConstants",
        "//third_party/java/junit",
        "//third_party/java/truth",
    ],
)

java_test(
    name = "JavaUtilTest",
    srcs = ["JavaUtilTest.java"],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/collect/nestedset",
        "//src/main/java/com/google/devtools/build/lib/rules/java:java-compilation",
        "//src/main/java/com/google/devtools/build/lib/vfs:pathfragment",
        "//third_party/java/guava:collect",
        "//third_party/java/junit",
        "//third_party/java/truth",
    ],
)

java_test(
    name = "JavaInfoCodecTest",
    srcs = ["JavaInfoCodecTest.java"],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/compress:compression_service_impl",
        "//src/main/java/com/google/devtools/build/lib/skyframe/serialization",
        "//src/main/java/com/google/devtools/build/lib/skyframe/serialization/testutils",
        "//src/main/java/com/google/devtools/build/lib/skyframe/serialization/testutils:depsutils",
        "//src/main/java/com/google/devtools/build/lib/skyframe/serialization/testutils:dumper",
        "//src/main/java/com/google/devtools/build/lib/skyframe/serialization/testutils:round-tripping",
        "//src/main/java/net/starlark/java/eval",
        "//src/test/java/com/google/devtools/build/lib/analysis/util",
        "//src/test/java/com/google/devtools/build/lib/rules/java:java-info",
        "//third_party/java/guava:collect",
        "//third_party/java/junit",
        "//third_party/java/truth",
    ],
)

java_library(
    name = "java_compile_action_test_helper",
    testonly = 1,
    srcs = ["JavaCompileActionTestHelper.java"],
    visibility = ["//visibility:public"],
    deps = [
        "//src/java_tools/buildjar/java/com/google/devtools/build/buildjar:optionsparser",
        "//src/main/java/com/google/devtools/build/lib/analysis/config:core_option_converters",
        "//src/main/java/com/google/devtools/build/lib/rules/java:java-compilation",
        "//third_party/java/guava:base",
    ],
)
