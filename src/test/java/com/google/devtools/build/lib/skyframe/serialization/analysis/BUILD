load("@rules_java//java:defs.bzl", "java_library", "java_test")

package(
    default_applicable_licenses = ["//:license"],
    default_visibility = ["//src:__subpackages__"],
)

filegroup(
    name = "srcs",
    srcs = glob(["**"]),
    visibility = ["//src:__subpackages__"],
)

java_test(
    name = "AnalysisCacheInvalidatorTest",
    srcs = ["AnalysisCacheInvalidatorTest.java"],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/events",
        "//src/main/java/com/google/devtools/build/lib/skyframe/serialization",
        "//src/main/java/com/google/devtools/build/lib/skyframe/serialization:visible-for-serialization",
        "//src/main/java/com/google/devtools/build/lib/skyframe/serialization/analysis:analysis_cache_invalidator",
        "//src/main/java/com/google/devtools/build/lib/skyframe/serialization/analysis:client_id",
        "//src/main/java/com/google/devtools/build/lib/skyframe/serialization/analysis:remote_analysis_cache_client",
        "//src/main/java/com/google/devtools/build/lib/skyframe/serialization/analysis:remote_analysis_caching_state",
        "//src/main/java/com/google/devtools/build/lib/skyframe/serialization/autocodec",
        "//src/main/java/com/google/devtools/build/skyframe:skyframe-objects",
        "//third_party:guava",
        "//third_party:junit4",
        "//third_party:mockito",
        "//third_party:truth",
        "@com_google_protobuf//:protobuf_java",
    ],
)

java_library(
    name = "SkycacheIntegrationTestBase",
    testonly = True,
    srcs = ["SkycacheIntegrationTestBase.java"],
    deps = [
        ":SkycacheIntegrationTestHelpers",
        "//src/main/java/com/google/devtools/build/lib:runtime",
        "//src/main/java/com/google/devtools/build/lib/actions:action_lookup_data",
        "//src/main/java/com/google/devtools/build/lib/actions:action_lookup_key",
        "//src/main/java/com/google/devtools/build/lib/analysis:blaze_directories",
        "//src/main/java/com/google/devtools/build/lib/analysis:build_view",
        "//src/main/java/com/google/devtools/build/lib/cmdline",
        "//src/main/java/com/google/devtools/build/lib/pkgcache",
        "//src/main/java/com/google/devtools/build/lib/runtime/commands",
        "//src/main/java/com/google/devtools/build/lib/skyframe:sky_functions",
        "//src/main/java/com/google/devtools/build/lib/skyframe/serialization/analysis:dependencies_provider",
        "//src/main/java/com/google/devtools/build/lib/util:abrupt_exit_exception",
        "//src/main/java/com/google/devtools/build/lib/util/io:out-err",
        "//src/main/java/com/google/devtools/build/lib/vfs",
        "//src/main/java/com/google/devtools/build/skyframe:skyframe-objects",
        "//src/test/java/com/google/devtools/build/lib/buildtool/util",
        "//src/test/java/com/google/devtools/build/lib/testutil:TestUtils",
        "//third_party:guava",
        "//third_party:jsr305",
        "//third_party:junit4",
        "//third_party:truth",
        "//third_party/pprof:profile_java_proto",
        "@com_google_protobuf//:protobuf_java",
    ],
)

java_library(
    name = "SkycacheIntegrationTestHelpers",
    testonly = True,
    srcs = ["SkycacheIntegrationTestHelpers.java"],
    deps = [
        "//src/main/java/com/google/devtools/build/lib:runtime",
        "//src/main/java/com/google/devtools/build/lib/vfs",
        "//third_party:guava",
        "//third_party:junit4",
        "//third_party:truth",
    ],
)

java_test(
    name = "BazelSkycacheIntegrationTest",
    srcs = ["BazelSkycacheIntegrationTest.java"],
    tags = ["no_windows"],  # b/380393822 - disable until PathFragmentPrefixTrie handles windows correctly.
    deps = [
        ":SkycacheIntegrationTestBase",
        "//src/main/java/com/google/devtools/build/lib:runtime",
        "//src/main/java/com/google/devtools/build/lib/skyframe:sky_functions",
        "//src/main/java/com/google/devtools/build/lib/skyframe/serialization",
        "//src/main/java/com/google/devtools/build/lib/skyframe/serialization:serialization_module",
        "//src/main/java/com/google/devtools/build/lib/skyframe/serialization/analysis:long_version_getter_test_injection",
        "//src/main/java/com/google/devtools/build/lib/skyframe/serialization/analysis:remote_analysis_caching_services_supplier",
        "//src/main/java/com/google/devtools/build/lib/versioning:long_version_getter",
        "//third_party:guava",
        "//third_party:junit4",
        "//third_party:mockito",
        "//third_party:truth",
    ],
)

java_test(
    name = "ValueOrFutureMapTest",
    srcs = ["ValueOrFutureMapTest.java"],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/concurrent:settable_future_keyed_value",
        "//src/main/java/com/google/devtools/build/lib/skyframe/serialization/analysis:value_or_future_map",
        "//third_party:guava",
        "//third_party:junit4",
        "//third_party:truth",
    ],
)

java_test(
    name = "FileOpNodeMemoizingLookupTest",
    srcs = ["FileOpNodeMemoizingLookupTest.java"],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/actions:action_lookup_data",
        "//src/main/java/com/google/devtools/build/lib/actions:action_lookup_key",
        "//src/main/java/com/google/devtools/build/lib/skyframe:filesystem_keys",
        "//src/main/java/com/google/devtools/build/lib/skyframe/serialization/analysis:file_op_node_map",
        "//src/main/java/com/google/devtools/build/skyframe",
        "//src/main/java/com/google/devtools/build/skyframe:skyframe-objects",
        "//src/test/java/com/google/devtools/build/lib/buildtool/util",
        "//third_party:guava",
        "//third_party:junit4",
        "//third_party:truth",
    ],
)

java_test(
    name = "VersionedChangesTest",
    srcs = ["VersionedChangesTest.java"],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/skyframe/serialization/analysis:versioned_changes",
        "//third_party:guava",
        "//third_party:junit4",
        "//third_party:truth",
    ],
)

java_test(
    name = "FileSystemDependenciesTest",
    srcs = ["FileSystemDependenciesTest.java"],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/skyframe/serialization/analysis:file_dependency_deserializer",
        "//src/main/java/com/google/devtools/build/lib/skyframe/serialization/analysis:versioned_changes",
        "//third_party:guava",
        "//third_party:junit4",
        "//third_party:truth",
    ],
)

java_test(
    name = "FileOpMatchMemoizingLookupTest",
    srcs = ["FileOpMatchMemoizingLookupTest.java"],
    deps = [
        ":controllable_file_dependencies",
        "//src/main/java/com/google/devtools/build/lib/skyframe/serialization/analysis:file_dependency_deserializer",
        "//src/main/java/com/google/devtools/build/lib/skyframe/serialization/analysis:versioned_changes",
        "//src/main/java/com/google/devtools/build/lib/skyframe/serialization/analysis:versioned_changes_validator",
        "//third_party:guava",
        "//third_party:junit4",
        "//third_party:truth",
        "@maven//:com_google_testparameterinjector_test_parameter_injector",
    ],
)

java_test(
    name = "NestedMatchMemoizingLookupTest",
    srcs = ["NestedMatchMemoizingLookupTest.java"],
    deps = [
        ":controllable_file_dependencies",
        "//src/main/java/com/google/devtools/build/lib/skyframe/serialization/analysis:file_dependency_deserializer",
        "//src/main/java/com/google/devtools/build/lib/skyframe/serialization/analysis:versioned_changes",
        "//src/main/java/com/google/devtools/build/lib/skyframe/serialization/analysis:versioned_changes_validator",
        "//third_party:guava",
        "//third_party:junit4",
        "//third_party:truth",
        "@maven//:com_google_testparameterinjector_test_parameter_injector",
    ],
)

java_test(
    name = "VersionedChangesValidatorTest",
    srcs = ["VersionedChangesValidatorTest.java"],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/skyframe/serialization/analysis:file_dependency_deserializer",
        "//src/main/java/com/google/devtools/build/lib/skyframe/serialization/analysis:versioned_changes",
        "//src/main/java/com/google/devtools/build/lib/skyframe/serialization/analysis:versioned_changes_validator",
        "//third_party:guava",
        "//third_party:junit4",
        "//third_party:truth",
        "@maven//:com_google_testparameterinjector_test_parameter_injector",
    ],
)

java_library(
    name = "controllable_file_dependencies",
    srcs = ["ControllableFileDependencies.java"],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/skyframe/serialization/analysis:file_dependency_deserializer",
        "//src/main/java/com/google/devtools/build/lib/skyframe/serialization/analysis:versioned_changes",
        "//third_party:guava",
    ],
)
