load("@rules_java//java:defs.bzl", "java_library", "java_test")

package(
    default_applicable_licenses = ["//:license"],
    default_testonly = 1,
    default_visibility = ["//src:__subpackages__"],
)

filegroup(
    name = "srcs",
    testonly = 0,
    srcs = glob(["*"]) + [
        "//src/test/java/com/google/devtools/build/lib/bazel/repository/cache:srcs",
        "//src/test/java/com/google/devtools/build/lib/bazel/repository/decompressor:srcs",
        "//src/test/java/com/google/devtools/build/lib/bazel/repository/downloader:srcs",
        "//src/test/java/com/google/devtools/build/lib/bazel/repository/starlark:srcs",
    ],
    visibility = ["//src:__subpackages__"],
)

java_library(
    name = "RepositoryTests_lib",
    srcs = glob(["*.java"]),
    deps = [
        "//src/main/java/com/google/devtools/build/lib/actions:file_metadata",
        "//src/main/java/com/google/devtools/build/lib/analysis:analysis_cluster",
        "//src/main/java/com/google/devtools/build/lib/analysis:blaze_directories",
        "//src/main/java/com/google/devtools/build/lib/analysis:server_directories",
        "//src/main/java/com/google/devtools/build/lib/bazel/bzlmod:extension_eval_impl",
        "//src/main/java/com/google/devtools/build/lib/bazel/bzlmod:resolution_impl",
        "//src/main/java/com/google/devtools/build/lib/bazel/repository:exception",
        "//src/main/java/com/google/devtools/build/lib/bazel/repository:repo_definition",
        "//src/main/java/com/google/devtools/build/lib/bazel/repository:repo_definition_function",
        "//src/main/java/com/google/devtools/build/lib/bazel/repository:repo_definition_value",
        "//src/main/java/com/google/devtools/build/lib/bazel/repository:repository_fetch_function",
        "//src/main/java/com/google/devtools/build/lib/bazel/repository:repository_options",
        "//src/main/java/com/google/devtools/build/lib/bazel/repository:repository_resolved_event",
        "//src/main/java/com/google/devtools/build/lib/bazel/repository/cache",
        "//src/main/java/com/google/devtools/build/lib/clock",
        "//src/main/java/com/google/devtools/build/lib/cmdline",
        "//src/main/java/com/google/devtools/build/lib/events",
        "//src/main/java/com/google/devtools/build/lib/packages:autoload_symbols",
        "//src/main/java/com/google/devtools/build/lib/pkgcache",
        "//src/main/java/com/google/devtools/build/lib/rules:repository/repo_recorded_input",
        "//src/main/java/com/google/devtools/build/lib/rules:repository/repository_directory_dirtiness_checker",
        "//src/main/java/com/google/devtools/build/lib/rules:repository/repository_directory_value",
        "//src/main/java/com/google/devtools/build/lib/skyframe:bzl_compile",
        "//src/main/java/com/google/devtools/build/lib/skyframe:client_environment_function",
        "//src/main/java/com/google/devtools/build/lib/skyframe:containing_package_lookup_function",
        "//src/main/java/com/google/devtools/build/lib/skyframe:file_function",
        "//src/main/java/com/google/devtools/build/lib/skyframe:ignored_subdirectories_function",
        "//src/main/java/com/google/devtools/build/lib/skyframe:package_lookup_function",
        "//src/main/java/com/google/devtools/build/lib/skyframe:precomputed_function",
        "//src/main/java/com/google/devtools/build/lib/skyframe:precomputed_value",
        "//src/main/java/com/google/devtools/build/lib/skyframe:repository_mapping_function",
        "//src/main/java/com/google/devtools/build/lib/skyframe:sky_functions",
        "//src/main/java/com/google/devtools/build/lib/skyframe:skyframe_cluster",
        "//src/main/java/com/google/devtools/build/lib/util",
        "//src/main/java/com/google/devtools/build/lib/util/io",
        "//src/main/java/com/google/devtools/build/lib/vfs",
        "//src/main/java/com/google/devtools/build/lib/vfs:pathfragment",
        "//src/main/java/com/google/devtools/build/skyframe",
        "//src/main/java/com/google/devtools/build/skyframe:skyframe-objects",
        "//src/main/java/com/google/devtools/common/options",
        "//src/main/java/net/starlark/java/eval",
        "//src/test/java/com/google/devtools/build/lib/analysis/util",
        "//src/test/java/com/google/devtools/build/lib/bazel/bzlmod:util",
        "//src/test/java/com/google/devtools/build/lib/skyframe/util:SkyframeExecutorTestUtils",
        "//src/test/java/com/google/devtools/build/lib/testutil",
        "//src/test/java/com/google/devtools/build/lib/testutil:TestConstants",
        "//third_party:caffeine",
        "//third_party:guava",
        "//third_party:junit4",
        "//third_party:mockito",
        "//third_party:truth",
    ],
)

java_test(
    name = "RepositoryTests",
    jvm_flags = [
        "-Djava.lang.Thread.allowVirtualThreads=true",
    ],
    tags = [
        "rules",
    ],
    test_class = "com.google.devtools.build.lib.AllTests",
    runtime_deps = [
        ":RepositoryTests_lib",
        "//src/test/java/com/google/devtools/build/lib:test_runner",
    ],
)
