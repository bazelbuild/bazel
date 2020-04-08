load("@rules_java//java:defs.bzl", "java_library", "java_test")

licenses(["notice"])

package(
    default_testonly = 1,
    default_visibility = ["//src:__subpackages__"],
)

filegroup(
    name = "srcs",
    testonly = 0,
    srcs = glob(["**"]) + [
        "//src/test/java/com/google/devtools/build/lib/packages/util/mock:srcs",
    ],
    visibility = ["//src:__subpackages__"],
)

java_library(
    name = "PackageTestsUtil",
    srcs = ["WorkspaceFactoryTestHelper.java"],
    deps = [
        "//src/main/java/com/google/devtools/build/lib:build",
        "//src/main/java/com/google/devtools/build/lib:syntax",
        "//src/main/java/com/google/devtools/build/lib/packages",
        "//src/main/java/com/google/devtools/build/lib/packages:starlark_semantics_options",
        "//src/test/java/com/google/devtools/build/lib/testutil",
        "//third_party:guava",
        "//third_party:junit4",
        "//third_party:truth",
    ],
)

java_test(
    name = "PackagesTests",
    size = "small",
    srcs = glob(
        include = ["*.java"],
        exclude = [
            "BazelDocumentationTest.java",
            "ExternalPackageTest.java",
            "WorkspaceFactoryTestHelper.java",
        ],
    ),
    test_class = "com.google.devtools.build.lib.AllTests",
    runtime_deps = [
        "//src/test/java/com/google/devtools/build/lib:test_runner",
    ],
    deps = [
        ":PackageTestsUtil",
        ":testutil",
        "//src/main/java/com/google/devtools/build/lib:build",
        "//src/main/java/com/google/devtools/build/lib:syntax",
        "//src/main/java/com/google/devtools/build/lib/cmdline",
        "//src/main/java/com/google/devtools/build/lib/collect/nestedset",
        "//src/main/java/com/google/devtools/build/lib/packages",
        "//src/main/java/com/google/devtools/build/lib/packages:build_type",
        "//src/main/java/com/google/devtools/build/lib/packages:starlark_semantics_options",
        "//src/main/java/com/google/devtools/build/lib/packages:type",
        "//src/main/java/com/google/devtools/build/lib/pkgcache",
        "//src/main/java/com/google/devtools/build/lib/skyframe/serialization",
        "//src/main/java/com/google/devtools/build/lib/skyframe/serialization/testutils",
        "//src/main/java/com/google/devtools/build/lib/skylarkinterface",
        "//src/main/java/com/google/devtools/build/lib/util:filetype",
        "//src/main/java/com/google/devtools/build/lib/vfs:pathfragment",
        "//src/main/java/com/google/devtools/build/skyframe",
        "//src/main/java/com/google/devtools/build/skyframe:skyframe-objects",
        "//src/test/java/com/google/devtools/build/lib/analysis/util",
        "//src/test/java/com/google/devtools/build/lib/events:testutil",
        "//src/test/java/com/google/devtools/build/lib/testutil",
        "//src/test/java/com/google/devtools/build/lib/testutil:JunitUtils",
        "//src/test/java/com/google/devtools/build/lib/testutil:TestUtils",
        "//third_party:guava",
        "//third_party:guava-testlib",
        "//third_party:jsr305",
        "//third_party:junit4",
        "//third_party:mockito",
        "//third_party:truth",
    ],
)

java_test(
    name = "BazelDocumentationTests",
    size = "medium",
    srcs = ["BazelDocumentationTest.java"],
    data = [
        "//site:docs/user-manual.html",
    ],
    test_class = "com.google.devtools.build.lib.packages.BazelDocumentationTest",
    deps = [
        ":testutil",
        "//src/main/java/com/google/devtools/build/lib/bazel:main",
        "//src/main/java/com/google/devtools/build/lib/bazel/rules",
        "//third_party:guava",
        "//third_party:junit4",
        "@bazel_tools//tools/java/runfiles",
    ],
)

java_library(
    name = "testutil",
    srcs = glob(["util/*.java"]),
    resources = [
        "//src/test/java/com/google/devtools/build/lib/packages/util/mock:android_cc_toolchain_config.bzl",
        "//src/test/java/com/google/devtools/build/lib/packages/util/mock:osx_cc_toolchain_config.bzl",
        "//tools/build_defs/cc:action_names.bzl",
        "//tools/python:srcs",  # For BazelMockPythonSupport
        "@rules_cc//cc:srcs",
    ],
    deps = [
        "//src/main/java/com/google/devtools/build/docgen:docgen_javalib",
        "//src/main/java/com/google/devtools/build/lib:build",
        "//src/main/java/com/google/devtools/build/lib:runtime",
        "//src/main/java/com/google/devtools/build/lib:syntax",
        "//src/main/java/com/google/devtools/build/lib/clock",
        "//src/main/java/com/google/devtools/build/lib/cmdline",
        "//src/main/java/com/google/devtools/build/lib/packages",
        "//src/main/java/com/google/devtools/build/lib/packages:starlark_semantics_options",
        "//src/main/java/com/google/devtools/build/lib/pkgcache",
        "//src/main/java/com/google/devtools/build/lib/rules/cpp",
        "//src/main/java/com/google/devtools/build/lib/rules/proto",
        "//src/main/java/com/google/devtools/build/lib/skyframe/packages:PackageFactoryBuilderWithSkyframeForTesting",
        "//src/main/java/com/google/devtools/build/lib/util/io",
        "//src/main/java/com/google/devtools/build/lib/vfs:pathfragment",
        "//src/main/java/com/google/devtools/build/lib/vfs/inmemoryfs",
        "//src/test/java/com/google/devtools/build/lib/analysis/util:test-build-options",
        "//src/test/java/com/google/devtools/build/lib/events:testutil",
        "//src/test/java/com/google/devtools/build/lib/testutil",
        "//src/test/java/com/google/devtools/build/lib/testutil:SkyframeExecutorTestHelper",
        "//src/test/java/com/google/devtools/build/lib/testutil:TestConstants",
        "//src/test/java/com/google/devtools/build/lib/testutil:TestPackageFactoryBuilderFactory",
        "//src/test/java/com/google/devtools/build/lib/testutil:TestUtils",
        "//src/test/java/com/google/devtools/build/lib/vfs/util",
        "//third_party:guava",
        "//third_party:jsr305",
        "//third_party:junit4",
        "//third_party:truth",
    ],
)
