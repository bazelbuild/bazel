load("@rules_java//java:defs.bzl", "java_library", "java_test")

package(
    default_testonly = True,
    default_visibility = ["//src:__subpackages__"],
)

filegroup(
    name = "srcs",
    testonly = 0,
    srcs = glob(["*"]),
    visibility = ["//src:__subpackages__"],
)

test_suite(
    name = "PythonTests",
    tests = [
        ":PyBinaryConfiguredTargetTest",
        ":PyInfoTest",
        ":PyLibraryConfiguredTargetTest",
        ":PyProviderUtilsTest",
        ":PyRuntimeConfiguredTargetTest",
        ":PyRuntimeInfoTest",
        ":PyStarlarkTransitionsTest",
        ":PyStructUtilsTest",
        ":PyTestConfiguredTargetTest",
        ":PythonConfigurationTest",
        ":PythonSrcsVersionAspectTest",
        ":PythonStarlarkApiTest",
        ":PythonToolchainTest",
        ":PythonVersionSelectTest",
        ":PythonVersionTest",
    ],
)

java_library(
    name = "PythonTestUtils",
    srcs = ["PythonTestUtils.java"],
)

java_test(
    name = "PythonConfigurationTest",
    srcs = ["PythonConfigurationTest.java"],
    deps = [
        ":PythonTestUtils",
        "//src/main/java/com/google/devtools/build/lib/analysis:config/build_configuration",
        "//src/main/java/com/google/devtools/build/lib/rules/python",
        "//src/main/java/com/google/devtools/common/options",
        "//src/test/java/com/google/devtools/build/lib/analysis/util",
        "//third_party:junit4",
        "//third_party:truth",
    ],
)

java_test(
    name = "PyRuntimeConfiguredTargetTest",
    srcs = ["PyRuntimeConfiguredTargetTest.java"],
    deps = [
        ":PythonTestUtils",
        "//src/main/java/com/google/devtools/build/lib/rules/python",
        "//src/test/java/com/google/devtools/build/lib/actions/util",
        "//src/test/java/com/google/devtools/build/lib/analysis/util",
        "//third_party:junit4",
        "//third_party:truth",
    ],
)

java_library(
    name = "PyBaseTestBase",
    srcs = ["PyBaseConfiguredTargetTestBase.java"],
    deps = [
        ":PythonTestUtils",
        "//src/main/java/com/google/devtools/build/lib/analysis:configured_target",
        "//src/main/java/com/google/devtools/build/lib/rules/python",
        "//src/test/java/com/google/devtools/build/lib/analysis/util",
        "//src/test/java/com/google/devtools/build/lib/packages:testutil",
        "//third_party:junit4",
        "//third_party:truth",
    ],
)

java_library(
    name = "PyExecutableTestBase",
    srcs = ["PyExecutableConfiguredTargetTestBase.java"],
    deps = [
        ":PyBaseTestBase",
        ":PythonTestUtils",
        "//src/main/java/com/google/devtools/build/lib/actions",
        "//src/main/java/com/google/devtools/build/lib/analysis:configured_target",
        "//src/main/java/com/google/devtools/build/lib/rules/python",
        "//third_party:guava",
        "//third_party:junit4",
        "//third_party:truth",
    ],
)

java_test(
    name = "PyBinaryConfiguredTargetTest",
    srcs = ["PyBinaryConfiguredTargetTest.java"],
    tags = ["no_windows"],
    deps = [
        ":PyExecutableTestBase",
        "//src/main/java/com/google/devtools/build/lib/analysis:analysis_cluster",
        "//src/main/java/com/google/devtools/build/lib/analysis:configured_target",
        "//src/test/java/com/google/devtools/build/lib/testutil:TestConstants",
        "//third_party:junit4",
        "//third_party:truth",
    ],
)

java_test(
    name = "PyTestConfiguredTargetTest",
    srcs = ["PyTestConfiguredTargetTest.java"],
    deps = [
        ":PyExecutableTestBase",
        "//third_party:junit4",
    ],
)

java_test(
    name = "PyLibraryConfiguredTargetTest",
    srcs = ["PyLibraryConfiguredTargetTest.java"],
    deps = [
        ":PyBaseTestBase",
        ":PythonTestUtils",
        "//src/main/java/com/google/devtools/build/lib/analysis:analysis_cluster",
        "//src/main/java/com/google/devtools/build/lib/analysis:configured_target",
        "//src/main/java/com/google/devtools/build/lib/rules/python",
        "//src/test/java/com/google/devtools/build/lib/actions/util",
        "//third_party:junit4",
        "//third_party:truth",
    ],
)

java_test(
    name = "PythonVersionTest",
    srcs = ["PythonVersionTest.java"],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/rules/python",
        "//third_party:junit4",
        "//third_party:truth",
    ],
)

java_test(
    name = "PythonVersionSelectTest",
    srcs = ["PythonVersionSelectTest.java"],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/actions",
        "//src/main/java/com/google/devtools/build/lib/analysis:file_provider",
        "//src/main/java/com/google/devtools/build/lib/collect/nestedset",
        "//src/test/java/com/google/devtools/build/lib/analysis/util",
        "//src/test/java/com/google/devtools/build/lib/testutil:TestConstants",
        "//third_party:junit4",
        "//third_party:truth",
    ],
)

java_test(
    name = "PythonSrcsVersionAspectTest",
    srcs = ["PythonSrcsVersionAspectTest.java"],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/actions",
        "//src/main/java/com/google/devtools/build/lib/analysis:analysis_cluster",
        "//src/main/java/com/google/devtools/build/lib/analysis:configured_target",
        "//src/test/java/com/google/devtools/build/lib/analysis/util",
        "//src/test/java/com/google/devtools/build/lib/testutil:TestConstants",
        "//third_party:junit4",
        "//third_party:truth",
    ],
)

java_test(
    name = "PythonToolchainTest",
    srcs = ["PythonToolchainTest.java"],
    deps = [
        "//src/main/java/com/google/devtools/build/lib:syntax",
        "//src/main/java/com/google/devtools/build/lib/analysis/platform",
        "//src/test/java/com/google/devtools/build/lib/analysis/util",
        "//src/test/java/com/google/devtools/build/lib/testutil:TestConstants",
        "//third_party:junit4",
        "//third_party:truth",
    ],
)

java_test(
    name = "PyProviderUtilsTest",
    srcs = ["PyProviderUtilsTest.java"],
    deps = [
        "//src/main/java/com/google/devtools/build/lib:syntax",
        "//src/main/java/com/google/devtools/build/lib/analysis:configured_target",
        "//src/main/java/com/google/devtools/build/lib/rules/python",
        "//src/test/java/com/google/devtools/build/lib/analysis/util",
        "//third_party:junit4",
        "//third_party:truth",
    ],
)

java_test(
    name = "PyStructUtilsTest",
    srcs = ["PyStructUtilsTest.java"],
    deps = [
        "//src/main/java/com/google/devtools/build/lib:syntax",
        "//src/main/java/com/google/devtools/build/lib/actions",
        "//src/main/java/com/google/devtools/build/lib/collect/nestedset",
        "//src/main/java/com/google/devtools/build/lib/packages",
        "//src/main/java/com/google/devtools/build/lib/rules/python",
        "//src/main/java/com/google/devtools/build/lib/vfs",
        "//src/test/java/com/google/devtools/build/lib/actions/util",
        "//src/test/java/com/google/devtools/build/lib/testutil",
        "//third_party:guava",
        "//third_party:junit4",
        "//third_party:truth",
    ],
)

java_test(
    name = "PyInfoTest",
    srcs = ["PyInfoTest.java"],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/actions",
        "//src/main/java/com/google/devtools/build/lib/collect/nestedset",
        "//src/main/java/com/google/devtools/build/lib/rules/python",
        "//src/main/java/com/google/devtools/build/lib/syntax:frontend",
        "//src/test/java/com/google/devtools/build/lib/analysis/util",
        "//src/test/java/com/google/devtools/build/lib/skylark/util",
        "//src/test/java/com/google/devtools/build/lib/syntax/util",
        "//third_party:junit4",
        "//third_party:truth",
    ],
)

java_test(
    name = "PyRuntimeInfoTest",
    srcs = ["PyRuntimeInfoTest.java"],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/actions",
        "//src/main/java/com/google/devtools/build/lib/collect/nestedset",
        "//src/main/java/com/google/devtools/build/lib/rules/python",
        "//src/main/java/com/google/devtools/build/lib/syntax:frontend",
        "//src/main/java/com/google/devtools/build/lib/vfs:pathfragment",
        "//src/test/java/com/google/devtools/build/lib/analysis/util",
        "//src/test/java/com/google/devtools/build/lib/skylark/util",
        "//src/test/java/com/google/devtools/build/lib/syntax/util",
        "//third_party:junit4",
        "//third_party:truth",
    ],
)

java_test(
    name = "PyStarlarkTransitionsTest",
    srcs = ["PyStarlarkTransitionsTest.java"],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/analysis:configured_target",
        "//src/main/java/com/google/devtools/build/lib/cmdline",
        "//src/main/java/com/google/devtools/build/lib/packages",
        "//src/main/java/com/google/devtools/build/lib/rules/python",
        "//src/test/java/com/google/devtools/build/lib/analysis/util",
        "//third_party:guava",
        "//third_party:junit4",
        "//third_party:truth",
    ],
)

java_test(
    name = "PythonStarlarkApiTest",
    srcs = ["PythonStarlarkApiTest.java"],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/actions",
        "//src/main/java/com/google/devtools/build/lib/analysis:configured_target",
        "//src/main/java/com/google/devtools/build/lib/packages",
        "//src/main/java/com/google/devtools/build/lib/rules/python",
        "//src/test/java/com/google/devtools/build/lib/analysis/util",
        "//src/test/java/com/google/devtools/build/lib/testutil:TestConstants",
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
