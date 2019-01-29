package(
    default_testonly = True,
)

filegroup(
    name = "srcs",
    testonly = 0,
    srcs = glob(["**"]),
    visibility = ["//src:__subpackages__"],
)

test_suite(
    name = "PythonTests",
    tests = [
        ":BazelPythonConfigurationTest",
        ":PyBinaryConfiguredTargetTest",
        ":PyInfoTest",
        ":PyLibraryConfiguredTargetTest",
        ":PyProviderUtilsTest",
        ":PyStructUtilsTest",
        ":PyTestConfiguredTargetTest",
        ":PythonConfigurationTest",
        ":PythonSrcsVersionAspectTest",
        ":PythonStarlarkApiTest",
        ":PythonVersionSelectTest",
        ":PythonVersionTest",
    ],
)

java_library(
    name = "PythonTestUtils",
    srcs = ["PythonTestUtils.java"],
    deps = [
        "//src/main/java/com/google/devtools/build/lib:python-rules",
        "//third_party:junit4",
        "//third_party:truth",
    ],
)

java_test(
    name = "BazelPythonConfigurationTest",
    srcs = ["BazelPythonConfigurationTest.java"],
    deps = [
        "//src/main/java/com/google/devtools/build/lib:bazel-rules",
        "//src/main/java/com/google/devtools/build/lib:build-base",
        "//src/main/java/com/google/devtools/common/options",
        "//src/test/java/com/google/devtools/build/lib:analysis_testutil",
        "//src/test/java/com/google/devtools/build/lib:testutil",
        "//third_party:junit4",
        "//third_party:truth",
    ],
)

java_test(
    name = "PythonConfigurationTest",
    srcs = ["PythonConfigurationTest.java"],
    deps = [
        ":PythonTestUtils",
        "//src/main/java/com/google/devtools/build/lib:build-base",
        "//src/main/java/com/google/devtools/build/lib:python-rules",
        "//src/main/java/com/google/devtools/common/options",
        "//src/test/java/com/google/devtools/build/lib:analysis_testutil",
        "//src/test/java/com/google/devtools/build/lib:testutil",
        "//third_party:junit4",
        "//third_party:truth",
    ],
)

java_library(
    name = "PyBaseTestBase",
    srcs = ["PyBaseConfiguredTargetTestBase.java"],
    deps = [
        ":PythonTestUtils",
        "//src/main/java/com/google/devtools/build/lib:bazel-rules",
        "//src/main/java/com/google/devtools/build/lib:build-base",
        "//src/main/java/com/google/devtools/build/lib:python-rules",
        "//src/test/java/com/google/devtools/build/lib:analysis_testutil",
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
        "//src/main/java/com/google/devtools/build/lib:bazel-rules",
        "//src/main/java/com/google/devtools/build/lib:build-base",
        "//src/main/java/com/google/devtools/build/lib:python-rules",
        "//src/main/java/com/google/devtools/build/lib/actions",
        "//third_party:guava",
        "//third_party:junit4",
        "//third_party:truth",
    ],
)

java_test(
    name = "PyBinaryConfiguredTargetTest",
    srcs = ["PyBinaryConfiguredTargetTest.java"],
    deps = [
        ":PyExecutableTestBase",
        "//src/main/java/com/google/devtools/build/lib:build-base",
        "//src/test/java/com/google/devtools/build/lib:testutil",
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
        "//src/main/java/com/google/devtools/build/lib:build-base",
        "//src/main/java/com/google/devtools/build/lib:python-rules",
        "//src/test/java/com/google/devtools/build/lib:actions_testutil",
        "//src/test/java/com/google/devtools/build/lib:analysis_testutil",
        "//third_party:junit4",
        "//third_party:truth",
    ],
)

java_test(
    name = "PythonVersionTest",
    srcs = ["PythonVersionTest.java"],
    deps = [
        "//src/main/java/com/google/devtools/build/lib:python-rules",
        "//src/test/java/com/google/devtools/build/lib:testutil",
        "//third_party:junit4",
        "//third_party:truth",
    ],
)

java_test(
    name = "PythonVersionSelectTest",
    srcs = ["PythonVersionSelectTest.java"],
    deps = [
        ":PythonTestUtils",
        "//src/main/java/com/google/devtools/build/lib:build-base",
        "//src/main/java/com/google/devtools/build/lib/actions",
        "//src/main/java/com/google/devtools/build/lib/collect/nestedset",
        "//src/test/java/com/google/devtools/build/lib:analysis_testutil",
        "//src/test/java/com/google/devtools/build/lib:testutil",
        "//third_party:junit4",
        "//third_party:truth",
    ],
)

java_test(
    name = "PythonSrcsVersionAspectTest",
    srcs = ["PythonSrcsVersionAspectTest.java"],
    deps = [
        "//src/main/java/com/google/devtools/build/lib:build-base",
        "//src/main/java/com/google/devtools/build/lib/actions",
        "//src/test/java/com/google/devtools/build/lib:analysis_testutil",
        "//src/test/java/com/google/devtools/build/lib:testutil",
        "//third_party:junit4",
        "//third_party:truth",
    ],
)

java_test(
    name = "PyProviderUtilsTest",
    srcs = ["PyProviderUtilsTest.java"],
    deps = [
        "//src/main/java/com/google/devtools/build/lib:build-base",
        "//src/main/java/com/google/devtools/build/lib:python-rules",
        "//src/main/java/com/google/devtools/build/lib:syntax",
        "//src/test/java/com/google/devtools/build/lib:analysis_testutil",
        "//src/test/java/com/google/devtools/build/lib:testutil",
        "//third_party:guava",
        "//third_party:junit4",
        "//third_party:truth",
    ],
)

java_test(
    name = "PyStructUtilsTest",
    srcs = ["PyStructUtilsTest.java"],
    deps = [
        "//src/main/java/com/google/devtools/build/lib:packages-internal",
        "//src/main/java/com/google/devtools/build/lib:python-rules",
        "//src/main/java/com/google/devtools/build/lib:syntax",
        "//src/main/java/com/google/devtools/build/lib/actions",
        "//src/main/java/com/google/devtools/build/lib/collect/nestedset",
        "//src/main/java/com/google/devtools/build/lib/vfs",
        "//src/main/java/com/google/devtools/build/lib/vfs:pathfragment",
        "//src/test/java/com/google/devtools/build/lib:testutil",
        "//third_party:guava",
        "//third_party:junit4",
        "//third_party:truth",
    ],
)

java_test(
    name = "PyInfoTest",
    srcs = ["PyInfoTest.java"],
    deps = [
        "//src/main/java/com/google/devtools/build/lib:events",
        "//src/main/java/com/google/devtools/build/lib:python-rules",
        "//src/main/java/com/google/devtools/build/lib/actions",
        "//src/main/java/com/google/devtools/build/lib/collect/nestedset",
        "//src/main/java/com/google/devtools/build/lib/vfs:pathfragment",
        "//src/test/java/com/google/devtools/build/lib:analysis_testutil",
        "//src/test/java/com/google/devtools/build/lib:testutil",
        "//src/test/java/com/google/devtools/build/lib/skylark:testutil",
        "//third_party:junit4",
        "//third_party:truth",
    ],
)

java_test(
    name = "PythonStarlarkApiTest",
    srcs = ["PythonStarlarkApiTest.java"],
    deps = [
        "//src/main/java/com/google/devtools/build/lib:packages-internal",
        "//src/main/java/com/google/devtools/build/lib:python-rules",
        "//src/test/java/com/google/devtools/build/lib:analysis_testutil",
        "//third_party:junit4",
        "//third_party:truth",
    ],
)
