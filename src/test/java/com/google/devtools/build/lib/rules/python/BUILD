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
        ":PyLibraryConfiguredTargetTest",
        ":PyTestConfiguredTargetTest",
        ":PythonConfigurationTest",
        ":PythonVersionTest",
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
        "//src/main/java/com/google/devtools/build/lib:bazel-rules",
        "//src/main/java/com/google/devtools/build/lib:build-base",
        "//src/main/java/com/google/devtools/build/lib:python-rules",
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
        "//src/main/java/com/google/devtools/build/lib:build-base",
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
