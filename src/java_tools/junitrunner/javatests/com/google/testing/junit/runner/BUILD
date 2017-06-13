DEFAULT_VISIBILITY = [
    "//src/java_tools/junitrunner:__subpackages__",
]

# Libraries
# =========================================================
java_library(
    name = "tests",
    testonly = 1,
    srcs = glob(
        ["**/*.java"],
        exclude = ["testbed/*.java"],
    ),
    deps = [
        "//src/java_tools/junitrunner/java/com/google/testing/junit/runner:test_runner",
        "//src/java_tools/junitrunner/java/com/google/testing/junit/runner/internal",
        "//src/java_tools/junitrunner/java/com/google/testing/junit/runner/internal:junit4",
        "//src/java_tools/junitrunner/java/com/google/testing/junit/runner/junit4",
        "//src/java_tools/junitrunner/java/com/google/testing/junit/runner/model",
        "//src/java_tools/junitrunner/java/com/google/testing/junit/runner/sharding",
        "//src/java_tools/junitrunner/java/com/google/testing/junit/runner/sharding/api",
        "//src/java_tools/junitrunner/java/com/google/testing/junit/runner/sharding/testing",
        "//src/java_tools/junitrunner/java/com/google/testing/junit/runner/util",
        "//src/test/java/com/google/devtools/build/lib:testutil",
        "//third_party:dagger",
        "//third_party:guava",
        "//third_party:jsr305",
        "//third_party:jsr330_inject",
        "//third_party:junit4",
        "//third_party:mockito",
        "//third_party:truth",
    ],
)

java_binary(
    name = "TestbedBinary",
    testonly = 1,
    main_class = "com.google.testing.junit.runner.BazelTestRunner",
    runtime_deps = [
        "//src/java_tools/junitrunner/java/com/google/testing/junit/runner:test_runner",
        "//src/java_tools/junitrunner/javatests/com/google/testing/junit/runner/testbed",
    ],
)

# Tests
# =========================================================

java_test(
    name = "AllTests",
    args = glob(["**/*Test.java"]),
    main_class = "com.google.testing.junit.runner.TestRunner",
    use_testrunner = 0,
    runtime_deps = [":tests"],
)

# BazelTestRunner integration tests
sh_library(
    name = "testenv",
    srcs = ["testenv.sh"],
    deps = ["//src/test/shell:bashunit"],
)

sh_test(
    name = "junit4_testbridge_integration_test",
    size = "medium",
    srcs = ["junit4_testbridge_integration_tests.sh"],
    args = [
        "$(location :TestbedBinary)",
        "bazel.test_suite",
    ],
    data = [":TestbedBinary"],
    shard_count = 0,
    deps = [":testenv"],
)

sh_test(
    name = "utf8_test_log_test",
    size = "medium",
    srcs = ["utf8_test_log_test.sh"],
    args = [
        "$(location :TestbedBinary)",
        "bazel.test_suite",
    ],
    data = [":TestbedBinary"],
    shard_count = 0,
    deps = [":testenv"],
)

sh_test(
    name = "antxmlresultwriter_integration_test",
    size = "medium",
    srcs = ["antxmlresultwriter_integration_test.sh"],
    args = [
        "$(location :TestbedBinary)",
        "$(location //src/java_tools/junitrunner/javatests/com/google/testing/junit/runner/testbed:XmlOutputExercises.ant.xml)",
        "bazel.test_suite",
    ],
    data = [
        ":TestbedBinary",
        "//src/java_tools/junitrunner/javatests/com/google/testing/junit/runner/testbed:XmlOutputExercises.ant.xml",
    ],
    shard_count = 0,
    deps = [":testenv"],
)

sh_test(
    name = "deploy_jar_integration_test",
    size = "medium",
    srcs = ["deploy_jar_integration_tests.sh"],
    args = [
        "$(JAVABASE)",
        "$(location :TestbedBinary_deploy.jar)",
    ],
    data = [
        ":TestbedBinary_deploy.jar",
        "//tools/defaults:jdk",
    ],
    shard_count = 0,
    deps = [":testenv"],
)

filegroup(
    name = "srcs",
    srcs = glob(["**"]),
    visibility = ["//src:__pkg__"],
)

exports_files([
    "stack_trace_integration_tests.sh",
    "junit4_testbridge_integration_tests.sh",
    "utf8_test_log_test.sh",
    "deploy_jar_integration_tests.sh",
])
