package(default_visibility = [
    "//src/java_tools/junitrunner:junitrunner_package_group",
])

filegroup(
    name = "common_runner_java_files",
    srcs = glob(
        ["*.java"],
        exclude =
            [
                "BazelTestRunner.java",
                "ExperimentalTestRunner.java",
            ],
    ),
)

java_library(
    name = "test_runner",
    srcs = [
        "BazelTestRunner.java",
        ":common_runner_java_files",
    ],
    data = ["//tools:test_sharding_compliant"],
    deps = [
        "//src/java_tools/junitrunner/java/com/google/testing/junit/runner/internal",
        "//src/java_tools/junitrunner/java/com/google/testing/junit/runner/junit4",
        "//src/java_tools/junitrunner/java/com/google/testing/junit/runner/model",
        "//src/java_tools/junitrunner/java/com/google/testing/junit/runner/sharding",
        "//src/java_tools/junitrunner/java/com/google/testing/junit/runner/util",
        "//third_party:junit4",
    ],
)

java_library(
    name = "experimental_test_runner",
    srcs = [
        "ExperimentalTestRunner.java",
        ":common_runner_java_files",
    ],
    data = ["//tools:test_sharding_compliant"],
    deps = [
        "//src/java_tools/junitrunner/java/com/google/testing/junit/runner/internal",
        "//src/java_tools/junitrunner/java/com/google/testing/junit/runner/junit4",
        "//src/java_tools/junitrunner/java/com/google/testing/junit/runner/model",
        "//src/java_tools/junitrunner/java/com/google/testing/junit/runner/sharding",
        "//src/java_tools/junitrunner/java/com/google/testing/junit/runner/util",
        "//src/main/protobuf:worker_protocol_java_proto",
        "//third_party:junit4",
    ],
)

java_binary(
    name = "Runner",
    main_class = "com.google.testing.junit.runner.BazelTestRunner",
    visibility = [
        "//src:__subpackages__",
        "//src/java_tools/junitrunner:junitrunner_package_group",
    ],
    runtime_deps = [":test_runner"],
)

java_binary(
    name = "ExperimentalRunner",
    main_class = "com.google.testing.junit.runner.ExperimentalTestRunner",
    visibility = [
        "//src:__pkg__",
        "//src/java_tools/junitrunner:junitrunner_package_group",
    ],
    runtime_deps = [":experimental_test_runner"],
)

filegroup(
    name = "srcs",
    srcs = glob(["**/*.java"]) + [
        "BUILD",
        "//src/java_tools/junitrunner/java/com/google/testing/junit/runner/internal:srcs",
        "//src/java_tools/junitrunner/java/com/google/testing/junit/runner/junit4:srcs",
        "//src/java_tools/junitrunner/java/com/google/testing/junit/runner/model:srcs",
        "//src/java_tools/junitrunner/java/com/google/testing/junit/runner/sharding:srcs",
        "//src/java_tools/junitrunner/java/com/google/testing/junit/runner/util:srcs",
    ],
)
