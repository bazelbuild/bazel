package(
    default_testonly = 1,
    default_visibility = [
        "//src/test/java/com/google/devtools/build/lib:__subpackages__",
    ],
)

licenses(["notice"])  # Apache 2.0

filegroup(
    name = "srcs",
    testonly = 0,
    srcs = glob(
        ["**"],
        exclude = [
            "*~",
        ],
    ),
    visibility = ["//src:__pkg__"],
)

java_library(
    name = "ShellTests_lib",
    srcs = glob(
        ["*.java"],
        exclude = [
            "CommandUsingLinuxSandboxTest.java",
            "InterruptibleTest.java",
        ],
    ),
    javacopts = ["-Xlint:-deprecation"],
    deps = [
        "//src/main/java/com/google/devtools/build/lib:bazel-main",
        "//src/main/java/com/google/devtools/build/lib:os_util",
        "//src/main/java/com/google/devtools/build/lib:runtime",
        "//src/main/java/com/google/devtools/build/lib:unix",
        "//src/main/java/com/google/devtools/build/lib:util",
        "//src/main/java/com/google/devtools/build/lib/collect",
        "//src/main/java/com/google/devtools/build/lib/shell",
        "//src/main/java/com/google/devtools/build/lib/vfs",
        "//src/main/protobuf:execution_statistics_java_proto",
        "//src/test/java/com/google/devtools/build/lib:foundations_testutil",
        "//src/test/java/com/google/devtools/build/lib:test_runner",
        "//src/test/java/com/google/devtools/build/lib:testutil",
        "//third_party:guava",
        "//third_party:guava-testlib",
        "//third_party:jsr305",
        "//third_party:junit4",
        "//third_party:mockito",
        "//third_party:truth",
        "//third_party:truth8",
    ],
)

java_test(
    name = "ShellTests",
    size = "medium",
    data = [
        ":killmyself",
        "//src/main/tools:process-wrapper",
        "//src/test/shell/integration:spend_cpu_time",
    ],
    flaky = 1,
    shard_count = 4,
    tags = [
        "no_windows",
        "shell",
    ],
    test_class = "com.google.devtools.build.lib.AllTests",
    runtime_deps = [
        ":ShellTests_lib",
        "//src/test/java/com/google/devtools/build/lib:test_runner",
    ],
)

java_test(
    name = "InterruptibleTest",
    size = "small",
    srcs = ["InterruptibleTest.java"],
    tags = [
        "no_windows",
        "shell",
    ],
    deps = [
        "//src/main/java/com/google/devtools/build/lib:os_util",
        "//src/main/java/com/google/devtools/build/lib/shell",
        "//third_party:junit4",
        "//third_party:truth",
        "//third_party:truth8",
    ],
)

java_test(
    name = "CommandUsingLinuxSandboxTest",
    size = "small",
    srcs = [
        "CommandUsingLinuxSandboxTest.java",
        "ExecutionStatisticsTestUtil.java",
    ],
    data = [
        "//src/main/tools:linux-sandbox",
        "//src/test/shell/integration:spend_cpu_time",
    ],
    tags = [
        "no_windows",
        "shell",
    ],
    deps = [
        "//src/main/java/com/google/devtools/build/lib:bazel-main",
        "//src/main/java/com/google/devtools/build/lib:os_util",
        "//src/main/java/com/google/devtools/build/lib:runtime",
        "//src/main/java/com/google/devtools/build/lib:unix",
        "//src/main/java/com/google/devtools/build/lib:util",
        "//src/main/java/com/google/devtools/build/lib/collect",
        "//src/main/java/com/google/devtools/build/lib/sandbox",
        "//src/main/java/com/google/devtools/build/lib/shell",
        "//src/main/java/com/google/devtools/build/lib/vfs",
        "//src/main/protobuf:execution_statistics_java_proto",
        "//src/test/java/com/google/devtools/build/lib:foundations_testutil",
        "//src/test/java/com/google/devtools/build/lib:test_runner",
        "//src/test/java/com/google/devtools/build/lib:testutil",
        "//third_party:guava",
        "//third_party:guava-testlib",
        "//third_party:jsr305",
        "//third_party:junit4",
        "//third_party:mockito",
        "//third_party:truth",
        "//third_party:truth8",
    ],
)

cc_binary(
    name = "killmyself",
    srcs = ["killmyself.cc"],
)

test_suite(
    name = "windows_tests",
    tags = [
        "-no_windows",
        "-slow",
    ],
    visibility = ["//visibility:private"],
)

test_suite(
    name = "all_windows_tests",
    tests = [
        ":windows_tests",
    ],
    visibility = ["//src/test/java/com/google/devtools/build/lib:__pkg__"],
)
