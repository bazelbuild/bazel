package(default_testonly = 1)

filegroup(
    name = "srcs",
    testonly = 0,
    srcs = glob(["**"]),
    visibility = ["//src/test/java/com/google/devtools/build/lib/blackbox/tests:__pkg__"],
)

common_deps = [
    "//src/test/java/com/google/devtools/build/lib/blackbox/tests:common_deps",
]

java_test(
    name = "RepoWithRuleWritingTextGeneratorTest",
    timeout = "moderate",
    srcs = [
        "RepoWithRuleWritingTextGenerator.java",
        "RepoWithRuleWritingTextGeneratorTest.java",
    ],
    tags = ["black_box_test"],
    deps = common_deps,
)

java_test(
    name = "WorkspaceBlackBoxTest",
    timeout = "moderate",
    srcs = [
        "RepoWithRuleWritingTextGenerator.java",
        "WorkspaceBlackBoxTest.java",
    ],
    tags = ["black_box_test"],
    deps = common_deps,
)

test_suite(
    name = "ws_black_box_tests",
    tags = ["black_box_test"],
    tests = [
        "RepoWithRuleWritingTextGeneratorTest",
        "WorkspaceBlackBoxTest",
    ],
)
