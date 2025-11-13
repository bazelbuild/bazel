load("@rules_java//java:defs.bzl", "java_test")

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
        "WorkspaceTestUtils.java",
    ],
    tags = ["black_box_test"],
    deps = common_deps,
)

java_test(
    name = "BazelEmbeddedStarlarkBlackBoxTest",
    timeout = "moderate",
    srcs = [
        "BazelEmbeddedStarlarkBlackBoxTest.java",
        "RepoWithRuleWritingTextGenerator.java",
        "WorkspaceTestUtils.java",
    ],
    tags = [
        "black_box_test",
    ],
    deps = common_deps + [
        "//src/main/java/com/google/devtools/build/lib/bazel/repository/decompressor",
        "//src/main/java/com/google/devtools/build/lib/vfs",
        "//src/test/java/com/google/devtools/build/lib/events:testutil",
        "//src/test/java/com/google/devtools/build/lib/vfs/util",
    ],
)

java_test(
    name = "GitRepositoryBlackBoxTest",
    timeout = "moderate",
    srcs = [
        "GitRepositoryBlackBoxTest.java",
        "GitRepositoryHelper.java",
        "RepoWithRuleWritingTextGenerator.java",
        "WorkspaceTestUtils.java",
    ],
    tags = ["black_box_test"],
    deps = common_deps + [
        "//src/main/java/com/google/devtools/build/lib/vfs",
        "//src/test/java/com/google/devtools/build/lib/events:testutil",
        "//src/test/java/com/google/devtools/build/lib/vfs/util",
    ],
)

java_test(
    name = "PatchApiBlackBoxTest",
    timeout = "moderate",
    srcs = [
        "PatchApiBlackBoxTest.java",
        "WorkspaceTestUtils.java",
    ],
    tags = ["black_box_test"],
    deps = common_deps + [
        "//src/main/java/com/google/devtools/build/lib/vfs",
        "//src/test/java/com/google/devtools/build/lib/events:testutil",
        "//src/test/java/com/google/devtools/build/lib/vfs/util",
    ],
)

test_suite(
    name = "ws_black_box_tests",
    tags = ["black_box_test"],
    tests = [
        "BazelEmbeddedStarlarkBlackBoxTest",
        "GitRepositoryBlackBoxTest",
        "PatchApiBlackBoxTest",
        "RepoWithRuleWritingTextGeneratorTest",
        "WorkspaceBlackBoxTest",
    ],
)
