DEFAULT_VISIBILITY = [
    "//src/java_tools/junitrunner/java/com/google/testing/junit/runner:__subpackages__",
    "//src/test/junitrunner/javatests/com/google/testing/junit/runner:__subpackages__",
]

licenses(["notice"])  # Apache 2.0

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

# Tests
# =========================================================
java_test(
    name = "AllTests",
    args = glob(["**/*Test.java"]),
    main_class = "com.google.testing.junit.runner.TestRunner",
    use_testrunner = 0,
    runtime_deps = [":tests"],
)

filegroup(
    name = "srcs",
    srcs = glob(["**"]),
    visibility = ["//src:__pkg__"],
)
