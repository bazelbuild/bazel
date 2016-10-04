DEFAULT_VISIBILITY = [
    "//java/com/google/testing/junit/runner:__subpackages__",
    "//javatests/com/google/testing/junit/runner:__subpackages__",
    "//third_party/bazel/src/java_tools/junitrunner/java/com/google/testing/junit/runner:__subpackages__",
    "//third_party/bazel/src/java_tools/junitrunner/javatests/com/google/testing/junit/runner:__subpackages__",
]

package(default_visibility = ["//src:__subpackages__"])

java_library(
    name = "test_runner",
    srcs = glob(["*.java"]),
    data = ["//tools:test_sharding_compliant"],
    # Disable sunapi warnings about sun.misc.Signal.
    # There are no non-Sun APIs for doing this.
    javacopts = ["-Xlint:-sunapi"],
    runtime_deps = ["//src/java_tools/junitrunner/java/com/google/testing/junit/runner/sharding/weighted"],
    deps = [
        "//src/java_tools/junitrunner/java/com/google/testing/junit/runner/internal",
        "//src/java_tools/junitrunner/java/com/google/testing/junit/runner/junit4",
        "//src/java_tools/junitrunner/java/com/google/testing/junit/runner/model",
        "//src/java_tools/junitrunner/java/com/google/testing/junit/runner/sharding",
        "//src/java_tools/junitrunner/java/com/google/testing/junit/runner/util",
        "//third_party:junit4",
    ],
)

java_binary(
    name = "Runner",
    main_class = "com.google.testing.junit.runner.BazelTestRunner",
    runtime_deps = [":test_runner"],
)

filegroup(
    name = "srcs",
    srcs = glob(["**"]),
)
