DEFAULT_VISIBILITY = [
    "//java/com/google/testing/junit/runner:__subpackages__",
    "//javatests/com/google/testing/junit/runner:__subpackages__",
    "//third_party/bazel/src/java_tools/junitrunner/java/com/google/testing/junit/runner:__subpackages__",
    "//third_party/bazel/src/java_tools/junitrunner/javatests/com/google/testing/junit/runner:__subpackages__",
]

package(default_visibility = ["//src:__subpackages__"])

java_library(
    name = "runner",
    srcs = ["BazelTestRunner.java"],
    data = ["//tools:test_sharding_compliant"],
    # Disable sunapi warnings about sun.misc.Signal.
    # There are no non-Sun APIs for doing this.
    javacopts = ["-Xlint:-sunapi"],
    deps = [
        "//src/java_tools/junitrunner/java/com/google/testing/junit/runner/internal",
        "//src/java_tools/junitrunner/java/com/google/testing/junit/runner/junit4",
        "//src/java_tools/junitrunner/java/com/google/testing/junit/runner/model",
        "//third_party:guava",
        "//third_party:guice",
        "//third_party:joda_time",
        "//third_party:jsr305",
        "//third_party:junit4",
    ],
)

java_binary(
    name = "Runner",
    main_class = "com.google.testing.junit.runner.BazelTestRunner",
    runtime_deps = [":runner"],
)

filegroup(
    name = "srcs",
    srcs = glob(["**"]),
)
