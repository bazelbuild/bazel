package(default_visibility = [
    "//src/java_tools/junitrunner:junitrunner_package_group",
])

# TODO(bazel-team): This should be testonly = 1.
java_library(
    name = "sharding",
    srcs = glob(["*.java"]),
    deps = [
        "//src/java_tools/junitrunner/java/com/google/testing/junit/runner/sharding/api",
        "//src/java_tools/junitrunner/java/com/google/testing/junit/runner/util",
        "//third_party:jsr305",
        "//third_party:jsr330_inject",
        "//third_party:junit4",
    ],
)

filegroup(
    name = "srcs",
    srcs = glob(["**/*.java"]) + [
        "BUILD",
        "//src/java_tools/junitrunner/java/com/google/testing/junit/runner/sharding/api:srcs",
        "//src/java_tools/junitrunner/java/com/google/testing/junit/runner/sharding/testing:srcs",
    ],
)
