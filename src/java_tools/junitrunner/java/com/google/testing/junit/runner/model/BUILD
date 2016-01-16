package(default_visibility = ["//src:__subpackages__"])

java_library(
    name = "model",
    srcs = glob(["*.java"]),
    deps = [
        "//src/java_tools/junitrunner/java/com/google/testing/junit/junit4:runner",
        "//src/java_tools/junitrunner/java/com/google/testing/junit/runner/sharding",
        "//src/java_tools/junitrunner/java/com/google/testing/junit/runner/util",
        "//third_party:guava",
        "//third_party:guice",
        "//third_party:joda_time",
        "//third_party:jsr305",
        "//third_party:junit4",
    ],
)

filegroup(
    name = "srcs",
    srcs = glob(["**"]),
)
