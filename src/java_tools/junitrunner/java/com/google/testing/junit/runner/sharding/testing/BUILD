package(default_visibility = [
    "//src/java_tools/junitrunner:junitrunner_package_group",
])

java_library(
    name = "testing",
    testonly = 1,
    srcs = glob(["*.java"]),
    deps = [
        "//src/java_tools/junitrunner/java/com/google/testing/junit/runner/sharding",
        "//src/java_tools/junitrunner/java/com/google/testing/junit/runner/sharding/api",
        "//src/test/java/com/google/devtools/build/lib:testutil",
        "//third_party:guava",
        "//third_party:junit4",
        "//third_party:truth",
    ],
)

filegroup(
    name = "srcs",
    srcs = glob(["**/*.java"]) + ["BUILD"],
)
