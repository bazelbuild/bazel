package(default_visibility = [
    "//src/java_tools/junitrunner:junitrunner_package_group",
])

java_library(
    name = "model",
    srcs = glob(["*.java"]),
    deps = [
        "//src/java_tools/junitrunner/java/com/google/testing/junit/junit4:runner",
        "//src/java_tools/junitrunner/java/com/google/testing/junit/runner/sharding",
        "//src/java_tools/junitrunner/java/com/google/testing/junit/runner/util",
        "//third_party:jsr305",
        "//third_party:jsr330_inject",
        "//third_party:junit4",
    ],
)

filegroup(
    name = "srcs",
    srcs = glob(["**/*.java"]) + ["BUILD"],
)
