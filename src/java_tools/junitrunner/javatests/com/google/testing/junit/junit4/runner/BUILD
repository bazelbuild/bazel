package(default_visibility = [
    "//src/java_tools/junitrunner:junitrunner_package_group",
])

filegroup(
    name = "srcs",
    srcs = glob(["*"]),
)

java_test(
    name = "RegExTestCaseFilterTest",
    size = "small",
    srcs = ["RegExTestCaseFilterTest.java"],
    deps = [
        "//src/java_tools/junitrunner/java/com/google/testing/junit/junit4:runner",
        "//third_party:guava",
        "//third_party:junit4",
        "//third_party:truth",
    ],
)
