# Description:
#   Run JUnit tests in a controlled environment.
package(default_visibility = ["//visibility:private"])

filegroup(
    name = "srcs",
    srcs = [
        "BUILD",
        "//src/java_tools/junitrunner/java/com/google/testing/coverage:srcs",
        "//src/java_tools/junitrunner/java/com/google/testing/junit/junit4:srcs",
        "//src/java_tools/junitrunner/java/com/google/testing/junit/runner:srcs",
        "//src/java_tools/junitrunner/javatests/com/google/testing/junit/junit4/runner:srcs",
        "//src/java_tools/junitrunner/javatests/com/google/testing/junit/runner:srcs",
    ],
    visibility = ["//:__subpackages__"],
)

package_group(
    name = "junitrunner_package_group",
    packages = ["//src/java_tools/junitrunner/..."],
)
