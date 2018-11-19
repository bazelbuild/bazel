# Description:
#   Testing tools for the devtools-common options parser.
package(
    default_testonly = 1,
    default_visibility = ["//src:__subpackages__"],
)

licenses(["notice"])  # Apache 2.0

filegroup(
    name = "srcs",
    testonly = 0,
    srcs = glob(
        ["**"],
    ),
    visibility = ["//src/main/java/com/google/devtools/common/options:__pkg__"],
)

java_library(
    name = "testing",
    srcs = glob(["*.java"]),
    deps = [
        "//src/main/java/com/google/devtools/common/options",
        "//third_party:guava",
        "//third_party:guava-testlib",
        "//third_party:truth",
    ],
)
