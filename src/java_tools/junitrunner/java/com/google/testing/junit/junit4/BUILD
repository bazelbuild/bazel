# Description:
#
#   JUnit 4.x extensions

package(
    default_testonly = 1,
    default_visibility = ["//src:__subpackages__"],
)

# Libraries
# =========================================================

# Extensions for writing custom JUnit4 runners
java_library(
    name = "runner",
    testonly = 0,  # TODO(bazel-team): make it testonly
    srcs = glob(["runner/*.java"]),
    deps = [
        "//third_party:guava",
        "//third_party:junit4",
    ],
)

filegroup(
    name = "srcs",
    srcs = glob(["**"]),
)
