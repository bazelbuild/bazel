# Description:
#   Rules to define Platforms and related Constraints.

package(
    default_visibility = ["//src:__subpackages__"],
)

java_library(
    name = "platform",
    srcs = glob([
        "*.java",
    ]),
    deps = [
        "//src/main/java/com/google/devtools/build/lib:build-base",
        "//src/main/java/com/google/devtools/build/lib:packages",
        "//third_party:auto_value",
        "//third_party:guava",
    ],
)

filegroup(
    name = "srcs",
    testonly = 0,  # All srcs should be not test only, overwrite package default.
    srcs = glob(["**"]),
)
