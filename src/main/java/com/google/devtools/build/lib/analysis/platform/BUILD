# Description:
#   Providers defined for platforms, constraints, and toolchains.

package(
    default_visibility = ["//src:__subpackages__"],
)

UTIL_SRCS = ["PlatformProviderUtils.java"]

java_library(
    name = "platform",
    srcs = glob(
        [
            "*.java",
        ],
        exclude = UTIL_SRCS,
    ),
    deps = [
        "//src/main/java/com/google/devtools/build/lib:packages",
        "//src/main/java/com/google/devtools/build/lib:skylarkinterface",
        "//src/main/java/com/google/devtools/build/lib:transitive-info-provider",
        "//third_party:auto_value",
        "//third_party:guava",
    ],
)

java_library(
    name = "utils",
    srcs = UTIL_SRCS,
    deps = [
        ":platform",
        "//src/main/java/com/google/devtools/build/lib:preconditions",
        "//src/main/java/com/google/devtools/build/lib:skylark-provider-collection",
        "//third_party:guava",
    ],
)

filegroup(
    name = "srcs",
    testonly = 0,  # All srcs should be not test only, overwrite package default.
    srcs = glob(["**"]),
)
