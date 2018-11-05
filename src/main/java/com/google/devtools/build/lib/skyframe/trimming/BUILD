# Classes which provide support for automatically trimming configuration to avoid wasted work during a build.

package(
    default_visibility = ["//src:__subpackages__"],
)

filegroup(
    name = "srcs",
    srcs = glob(["**"]),
)

java_library(
    name = "trimmed_configuration_cache",
    srcs = [
        "ConfigurationComparer.java",
        "KeyAndState.java",
        "TrimmedConfigurationCache.java",
    ],
    deps = [
        "//third_party:auto_value",
        "//third_party:guava",
    ],
)
