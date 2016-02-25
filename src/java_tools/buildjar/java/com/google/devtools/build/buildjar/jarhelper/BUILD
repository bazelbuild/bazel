package_group(
    name = "packages",
    packages = [
        "//src/java_tools/buildjar/...",
    ],
)

package(default_visibility = [":packages"])

JARHELPER_SRCS = [
    "JarHelper.java",
    "JarCreator.java",
]

filegroup(
    name = "srcs",
    # include this BUILD file for the default android tools repository
    srcs = glob(["**"]),
    visibility = ["//src:__pkg__"],
)

# JarHelper library for GenClass so it can create the gen jar.
java_library(
    name = "jarhelper",
    srcs = JARHELPER_SRCS,
    visibility = [
        ":packages",
        "//src/tools/android/java/com/google/devtools/build/android/idlclass:__pkg__",
    ],
    deps = [
        "//third_party:guava",
    ],
)

#
## Bootstrapping using Skylark rules
#

load("//tools/build_rules:java_rules_skylark.bzl", "bootstrap_java_library", "bootstrap_java_binary")

bootstrap_java_library(
    name = "bootstrap_jarhelper",
    srcs = JARHELPER_SRCS,
    deps = [
        "//src/java_tools/buildjar/java/com/google/devtools/build/buildjar:skylark-deps",
    ],
)
