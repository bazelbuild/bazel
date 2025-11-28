load("@rules_java//java:defs.bzl", "java_library")

package(
    default_applicable_licenses = ["//:license"],
    default_visibility = ["//src:__subpackages__"],
)

filegroup(
    name = "srcs",
    srcs = glob(["**"]),
    visibility = ["//src:__subpackages__"],
)

java_library(
    name = "toml",
    srcs = ["TomlParser.java"],
    visibility = ["//src/main/java/net/starlark/java:clients"],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/packages:provider",
        "//src/main/java/net/starlark/java/annot",
        "//src/main/java/net/starlark/java/eval",
        "//third_party:guava",
        "@maven//:com_fasterxml_jackson_core_jackson_core",
        "@maven//:com_fasterxml_jackson_dataformat_jackson_dataformat_toml",
    ],
)
