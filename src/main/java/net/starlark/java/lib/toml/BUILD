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
        "//src/main/java/net/starlark/java/annot",
        "//src/main/java/net/starlark/java/eval",
        "//src/main/java/net/starlark/java/lib",
        "//third_party/java/jsr305_annotations",
        "@maven//:org_tomlj_tomlj",
    ],
)
