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
        "//src/main/java/net/starlark/java/lib/json",
        "//third_party:guava",
        "@maven//:org_tomlj_tomlj",
    ],
)
