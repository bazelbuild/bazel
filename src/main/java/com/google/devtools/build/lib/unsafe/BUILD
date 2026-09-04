load("@rules_java//java:defs.bzl", "java_library")

package(
    default_applicable_licenses = ["//:license"],
    default_visibility = ["//src:__subpackages__"],
)

filegroup(
    name = "srcs",
    srcs = glob(["**"]),
    visibility = ["//src/main/java/com/google/devtools/build/lib:__pkg__"],
)

java_library(
    name = "unsafe-provider",
    srcs = ["UnsafeProvider.java"],
)

java_library(
    name = "string",
    srcs = ["StringUnsafe.java"],
    add_opens = [
        "java.base/java.lang",
    ],
    deps = ["//third_party/java/guava:base"],
)
