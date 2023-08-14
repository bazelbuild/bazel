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
    name = "util",
    testonly = 1,
    srcs = [
        "StarlarkUtil.java",
    ],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/vfs",
        "//src/test/java/com/google/devtools/build/lib/packages:testutil",
        "//src/test/java/com/google/devtools/build/lib/testutil",
        "//third_party:guava",
    ],
)
