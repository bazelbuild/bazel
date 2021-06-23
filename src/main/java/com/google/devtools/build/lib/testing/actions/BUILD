load("@rules_java//java:defs.bzl", "java_library")

package(
    default_testonly = True,
    default_visibility = ["//src:__subpackages__"],
)

licenses(["notice"])

filegroup(
    name = "srcs",
    testonly = False,
    srcs = glob(["**"]),
    visibility = ["//src:__subpackages__"],
)

java_library(
    name = "artifact_expander_helper",
    srcs = ["ArtifactExpanderHelper.java"],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/actions",
        "//src/main/java/com/google/devtools/build/lib/actions:artifacts",
        "//src/main/java/com/google/devtools/build/lib/actions:middleman_type",
        "//third_party:guava",
    ],
)
