load("@rules_java//java:defs.bzl", "java_library")

package(
    default_applicable_licenses = ["//:license"],
    default_visibility = ["//src:__subpackages__"],
)

filegroup(
    name = "srcs",
    srcs = glob(["*"]),
    visibility = ["//src:__subpackages__"],
)

# This package defines the StarlarkSemantics keys used by Bazel, and their command-line flags.
# Logically it is lower-level than lib.packages, lib.starlarkbuildapi, and even lib.cmdline,
# so it may not depend on (say) Label.
java_library(
    name = "semantics",
    srcs = [
        "BuildLanguageOptions.java",
        "FlagConstants.java",
    ],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/concurrent",
        "//src/main/java/com/google/devtools/common/options",
        "//src/main/java/net/starlark/java/eval",
        "//third_party:guava",
    ],
)
