load("@rules_java//java:defs.bzl", "java_library")

package(
    default_applicable_licenses = ["//:license"],
    default_visibility = ["//src:__subpackages__"],
)

filegroup(
    name = "srcs",
    srcs = glob(["*"]),
    visibility = ["//src/main/java/com/google/devtools/build/lib:__pkg__"],
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
    visibility = [
        "//src/main/java/com/google/devtools/build/lib:build_lib",
        "//src/main/java/com/google/devtools/build/lib:starlark_access",
    ],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/concurrent",
        "//src/main/java/com/google/devtools/build/lib/util",
        "//src/main/java/com/google/devtools/common/options",
        "//src/main/java/net/starlark/java/eval",
        "//third_party/java/guava:cache",
        "//third_party/java/guava:collect",
        "//third_party/java/protobuf:protobuf_lite",
    ],
)
