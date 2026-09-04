load("@rules_java//java:defs.bzl", "java_library")

package(
    default_applicable_licenses = ["//:license"],
    default_visibility = ["//src:__subpackages__"],
)

filegroup(
    name = "srcs",
    srcs = glob(["**"]) + ["//src/main/java/net/starlark/java/annot/processor:srcs"],
    visibility = ["//src/main/java/net/starlark/java:__pkg__"],
)

# Annotations to make Java classes and methods accessible from Starlark.
# Depending on this library adds annotations processing to the build.
java_library(
    name = "annot",
    exported_plugins = ["//src/main/java/net/starlark/java/annot/processor"],
    visibility = ["//src/main/java/net/starlark/java:clients"],
    exports = [":annot_sans_processor"],
)

# Annotations without processor, to avoid a dependency cycle in the processor itself.
java_library(
    name = "annot_sans_processor",
    srcs = glob(["*.java"]),
    visibility = [
        "//src/main/java/net/starlark/java/annot/processor:__pkg__",
        "//src/test/java/com/google/devtools/build/lib/bazel/repository/starlark:__subpackages__",
    ],
    deps = [
        "//third_party/java/error_prone:annotations",
        "//third_party/java/jsr305_annotations",
    ],
)
