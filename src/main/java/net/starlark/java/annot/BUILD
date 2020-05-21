load("@rules_java//java:defs.bzl", "java_library")

package(default_visibility = ["//src:__subpackages__"])

filegroup(
    name = "srcs",
    srcs = glob(["**"]),
    visibility = ["//src:__subpackages__"],
)

# Annotations to make Java classes and methods accessible from Starlark.
# Depending on this library adds annotations processing to the build.
java_library(
    name = "annot",
    exported_plugins = ["//src/main/java/net/starlark/java/annot/processor"],
    visibility = ["//visibility:public"],
    exports = [":annot_sans_processor"],
)

# Annotations without processor, to avoid a dependency cycle in the processor itself.
java_library(
    name = "annot_sans_processor",
    srcs = glob(["*.java"]),
    deps = ["//third_party:jsr305"],
)
