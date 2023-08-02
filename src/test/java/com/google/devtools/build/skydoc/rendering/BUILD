load("@rules_java//java:defs.bzl", "java_test")

package(
    default_applicable_licenses = ["//:license"],
    default_visibility = ["//src:__subpackages__"],
)

filegroup(
    name = "srcs",
    testonly = 0,
    srcs = glob(["*"]),
    visibility = ["//src:__subpackages__"],
)

java_test(
    name = "LabelRendererTest",
    size = "small",
    srcs = ["LabelRendererTest.java"],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/cmdline",
        "//src/main/java/com/google/devtools/build/skydoc/rendering:rendering_util",
        "//src/main/java/net/starlark/java/eval",
        "//third_party:guava",
        "//third_party:junit4",
        "//third_party:truth",
    ],
)
