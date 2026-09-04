load("@rules_java//java:defs.bzl", "java_test")

package(
    default_applicable_licenses = ["//:license"],
    default_testonly = 1,
    default_visibility = ["//src:__subpackages__"],
)

filegroup(
    name = "srcs",
    testonly = 0,
    srcs = glob(["**"]),
    visibility = ["//src/test/java/com/google/devtools/build/lib/analysis:__pkg__"],
)

java_test(
    name = "TemplateExpanderTest",
    srcs = ["TemplateExpanderTest.java"],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/analysis/stringtemplate",
        "//third_party/java/guava:collect",
        "//third_party/java/junit",
        "//third_party/java/truth",
    ],
)
