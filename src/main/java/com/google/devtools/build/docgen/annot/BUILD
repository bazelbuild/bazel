load("@rules_java//java:defs.bzl", "java_library")

package(
    default_applicable_licenses = ["//:license"],
    default_visibility = ["//visibility:public"],
)

filegroup(
    name = "srcs",
    srcs = glob(["*"]),
    visibility = ["//visibility:public"],
)

java_library(
    name = "annot",
    srcs = [
        "DocCategory.java",
        "GlobalMethodDocs.java",
        "StarlarkConstructor.java",
    ],
    deps = ["//third_party/java/guava:base"],
)
