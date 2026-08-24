load("@rules_java//java:defs.bzl", "java_library")

package(
    default_applicable_licenses = ["//:license"],
    default_visibility = ["//visibility:public"],
)

java_library(
    name = "statistics",
    srcs = ["BlazeJavacStatistics.java"],
    javacopts = [
        "--add-exports=jdk.compiler/com.sun.tools.javac.util=ALL-UNNAMED",
    ],
    deps = [
        "//third_party/java/auto:auto_value",
        "//third_party/java/caffeine",
        "//third_party/java/guava:collect",
    ],
)

filegroup(
    name = "srcs",
    srcs = glob(["**/*.java"]) + [
        "BUILD",
    ],
    visibility = ["//src:__subpackages__"],
)
