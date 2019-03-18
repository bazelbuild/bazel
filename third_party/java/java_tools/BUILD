package(default_visibility = ["//visibility:public"])

licenses(["notice"])

filegroup(
    name = "srcs",
    srcs = glob(["**"]),
)

filegroup(
    name = "bazel_embedded_java_tools",
    srcs = [
        "BUILD.javalangtools",
        "BUILD.pkg",
    ],
)

filegroup(
    name = "java_tools",
    srcs = glob(
        ["*.jar"],
        exclude = ["bazel-singlejar_deploy.jar"],
    ) + select({
        "//src/conditions:arm": ["bazel-singlejar_deploy.jar"],
        "//conditions:default": [],
    }) + [
        "BUILD.pkg",
    ],
)

genrule(
    name = "java_tools_zip",
    srcs = [
        ":java_tools",
        "//third_party/java/jdk/langtools:java_compiler_jar",
        "//third_party/java/jdk/langtools:javac_jar",
        "//third_party/java/jdk/langtools:jdk_compiler_jar",
    ],
    outs = ["java_tools.zip"],
    cmd = "zip -q -j $@ cat $(SRCS) | sort",
    visibility = ["//third_party:__pkg__"],
)
