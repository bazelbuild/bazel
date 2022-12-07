package(default_visibility = ["//visibility:private"])

filegroup(
    name = "srcs",
    srcs = [
        "BUILD",
        "BUILD.tools",
        "//tools/java/runfiles:srcs",
    ],
    visibility = ["//tools:__pkg__"],
)

genrule(
    name = "copy_java_stub_template",
    srcs = [
        "//src/main/java/com/google/devtools/build/lib/bazel/rules/java:java_stub_template.txt",
    ],
    outs = ["java_stub_template.txt"],
    cmd = "cp $< $@",
    visibility = [
        "//tools:__pkg__",
        "//tools/jdk:__pkg__",
    ],
)

filegroup(
    name = "embedded_tools",
    srcs = [
        "BUILD.tools",
        ":java_stub_template.txt",
        "//tools/java/runfiles:embedded_tools",
    ],
    visibility = ["//tools:__pkg__"],
)
