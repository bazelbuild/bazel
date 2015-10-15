package(default_visibility = ["//src:__subpackages__"])

java_library(
    name = "genquery",
    srcs = glob([
        "*.java",
    ]),
    deps = [
        "//src/main/java:analysis-exec-rules-skyframe",
        "//src/main/java:collect",
        "//src/main/java:common",
        "//src/main/java:concurrent",
        "//src/main/java:events",
        "//src/main/java:graph",
        "//src/main/java:options",
        "//src/main/java:packages",
        "//src/main/java:query2",
        "//src/main/java:shell",
        "//src/main/java:skyframe-base",
        "//src/main/java:util",
        "//src/main/java:vfs",
        "//src/main/java/com/google/devtools/build/lib/actions",
        "//src/main/protobuf:crosstool_config_proto",
        "//src/main/protobuf:extra_actions_base_proto",
        "//third_party:guava",
        "//third_party:jsr305",
        "//third_party:protobuf",
    ],
)

filegroup(
    name = "srcs",
    srcs = glob(["**"]),
)
