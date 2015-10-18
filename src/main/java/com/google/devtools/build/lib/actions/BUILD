package(default_visibility = ["//src:__subpackages__"])

java_library(
    name = "actions",
    srcs = glob([
        "**/*.java",
    ]),
    deps = [
        "//src/main/java:base-util",
        "//src/main/java:clock",
        "//src/main/java:collect",
        "//src/main/java:common",
        "//src/main/java:concurrent",
        "//src/main/java:events",
        "//src/main/java:io",
        "//src/main/java:packages",
        "//src/main/java:shell",
        "//src/main/java:skyframe-base",
        "//src/main/java:util",
        "//src/main/java:vfs",
        "//src/main/java/com/google/devtools/common/options",
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
