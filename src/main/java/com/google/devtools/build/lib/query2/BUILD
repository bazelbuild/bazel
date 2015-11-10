package(
    default_visibility = [
        "//src:__subpackages__",
    ],
)

java_library(
    name = "query2",
    srcs = glob([
        "**/*.java",
    ]),
    deps = [
        "//src/main/java/com/google/devtools/build/lib:analysis-exec-rules-skyframe",
        "//src/main/java/com/google/devtools/build/lib:collect",
        "//src/main/java/com/google/devtools/build/lib:common",
        "//src/main/java/com/google/devtools/build/lib:concurrent",
        "//src/main/java/com/google/devtools/build/lib:events",
        "//src/main/java/com/google/devtools/build/lib:graph",
        "//src/main/java/com/google/devtools/build/lib:packages",
        "//src/main/java/com/google/devtools/build/lib:util",
        "//src/main/java/com/google/devtools/build/lib:vfs",
        "//src/main/java/com/google/devtools/build/skyframe",
        "//src/main/java/com/google/devtools/common/options",
        "//src/main/protobuf:build_proto",
        "//third_party:guava",
        "//third_party:jsr305",
    ],
)

filegroup(
    name = "srcs",
    srcs = glob(["**"]),
)
