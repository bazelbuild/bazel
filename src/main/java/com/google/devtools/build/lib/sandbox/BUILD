package(
    default_visibility = [
        "//src:__subpackages__",
    ],
)

java_library(
    name = "sandbox",
    srcs = glob(["*.java"]),
    deps = [
        "//src/main/java:analysis-exec-rules-skyframe",
        "//src/main/java:buildtool-runtime",
        "//src/main/java:common",
        "//src/main/java:events",
        "//src/main/java:io",
        "//src/main/java:os_util",
        "//src/main/java:packages",
        "//src/main/java:shell",
        "//src/main/java:unix",
        "//src/main/java:util",
        "//src/main/java:vfs",
        "//src/main/java/com/google/devtools/build/lib/actions",
        "//src/main/java/com/google/devtools/build/lib/standalone",
        "//src/main/java/com/google/devtools/common/options",
        "//third_party:guava",
    ],
)

filegroup(
    name = "srcs",
    srcs = glob(["**"]),
)
