package(
    default_visibility = [
        "//src:__subpackages__",
    ],
)

java_library(
    name = "sandbox",
    srcs = glob(["*.java"]),
    deps = [
        "//src/main/java:actions",
        "//src/main/java:analysis-exec-rules-skyframe",
        "//src/main/java:buildtool-runtime",
        "//src/main/java:common",
        "//src/main/java:events",
        "//src/main/java:packages",
        "//src/main/java:shell",
        "//src/main/java:unix",
        "//src/main/java:vfs",
        "//src/main/java/com/google/devtools/build/lib/standalone",
        "//third_party:guava",
    ],
)

filegroup(
    name = "srcs",
    srcs = glob(["**"]),
    visibility = ["//src/main/java:__pkg__"],
)
