package(
    default_visibility = [
        "//src:__subpackages__",
    ],
)

java_library(
    name = "sandbox",
    srcs = glob(["*.java"]),
    deps = [
        "//src/main/java/com/google/devtools/build/lib:analysis-exec-rules-skyframe",
        "//src/main/java/com/google/devtools/build/lib:buildtool-runtime",
        "//src/main/java/com/google/devtools/build/lib:common",
        "//src/main/java/com/google/devtools/build/lib:concurrent",
        "//src/main/java/com/google/devtools/build/lib:events",
        "//src/main/java/com/google/devtools/build/lib:io",
        "//src/main/java/com/google/devtools/build/lib:os_util",
        "//src/main/java/com/google/devtools/build/lib:packages",
        "//src/main/java/com/google/devtools/build/lib:shell",
        "//src/main/java/com/google/devtools/build/lib:unix",
        "//src/main/java/com/google/devtools/build/lib:util",
        "//src/main/java/com/google/devtools/build/lib:vfs",
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
