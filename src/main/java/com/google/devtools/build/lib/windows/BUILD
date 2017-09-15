package(default_visibility = ["//src:__subpackages__"])

filegroup(
    name = "srcs",
    srcs = glob(["**"]),
    visibility = ["//src/main/java/com/google/devtools/build/lib:__pkg__"],
)

java_library(
    name = "windows",
    srcs = glob([
        "*.java",
    ]),
    visibility = ["//visibility:public"],
    deps = [
        "//src/main/java/com/google/devtools/build/lib:base-util",
        "//src/main/java/com/google/devtools/build/lib:os_util",
        "//src/main/java/com/google/devtools/build/lib:preconditions",
        "//src/main/java/com/google/devtools/build/lib/clock",
        "//src/main/java/com/google/devtools/build/lib/concurrent",
        "//src/main/java/com/google/devtools/build/lib/shell",
        "//src/main/java/com/google/devtools/build/lib/vfs",
        "//src/main/java/com/google/devtools/build/lib/windows/jni",
        "//src/main/java/com/google/devtools/common/options",
        "//third_party:guava",
        "//third_party:jsr305",
    ],
)
