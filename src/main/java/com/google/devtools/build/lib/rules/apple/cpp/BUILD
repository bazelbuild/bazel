package(default_visibility = ["//src:__subpackages__"])

filegroup(
    name = "srcs",
    srcs = glob(["**"]),
    visibility = ["//src/main/java/com/google/devtools/build/lib:__pkg__"],
)

# Description:
#   Support for building cpp for Apple platform
java_library(
    name = "cpp",
    srcs = glob(["*.java"]),
    deps = [
        "//src/main/java/com/google/devtools/build/lib:build-base",
        "//src/main/java/com/google/devtools/build/lib:packages-internal",
        "//src/main/java/com/google/devtools/build/lib/rules/apple",
        "//src/main/java/com/google/devtools/build/lib/rules/cpp",
        "//third_party:guava",
    ],
)
