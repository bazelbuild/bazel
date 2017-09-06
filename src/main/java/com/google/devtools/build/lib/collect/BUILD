package(default_visibility = ["//src:__subpackages__"])

filegroup(
    name = "srcs",
    srcs = glob(["**"]),
    visibility = ["//src/main/java/com/google/devtools/build/lib:__pkg__"],
)

# Library of collection utilities.
java_library(
    name = "collect",
    srcs = glob([
        "*.java",
    ]),
    deps = [
        "//src/main/java/com/google/devtools/build/lib:preconditions",
        "//src/main/java/com/google/devtools/build/lib/collect/nestedset",
        "//src/main/java/com/google/devtools/build/lib/concurrent",
        "//third_party:guava",
        "//third_party:jsr305",
    ],
)
