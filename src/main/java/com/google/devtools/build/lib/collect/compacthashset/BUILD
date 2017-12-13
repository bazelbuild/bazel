package(default_visibility = ["//src:__subpackages__"])

filegroup(
    name = "srcs",
    srcs = glob(["**"]),
    visibility = ["//src/main/java/com/google/devtools/build/lib:__pkg__"],
)

# Library of collection utilities.
java_library(
    name = "compacthashset",
    srcs = glob([
        "*.java",
    ]),
    deps = [
        "//third_party:guava",
        "//third_party:jsr305",
    ],
)
