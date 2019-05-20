package(default_visibility = ["//src:__subpackages__"])

java_library(
    name = "supplier",
    srcs = glob(["*.java"]),
    deps = [
        "//third_party:error_prone_annotations",
        "//third_party:guava",
        "//third_party:jsr305",
    ],
)

filegroup(
    name = "srcs",
    srcs = glob(["**"]),
    visibility = ["//src/main/java/com/google/devtools/build/lib:__pkg__"],
)
