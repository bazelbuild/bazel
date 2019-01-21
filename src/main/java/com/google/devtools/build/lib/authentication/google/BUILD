package(default_visibility = ["//src:__subpackages__"])

filegroup(
    name = "srcs",
    srcs = glob(["**"]),
    visibility = ["//src/main/java/com/google/devtools/build/lib:__pkg__"],
)

java_library(
    name = "google",
    srcs = glob(["*.java"]),
    deps = [
        "//src/main/java/com/google/devtools/common/options",
        "//src/main/java/com/google/devtools/build/lib:runtime",
        "//src/main/java/com/google/devtools/build/lib:exitcode-external",
        "//src/main/java/com/google/devtools/build/lib:util",
        "//third_party:auth",
        "//third_party:guava",
        "//third_party:jsr305",
    ],
)
