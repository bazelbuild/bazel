package(default_visibility = ["//src:__subpackages__"])

filegroup(
    name = "srcs",
    srcs = glob(["**"]),
    visibility = ["//src/main/java/com/google/devtools/build/lib:__pkg__"],
)

# Description:
#   Rules to define Platforms and related Constraints.
java_library(
    name = "platform",
    srcs = glob([
        "*.java",
    ]),
    deps = [
        "//src/main/java/com/google/devtools/build/lib:build-base",
        "//src/main/java/com/google/devtools/build/lib:core-rules",
        "//src/main/java/com/google/devtools/build/lib:packages",
        "//src/main/java/com/google/devtools/build/lib:skylarkinterface",
        "//src/main/java/com/google/devtools/build/lib/actions",
        "//src/main/java/com/google/devtools/build/lib/analysis/platform",
        "//src/main/java/com/google/devtools/build/lib/analysis/platform:utils",
        "//src/main/java/com/google/devtools/build/lib/skylarkbuildapi/platform",
        "//src/main/java/com/google/devtools/build/skyframe",
        "//third_party:guava",
    ],
)
