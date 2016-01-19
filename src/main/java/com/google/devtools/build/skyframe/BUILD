# Description:
#   Skyframe Java code
package(
    default_visibility = ["//src:__subpackages__"],
)

java_library(
    name = "skyframe",
    srcs = glob(["**/*.java"]),
    deps = [
        "//src/main/java/com/google/devtools/build/lib:clock",  # keep
        "//src/main/java/com/google/devtools/build/lib:collect",
        "//src/main/java/com/google/devtools/build/lib:concurrent",
        "//src/main/java/com/google/devtools/build/lib:events",
        "//src/main/java/com/google/devtools/build/lib:util",
        "//src/main/java/com/google/devtools/build/lib:vfs",
        "//src/main/java/com/google/devtools/common/options",
        "//third_party:guava",
        "//third_party:jsr305",
    ],
)

filegroup(
    name = "srcs",
    srcs = glob(["**"]),
)
