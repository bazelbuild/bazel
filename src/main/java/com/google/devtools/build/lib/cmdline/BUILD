########################################################################
#
# The "cmdline" library: Label validation and target pattern
# parsing. Both the label syntax and the target pattern syntax are a
# public interface, and all changes have to be made in a
# backwards-compatible manner. This library is intentionally
# standalone so it can be used outside without pulling in any internal
# code.
#

java_library(
    name = "cmdline",
    srcs = glob(["*.java"]),
    visibility = ["//src:__subpackages__"],
    deps = [
        "//src/main/java/com/google/devtools/build/lib:base-util",
        "//src/main/java/com/google/devtools/build/lib:concurrent",
        "//src/main/java/com/google/devtools/build/lib:skylarkinterface",
        "//src/main/java/com/google/devtools/build/lib:util",
        "//src/main/java/com/google/devtools/build/lib:vfs",
        "//src/main/java/com/google/devtools/build/skyframe:skyframe-objects",
        "//third_party:guava",
        "//third_party:jsr305",
    ],
)

filegroup(
    name = "srcs",
    srcs = glob(["*"]),
    visibility = ["//visibility:public"],
)
