# Description:
#   Skyframe Java code
package(
    default_visibility = ["//src:__subpackages__"],
)

SKYFRAME_OBJECT_SRCS = [
    "SkyValue.java",
    "SkyKey.java",
    "LegacySkyKey.java",
    "SkyFunctionName.java",
]

java_library(
    name = "skyframe-objects",
    srcs = SKYFRAME_OBJECT_SRCS,
    visibility = ["//visibility:public"],
    deps = [
        "//src/main/java/com/google/devtools/build/lib:collect",
        "//src/main/java/com/google/devtools/build/lib:concurrent",
        "//src/main/java/com/google/devtools/build/lib:preconditions",
        "//third_party:guava",
    ],
)

java_library(
    name = "skyframe",
    srcs = glob(
        ["**/*.java"],
        exclude = SKYFRAME_OBJECT_SRCS,
    ),
    deps = [
        ":skyframe-objects",
        "//src/main/java/com/google/devtools/build/lib:clock",  # keep
        "//src/main/java/com/google/devtools/build/lib:collect",
        "//src/main/java/com/google/devtools/build/lib:concurrent",
        "//src/main/java/com/google/devtools/build/lib:events",
        "//src/main/java/com/google/devtools/build/lib:util",
        "//src/main/java/com/google/devtools/build/lib:vfs",
        "//src/main/java/com/google/devtools/build/lib/profiler",
        "//src/main/java/com/google/devtools/common/options",
        "//third_party:guava",
        "//third_party:jsr305",
    ],
)

filegroup(
    name = "srcs",
    testonly = 0,  # All srcs should be not test only, overwrite package default.
    srcs = glob(["**"]),
)
