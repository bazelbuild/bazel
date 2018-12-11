# Description:
#   Skyframe Java code
package(
    default_visibility = ["//src:__subpackages__"],
)

SKYFRAME_OBJECT_SRCS = [
    "AbstractSkyKey.java",
    "FunctionHermeticity.java",
    "ShareabilityOfValue.java",
    "SkyFunctionName.java",
    "SkyKey.java",
    "SkyValue.java",
]

java_library(
    name = "skyframe-objects",
    srcs = SKYFRAME_OBJECT_SRCS,
    visibility = ["//visibility:public"],
    deps = [
        "//third_party:guava",
        "//third_party:jsr305",
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
        "//src/main/java/com/google/devtools/build/lib:events",
        "//src/main/java/com/google/devtools/build/lib:util",
        "//src/main/java/com/google/devtools/build/lib/clock",  # keep
        "//src/main/java/com/google/devtools/build/lib/collect/compacthashset",
        "//src/main/java/com/google/devtools/build/lib/collect/nestedset",
        "//src/main/java/com/google/devtools/build/lib/concurrent",
        "//src/main/java/com/google/devtools/build/lib/profiler",
        "//src/main/java/com/google/devtools/build/lib/skyframe/serialization/autocodec",
        "//src/main/java/com/google/devtools/common/options",
        "//third_party:guava",
        "//third_party:jsr305",
    ],
)

filegroup(
    name = "srcs",
    srcs = glob(["**"]),
)
