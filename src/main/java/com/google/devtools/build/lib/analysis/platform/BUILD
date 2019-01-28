package(default_visibility = ["//src:__subpackages__"])

filegroup(
    name = "srcs",
    srcs = glob(["**"]),
    visibility = ["//src/main/java/com/google/devtools/build/lib:__pkg__"],
)

UTIL_SRCS = ["PlatformProviderUtils.java"]

# Description:
#   Providers defined for platforms, constraints, and toolchains.
java_library(
    name = "platform",
    srcs = glob(
        [
            "*.java",
        ],
        exclude = UTIL_SRCS,
    ),
    deps = [
        "//src/main/java/com/google/devtools/build/lib:packages",
        "//src/main/java/com/google/devtools/build/lib:skylarkinterface",
        "//src/main/java/com/google/devtools/build/lib:string_util",
        "//src/main/java/com/google/devtools/build/lib:transitive-info-provider",
        "//src/main/java/com/google/devtools/build/lib/concurrent",
        "//src/main/java/com/google/devtools/build/lib/skyframe/serialization",
        "//src/main/java/com/google/devtools/build/lib/skyframe/serialization/autocodec",
        "//src/main/java/com/google/devtools/build/lib/skylarkbuildapi/platform",
        "//third_party:auto_value",
        "//third_party:guava",
    ],
)

java_library(
    name = "utils",
    srcs = UTIL_SRCS,
    deps = [
        ":platform",
        "//src/main/java/com/google/devtools/build/lib:provider-collection",
        "//third_party:guava",
    ],
)
