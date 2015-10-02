filegroup(
    name = "srcs",
    # include this BUILD file for the default android tools repository
    srcs = glob(["**"]),
    visibility = ["//src:__pkg__"],
)

JARHELPER_SRCS = [
    "JarHelper.java",
    "JarCreator.java",
]

# for bootstrap_java_binary
exports_files(
    JARHELPER_SRCS,
    visibility = ["//src/java_tools/buildjar:__pkg__"],
)

# JarHelper library for GenClass so it can create the gen jar.
java_library(
    name = "jarhelper",
    srcs = JARHELPER_SRCS,
    visibility = [
        "//src/java_tools/buildjar:__pkg__",
        "//src/java_tools/buildjar/java/com/google/devtools/build/buildjar/genclass:__pkg__",
        "//src/tools/android/java/com/google/devtools/build/android/idlclass:__pkg__",
    ],
    deps = [
        "//third_party:guava",
    ],
)
