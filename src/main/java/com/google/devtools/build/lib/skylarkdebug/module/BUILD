package(default_visibility = ["//src:__subpackages__"])

filegroup(
    name = "srcs",
    srcs = glob(["**"]),
    visibility = ["//src/main/java/com/google/devtools/build/lib:__pkg__"],
)

java_library(
    name = "module",
    srcs = ["SkylarkDebuggerModule.java"],
    deps = [
        ":options",
        "//src/main/java/com/google/devtools/build/lib:events",
        "//src/main/java/com/google/devtools/build/lib:runtime",
        "//src/main/java/com/google/devtools/build/lib:syntax",
        "//src/main/java/com/google/devtools/build/lib/skylarkdebug/server",
        "//src/main/java/com/google/devtools/common/options",
        "//third_party:guava",
    ],
)

java_library(
    name = "options",
    srcs = ["SkylarkDebuggerOptions.java"],
    deps = [
        "//src/main/java/com/google/devtools/common/options",
    ],
)
