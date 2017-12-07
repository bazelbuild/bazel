# The Skylark interpreter
# Open-sourced as part of Bazel.

package(default_visibility = ["//src:__subpackages__"])

java_binary(
    name = "Skylark",
    srcs = ["Skylark.java"],
    main_class = "com.google.devtools.skylark.Skylark",
    deps = [
        "//src/main/java/com/google/devtools/build/lib:events",
        "//src/main/java/com/google/devtools/build/lib:packages-internal",
    ],
)

filegroup(
    name = "srcs",
    srcs = glob(["**"]),
)
