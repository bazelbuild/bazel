# The Skylint linter for BUILD and Skylark files
# Open-sourced as part of Bazel.

java_binary(
    name = "Skylint",
    srcs = [],
    main_class = "com.google.devtools.skylark.skylint.Skylint",
    visibility = ["//src/test/skylark/skylint:__pkg__"],
    runtime_deps = [
        ":skylint_lib",
    ],
)

java_library(
    name = "skylint_lib",
    srcs = glob(["**/*.java"]),
    visibility = [
        # For docstring parsing libraries.
        "//src/main/java/com/google/devtools/build/skydoc:__subpackages__",
        "//src/tools/skylark/javatests/com/google/devtools/skylark/skylint:__pkg__",
    ],
    deps = [
        # TODO(bazel-team): Once BazelLibrary has a Build API interface, depend
        # on lib:skylarkbuildapi instead of on lib:packages.
        "//src/main/java/com/google/devtools/build/lib:packages",
        "//src/main/java/com/google/devtools/build/lib:skylarkinterface",
        "//third_party:guava",
    ],
)

filegroup(
    name = "srcs",
    srcs = glob(
        ["**"],
    ),
    visibility = ["//src:__pkg__"],
)
