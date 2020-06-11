# The spelling checker used by the Starlark interpreter.

# The SpellChecker is available to Starlark and Bazel, but not the world.
package(default_visibility = ["//src:__subpackages__"])

filegroup(
    name = "srcs",
    srcs = glob(["**"]),
    visibility = ["//src:__subpackages__"],
)

java_library(
    name = "spelling",
    srcs = ["SpellChecker.java"],
    deps = ["//third_party:jsr305"],
)
