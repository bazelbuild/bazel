# Bazel's Starlark interpreter

licenses(["notice"])

filegroup(
    name = "srcs",
    srcs = [
        "BUILD",
        "//src/main/java/net/starlark/java/annot:srcs",
        "//src/main/java/net/starlark/java/cmd:srcs",
        "//src/main/java/net/starlark/java/eval:srcs",
        "//src/main/java/net/starlark/java/lib:srcs",
        "//src/main/java/net/starlark/java/spelling:srcs",
        "//src/main/java/net/starlark/java/syntax:srcs",
    ],
    visibility = ["//src:__pkg__"],
)

# The java.starlark.net project, including its tests.
package_group(
    name = "starlark",
    packages = [
        "//src/main/java/net/starlark/java/...",
        "//src/test/java/net/starlark/java/...",
    ],
)

# Bazel and its tests.
package_group(
    name = "bazel",
    packages = ["//src/..."],
)

# The Java Starlark interpreter is not supported for general use as a public
# API. However, Copybara and Stardoc are approved clients.
#
# We do not use visibility to prevent other dependencies, because cross-repo
# visibility restrictions do not make sense. Just be aware this is not
# guaranteed as a stable API.
package_group(
    name = "clients",
    packages = ["public"],
)
