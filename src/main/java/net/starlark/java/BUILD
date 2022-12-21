# Bazel's Starlark interpreter

licenses(["notice"])

filegroup(
    name = "srcs",
    srcs = ["BUILD"],
    visibility = [":bazel"],
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
