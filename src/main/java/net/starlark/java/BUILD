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

# Approved clients of java.starlark.net.
package_group(
    name = "clients",
    packages = ["//..."],  # anyone
)
