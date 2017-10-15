package(default_visibility = [":packages"])

package_group(
    name = "packages",
    packages = ["//src/java_tools/buildjar/..."],
)

filegroup(
    name = "srcs",
    srcs = glob(["**"]) + [
        "//src/java_tools/buildjar/javatests/com/google/devtools/build/java/turbine/javac:srcs",
    ],
)
