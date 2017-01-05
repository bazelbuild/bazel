filegroup(
    name = "srcs",
    srcs = glob(["**"]) + ["//src/java_tools/buildjar/javatests/com/google/devtools/build/java/turbine/javac:srcs"],
    visibility = ["//src/java_tools/buildjar:__pkg__"],
)

package_group(
    name = "packages",
    packages = ["//src/java_tools/buildjar/..."],
)

package(default_visibility = [":packages"])
