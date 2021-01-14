package(default_visibility = [
    "//src/java_tools:__subpackages__",
])

filegroup(
    name = "srcs",
    srcs = glob(["*"]) + [
        "//src/java_tools/singlejar/java/com/google/devtools/build/zip:srcs",
        "//src/java_tools/singlejar/javatests/com/google/devtools/build/zip:srcs",
    ],
    visibility = ["//src:__subpackages__"],
)

package_group(
    name = "singlejar_package_group",
    packages = ["//src/java_tools/singlejar/..."],
)
