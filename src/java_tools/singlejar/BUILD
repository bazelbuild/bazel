# Description:
#   SingleJar combines multiple zip files and additional files
#   into a single zip file.
package(default_visibility = ["//src/java_tools:__subpackages__"])

filegroup(
    name = "srcs",
    srcs = glob(["*"]) + [
        "//src/java_tools/singlejar/java/com/google/devtools/build/singlejar:srcs",
        "//src/java_tools/singlejar/java/com/google/devtools/build/zip:srcs",
        "//src/java_tools/singlejar/javatests/com/google/devtools/build/singlejar:srcs",
        "//src/java_tools/singlejar/javatests/com/google/devtools/build/zip:srcs",
    ],
    visibility = ["//:__subpackages__"],
)

package_group(
    name = "singlejar_package_group",
    packages = ["//src/java_tools/singlejar/..."],
)

alias(
    name = "libSingleJar",
    actual = "//src/java_tools/singlejar/java/com/google/devtools/build/singlejar:libSingleJar",
    visibility = ["//visibility:public"],
)

alias(
    name = "SingleJar",
    actual = "//src/java_tools/singlejar/java/com/google/devtools/build/singlejar:bazel-singlejar",
    visibility = ["//visibility:public"],
)

alias(
    name = "SingleJar_deploy.jar",
    actual = "//src/java_tools/singlejar/java/com/google/devtools/build/singlejar:bazel-singlejar_deploy.jar",
    visibility = ["//visibility:public"],
)

alias(
    name = "bootstrap",
    actual = "//src/java_tools/singlejar/java/com/google/devtools/build/singlejar:bootstrap",
)

alias(
    name = "bootstrap_deploy.jar",
    actual = "//src/java_tools/singlejar/java/com/google/devtools/build/singlejar:bootstrap_deploy.jar",
)
