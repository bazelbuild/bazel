# Description:
#   SingleJar combines multiple zip files and additional files
#   into a single zip file.
package(
    default_applicable_licenses = ["//:license"],
    default_visibility = [
        "//scripts/bootstrap:__subpackages__",
        "//src/java_tools:__subpackages__",
    ],
)

filegroup(
    name = "srcs",
    srcs = glob(["*"]) + [
        "//src/java_tools/singlejar/java/com/google/devtools/build/singlejar:srcs",
        "//src/java_tools/singlejar/java/com/google/devtools/build/zip:srcs",
        "//src/java_tools/singlejar/javatests/com/google/devtools/build/singlejar:srcs",
        "//src/java_tools/singlejar/javatests/com/google/devtools/build/zip:srcs",
    ],
    visibility = ["//src:__subpackages__"],
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
