# Description:
#   Zip provides a library for reading and writing zip files, allowing more
#   advanced manipulation than the JDK equivalents by providing detailed zip
#   entry data and raw file access.
package(default_visibility = ["//src:__subpackages__"])

package_group(
    name = "packages",
    packages = [
        "//src/java_tools/singlejar/java/com/google/devtools/build/zip",
        "//src/java_tools/singlejar/javatests/com/google/devtools/build/zip",
    ],
)

filegroup(
    name = "srcs",
    srcs = glob(["*.java"]) + ["BUILD"],
)

# Necessary for the bootstrap compilation.
filegroup(
    name = "java-srcs",
    srcs = glob(["*.java"]),
)

java_library(
    name = "zip",
    srcs = glob(["*.java"]),
    deps = ["//third_party:jsr305"],
)
