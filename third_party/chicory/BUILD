load("@rules_java//java:defs.bzl", "java_library")

package(default_visibility = ["//:__subpackages__"])

filegroup(
    name = "srcs",
    srcs = glob(["**"]),
)

java_library(
    name = "chicory",
    exports = [
        "@chicory//log",
        "@chicory//runtime",
        "@chicory//wasm",
    ],
)

# for bootstrapping
filegroup(
    name = "dist_jars",
    srcs = [
        "@chicory//log",
        "@chicory//runtime",
        "@chicory//wasm",
    ],
)
