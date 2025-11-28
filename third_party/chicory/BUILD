load("@rules_java//java:defs.bzl", "java_library")

package(default_visibility = ["//:__subpackages__"])

filegroup(
    name = "srcs",
    srcs = glob(["**"]),
)

java_library(
    name = "chicory",
    exports = [
        "@maven//:com_dylibso_chicory_compiler",
        "@maven//:com_dylibso_chicory_dircache_experimental",
        "@maven//:com_dylibso_chicory_log",
        "@maven//:com_dylibso_chicory_runtime",
        "@maven//:com_dylibso_chicory_wasm",
    ],
)

# for bootstrapping
filegroup(
    name = "dist_jars",
    srcs = [
        "@maven//:com_dylibso_chicory_compiler",
        "@maven//:com_dylibso_chicory_dircache_experimental",
        "@maven//:com_dylibso_chicory_log",
        "@maven//:com_dylibso_chicory_runtime",
        "@maven//:com_dylibso_chicory_wasm",
    ],
)
