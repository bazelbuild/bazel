# Description:
#   This is the extension package to support desugaring try-with-resources statements.
package(
    default_visibility = [
        "//src/test/java/com/google/devtools/build/android/desugar:__subpackages__",
        "//src/tools/android/java/com/google/devtools/build/android/desugar:__pkg__",
    ],
)

java_library(
    name = "throwable_extension",
    srcs = ["ThrowableExtension.java"],
    # This library must be compiled with java7, as we directly copy it to the desugared jar.
    javacopts = [
        "-source 7",
        "-target 7",
    ],
)

filegroup(
    name = "srcs",
    srcs = glob(["**"]),
)
