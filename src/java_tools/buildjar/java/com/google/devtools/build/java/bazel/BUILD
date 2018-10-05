# Description:
#   Tools for working with Java source code, using the Sun javac API.
#
# This is not the source code for javac itself.

package(
    default_visibility = ["//src/java_tools/buildjar:buildjar_package_group"],
)

# Exports the canonical javac bootclasspath artifact locations
genrule(
    name = "javac-bootclasspath-locations",
    srcs = ["//tools/jdk:platformclasspath"],
    outs = ["JavacBootclasspathLocations.java"],
    cmd = "$(location javac-bootclasspath-locations.sh) $@ $(GENDIR) $(SRCS)",
    tools = ["javac-bootclasspath-locations.sh"],
)

# Provides java-level access to the canonical javac bootclasspath
java_library(
    name = "JavacBootclasspath",
    srcs = [
        "JavacBootclasspath.java",
        ":javac-bootclasspath-locations",
    ],
    data = ["//tools/jdk:platformclasspath"],
    visibility = ["//visibility:public"],
    deps = ["//third_party:guava"],
)

filegroup(
    name = "srcs",
    srcs = glob([
        "*.java",
        "*.sh",
    ]) + [
        "BUILD",
    ],
)
