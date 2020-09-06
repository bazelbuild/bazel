# Description:
#   Android data binding buildtime libs and common lib shared with Android
#   clients: implementation.

package(default_visibility = ["//third_party/java/android_databinding:__pkg__"])

licenses(["notice"])  # Apache License 2.0

exports_files(["LICENSE"])

filegroup(
  name = "srcs",
  srcs = glob(["**"]),
)

# Base library shared by build tools and the data binding runtime.
java_import(
    name = "base",
    jars = ["base.jar"],
)

# Resource processor implementation.
java_import(
    name = "exec",
    jars = ["exec.jar"],
    deps = [
        ":compiler",
    ],
)

# Data binding XML expressions -> Java code implementation.
java_import(
    name = "compiler",
    jars = ["compiler.jar"],
    deps = [
        ":base",
        ":compiler_common",
        ":x_common",
        "//third_party:apache_commons_codec",
        "//third_party/jaxb:jaxb-2.3.0",
    ],
)

# Support library for the compiler (distributed separately).
java_import(
    name = "compiler_common",
    jars = ["compiler_common.jar"],
    deps = [
        ":base",
        ":x_common",
        "//third_party:android-annotations",
        "//third_party:apache_commons_io",
        "//third_party:checker_framework_annotations",
        "//third_party:error_prone_annotations",
        "//third_party:guava",
        "//third_party:jsr305",
        "//third_party/animal_sniffer",
        "//third_party/antlr",
        "//third_party/java/j2objc-annotations",
        "//third_party/java/javapoet",
        "//third_party/jetifier",
        "//third_party/juniversalchardet",
        "//third_party/kotlin_stdlib",
    ],
)

# Android X library for the compiler (distributed separately).
# in support world, this is equal to base
java_import(
    name = "x_common",
    jars = ["x_common.jar"],
)
