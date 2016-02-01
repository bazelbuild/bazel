load("//tools/build_defs/scala:scala.bzl", "scala_binary", "scala_library")

# The examples below show how to combine Scala and Java rules.
# ScalaBinary is the Scala equivalent of JavaBinary.

java_binary(
    name = "JavaBinary",
    srcs = ["JavaBinary.java"],
    main_class = "scala.test.JavaBinary",
    deps = [":lib_import"],
)

# TODO(bazel-team): Allow java rules to depend directly on scala_library.
# For now, we have to use a java_import proxy.
java_import(
    name = "lib_import",
    jars = [":HelloLib_deploy.jar"],
)

scala_binary(
    name = "ScalaBinary",
    srcs = ["ScalaBinary.scala"],
    main_class = "scala.test.ScalaBinary",
    deps = [":HelloLib"],
)

scala_library(
    name = "HelloLib",
    srcs = ["HelloLib.scala"],
    deps = [
        "OtherJavaLib",
        "OtherLib",
    ],
)

scala_library(
    name = "OtherLib",
    srcs = ["OtherLib.scala"],
)

java_library(
    name = "OtherJavaLib",
    srcs = ["OtherJavaLib.java"],
)

scala_library(
    name = "ScalaLibResources",
    srcs = ["src/main/scala/scala/test/ScalaLibResources.scala"],
    resources = ["src/main/resources/scala/test/hellos"],
)

scala_binary(
    name = "ScalaLibBinary",
    srcs = ["src/main/scala/scala/test/ScalaLibBinary.scala"],
    main_class = "scala.test.ScalaLibBinary",
    deps = ["ScalaLibResources"],
)
