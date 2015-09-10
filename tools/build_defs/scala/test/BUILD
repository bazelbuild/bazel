load("/tools/build_defs/scala/scala", "scala_library")

java_binary(
    name = "Hello",
    srcs = ["Hello.java"],
    main_class = "scala.test.Hello",
    deps = [":lib_import"],
)

# TODO(bazel-team): Allow java rules to depend directly on scala_library.
# For now, we have to use a java_import proxy.
java_import(
    name = "lib_import",
    jars = [":HelloLib"],
)

scala_library(
    name = "HelloLib",
    srcs = ["HelloLib.scala"],
    deps = ["OtherLib"],
)

scala_library(
    name = "OtherLib",
    srcs = ["OtherLib.scala"],
)
