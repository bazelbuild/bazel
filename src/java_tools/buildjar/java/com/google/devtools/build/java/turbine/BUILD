package_group(
    name = "packages",
    packages = ["//src/java_tools/buildjar/..."],
)

package(default_visibility = [":packages"])

java_binary(
    name = "turbine",
    main_class = "com.google.devtools.build.java.turbine.javac.JavacTurbine",
    runtime_deps = [
        "//src/java_tools/buildjar/java/com/google/devtools/build/java/turbine/javac:javac_turbine",
    ],
)

java_library(
    name = "turbine_options",
    srcs = ["TurbineOptions.java"],
    deps = [
        "//third_party:guava",
        "//third_party:jsr305",
    ],
)

java_library(
    name = "turbine_options_parser",
    srcs = ["TurbineOptionsParser.java"],
    deps = [
        ":turbine_options",
        "//third_party:guava",
        "//third_party:jsr305",
    ],
)

filegroup(
    name = "srcs",
    srcs = glob(["**"]) + ["//src/java_tools/buildjar/java/com/google/devtools/build/java/turbine/javac:srcs"],
    visibility = ["//src/java_tools/buildjar:__pkg__"],
)
