package(default_visibility = [":packages"])

package_group(
    name = "packages",
    packages = ["//src/java_tools/buildjar/..."],
)

java_binary(
    name = "turbine",
    main_class = "com.google.devtools.build.java.turbine.Turbine",
    visibility = ["//visibility:public"],
    runtime_deps = [":turbine_main"],
)

java_library(
    name = "turbine_main",
    srcs = ["Turbine.java"],
    deps = [
        "//src/java_tools/buildjar/java/com/google/devtools/build/java/turbine/javac:javac_turbine",
        "//third_party:guava",
        "//third_party:turbine",
    ],
)

filegroup(
    name = "srcs",
    srcs = glob(["**"]) + [
        "//src/java_tools/buildjar/java/com/google/devtools/build/java/turbine/javac:srcs",
    ],
)
