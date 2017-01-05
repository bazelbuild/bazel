package_group(
    name = "packages",
    packages = ["//src/java_tools/buildjar/..."],
)

package(default_visibility = [":packages"])

java_binary(
    name = "turbine",
    srcs = ["Turbine.java"],
    main_class = "com.google.devtools.build.java.turbine.Turbine",
    visibility = ["//visibility:public"],
    deps = [
        "//src/java_tools/buildjar/java/com/google/devtools/build/java/turbine/javac:javac_turbine",
        "//third_party:guava",
        "//third_party:turbine",
    ],
)

filegroup(
    name = "srcs",
    srcs = glob(["**"]) + ["//src/java_tools/buildjar/java/com/google/devtools/build/java/turbine/javac:srcs"],
    visibility = ["//src/java_tools/buildjar:__pkg__"],
)
