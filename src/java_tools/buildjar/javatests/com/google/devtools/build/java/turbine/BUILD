package_group(
    name = "packages",
    packages = ["//src/java_tools/buildjar/..."],
)

package(default_visibility = [":packages"])

java_test(
    name = "TurbineOptionsTest",
    srcs = ["TurbineOptionsTest.java"],
    deps = [
        "//src/java_tools/buildjar/java/com/google/devtools/build/java/turbine:turbine_options",
        "//src/java_tools/buildjar/java/com/google/devtools/build/java/turbine:turbine_options_parser",
        "//third_party:guava",
        "//third_party:junit4",
        "//third_party:truth",
    ],
)
