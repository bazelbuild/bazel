package(default_visibility = ["//visibility:public"])

licenses(["unencumbered"])  # google owned

filegroup(
    name = "srcs",
    srcs = glob(["**"]),
)

java_import(
    name = "jetifier",
    jars = glob(["*.jar"]),
    deps = [
      "//third_party:gson",
      "//third_party/jetbrains_annotations",
      "//third_party/kotlin_stdlib",
    ],
)
