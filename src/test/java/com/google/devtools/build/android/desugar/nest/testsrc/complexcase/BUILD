package(
    default_testonly = 1,
    default_visibility = ["//src/test/java/com/google/devtools/build/android/desugar:__subpackages__"],
)

licenses(["notice"])  # Apache 2.0

java_import(
    name = "complexcase",
    jars = ["libcomplexcase.jar"],
)

filegroup(
    name = "srcs",
    testonly = 0,
    srcs = glob(["*"]),
)
