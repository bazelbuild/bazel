DEFAULT_VISIBILITY = [
    "//src/java_tools/junitrunner:__subpackages__",
]

package(default_visibility = DEFAULT_VISIBILITY)

java_library(
    name = "testbed",
    testonly = 1,
    srcs = glob(["*.java"]),
    deps = [
        "//third_party:guava",
        "//third_party:junit4",
        "//third_party:truth",
    ],
)

exports_files([
    "XmlOutputExercises.ant.xml",
])

filegroup(
    name = "srcs",
    srcs = glob(["*.java"]) + [
        "XmlOutputExercises.ant.xml",
        "BUILD",
    ],
    visibility = ["//src:__pkg__"],
)
