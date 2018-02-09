# Description:
#   A checker to check the completeness of the deps of java_import or aar_import targets.

package(
    default_visibility = ["//src:__subpackages__"],
)

filegroup(
    name = "srcs",
    srcs = glob(["**"]),
)

java_library(
    name = "import_deps_checker",
    srcs = glob(
        ["*.java"],
        exclude = ["Main.java"],
    ),
    deps = [
        "//third_party:asm",
        "//third_party:asm-commons",
        "//third_party:asm-tree",
        "//third_party:auto_value",
        "//third_party:guava",
        "//third_party:jsr305",
        "//third_party:junit4",
    ],
)

java_binary(
    name = "ImportDepsChecker",
    srcs = ["Main.java"],
    main_class = "com.google.devtools.build.importdeps.Main",
    deps = [
        ":import_deps_checker",
        "//src/main/java/com/google/devtools/common/options",
        "//src/main/protobuf:worker_protocol_java_proto",
        "//third_party:guava",
    ],
)
