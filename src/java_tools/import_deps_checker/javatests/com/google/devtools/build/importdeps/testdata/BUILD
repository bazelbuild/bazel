# Description:
#   Test data for testing dependency checking.
package(
    default_testonly = 1,
    default_visibility = ["//src/java_tools/import_deps_checker/javatests/com/google/devtools/build/importdeps:__subpackages__"],
)

licenses(["notice"])  # Apache 2.0

filegroup(
    name = "srcs",
    testonly = False,
    srcs = glob(["**"]),
)

java_library(
    name = "testdata_client",
    srcs = ["Client.java"],
    deps = [":testdata_lib_neverlink"],
)

java_library(
    name = "testdata_lib_neverlink",
    srcs = glob(["Library*.java"]),
    neverlink = 1,
)

java_library(
    name = "testdata_lib_Library",
    srcs = ["Library.java"],
    deps = [":testdata_lib_LibraryInterface"],
)

java_library(
    name = "testdata_lib_Library_no_members",
    srcs = ["library_no_members/com/google/devtools/build/importdeps/testdata/Library.java"],
    deps = [":testdata_lib_LibraryInterface"],
)

java_library(
    name = "testdata_lib_LibraryAnnotations",
    srcs = ["LibraryAnnotations.java"],
)

java_library(
    name = "testdata_lib_LibraryException",
    srcs = ["LibraryException.java"],
)

java_library(
    name = "testdata_lib_LibraryInterface",
    srcs = ["LibraryInterface.java"],
)

java_library(
    name = "testdata_lib_LibraryModuleInfo",
    srcs = ["LibraryModuleInfo.java", "module-info.java"],
    javacopts = ["-source 9", "-target 9"],
)

java_library(
    name = "OneAnnotationValue",
    srcs = ["OneAnnotationValue.java"],
    deps = [":StringAnnotation"],
)

java_library(
    name = "StringAnnotation",
    srcs = ["StringAnnotation.java"],
)
