# Description:
#   Test data for testing dependency checking.
package(
    default_testonly = 1,
    default_visibility = ["//src:__subpackages__"],
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
)

java_library(
    name = "testdata_lib_Library_no_members",
    srcs = ["library_no_members/com/google/devtools/build/importdeps/testdata/Library.java"],
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
