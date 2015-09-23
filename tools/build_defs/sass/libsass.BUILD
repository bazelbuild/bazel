package(default_visibility = ["@sassc//:__pkg__"])

BASE_DIR = "libsass-3.3.0-beta1/"

filegroup(
    name = "srcs",
    srcs = glob([
         BASE_DIR + "src/**/*.h*",
         BASE_DIR + "src/**/*.c*",
    ]),
)

cc_library(
    name = "headers",
    includes = [BASE_DIR + "include"],
)