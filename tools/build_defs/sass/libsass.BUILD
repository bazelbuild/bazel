package(default_visibility = ["@sassc//:__pkg__"])

filegroup(
    name = "srcs",
    srcs = glob([
         "src/**/*.h*",
         "src/**/*.c*",
    ]),
)

# Includes directive may seem unnecessary here, but its needed for the weird
# interplay between libsass/sassc projects. This is intentional.
cc_library(
    name = "headers",
    includes = ["include"],
    hdrs = glob(["include/**/*.h"]),
)