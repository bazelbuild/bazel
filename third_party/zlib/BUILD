licenses(["notice"])  #  BSD/MIT-like license

filegroup(
    name = "srcs",
    srcs = glob(["**"]),
    visibility = ["//third_party:__pkg__"],
)

filegroup(
    name = "embedded_tools",
    srcs = glob(["*.c"]) + glob(["*.h"]) + ["BUILD"] + ["LICENSE.txt"],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "zlib",
    srcs = glob(["*.c"]),
    hdrs = glob(["*.h"]),
    copts = ["-w"],
    includes = ["."],
    visibility = ["//visibility:public"],
)
