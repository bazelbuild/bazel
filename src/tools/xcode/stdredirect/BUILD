package(default_visibility = ["//src:__subpackages__"])

# This target will only build on a Mac.
genrule(
    name = "stdredirect_genrule",
    srcs = ["StdRedirect.c"],
    outs = ["StdRedirect.dylib"],
    cmd = "/usr/bin/xcrun clang -arch i386 -arch x86_64 -mios-simulator-version-min=7.0 -dynamiclib " +
          "  -isysroot \"$$(/usr/bin/xcrun --sdk iphonesimulator --show-sdk-path)\" " +
          "  -o $@ $<",
    output_to_bindir = 1,
    visibility = ["//visibility:public"],
)

filegroup(
    name = "srcs",
    srcs = glob(["**"]),
    visibility = ["//src:__pkg__"],
)
