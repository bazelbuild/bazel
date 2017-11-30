package(default_visibility = ["//visibility:public"])

filegroup(
    name = "zipper",
    srcs = select({
        "//src/conditions:remote": ["//third_party/ijar:zipper"],
        "//conditions:default": glob(["zipper/*"]),
   }),
)
