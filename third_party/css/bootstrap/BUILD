licenses(["notice"])  # MIT

filegroup(
    name = "srcs",
    srcs = glob(["**"]),
    visibility = ["//third_party:__pkg__"],
)

filegroup(
    name = "bootstrap_css",
    srcs = [
        "css/bootstrap.css",
        "css/bootstrap.css.map",
        "css/bootstrap.min.css",
        "css/bootstrap.min.css.map",
    ],
    visibility = ["//site:__pkg__"],
)

filegroup(
    name = "bootstrap_images",
    srcs = glob(["fonts/glyphicons-halflings-regular.*"]),
    visibility = ["//site:__pkg__"],
)
