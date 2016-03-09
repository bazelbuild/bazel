package(default_visibility = ["//visibility:public"])

exports_files(glob(["**"]))

filegroup(
    name = "srcs",
    srcs = glob(["**"]),
)

java_binary(
    name = "j2objc",
    main_class = "com.google.devtools.j2objc.J2ObjC",
    runtime_deps = ["@bazel_j2objc//:j2objc"],
)

filegroup(
    name = "j2objc_wrapper",
    srcs = ["j2objc_wrapper.py"],
)
