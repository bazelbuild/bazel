filegroup(
    name = "srcs",
    srcs = glob(["**"]),
    visibility = ["//examples:__pkg__"],
)

cc_library(
    name = "hellolib",
    srcs = [
        "hello-library.cpp",
    ],
    hdrs = [
        "hello-library.h",
    ],
    # Define COMPILING_DLL to export symbols during compiling the DLL.
    # See hello-library.h
    copts = ["/DCOMPILING_DLL"],
)

# **Explicitly link to hellolib.dll**

# Declare hellolib.dll as data dependency and load it explicitly in code.
cc_binary(
    name = "hello_world-load-dll-at-runtime",
    srcs = [
        "hello_world-load-dll-at-runtime.cpp",
    ],
    data = [":hellolib"],
)

# **Implicitly link to hellolib.dll**

# Consume hellolib as a dependency in shared way.
cc_binary(
    name = "hello_world-link-to-dll-via-lib",
    srcs = [
        "hello_world-link-to-dll-via-lib.cpp",
    ],
    deps = [":hellolib"],
    linkstatic = 0,
)
