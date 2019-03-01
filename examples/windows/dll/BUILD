filegroup(
    name = "srcs",
    srcs = glob(["**"]),
    visibility = ["//examples:__pkg__"],
)

cc_library(
    name = "hello-library-header",
    hdrs = ["hello-library.h"],
)

cc_binary(
    name = "hellolib.dll",
    srcs = [
        "hello-library.cpp",
    ],
    # Define COMPILING_DLL to export symbols during compiling the DLL.
    # See hello-library.h
    copts = ["/DCOMPILING_DLL"],
    linkshared = 1,
    deps = [
        ":hello-library-header",
    ],
)

# **Explicitly link to hellolib.dll**

# Declare hellolib.dll as data dependency and load it explicitly in code.
cc_binary(
    name = "hello_world-load-dll-at-runtime",
    srcs = [
        "hello_world-load-dll-at-runtime.cpp",
    ],
    data = [":hellolib.dll"],
)

# **Implicitly link to hellolib.dll**

# Get the import library for hellolib.dll
filegroup(
    name = "hello_lib_import_lib",
    srcs = [":hellolib.dll"],
    output_group = "interface_library",
)

# Because we cannot directly depend on cc_binary from other cc rules in deps attribute,
# we use cc_import as a bridge to depend on hellolib.dll
cc_import(
    name = "hellolib_dll_import",
    interface_library = ":hello_lib_import_lib",
    shared_library = ":hellolib.dll",
)

# Create a new cc_library to also include the headers needed for hellolib.dll
cc_library(
    name = "hellolib_dll",
    deps = [
        ":hello-library-header",
        ":hellolib_dll_import",
    ],
)

cc_binary(
    name = "hello_world-link-to-dll-via-lib",
    srcs = [
        "hello_world-link-to-dll-via-lib.cpp",
    ],
    deps = [":hellolib_dll"],
)
