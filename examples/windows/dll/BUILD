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
    deps = [
        ":hello-library-header",
    ],
    # Define COMPILING_DLL to export symbols during compiling the DLL.
    # See hello-library.h
    copts = ["/DCOMPILING_DLL"],
    linkshared = 1,
)


# **Explicitly link to hellolib.dll**

# Declare hellolib.dll as data dependency and load it explicitly in code.
cc_binary(
    name = "hello-explicitly",
    srcs = [
        "hello-world-explicitly.cpp",
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

# Because we cannot directly depend on cc_binary, we use cc_import as a bridge to
# depend on hellolib.dll
cc_import(
    name = "hellolib_dll_import",
    shared_library = ":hellolib.dll",
    interface_library = ":hello_lib_import_lib",
)

# Create a new cc_library to also include the headers needed for hellolib.dll
cc_library(
    name = "hellolib_dll",
    deps = [
        ":hellolib_dll_import",
        ":hello-library-header",
    ],
)

# Just depend on :hellolib_dll
cc_binary(
    name = "hello-implicitly",
    srcs = [
        "hello-world-implicitly.cpp",
    ],
    deps = [":hellolib_dll"],
)
