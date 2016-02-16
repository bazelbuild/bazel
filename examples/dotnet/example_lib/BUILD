load("//tools/build_defs/dotnet:csharp.bzl", "csharp_library")

csharp_library(
    name = "MyClass",
    srcs = [
        "MyClass.cs",
        "Properties/AssemblyInfo.cs",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//examples/dotnet/example_transitive_lib:TransitiveClass",
    ],
)
