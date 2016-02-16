load("//tools/build_defs/dotnet:csharp.bzl", "csharp_binary")

csharp_binary(
    name = "hello",
    srcs = [
        "Program.cs",
        "Properties/AssemblyInfo.cs",
    ],
    deps = [
        "//examples/dotnet/example_lib:MyClass",
    ],
)
