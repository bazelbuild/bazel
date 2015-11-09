load("/tools/build_defs/dotnet/csharp", "csharp_binary")

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
