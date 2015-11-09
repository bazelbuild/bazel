load("/tools/build_defs/dotnet/csharp", "csharp_nunit_test")

csharp_nunit_test(
    name = "MyTest",
    srcs = [
        "MyTest.cs",
    ],
    deps = [
        "//examples/dotnet/example_lib:MyClass",
    ],
)
