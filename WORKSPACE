load("/tools/build_defs/d/d", "d_repositories")
load("/tools/build_defs/dotnet/csharp", "csharp_repositories")
load("/tools/build_defs/jsonnet/jsonnet", "jsonnet_repositories")
load("/tools/build_defs/sass/sass", "sass_repositories")
load("/tools/build_rules/go/def", "go_repositories")
load("/tools/build_rules/rust/rust", "rust_repositories")

csharp_repositories()
d_repositories()
go_repositories()
jsonnet_repositories()
rust_repositories()
sass_repositories()

# In order to run the Android integration tests, uncomment these rules, point
# them to the Android NDK and the SDK, and point the bind rules under them
# to @repository//:files .
# android_sdk_repository(
#     name = "androidsdk",
#     path = "/path/to/sdk",
#     # Available versions are under /path/to/sdk/build-tools/.
#     build_tools_version = "21.1.1",
#     # Available versions are under /path/to/sdk/platforms/.
#     api_level = 19,
# )

# android_ndk_repository(
#     name = "androidndk",
#     path = "/path/to/ndk",
#     api_level = 19,  # Set this to the SDK's api_level.
# )

bind(
    name = "android_sdk_for_testing",
    # Uncomment and delete the //:dummy line to run integration tests.
    #   actual = "@androidsdk//:files",
    actual = "//:dummy",
)

bind(
    name = "android_ndk_for_testing",
    # Uncomment and delete the //:dummy line to run integration tests.
    #   actual = "@androidndk//:files",
    actual = "//:dummy",
)
