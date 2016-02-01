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

# only used for the scala rule
new_http_archive(
    name = "scala",
    strip_prefix = "scala-2.11.7",
    sha256 = "ffe4196f13ee98a66cf54baffb0940d29432b2bd820bd0781a8316eec22926d0",
    url = "http://downloads.typesafe.com/scala/2.11.7/scala-2.11.7.tgz",
    build_file = "tools/build_defs/scala/scala.BUILD",
)

# only used for the scala test rule
http_file(
    name = "scalatest",
    url = "https://oss.sonatype.org/content/groups/public/org/scalatest/scalatest_2.11/2.2.6/scalatest_2.11-2.2.6.jar",
    sha256 = "f198967436a5e7a69cfd182902adcfbcb9f2e41b349e1a5c8881a2407f615962",
)

