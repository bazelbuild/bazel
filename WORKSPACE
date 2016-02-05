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

# Bind to dummy targets if no android SDK/NDK is present.
bind(
    name = "android_sdk_for_testing",
    actual = "//:dummy",
)

bind(
    name = "android_ndk_for_testing",
    actual = "//:dummy",
)

# In order to run the Android integration tests, run
# scripts/workspace_user.sh and uncomment the next two lines.
# load("/WORKSPACE.user", "android_repositories")
# android_repositories()

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

