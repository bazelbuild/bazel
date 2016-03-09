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

# For tools/cpp/test/...
load("//tools/cpp/test:docker_repository.bzl", "docker_repository")
docker_repository()

# In order to run the Android integration tests, run
# scripts/workspace_user.sh and uncomment the next two lines.
# load("/WORKSPACE.user", "android_repositories")
# android_repositories()
