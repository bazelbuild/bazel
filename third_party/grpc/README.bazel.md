# Bazel + gRPC

This directory contains the gRPC libraries needed by Bazel, sourced from
<https://github.com/grpc/grpc>.

| Repo             | Current   |
| ---------------- | --------- |
| `grpc/grpc`      | `v1.48.1` |

## Updating `third_party/grpc`

This requires 3 pull requests:

1. Update `third_party/grpc` to include files from new version

2. Switch `distdir_deps.bzl`, `WORKSPACE`, and any other references to new version

3. Remove older version from `third_party/grpc`

### How to update the C++ sources of gRPC

1. Update the gRPC patch file if necessary, it mostly helps avoid unnecessary dependencies.
2. Update `third_party/grpc/BUILD` to redirect targets to `@com_github_grpc_grpc` if necessary.
3. In a separate PR, update the gRPC definitions in the `distdir_deps.bzl` file.

### How to update the BUILD/bzl sources of gRPC

1. `git clone http://github.com/grpc/grpc.git` in a convenient directory
2. `git checkout v${GRPC_VERSION_NUM}`
3. `mkdir -p third_party/grpc/bazel`
4. `cp <gRPC git tree>/bazel/{BUILD,cc_grpc_library.bzl,generate_cc.bzl,protobuf.bzl} third_party/grpc/bazel`
5. In the `third_party/grpc/grpc` directory, apply local patches:
   `patch -p4 < bazel_${GRPC_VERSION_NUM}.patch`
