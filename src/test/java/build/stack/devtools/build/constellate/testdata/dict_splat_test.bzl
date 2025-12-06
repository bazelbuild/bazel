"""Test file for dict splat operator with fake objects."""

# Simulate proto_toolchains module
proto_toolchains = struct(
    if_legacy_toolchain = lambda x: x,
    use_toolchain = lambda x: [],
)

# Test dict merging with ** operator
test_attrs = dict({
    "name": attr.string(),
    "srcs": attr.label_list(),
}, **proto_toolchains.if_legacy_toolchain({
    "protoc": attr.label(),
}))

def test_function():
    """Function that uses dict splat."""
    base_dict = {"a": 1, "b": 2}
    extension = proto_toolchains.if_legacy_toolchain({"c": 3})
    merged = dict(base_dict, **extension)
    return merged
