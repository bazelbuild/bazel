"""Generates and compiles C++ grpc stubs from proto_library rules."""

load(":generate_cc.bzl", "generate_cc")

# Simplified version of gRPC upstream's cc_grpc_library.
def cc_grpc_library(
        name,
        srcs,
        deps,
        generate_mocks = False,
        extra_cc_library_kwargs = {},
        **kwargs):
    """Generates C++ grpc classes for services defined in proto_library rules.

    This rule expects a singleton list containing a proto_library target for its
    srcs argument, and expects a list (of arbitrary size) of cc_proto_library
    targets for its deps argument.

    It generates only grpc library classes.

    Assumes the generated classes will be used in cc_api_version = 2.

    Args:
        name (str): Name of rule.
        srcs (list): A single proto_library which contains services descriptors.
        deps (list): A list of cc_proto_library targets which
          provide the compiled code of any message that the services depend on.
        generate_mocks (bool): when True, Google Mock code for client stub is
          generated. False by default.
        extra_cc_library_kwargs (map): extra arguments to pass to the cc_library
          rule
        **kwargs: extra arguments to pass to all rules instantiated by this
          macro. Must be common to all build rules. See
          https://docs.bazel.build/versions/master/be/common-definitions.html#common-attributes
    """
    if len(srcs) != 1:
        fail("The srcs attribute must be a singleton list but was " + str(srcs),
             "srcs")

    codegen_grpc_target = "_" + name + "_grpc_codegen"
    generate_cc(
        name = codegen_grpc_target,
        srcs = srcs,
        plugin = "//third_party/grpc:cpp_plugin",
        generate_mocks = generate_mocks,
        protoc = "//third_party/protobuf:protoc",
        **kwargs
    )

    cc_library_kwargs = dict(**extra_cc_library_kwargs)
    cc_library_kwargs.update(**kwargs)
    native.cc_library(
        name = name,
        srcs = [":" + codegen_grpc_target],
        hdrs = [":" + codegen_grpc_target],
        deps = deps + ["//third_party/grpc:grpc++_codegen_proto"],
        **cc_library_kwargs
    )
