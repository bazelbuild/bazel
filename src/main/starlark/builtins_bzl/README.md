The .bzl files in this directory are packaged as Bazel's builtins zip. They use
a modified dialect of Bazel's Build Language, and cannot be loaded as regular
.bzl files. See StarlarkBuiltinsFunction.java and BzlLoadFunction.java for more
information.

When updating .bzl files in this directory, the effect may be observed
immediately if --experimental_builtins_bzl_path is set to "%workspace%", or
after a Bazel server restart if it's set to "%bundled%".
