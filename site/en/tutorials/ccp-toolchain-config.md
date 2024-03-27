Project: /_project.yaml
Book: /_book.yaml

# Bazel Tutorial: Configure C++ Toolchains

{% include "_buttons.html" %}

This tutorial uses an example scenario to describe how to configure C++
toolchains for a project.

## What you'll learn {: #what-you-learn }

In this tutorial you learn how to:

*  Set up the build environment
*  Use `--toolchain_resolution_debug` to debug toolchain resolution
*  Configure the C++ toolchain
*  Create a Starlark rule that provides additional
configuration for the `cc_toolchain` so that Bazel can build the application
with `clang`
*  Build the C++ binary for by running
`bazel build //main:hello-world` on a Linux machine
* Cross-compile the binary for android by running `bazel build //main:hello-world --platforms=//:android_x86_64`

## Before you begin {: #before-you-begin }

This tutorial assumes you are on Linux and have successfully built
C++ applications and installed the appropriate tooling and libraries.
The tutorial uses `clang version 16`, which you can install on your system.

### Set up the build environment {: #setup-build-environment }

Set up your build environment as follows:

1.  If you have not already done so,
   [download and install Bazel 7.0.2](https://bazel.build/install) or later.

2.  Add an empty `WORKSPACE` file at the root folder.

3.  Add the following `cc_binary` target to the `main/BUILD` file:

    ```python
    load("@rules_cc//cc:defs.bzl", "cc_binary")

    cc_binary(
        name = "hello-world",
        srcs = ["hello-world.cc"],
    )
    ```

    Because Bazel uses many internal tools written in C++ during the build, such
    as `process-wrapper`, the pre-existing default C++ toolchain is specified for
    the host platform. This enables these internal tools to build using that
    toolchain of the one created in this tutorial. Hence, the `cc_binary` target
    is also built with the default toolchain.

4.  Run the build with the following command:

    ```bash
    bazel build //main:hello-world
    ```

    The build succeeds without any toolchain registered in the `WORKSPACE`.

    To further see what's under the hood, run:

    ```bash
    bazel build //main:hello-world --toolchain_resolution_debug='@bazel_tools//tools/cpp:toolchain_type'

    INFO: ToolchainResolution: Target platform @@local_config_platform//:host: Selected execution platform @@local_config_platform//:host, type @@bazel_tools//tools/cpp:toolchain_type -> toolchain @@bazel_tools~cc_configure_extension~local_config_cc//:cc-compiler-k8
    ```

    Without specifying `--platforms`, Bazel builds the target for
    `@local_config_platform//:host` using
    `@bazel_tools//cc_configure_extension/local_config_cc//:cc-compiler-k8`

## Configure the C++ toolchain {: #configure-cc-toolchain }

To configure the C++ toolchain, repeatedly build the application and eliminate
each error one by one as described as following.

Note: This tutorial assumes you're using Bazel 7.0.2 or later. If you're
using an older release of Bazel, use
`--incompatible_enable_cc_toolchain_resolution` flag to enable C++ toolchain
resolution.

It also assumes `clang version 9.0.1`, although the details should only change
slightly between different versions of clang.

1.  Add `toolchain/BUILD` with

    ```python
    filegroup(name = "empty")

    cc_toolchain(
        name = "linux_x86_64_toolchain",
        toolchain_identifier = "linux_x86_64-toolchain",
        toolchain_config = ":linux_x86_64_toolchain_config",
        all_files = ":empty",
        compiler_files = ":empty",
        dwp_files = ":empty",
        linker_files = ":empty",
        objcopy_files = ":empty",
        strip_files = ":empty",
        supports_param_files = 0,
    )

    toolchain(
        name = "cc_toolchain_for_linux_x86_64",
        toolchain = ":linux_x86_64_toolchain",
        toolchain_type = "@bazel_tools//tools/cpp:toolchain_type",
        exec_compatible_with = [
            "@platforms//cpu:x86_64",
            "@platforms//os:linux",
        ],
        target_compatible_with = [
            "@platforms//cpu:x86_64",
            "@platforms//os:linux",
        ],
    )
    ```
    Then register the toolchain with the `WORKSPACE` with

    ```python
    register_toolchains(
        "//toolchain:cc_toolchain_for_linux_x86_64"
    )
    ```

    This step defines a `cc_toolchain` and binds it to a `toolchain` target for
    the host configuration.

2.  Run the build again. Because the `toolchain` package doesn't yet define the
    `linux_x86_64_toolchain_config` target, Bazel throws the following error:

    ```bash
    ERROR: toolchain/BUILD:4:13: in toolchain_config attribute of cc_toolchain rule //toolchain:linux_x86_64_toolchain: rule '//toolchain:linux_x86_64_toolchain_config' does not exist.
    ```

3.  In the `toolchain/BUILD` file, define an empty filegroup as follows:

    ```python
    package(default_visibility = ["//visibility:public"])

    filegroup(name = "linux_x86_64_toolchain_config")
    ```

4.  Run the build again. Bazel throws the following error:

    ```bash
    '//toolchain:linux_x86_64_toolchain_config' does not have mandatory providers: 'CcToolchainConfigInfo'.
    ```

    `CcToolchainConfigInfo` is a provider that you use to configure
    your C++ toolchains. To fix this error, create a Starlark rule
    that provides `CcToolchainConfigInfo` to Bazel by making a
    `toolchain/cc_toolchain_config.bzl` file with the following content:

    ```python
    def _impl(ctx):
        return cc_common.create_cc_toolchain_config_info(
            ctx = ctx,
            toolchain_identifier = "k8-toolchain",
            host_system_name = "local",
            target_system_name = "local",
            target_cpu = "k8",
            target_libc = "unknown",
            compiler = "clang",
            abi_version = "unknown",
            abi_libc_version = "unknown",
        )

    cc_toolchain_config = rule(
        implementation = _impl,
        attrs = {},
        provides = [CcToolchainConfigInfo],
    )
    ```

    `cc_common.create_cc_toolchain_config_info()` creates the needed provider
    `CcToolchainConfigInfo`. To use the `cc_toolchain_config` rule, add a load
    statement to `toolchain/BUILD` right below the package statement:

    ```python
    load(":cc_toolchain_config.bzl", "cc_toolchain_config")
    ```

    And replace the "linux_x86_64_toolchain_config" filegroup with a declaration
    of a `cc_toolchain_config` rule:

    ```python
    cc_toolchain_config(name = "linux_x86_64_toolchain_config")
    ```

5.  Run the build again. Bazel throws the following error:

    ```bash
    .../BUILD:1:1: C++ compilation of rule '//:hello-world' failed (Exit 1)
    src/main/tools/linux-sandbox-pid1.cc:421:
    "execvp(toolchain/DUMMY_GCC_TOOL, 0x11f20e0)": No such file or directory
    Target //:hello-world failed to build`
    ```

    At this point, Bazel has enough information to attempt building the code but
    it still does not know what tools to use to complete the required build
    actions. You will modify the Starlark rule implementation to tell Bazel what
    tools to use. For that, you need the tool_path() constructor from
    [`@bazel_tools//tools/cpp:cc_toolchain_config_lib.bzl`](https://source.bazel.build/bazel/+/4eea5c62a566d21832c93e4c18ec559e75d5c1ce:tools/cpp/cc_toolchain_config_lib.bzl;l=400):

    ```python
    # toolchain/cc_toolchain_config.bzl:
    # NEW
    load("@bazel_tools//tools/cpp:cc_toolchain_config_lib.bzl", "tool_path")

    def _impl(ctx):
        tool_paths = [ # NEW
            tool_path(
                name = "gcc",
                path = "/usr/bin/clang",
            ),
            tool_path(
                name = "ld",
                path = "/usr/bin/ld",
            ),
            tool_path(
                name = "ar",
                path = "/usr/bin/ar",
            ),
            tool_path(
                name = "cpp",
                path = "/bin/false",
            ),
            tool_path(
                name = "gcov",
                path = "/bin/false",
            ),
            tool_path(
                name = "nm",
                path = "/bin/false",
            ),
            tool_path(
                name = "objdump",
                path = "/bin/false",
            ),
            tool_path(
                name = "strip",
                path = "/bin/false",
            ),
        ]

        return cc_common.create_cc_toolchain_config_info(
            ctx = ctx,
            toolchain_identifier = "local",
            host_system_name = "local",
            target_system_name = "local",
            target_cpu = "k8",
            target_libc = "unknown",
            compiler = "clang",
            abi_version = "unknown",
            abi_libc_version = "unknown",
            tool_paths = tool_paths, # NEW
        )
    ```

    Make sure that `/usr/bin/clang` and `/usr/bin/ld` are the correct paths
    for your system.

6.  Run the build again. Bazel throws the following error:

    ```bash
    ERROR: main/BUILD:3:10: Compiling main/hello-world.cc failed: absolute path inclusion(s) found in rule '//main:hello-world':
    the source file 'main/hello-world.cc' includes the following non-builtin files with absolute paths (if these are builtin files, make sure these paths are in your toolchain):
      '/usr/include/c++/13/ctime'
      '/usr/include/x86_64-linux-gnu/c++/13/bits/c++config.h'
      '/usr/include/x86_64-linux-gnu/c++/13/bits/os_defines.h'
      ...
    ```

    Bazel needs to know where to search for included headers. There are
    multiple ways to solve this, such as using the `includes` attribute of
    `cc_binary`, but here this is solved at the toolchain level with the
    [`cxx_builtin_include_directories`](/rules/lib/toplevel/cc_common#create_cc_toolchain_config_info)
    parameter of `cc_common.create_cc_toolchain_config_info`. Beware that if
    you are using a different version of `clang`, the include path will be
    different. These paths may also be different depending on the distribution.

    Modify the return value in `toolchain/cc_toolchain_config.bzl` to look
    like this:

    ```python
    return cc_common.create_cc_toolchain_config_info(
        ctx = ctx,
        cxx_builtin_include_directories = [ # NEW
            "/usr/lib/llvm-16/lib/clang/16/include",
            "/usr/include",
        ],
        toolchain_identifier = "local",
        host_system_name = "local",
        target_system_name = "local",
        target_cpu = "k8",
        target_libc = "unknown",
        compiler = "clang",
        abi_version = "unknown",
        abi_libc_version = "unknown",
        tool_paths = tool_paths,
    )
    ```

7. Run the build command again, you will see an error like:

    ```bash
    /usr/bin/ld: bazel-out/k8-fastbuild/bin/main/_objs/hello-world/hello-world.o: in function `print_localtime()':
    hello-world.cc:(.text+0x68): undefined reference to `std::cout'
    ```

    The reason for this is because the linker is missing the C++ standard
    library and it can't find its symbols. There are many ways to solve this,
    such as using the `linkopts` attribute of `cc_binary`. Here it is solved by
    making sure that any target using the toolchain doesn't have to specify
    this flag.

    Copy the following code to `toolchain/cc_toolchain_config.bzl`:

    ```python
    # NEW
    load("@bazel_tools//tools/build_defs/cc:action_names.bzl", "ACTION_NAMES")
    # NEW
    load(
        "@bazel_tools//tools/cpp:cc_toolchain_config_lib.bzl",
        "feature",    # NEW
        "flag_group", # NEW
        "flag_set",   # NEW
        "tool_path",
    )

    all_link_actions = [ # NEW
        ACTION_NAMES.cpp_link_executable,
        ACTION_NAMES.cpp_link_dynamic_library,
        ACTION_NAMES.cpp_link_nodeps_dynamic_library,
    ]

    def _impl(ctx):
        tool_paths = [
            tool_path(
                name = "gcc",
                path = "/usr/bin/clang",
            ),
            tool_path(
                name = "ld",
                path = "/usr/bin/ld",
            ),
            tool_path(
                name = "ar",
                path = "/bin/false",
            ),
            tool_path(
                name = "cpp",
                path = "/bin/false",
            ),
            tool_path(
                name = "gcov",
                path = "/bin/false",
            ),
            tool_path(
                name = "nm",
                path = "/bin/false",
            ),
            tool_path(
                name = "objdump",
                path = "/bin/false",
            ),
            tool_path(
                name = "strip",
                path = "/bin/false",
            ),
        ]

        features = [ # NEW
            feature(
                name = "default_linker_flags",
                enabled = True,
                flag_sets = [
                    flag_set(
                        actions = all_link_actions,
                        flag_groups = ([
                            flag_group(
                                flags = [
                                    "-lstdc++",
                                ],
                            ),
                        ]),
                    ),
                ],
            ),
        ]

        return cc_common.create_cc_toolchain_config_info(
            ctx = ctx,
            features = features, # NEW
            cxx_builtin_include_directories = [
                "/usr/lib/llvm-9/lib/clang/9.0.1/include",
                "/usr/include",
            ],
            toolchain_identifier = "local",
            host_system_name = "local",
            target_system_name = "local",
            target_cpu = "k8",
            target_libc = "unknown",
            compiler = "clang",
            abi_version = "unknown",
            abi_libc_version = "unknown",
            tool_paths = tool_paths,
        )

    cc_toolchain_config = rule(
        implementation = _impl,
        attrs = {},
        provides = [CcToolchainConfigInfo],
    )
    ```

8.  Running `bazel build //main:hello-world`, it should finally build the binary successfully for host.

9.  In `toolchain/BUILD`, copy the `cc_toolchain_config`, `cc_toolchain`, and
    `toolchain` targets and replace `linux_x86_64` with `android_x86_64`in target
    names.

    In `WORKSPACE`, register the toolchain for android

    ```python
    register_toolchains(
        "//toolchain:cc_toolchain_for_linux_x86_64",
        "//toolchain:cc_toolchain_for_android_x86_64"
    )
    ```

10. Run `bazel build //main:hello-world --android_platforms=//toolchain:android_x86_64` to build the binary for Android.

In practice, Linux and Android should have different C++ toolchain configs. You
can either modify the existing `cc_toolchain_config` for the differences or
create a separate rules (i.e. `CcToolchainConfigInfo` provider) for separate
platforms.

## Review your work {: #review-your-work }

In this tutorial you learned how to configure a basic C++ toolchain, but
toolchains are more powerful than this simple example.

The key take-aways are:

- You need to specify a matching `platforms` flag in the command line for Bazel to
  resolve to the toolchain for the same constraint values on the platform.
  The documentation holds more [information about language specific configuration flags](/concepts/platforms).
- You have to let the toolchain know where the tools live. In this tutorial
  there is a simplified version where you access the tools from the system. If
  you are interested in a more self-contained approach, you can read about
  [workspaces](/reference/be/workspace). Your tools could come from a
  different workspace and you would have to make their files available
  to the `cc_toolchain` with target dependencies on attributes, such as
  `compiler_files`. The `tool_paths` would need to be changed as well.
- You can create features to customize which flags should be passed to
  different actions, be it linking or any other type of action.

## Further reading {: #further-reading }

For more details, see
[C++ toolchain configuration](/docs/cc-toolchain-config-reference)
