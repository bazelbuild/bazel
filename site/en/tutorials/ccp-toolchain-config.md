Project: /_project.yaml
Book: /_book.yaml

# Bazel Tutorial: Configure C++ Toolchains

{% include "_buttons.html" %}

This tutorial uses an example scenario to describe how to configure C++
toolchains for a project. It's based on an
[example C++ project](https://github.com/bazelbuild/examples/tree/master/cpp-tutorial/stage1)
that builds error-free using `clang`.

## What you'll learn

In this tutorial you learn how to:

*  Set up the build environment
*  Configure the C++ toolchain
*  Create a Starlark rule that provides additional
configuration for the `cc_toolchain` so that Bazel can build the application
with `clang`
*  Confirm expected outcome by running
`bazel build --config=clang_config //main:hello-world` on a Linux machine
*  Build the C++ application

## Before you begin

### Set up the build environment

This tutorial assumes you are on Linux and have successfully built
C++ applications and installed the appropriate tooling and libraries.
The tutorial uses `clang version 9.0.1`, which you can install on your system.

Set up your build environment as follows:

1.  If you have not already done so,
   [download and install Bazel 0.23](/install/ubuntu) or later.

2.  Download the
    [example C++ project](https://github.com/bazelbuild/examples/tree/master/cpp-tutorial/stage1)
    from GitHub and place it in an empty directory on your local machine.


3.  Add the following `cc_binary` target to the `main/BUILD` file:

    ```python
    cc_binary(
        name = "hello-world",
        srcs = ["hello-world.cc"],
    )
    ```

4.  Create a `.bazelrc` file at the root of the workspace directory with the
    following contents to enable the use of the `--config` flag:

    ```
    # Use our custom-configured c++ toolchain.

    build:clang_config --crosstool_top=//toolchain:clang_suite

    # Use --cpu as a differentiator.

    build:clang_config --cpu=k8

    # Use the default Bazel C++ toolchain to build the tools used during the
    # build.

    build:clang_config --host_crosstool_top=@bazel_tools//tools/cpp:toolchain
    ```

For an entry `build:{config_name} --flag=value`, the command line flag
`--config={config_name}` is associated with that particular flag. See
documentation for the flags used:
[`crosstool_top`](/docs/user-manual#crosstool-top),
[`cpu`](/docs/user-manual#cpu) and
[`host_crosstool_top`](/docs/user-manual#host-crosstool-top).

When you build your [target](/concepts/build-ref#targets)
with `bazel build --config=clang_config //main:hello-world`, Bazel uses your
custom toolchain from the
[cc_toolchain_suite](/reference/be/c-cpp#cc_toolchain_suite)
`//toolchain:clang_suite`. The suite may list different
[toolchains](/reference/be/c-cpp#cc_toolchain) for different CPUs,
and that's why it is differentiated with the flag `--cpu=k8`.

Because Bazel uses many internal tools written in C++ during the build, such as
process-wrapper, the pre-existing default C++ toolchain is specified for
the host platform, so that these tools are built using that toolchain instead of
the one created in this tutorial.

## Configuring the C++ toolchain

To configure the C++ toolchain, repeatedly build the application and eliminate
each error one by one as described below.

Note: This tutorial assumes you're using Bazel 0.23 or later. If you're
using an older release of Bazel, look for the "Configuring CROSSTOOL" tutorial.
It also assumes `clang version 9.0.1`, although the details should only change
slightly between different versions of clang.

1.  Run the build with the following command:

    ```
    bazel build --config=clang_config //main:hello-world
    ```

    Because you specified `--crosstool_top=//toolchain:clang_suite` in the
    `.bazelrc` file, Bazel throws the following error:

    ```
    No such package `toolchain`: BUILD file not found on package path.
    ```

    In the workspace directory, create the `toolchain` directory for the package
    and an empty `BUILD` file inside the `toolchain` directory.

2.  Run the build again. Because the `toolchain` package does not yet define the
    `clang_suite` target, Bazel throws the following error:

    ```
    No such target '//toolchain:clang_suite': target 'clang_suite' not declared
    in package 'toolchain' defined by .../toolchain/BUILD
    ```

    In the `toolchain/BUILD` file, define an empty filegroup as follows:

    ```python
    package(default_visibility = ["//visibility:public"])

    filegroup(name = "clang_suite")
    ```

3.  Run the build again. Bazel throws the following error:

    ```
    '//toolchain:clang_suite' does not have mandatory providers: 'ToolchainInfo'
    ```

    Bazel discovered that the `--crosstool_top` flag points to a rule that
    doesn't provide the necessary [`ToolchainInfo`](/rules/lib/providers/ToolchainInfo)
    provider. So you need to point `--crosstool_top` to a rule that does provide
    `ToolchainInfo` - that is the `cc_toolchain_suite` rule. In the
    `toolchain/BUILD` file, replace the empty filegroup with the following:

    ```python
    cc_toolchain_suite(
        name = "clang_suite",
        toolchains = {
            "k8": ":k8_toolchain",
        },
    )
    ```

    The `toolchains` attribute automatically maps the `--cpu` (and also
    `--compiler` when specified) values to  `cc_toolchain`. You have not yet
    defined any `cc_toolchain` targets and Bazel will complain about that
    shortly.

4.  Run the build again. Bazel throws the following error:

    ```
    Rule '//toolchain:k8_toolchain' does not exist
    ```

    Now you need to define `cc_toolchain` targets for every value in the
    `cc_toolchain_suite.toolchains` attribute. Add the following to the
    `toolchain/BUILD` file:

    ```python
    filegroup(name = "empty")

    cc_toolchain(
        name = "k8_toolchain",
        toolchain_identifier = "k8-toolchain",
        toolchain_config = ":k8_toolchain_config",
        all_files = ":empty",
        compiler_files = ":empty",
        dwp_files = ":empty",
        linker_files = ":empty",
        objcopy_files = ":empty",
        strip_files = ":empty",
        supports_param_files = 0,
    )
    ```

5.  Run the build again. Bazel throws the following error:

    ```
    Rule '//toolchain:k8_toolchain_config' does not exist
    ```

    Next, add a ":k8_toolchain_config" target to the `toolchain/BUILD` file:

    ```python
    filegroup(name = "k8_toolchain_config")
    ```

6.  Run the build again. Bazel throws the following error:

    ```
    '//toolchain:k8_toolchain_config' does not have mandatory providers:
    'CcToolchainConfigInfo'
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

    And replace the "k8_toolchain_config" filegroup with a declaration of a
    `cc_toolchain_config` rule:

    ```python
    cc_toolchain_config(name = "k8_toolchain_config")
    ```

7.  Run the build again. Bazel throws the following error:

    ```
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

8.  Run the build again. Bazel throws the following error:

     ```
     ..../BUILD:3:1: undeclared inclusion(s) in rule '//main:hello-world':
     this rule is missing dependency declarations for the following files included by 'main/hello-world.cc':
     '/usr/include/c++/9/ctime'
     '/usr/include/x86_64-linux-gnu/c++/9/bits/c++config.h'
     '/usr/include/x86_64-linux-gnu/c++/9/bits/os_defines.h'
     ....
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
     ```

9. Run the build command again, you will see an error like:

    ```
    /usr/bin/ld: bazel-out/k8-fastbuild/bin/main/_objs/hello-world/hello-world.o: in function `print_localtime()':
    hello-world.cc:(.text+0x68): undefined reference to `std::cout'
    ```
    The reason for this is because the linker is missing the C++ standard
    library and it can't find its symbols. There are many ways to solve this,
    such as using the `linkopts` attribute of `cc_binary`. Here it is solved by
    making sure that any target using the toolchain doesn't have to specify
    this flag.

    Copy the following code to `cc_toolchain_config.bzl`:

     ```python
      # NEW
      load("@bazel_tools//tools/build_defs/cc:action_names.bzl", "ACTION_NAMES")
      # NEW
      load(
          "@bazel_tools//tools/cpp:cc_toolchain_config_lib.bzl",
          "feature",
          "flag_group",
          "flag_set",
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
10. If you run `bazel build --config=clang_config //main:hello-world`, it should
    finally build.

## Review your work

In this tutorial you learned how to configure a basic C++ toolchain, but
toolchains are more powerful than this simple example.

The key take-aways are:
- You need to specify a `--crosstool_top` flag in the command line which should
  point to a `cc_toolchain_suite`
- You can create a shortcut for a particular configuration using the `.bazelrc`
  file
- The cc_toolchain_suite may list `cc_toolchains` for different CPUs and
  compilers. You can use command line flags like `--cpu` to differentiate.
- You have to let the toolchain know where the tools live. In this tutorial
  there is a simplified version where you access the tools from the system. If
  you are interested in a more self-contained approach, you can read about
  workspaces [here](/reference/be/workspace). Your tools could come from a
  different workspace and you would have to make their files available
  to the `cc_toolchain` with target dependencies on attributes, such as
  `compiler_files`. The `tool_paths` would need to be changed as well.
- You can create features to customize which flags should be passed to
  different actions, be it linking or any other type of action.

## Further reading

For more details, see
[C++ toolchain configuration](/docs/cc-toolchain-config-reference)
