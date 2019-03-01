---
layout: documentation
title: Configuring C++ toolchains
---

# Configuring C++ toolchains

* ToC
{:toc}

## Overview

This tutorial uses an example scenario to describe how to configure C++
toolchains for a project. It's based on an
[example C++ project](https://github.com/bazelbuild/examples/tree/master/cpp-tutorial/stage1)
that builds error-free using `gcc`, `clang`, and `msvc`.

In this tutorial, you will create a Starlark rule that provides additional
configuration for the `cc_toolchain` so that Bazel can build the application
with `emscripten`. The expected outcome is to run
`bazel build --config=asmjs //main:helloworld.js` on a Linux machine and build the
C++ application using [`emscripten`](https://kripken.github.io/emscripten-site/)
targeting [`asm.js`](http://asmjs.org/).

## Setting up the build environment

This tutorial assumes you are on Linux on which you have successfully built
C++ applications - in other words, we assume that appropriate tooling and
libraries have been installed.

Set up your build environment as follows:

1.  If you have not already done so,
   [download and install Bazel 0.23](../install-ubuntu.html) or later.

2.  Download the
    [example C++ project](https://github.com/bazelbuild/examples/tree/master/cpp-tutorial/stage1)
    from GitHub and place it in an empty directory on your local machine.


3.  Add the following `cc_binary` target to the `main/BUILD` file:

    ```
    cc_binary(
        name = "helloworld.js",
        srcs = ["hello-world.cc"],
    )
    ```

4.  Create a `.bazelrc` file at the root of the workspace directory with the
    following contents to enable the use of the `--config` flag:

    ```
    # Use our custom-configured c++ toolchain.

    build:asmjs --crosstool_top=//toolchain:emscripten

    # Use --cpu as a differentiator.

    build:asmjs --cpu=asmjs

    # Use the default Bazel C++ toolchain to build the tools used during the
    # build.

    build:asmjs --host_crosstool_top=@bazel_tools//tools/cpp:toolchain
    ```

In this example, we are using the `--cpu` flag as a differentiator, since
`emscripten` can target both `asmjs` and Web assembly. We are not configuring a
Web assembly toolchain, however. Since Bazel uses many internal tools written in
C++, such as process-wrapper, we are specifying a "sane" C++ toolchain for the
host platform.

## Configuring the C++ toolchain

To configure the C++ toolchain, repeatedly build the application and eliminate
each error one by one as described below.

**Note:** This tutorial assumes you're using Bazel 0.23 or later. If you're
using an older release of Bazel, look for the "Configuring CROSSTOOL" tutorial.

1.  Run the build with the following command:

    ```
    bazel build --config=asmjs //main:helloworld.js
    ```

    Because you specified `--crosstool_top=//toolchain:emscripten` in the
    `.bazelrc` file, Bazel throws the following error:

    ```
    No such package `toolchain`: BUILD file not found on package path.
    ```

    In the workspace directory, create the `toolchain` directory for the package
    and an empty `BUILD` file inside the `toolchain` directory.

2.  Run the build again. Because the `toolchain` package does not yet define the
    `emscripten` target, Bazel throws the following error:

    ```
    No such target '//toolchain:emscripten': target 'emscripten' not declared in
    package 'toolchain' defined by .../toolchain/BUILD
    ```

    In the `toolchain/BUILD` file, define an empty filegroup as follows:

    ```
    package(default_visibility = ['//visibility:public'])
    filegroup(name = "emscripten")
    ```

3.  Run the build again. Bazel throws the following error:

    ```
    '//toolchain:emscripten' does not have mandatory providers: 'ToolchainInfo'
    ```

    Bazel discovered that the `--crosstool_top` flag points to a rule that
    doesn't provide the necessary `ToolchainInfo` provider. So we need to point
    `--crosstool_top` to a rule that does provide `ToolchainInfo` - that is the
    `cc_toolchain_suite` rule. In the `toolchain/BUILD` file, replace the empty
    filegroup with the following:

    ```
    cc_toolchain_suite(
        name = "emscripten",
        toolchains = {
            "asmjs": ":asmjs_toolchain",
        },
    )
    ```

    The `toolchains` attribute automatically maps the `--cpu` (and also
    `--compiler` when specified) values to  `cc_toolchain`. You have not yet
    defined any `cc_toolchain` targets and Bazel will complain about that
    shortly.

4.  Run the build again. Bazel throws the following error:

    ```
    Rule '//toolchain:asmjs_toolchain' does not exist
    ```

    Now you need to define `cc_toolchain` targets for every value in the
    `cc_toolchain_suite.toolchains` attribute. This is where you specify the
    files that comprise the toolchain so that Bazel can set up sandboxing. Add
    the following to the `toolchain/BUILD` file:

    ```
    filegroup(name = "empty")

    cc_toolchain(
        name = "asmjs_toolchain",
        toolchain_identifier = "asmjs-toolchain",
        toolchain_config = ":asmjs_toolchain_config",
        all_files = ":empty",
        compiler_files = ":empty",
        cpu = "asmjs",
        dwp_files = ":empty",
        linker_files = ":empty",
        objcopy_files = ":empty",
        strip_files = ":empty",
        supports_param_files = 0,
    )
    ```

5.  Run the build again. Bazel throws the following error:

    ```
    Rule '//toolchain:asmjs-toolchain' does not exist
    ```

    Let's add a ":asmjs-toolchain-config" target to the `toolchain/BUILD` file:

    ```
    filegroup(name = "asmjs_toolchain_config")
    ```

6.  Run the build again. Bazel throws the following error:

    ```
    '//toolchain:asmjs_toolchain_config' does not have mandatory providers:
    'CcToolchainConfigInfo'
    ```

    `CcToolchainConfigInfo` is a provider that we use to configure our C++
    toolchains. We are going to create a Starlark rule that will provide
    `CcToolchainConfigInfo`. Create a `toolchains/cc_toolchain_config.bzl`
    file with the following content:
    ```
    def _impl(ctx):
        return cc_common.create_cc_toolchain_config_info(
            ctx = ctx,
            toolchain_identifier = "asmjs-toolchain",
            host_system_name = "i686-unknown-linux-gnu",
            target_system_name = "asmjs-unknown-emscripten",
            target_cpu = "asmjs",
            target_libc = "unknown",
            compiler = "emscripten",
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
    `CcToolchainConfigInfo`. Now let's declare a rule that will make use of
    the newly implemented `cc_toolchain_config` rule. Add a load statement to
    `toolchains/BUILD`:

    ```
    load(":cc_toolchain_config.bzl", "cc_toolchain_config")
    ```

    And replace the "asmjs_toolchain_config" filegroup with a declaration of a
    `cc_toolchain_config` rule:

    ```
    cc_toolchain_config(name = "asmjs_toolchain_config")
    ```

7.  Run the build again. Bazel throws the following error:

    ```
    .../BUILD:1:1: C++ compilation of rule '//:helloworld.js' failed (Exit 1)
    src/main/tools/linux-sandbox-pid1.cc:421:
    "execvp(toolchain/DUMMY_GCC_TOOL, 0x11f20e0)": No such file or directory
    Target //:helloworld.js failed to build`
    ```

    At this point, Bazel has enough information to attempt building the code but
    it still does not know what tools to use to complete the required build
    actions. We will modify our Starlark rule implementation to tell Bazel what
    tools to use. For that, we'll need the tool_path() constructor from
    [`@bazel_tools//tools/cpp:cc_toolchain_config_lib.bzl`](https://source.bazel.build/bazel/+/4eea5c62a566d21832c93e4c18ec559e75d5c1ce:tools/cpp/cc_toolchain_config_lib.bzl;l=400):

    ```
    # toolchain/cc_toolchain_config.bzl:
    load("@bazel_tools//tools/cpp:cc_toolchain_config_lib.bzl", "tool_path")

    def _impl(ctx):
        tool_paths = [
            tool_path(
                name = "gcc",
                path = "emcc.sh",
            ),
            tool_path(
                name = "ld",
                path = "emcc.sh",
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
        return cc_common.create_cc_toolchain_config_info(
            ctx = ctx,
            toolchain_identifier = "asmjs-toolchain",
            host_system_name = "i686-unknown-linux-gnu",
            target_system_name = "asmjs-unknown-emscripten",
            target_cpu = "asmjs",
            target_libc = "unknown",
            compiler = "emscripten",
            abi_version = "unknown",
            abi_libc_version = "unknown",
            tool_paths = tool_paths,
        )
    ```

    You may notice the `emcc.sh` wrapper script, which delegates to the external
    `emcc.py` file. Create the script in the `toolchain` package directory with
    the following contents and set its executable bit:

    ```
    #!/bin/bash
    set -euo pipefail
    python external/emscripten_toolchain/emcc.py "$@"
    ```

    Paths specified in the `tool_paths` list are relative to the package where
    the `cc_toolchain_config` target is specified.

    The `emcc.py` file does not yet exist in the workspace directory. To obtain
    it, you can either check the `emscripten` toolchain in with your project or
    pull it from its GitHub repository. This tutorial uses the latter approach.
    To pull the toolchain from the GitHub repository, add the following
    `http_archive` repository definitions to your `WORKSPACE` file:

    ```
    load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

    http_archive(
      name = 'emscripten_toolchain',
      url = 'https://github.com/kripken/emscripten/archive/1.37.22.tar.gz',
      build_file = '//:emscripten-toolchain.BUILD',
      strip_prefix = "emscripten-1.37.22",
    )

    http_archive(
      name = 'emscripten_clang',
      url = 'https://s3.amazonaws.com/mozilla-games/emscripten/packages/llvm/tag/linux_64bit/emscripten-llvm-e1.37.22.tar.gz',
      build_file = '//:emscripten-clang.BUILD',
      strip_prefix = "emscripten-llvm-e1.37.22",
    )
    ```

    In the workspace directory root, create the `emscripten-toolchain.BUILD` and
    `emscripten-clang.BUILD` files that expose these repositories as filegroups
    and establish their visibility across the build.

    First create the `emscripten-toolchain.BUILD` file with the following
    contents:

    ```
    package(default_visibility = ['//visibility:public'])

    filegroup(
      name = "all",
      srcs = glob(["**/*"]),
    )
    ```

    Next, create the `emscripten-clang.BUILD` file with the following contents:

    ```
    package(default_visibility = ['//visibility:public'])`

    filegroup(
      name = "all",
      srcs = glob(["**/*"]),
    )
    ```

    You may notice that the targets simply parse all of the files contained in
    the archives pulled by the `http_archive` repository rules. In a real
    world scenario, you would likely want to be more selective and granular by
    only parsing the files needed by the build and splitting them by action,
    such as compilation, linking, and so on. For the sake of simplicity, this
    tutorial omits this step.

8.  Run the build again. Bazel throws the following error:

    ```
    "execvp(toolchain/emcc.sh, 0x12bd0e0)": No such file or directory
    ```

    You now need to make Bazel aware of the artifacts you added in the previous
    step. In particular, the `emcc.sh` script must also be explicitly listed as
    a dependency of the corresponding `cc_toolchain` rule. Modify the
    `toolchain/BUILD` file to look as follows:

    ```
    package(default_visibility = ["//visibility:public"])

    load(":cc_toolchain_config.bzl", "cc_toolchain_config")

    cc_toolchain_config(name = "asmjs_toolchain_config")

    cc_toolchain_suite(
        name = "emscripten",
        toolchains = {
            "asmjs": ":asmjs_toolchain",
        },
    )

    filegroup(
        name = "all",
        srcs = [
            "emcc.sh",
            "@emscripten_clang//:all",
            "@emscripten_toolchain//:all",
        ],
    )

    cc_toolchain(
       name = "asmjs_toolchain",
       toolchain_identifier = "asmjs-toolchain",
       toolchain_config = ":asmjs_toolchain_config",
       all_files = ":all",
       compiler_files = ":all",
       cpu = "asmjs",
       dwp_files = ":empty",
       linker_files = ":all",
       objcopy_files = ":empty",
       strip_files = ":empty",
       supports_param_files = 0,
    )
    ```

    Congratulations! You are now using the `emscripten` toolchain to build your
    C++ sample code.  The next steps are optional but are included for
    completeness.


9.  (Optional) Run the build again. Bazel throws the following error:

     ```
     ERROR: .../BUILD:1:1: C++ compilation of rule '//:helloworld.js' failed (Exit 1)
     ```

     The next step is to make the toolchain deterministic and hermetic - that
     is, limit it to only touch files it's supposed to touch and ensure it
     doesn't write temporary data outside the sandbox.

     You also need to ensure the toolchain does not assume the existence of your
     home directory with its configuration files and that it does not depend on
     unspecified environment variables.

     For our example project, make the following modifications to the
     `toolchain/BUILD` file:

     ```
     filegroup(
       name = "all",
       srcs = [
         "emcc.sh",
         "@emscripten_toolchain//:all",
         "@emscripten_clang//:all",
         ":emscripten_cache_content"
         ],
      )

     filegroup(
       name = "emscripten_cache_content",
       srcs = glob(["emscripten_cache/**/*"]),
     )
     ```

     Since `emscripten` caches standard library files, you can save time by not
     compiling `stdlib` for every action and also prevent it from storing
     temporary data in random place, check in the precompiled bitcode files into
     the `toolchain/emscript_cache directory`. You can create them by calling
     the following from the `emscripten_clang` repository (or let `emscripten`
     create them in `~/.emscripten_cache`):

     ```
     python embuilder.py build dlmalloc libcxx libc gl libcxxabi libcxx_noexcept wasm-libc
     ```

     Copy those files to `toolchain/emscripten_cache`.
     Also update the `emcc.sh` script to look as follows:

     ```
     #!/bin/bash

     set -euo pipefail

     export LLVM_ROOT='external/emscripten_clang'
     export EMSCRIPTEN_NATIVE_OPTIMIZER='external/emscripten_clang/optimizer'
     export BINARYEN_ROOT='external/emscripten_clang/'
     export NODE_JS=''
     export EMSCRIPTEN_ROOT='external/emscripten_toolchain'
     export SPIDERMONKEY_ENGINE=''
     export EM_EXCLUSIVE_CACHE_ACCESS=1
     export EMCC_SKIP_SANITY_CHECK=1
     export EMCC_WASM_BACKEND=0

     mkdir -p "tmp/emscripten_cache"

     export EM_CACHE="tmp/emscripten_cache"
     export TEMP_DIR="tmp"

     # Prepare the cache content so emscripten doesn't keep rebuilding it
     cp -r toolchain/emscripten_cache/* tmp/emscripten_cache

     # Run emscripten to compile and link
     python external/emscripten_toolchain/emcc.py "$@"

     # Remove the first line of .d file
     find . -name "*.d" -exec sed -i '2d' {} \;
     ```

     Bazel can now properly compile the sample C++ code in `hello-world.cc`.


10.  (Optional) Run the build again. Bazel throws the following error:

     ```
     ..../BUILD:1:1: undeclared inclusion(s) in rule '//:helloworld.js':
     this rule is missing dependency declarations for the following files included by 'helloworld.cc':
     '.../external/emscripten_toolchain/system/include/libcxx/stdio.h'
     '.../external/emscripten_toolchain/system/include/libcxx/__config'
     '.../external/emscripten_toolchain/system/include/libc/stdio.h'
     '.../external/emscripten_toolchain/system/include/libc/features.h'
     '.../external/emscripten_toolchain/system/include/libc/bits/alltypes.h'
     ```

     At this point you have successfully compiled the example C++ code. The
     error above occurs because Bazel uses a `.d` file produced by the compiler
     to verify that all includes have been declared and to prune action inputs.

     In the `.d` file, Bazel discovered that our source code references system
     headers that have not been explicitly declared in the `BUILD` file. This in
     and of itself is not a problem and you can easily fix this by adding the
     target folders as `-isystem` directories. For this, you'll need to add
     a [`feature`](https://source.bazel.build/bazel/+/4eea5c62a566d21832c93e4c18ec559e75d5c1ce:tools/cpp/cc_toolchain_config_lib.bzl;l=336) to the `CcToolchainConfigInfo`.
     Modify `toolchain/cc_toolchain_config.bzl` to look like this:
     ```
     load("@bazel_tools//tools/cpp:cc_toolchain_config_lib.bzl",
          "feature",
          "flag_group",
          "flag_set",
          "tool_path")
     load("@bazel_tools//tools/build_defs/cc:action_names.bzl", "ACTION_NAMES")

     def _impl(ctx):
         tool_paths = [
             tool_path(
                 name = "gcc",
                 path = "emcc.sh",
             ),
             tool_path(
                 name = "ld",
                 path = "emcc.sh",
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
         toolchain_include_directories_feature = feature(
             name = "toolchain_include_directories",
             enabled = True,
             flag_sets = [
                 flag_set(
                     actions = [
                         "ACTION_NAMES.assemble",
                         "ACTION_NAMES.preprocess_assemble",
                         "ACTION_NAMES.linkstamp_compile",
                         "ACTION_NAMES.c_compile",
                         "ACTION_NAMES.cpp_compile",
                         "ACTION_NAMES.cpp_header_parsing",
                         "ACTION_NAMES.cpp_module_compile",
                         "ACTION_NAMES.cpp_module_codegen",
                         "ACTION_NAMES.lto_backend",
                         "ACTION_NAMES.clif_match",
                     ],
                     flag_groups = [
                         flag_group(
                             flags = [
                                 "-isystem",
                                 "external/emscripten_toolchain/system/include/libcxx",
                                 "-isystem",
                                 "external/emscripten_toolchain/system/include/libc",
                             ],
                         ),
                     ],
                 ),
             ],
         )

         return cc_common.create_cc_toolchain_config_info(
             ctx = ctx,
             toolchain_identifier = "asmjs-toolchain",
             host_system_name = "i686-unknown-linux-gnu",
             target_system_name = "asmjs-unknown-emscripten",
             target_cpu = "asmjs",
             target_libc = "unknown",
             compiler = "emscripten",
             abi_version = "unknown",
             abi_libc_version = "unknown",
             tool_paths = tool_paths,
             features = [toolchain_include_directories_feature],
         )

     cc_toolchain_config = rule(
         implementation = _impl,
         attrs = {},
         provides = [CcToolchainConfigInfo],
     )
     ```

11.  (Optional) Run the build again. With this final change, the build now
     completes error-free.
