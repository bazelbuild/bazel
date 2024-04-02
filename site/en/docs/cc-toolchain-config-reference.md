Project: /_project.yaml
Book: /_book.yaml

# C++ Toolchain Configuration

{% include "_buttons.html" %}

## Overview {:#overview}

To invoke the compiler with the right options, Bazel needs some knowledge about
the compiler internals, such as include directories and important flags.
In other words, Bazel needs a simplified model of the compiler to understand its
workings.

Bazel needs to know the following:

* Whether the compiler supports thinLTO, modules, dynamic linking, or PIC
  (position independent code).
* Paths to the required tools such as gcc, ld, ar, objcopy, and so on.
* The built-in system include directories. Bazel needs these to validate that
  all headers that were included in the source file were properly declared in
  the `BUILD` file.
* The default sysroot.
* Which flags to use for compilation, linking, archiving.
* Which flags to use for the supported compilation modes (opt, dbg, fastbuild).
* Make variables specifically required by the compiler.

If the compiler has support for multiple architectures, Bazel needs to configure
them separately.

[`CcToolchainConfigInfo`](/rules/lib/providers/CcToolchainConfigInfo) is a provider that provides the necessary level of
granularity for configuring the behavior of Bazel's C++ rules. By default,
Bazel automatically configures `CcToolchainConfigInfo` for your build, but you
have the option to configure it manually. For that, you need a Starlark rule
that provides the `CcToolchainConfigInfo` and you need to point the
[`toolchain_config`](/reference/be/c-cpp#cc_toolchain.toolchain_config) attribute of the
[`cc_toolchain`](/reference/be/c-cpp#cc_toolchain) to your rule.
You can create the `CcToolchainConfigInfo` by calling
[`cc_common.create_cc_toolchain_config_info()`](/rules/lib/toplevel/cc_common#create_cc_toolchain_config_info).
You can find Starlark constructors for all structs you'll need in the process in
[`@rules_cc//cc:cc_toolchain_config_lib.bzl`](https://github.com/bazelbuild/rules_cc/blob/master/cc/cc_toolchain_config_lib.bzl){: .external}.


When a C++ target enters the analysis phase, Bazel selects the appropriate
`cc_toolchain` target based on the `BUILD` file, and obtains the
`CcToolchainConfigInfo` provider from the target specified in the
`cc_toolchain.toolchain_config` attribute. The `cc_toolchain` target
passes this information to the C++ target through a `CcToolchainProvider`.

For example, a compile or link action, instantiated by a rule such as
`cc_binary` or `cc_library`, needs the following information:

*   The compiler or linker to use
*   Command-line flags for the compiler/linker
*   Configuration flags passed through the `--copt/--linkopt` options
*   Environment variables
*   Artifacts needed in the sandbox in which the action executes

All of the above information except the artifacts required in the sandbox is
specified in the Starlark target that the `cc_toolchain` points to.

The artifacts to be shipped to the sandbox are declared in the `cc_toolchain`
target. For example, with the `cc_toolchain.linker_files` attribute you can
specify the linker binary and toolchain libraries to ship into the sandbox.

## Toolchain selection {:#toolchain-selection}

The toolchain selection logic operates as follows:

1.  User specifies a `cc_toolchain_suite` target in the `BUILD` file and points
    Bazel to the target using the
    [`--crosstool_top` option](/docs/user-manual#flag--crosstool_top).

2.  The `cc_toolchain_suite` target references multiple toolchains. The
    values of the `--cpu` and `--compiler` flags determine which of those
    toolchains is selected, either based only on the `--cpu` flag value, or
    based on a joint `--cpu | --compiler` value. The selection process is as
    follows:

  * If the `--compiler` option is specified, Bazel selects the
        corresponding entry from the `cc_toolchain_suite.toolchains`
        attribute with `--cpu | --compiler`. If Bazel does not find
        a corresponding entry, it throws an error.

  * If the `--compiler` option is not specified, Bazel selects
    the corresponding entry from the `cc_toolchain_suite.toolchains`
    attribute with just `--cpu`.

  * If no flags are specified, Bazel inspects the host system and selects a
    `--cpu` value based on its findings. See the
    [inspection mechanism code](https://source.bazel.build/bazel/+/1b73bc37e184e71651eb631223dcce321ba16211:src/main/java/com/google/devtools/build/lib/analysis/config/AutoCpuConverter.java).

Once a toolchain has been selected, corresponding `feature` and `action_config`
objects in the Starlark rule govern the configuration of the build (that is,
items described later). These messages allow the implementation of
fully fledged C++ features in Bazel without modifying the
Bazel binary. C++ rules support multiple unique actions documented in detail
[in the Bazel source code](https://source.bazel.build/bazel/+/4f547a7ea86df80e4c76145ffdbb0c8b75ba3afa:tools/build_defs/cc/action_names.bzl).

## Features {:#features}

A feature is an entity that requires command-line flags, actions,
constraints on the execution environment, or dependency alterations. A feature
can be something as simple as allowing `BUILD` files to select configurations of
flags, such as `treat_warnings_as_errors`, or interact with the C++ rules and
include new compile actions and inputs to the compilation, such as
`header_modules` or `thin_lto`.

Ideally, `CcToolchainConfigInfo` contains a list of features, where each
feature consists of one or more flag groups, each defining a list of flags
that apply to specific Bazel actions.

A feature is specified by name, which allows full decoupling of the Starlark
rule configuration from Bazel releases. In other words, a Bazel release does not
affect the behavior of `CcToolchainConfigInfo` configurations as long as those
configurations do not require the use of new features.

A feature is enabled in one of the following ways:

*  The feature's `enabled` field is set to `true`.
*  Bazel or the rule owner explicitly enable it.
*  The user enables it through the `--feature` Bazel option or `features` rule
   attribute.

Features can have interdependencies, depend on command line flags, `BUILD` file
settings, and other variables.

### Feature relationships {:#feature-relationships}

Dependencies are typically managed directly with Bazel, which simply enforces
the requirements and manages conflicts intrinsic to the nature of the features
defined in the build. The toolchain specification allows for more granular
constraints for use directly within the Starlark rule that govern feature
support and expansion. These are:

<table>
  <col width="300">
  <col width="600">
  <tr>
   <td><strong>Constraint</strong>
   </td>
   <td><strong>Description</strong>
   </td>
  </tr>
  <tr>
   <td><pre>requires = [
   feature_set (features = [
       'feature-name-1',
       'feature-name-2'
   ]),
]</pre>
   </td>
   <td>Feature-level. The feature is supported only if the specified required
       features are enabled. For example, when a feature is only supported in
       certain build modes (<code>opt</code>, <code>dbg</code>, or
       <code>fastbuild</code>). If `requires` contains multiple `feature_set`s
       the feature is supported if any of the `feature_set`s is satisfied
       (when all specified features are enabled).
   </td>
  </tr>
  <tr>
   <td><pre>implies = ['feature']</pre>
   </td>
   <td><p>Feature-level. This feature implies the specified feature(s).
       Enabling a feature also implicitly enables all features implied by it
       (that is, it functions recursively).</p>
       <p>Also provides the ability to factor common subsets of functionality out of
       a set of features, such as the common parts of sanitizers. Implied
       features cannot be disabled.</p>
   </td>
  </tr>
  <tr>
   <td><pre>provides = ['feature']</pre>
   </td>
   <td><p>Feature-level. Indicates that this feature is one of several mutually
       exclusive alternate features. For example, all of the sanitizers could
       specify <code>provides = ["sanitizer"]</code>.</p>
       <p>This improves error handling by listing the alternatives if the user asks
       for two or more mutually exclusive features at once.</p>
   </td>
  </tr>
  <tr>
   <td><pre>with_features = [
  with_feature_set(
    features = ['feature-1'],
    not_features = ['feature-2'],
  ),
]</pre>
   </td>
   <td>Flag set-level. A feature can specify multiple flag sets with multiple.
     When <code>with_features</code> is specified, the flag set will only expand
     to the build command if there is at least one <code>with_feature_set</code>
     for which all of the features in the specified <code>features</code> set
     are enabled, and all the features specified in <code>not_features</code>
     set are disabled.
     If <code>with_features</code> is not specified, the flag set will be
     applied unconditionally for every action specified.
   </td>
  </tr>
</table>

## Actions {:#actions}

Actions provide the flexibility to modify the circumstances under
which an action executes without assuming how the action will be run. An
`action_config` specifies the tool binary that an action invokes, while a
`feature` specifies the configuration (flags) that determine how that tool
behaves when the action is invoked.

[Features](#features) reference actions to signal which Bazel actions
they affect since actions can modify the Bazel action graph. The
`CcToolchainConfigInfo` provider contains actions that have flags and tools
associated with them, such as `c++-compile`. Flags are assigned to each action
by associating them with a feature.

Each action name represents a single type of action performed by Bazel, such as
compiling or linking. There is, however, a many-to-one relationship between
actions and Bazel action types, where a Bazel action type refers to a Java class
that implements an action (such as `CppCompileAction`). In particular, the
"assembler actions" and "compiler actions" in the table below are
`CppCompileAction`, while the link actions are `CppLinkAction`.

### Assembler actions {:#assembler-actions}

<table>
  <col width="300">
  <col width="600">
  <tr>
   <td><strong>Action</strong>
   </td>
   <td><strong>Description</strong>
   </td>
  </tr>
  <tr>
   <td><code>preprocess-assemble</code>
   </td>
   <td>Assemble with preprocessing. Typically for <code>.S</code> files.
   </td>
  </tr>
  <tr>
   <td><code>assemble</code>
   </td>
   <td>Assemble without preprocessing. Typically for <code>.s</code> files.
   </td>
  </tr>
</table>

### Compiler actions {:#compiler-actions}

<table>
  <col width="300">
  <col width="600">
  <tr>
   <td><strong>Action</strong>
   </td>
   <td><strong>Description</strong>
   </td>
  </tr>
  <tr>
   <td><code>cc-flags-make-variable</code>
   </td>
   <td>Propagates <code>CC_FLAGS</code> to genrules.
   </td>
  </tr>
  <tr>
   <td><code>c-compile</code>
   </td>
   <td>Compile as C.
   </td>
  </tr>
  <tr>
   <td><code>c++-compile</code>
   </td>
   <td>Compile as C++.
   </td>
  </tr>
  <tr>
   <td><code>c++-header-parsing</code>
   </td>
   <td>Run the compiler's parser on a header file to ensure that the header is
     self-contained, as it will otherwise produce compilation errors. Applies
     only to toolchains that support modules.
   </td>
  </tr>
</table>

### Link actions {:#link-actions}

<table>
  <col width="300">
  <col width="600">
  <tr>
   <td><strong>Action</strong>
   </td>
   <td><strong>Description</strong>
   </td>
  </tr>
  <tr>
   <td><code>c++-link-dynamic-library</code>
   </td>
   <td>Link a shared library containing all of its dependencies.
   </td>
  </tr>
  <tr>
   <td><code>c++-link-nodeps-dynamic-library</code>
   </td>
   <td>Link a shared library only containing <code>cc_library</code> sources.
   </td>
  </tr>
  <tr>
   <td><code>c++-link-executable</code>
   </td>
   <td>Link a final ready-to-run library.
   </td>
  </tr>
</table>

### AR actions {:#ar-actions}

AR actions assemble object files into archive libraries (`.a` files) via `ar`
and encode some semantics into the name.

<table>
  <col width="300">
  <col width="600">
  <tr>
   <td><strong>Action</strong>
   </td>
   <td><strong>Description</strong>
   </td>
  </tr>
  <tr>
   <td><code>c++-link-static-library</code>
   </td>
   <td>Create a static library (archive).
   </td>
  </tr>
</table>

### LTO actions {:#lto-actions}

<table>
  <col width="300">
  <col width="600">
  <tr>
   <td><strong>Action</strong>
   </td>
   <td><strong>Description</strong>
   </td>
  </tr>
  <tr>
   <td><code>lto-backend</code>
   </td>
   <td>ThinLTO action compiling bitcodes into native objects.
   </td>
  </tr>
  <tr>
   <td><code>lto-index</code>
   </td>
   <td>ThinLTO action generating global index.
   </td>
  </tr>
</table>

## Using action_config {:#using-action-config}

The `action_config` is a Starlark struct that describes a Bazel
action by specifying the tool (binary) to invoke during the action and sets of
flags, defined by features. These flags apply constraints to the action's
execution.

The `action_config()` constructor has the following parameters:

<table>
  <col width="300">
  <col width="600">
  <tr>
   <td><strong>Attribute</strong>
   </td>
   <td><strong>Description</strong>
   </td>
  </tr>
  <tr>
   <td><code>action_name</code>
   </td>
    <td>The Bazel action to which this action corresponds.
        Bazel uses this attribute to discover per-action tool and execution
        requirements.
   </td>
  </tr>
  <tr>
   <td><code>tools</code>
   </td>
   <td>The executable to invoke. The tool applied to the action will be the
       first tool in the list with a feature set that matches the feature
       configuration. Default value must be provided.
   </td>
  </tr>
  <tr>
   <td><code>flag_sets</code>
   </td>
   <td>A list of flags that applies to a group of actions. Same as for a
       feature.
   </td>
  </tr>
  <tr>
   <td><code>env_sets</code>
   </td>
   <td>A list of environment constraints that applies to a group of actions.
       Same as for a feature.
   </td>
  </tr>
</table>

An `action_config` can require and imply other features and
<code>action_config</code>s as dictated by the
[feature relationships](#feature-relationships) described earlier. This behavior
is similar to that of a feature.

The last two attributes are redundant against the corresponding attributes on
features and are included because some Bazel actions require certain flags or
environment variables and the goal is to avoid unnecessary `action_config`+`feature`
pairs. Typically, sharing a single feature across multiple `action_config`s is
preferred.

You can not define more than one `action_config` with the same `action_name`
within the same toolchain. This prevents ambiguity in tool paths
and enforces the intention behind `action_config` - that an action's properties
are clearly described in a single place in the toolchain.

### Using tool constructor {:#using-tool-constructor}

An`action_config` can specify a set of tools via its `tools` parameter.
The `tool()` constructor takes in the following parameters:


<table>
  <col width="300">
  <col width="600">
  <tr>
   <td><strong>Field</strong>
   </td>
   <td><strong>Description</strong>
   </td>
  </tr>
  <tr>
   <td><code>path</code>
   </td>
   <td>Path to the tool in question (relative to the current location).
   </td>
  </tr>
  <tr>
   <td><code>with_features</code>
   </td>
   <td>A list of feature sets out of which at least one must be satisfied
       for this tool to apply.
   </td>
  </tr>
</table>

For a given `action_config`, only a single `tool` applies
its tool path and execution requirements to the Bazel action. A tool is selected
by iterating through the `tools` attribute on an `action_config` until a tool
with a `with_feature` set matching the feature configuration is found
(see [Feature relationships](#feature-relationships) earlier on this page
for more information). You should end your tool lists with a default
tool that corresponds to an empty feature configuration.

### Example usage {:#example-usage}

Features and actions can be used together to implement Bazel actions
with diverse cross-platform semantics. For example, debug symbol generation on
macOS requires generating symbols in the compile action, then invoking a
specialized tool during the link action to create  compressed dsym archive, and
then decompressing that archive to produce the application bundle and `.plist`
files consumable by Xcode.

With Bazel, this process can instead be implemented as follows, with
`unbundle-debuginfo` being a Bazel action:

    load("@rules_cc//cc:defs.bzl", "ACTION_NAMES")

    action_configs = [
        action_config (
            action_name = ACTION_NAMES.cpp_link_executable,
            tools = [
                tool(
                    with_features = [
                        with_feature(features=["generate-debug-symbols"]),
                    ],
                    path = "toolchain/mac/ld-with-dsym-packaging",
                ),
                tool (path = "toolchain/mac/ld"),
            ],
        ),
    ]

    features = [
        feature(
            name = "generate-debug-symbols",
            flag_sets = [
                flag_set (
                    actions = [
                        ACTION_NAMES.c_compile,
                        ACTION_NAMES.cpp_compile
                    ],
                    flag_groups = [
                        flag_group(
                            flags = ["-g"],
                        ),
                    ],
                )
            ],
            implies = ["unbundle-debuginfo"],
       ),
    ]


This same feature can be implemented entirely differently for Linux, which uses
`fission`, or for Windows, which produces `.pdb` files. For example, the
implementation for `fission`-based debug symbol generation might look as
follows:

    load("@rules_cc//cc:defs.bzl", "ACTION_NAMES")

    action_configs = [
        action_config (
            name = ACTION_NAMES.cpp_compile,
            tools = [
                tool(
                    path = "toolchain/bin/gcc",
                ),
            ],
        ),
    ]

    features = [
        feature (
            name = "generate-debug-symbols",
            requires = [with_feature_set(features = ["dbg"])],
            flag_sets = [
                flag_set(
                    actions = [ACTION_NAMES.cpp_compile],
                    flag_groups = [
                        flag_group(
                            flags = ["-gsplit-dwarf"],
                        ),
                    ],
                ),
                flag_set(
                    actions = [ACTION_NAMES.cpp_link_executable],
                    flag_groups = [
                        flag_group(
                            flags = ["-Wl", "--gdb-index"],
                        ),
                    ],
                ),
          ],
        ),
    ]


### Flag groups {:#flag-groups}

`CcToolchainConfigInfo` allows you to bundle flags into groups that serve a
specific purpose. You can specify a flag within using pre-defined variables
within the flag value, which the compiler expands when adding the flag to the
build command. For example:

    flag_group (
        flags = ["%{output_execpath}"],
    )


In this case, the contents of the flag will be replaced by the output file path
of the action.

Flag groups are expanded to the build command in the order in which they appear
in the list, top-to-bottom, left-to-right.

For flags that need to repeat with different values when added to the build
command, the flag group can iterate variables of type `list`. For example, the
variable `include_path` of type `list`:

    flag_group (
        iterate_over = "include_paths",
        flags = ["-I%{include_paths}"],
    )

expands to `-I<path>` for each path element in the `include_paths` list. All
flags (or `flag_group`s) in the body of a flag group declaration are expanded as
a unit. For example:

    flag_group (
        iterate_over = "include_paths",
        flags = ["-I", "%{include_paths}"],
    )

expands to `-I <path>` for each path element in the `include_paths` list.

A variable can repeat multiple times. For example:

    flag_group (
        iterate_over = "include_paths",
        flags = ["-iprefix=%{include_paths}", "-isystem=%{include_paths}"],
    )

expands to:

    -iprefix=<inc0> -isystem=<inc0> -iprefix=<inc1> -isystem=<inc1>

Variables can correspond to structures accessible using dot-notation. For
example:

    flag_group (
        flags = ["-l%{libraries_to_link.name}"],
    )

Structures can be nested and may also contain sequences. To prevent name clashes
and to be explicit, you must specify the full path through the fields. For
example:

    flag_group (
        iterate_over = "libraries_to_link",
        flag_groups = [
            flag_group (
                iterate_over = "libraries_to_link.shared_libraries",
                flags = ["-l%{libraries_to_link.shared_libraries.name}"],
            ),
        ],
    )


### Conditional expansion {:#conditional-expansion}

Flag groups support conditional expansion based on the presence of a particular
variable or its field using the `expand_if_available`, `expand_if_not_available`,
`expand_if_true`, `expand_if_false`, or `expand_if_equal` attributes. For example:


    flag_group (
        iterate_over = "libraries_to_link",
        flag_groups = [
            flag_group (
                iterate_over = "libraries_to_link.shared_libraries",
                flag_groups = [
                    flag_group (
                        expand_if_available = "libraries_to_link.shared_libraries.is_whole_archive",
                        flags = ["--whole_archive"],
                    ),
                    flag_group (
                        flags = ["-l%{libraries_to_link.shared_libraries.name}"],
                    ),
                    flag_group (
                        expand_if_available = "libraries_to_link.shared_libraries.is_whole_archive",
                        flags = ["--no_whole_archive"],
                    ),
                ],
            ),
        ],
    )

Note: The `--whole_archive` and `--no_whole_archive` options are added to
the build command only when a currently iterated library has an
`is_whole_archive` field.

## CcToolchainConfigInfo reference {:#cctoolchainconfiginfo-reference}

This section provides a reference of build variables, features, and other
information required to successfully configure C++ rules.

### CcToolchainConfigInfo build variables {:#cctoolchainconfiginfo-build-variables}

The following is a reference of `CcToolchainConfigInfo` build variables.

Note: The **Action** column indicates the relevant action type, if applicable.

<table>
  <tr>
   <td><strong>Variable</strong>
   </td>
   <td><strong>Action</strong>
   </td>
   <td><strong>Description</strong>
   </td>
  </tr>
  <tr>
   <td><strong><code>source_file</code></strong>
   </td>
   <td>compile</td>
   <td>Source file to compile.
   </td>
  </tr>
  <tr>
   <td><strong><code>input_file</code></strong>
   </td>
   <td>strip</td>
   <td>Artifact to strip.
   </td>
  </tr>
  <tr>
   <td><strong><code>output_file</code></strong>
   </td>
   <td>compile</td>
   <td>Compilation output.
   </td>
  </tr>
  <tr>
   <td><strong><code>output_assembly_file</code></strong>
   </td>
   <td>compile</td>
   <td>Emitted assembly file. Applies only when the
       <code>compile</code> action emits assembly text, typically when using the
       <code>--save_temps</code> flag. The contents are the same as for
       <code>output_file</code>.
   </td>
  </tr>
  <tr>
   <td><strong><code>output_preprocess_file</code></strong>
   </td>
   <td>compile</td>
   <td>Preprocessed output. Applies only to compile
       actions that only preprocess the source files, typically when using the
     <code>--save_temps</code> flag. The contents are the same as for
     <code>output_file</code>.
   </td>
  </tr>
  <tr>
   <td><strong><code>includes</code></strong>
   </td>
   <td>compile</td>
   <td>Sequence of files the compiler must
       unconditionally include in the compiled source.
   </td>
  </tr>
  <tr>
   <td><strong><code>include_paths</code></strong>
   </td>
   <td>compile</td>
   <td>Sequence directories in which the compiler
       searches for headers included using <code>#include&lt;foo.h&gt;</code>
       and <code>#include "foo.h"</code>.
   </td>
  </tr>
  <tr>
   <td><strong><code>quote_include_paths</code></strong>
   </td>
   <td>compile</td>
   <td>Sequence of <code>-iquote</code> includes -
       directories in which the compiler searches for headers included using
       <code>#include "foo.h"</code>.
   </td>
  </tr>
  <tr>
   <td><strong><code>system_include_paths</code></strong>
   </td>
   <td>compile</td>
   <td>Sequence of <code>-isystem</code> includes -
       directories in which the compiler searches for headers included using
       <code>#include &lt;foo.h&gt;</code>.
   </td>
  </tr>
  <tr>
   <td><strong><code>dependency_file</code></strong>
   </td>
   <td>compile</td>
   <td>The <code>.d</code> dependency file generated by the compiler.
   </td>
  </tr>
  <tr>
   <td><strong><code>preprocessor_defines</code></strong>
   </td>
   <td>compile</td>
   <td>Sequence of <code>defines</code>, such as <code>--DDEBUG</code>.
   </td>
  </tr>
  <tr>
   <td><strong><code>pic</code></strong>
   </td>
   <td>compile</td>
   <td>Compiles the output as position-independent code.
   </td>
  </tr>
  <tr>
   <td><strong><code>gcov_gcno_file</code></strong>
   </td>
   <td>compile</td>
   <td>The <code>gcov</code> coverage file.
   </td>
  </tr>
  <tr>
   <td><strong><code>per_object_debug_info_file</code></strong>
   </td>
   <td>compile</td>
   <td>The per-object debug info (<code>.dwp</code>) file.
   </td>
  </tr>
  <tr>
   <td><strong><code>stripotps</code></strong>
   </td>
   <td>strip</td>
   <td>Sequence of <code>stripopts</code>.
   </td>
  </tr>
  <tr>
   <td><strong><code>legacy_compile_flags</code></strong>
   </td>
   <td>compile</td>
   <td>Sequence of flags from legacy
       <code>CROSSTOOL</code> fields such as <code>compiler_flag</code>,
       <code>optional_compiler_flag</code>, <code>cxx_flag</code>, and
       <code>optional_cxx_flag</code>.
   </td>
  </tr>
  <tr>
   <td><strong><code>user_compile_flags</code></strong>
   </td>
   <td>compile</td>
   <td>Sequence of flags from either the
       <code>copt</code> rule attribute or the <code>--copt</code>,
       <code>--cxxopt</code>, and <code>--conlyopt</code> flags.
   </td>
  </tr>
  <tr>
   <td><strong><code>unfiltered_compile_flags</code></strong>
   </td>
   <td>compile</td>
   <td>Sequence of flags from the
     <code>unfiltered_cxx_flag</code> legacy <code>CROSSTOOL</code> field or the
       <code>unfiltered_compile_flags</code> feature. These are not filtered by
       the <code>nocopts</code> rule attribute.
   </td>
  </tr>
  <tr>
   <td><strong><code>sysroot</code></strong>
   </td>
   <td></td>
   <td>The <code>sysroot</code>.
   </td>
  </tr>
  <tr>
   <td><strong><code>runtime_library_search_directories</code></strong>
   </td>
   <td>link</td>
   <td>Entries in the linker runtime search path (usually
       set with the <code>-rpath</code> flag).
   </td>
  </tr>
  <tr>
   <td><strong><code>library_search_directories</code></strong>
   </td>
   <td>link</td>
   <td>Entries in the linker search path (usually set with
       the <code>-L</code> flag).
   </td>
  </tr>
  <tr>
   <td><strong><code>libraries_to_link</code></strong>
   </td>
   <td>link</td>
   <td>Flags providing files to link as inputs in the linker invocation.
   </td>
  </tr>
  <tr>
   <td><strong><code>def_file_path</code></strong>
   </td>
   <td>link</td>
   <td>Location of def file used on Windows with MSVC.
   </td>
  </tr>
  <tr>
   <td><strong><code>linker_param_file</code></strong>
   </td>
   <td>link</td>
   <td>Location of linker param file created by bazel to
       overcome command line length limit.
   </td>
  </tr>
  <tr>
   <td><strong><code>output_execpath</code></strong>
   </td>
   <td>link</td>
   <td>Execpath of the output of the linker.
   </td>
  </tr>
  <tr>
   <td><strong><code>generate_interface_library</code></strong>
   </td>
   <td>link</td>
   <td><code>"yes"</code> or <code>"no"</code> depending on whether interface library should
       be generated.
   </td>
  </tr>
  <tr>
   <td><strong><code>interface_library_builder_path</code></strong>
   </td>
   <td>link</td>
   <td>Path to the interface library builder tool.
   </td>
  </tr>
  <tr>
   <td><strong><code>interface_library_input_path</code></strong>
   </td>
   <td>link</td>
   <td>Input for the interface library <code>ifso</code> builder tool.
   </td>
  </tr>
  <tr>
   <td><strong><code>interface_library_output_path</code></strong>
   </td>
   <td>link</td>
   <td>Path where to generate interface library using the <code>ifso</code> builder tool.
   </td>
  </tr>
  <tr>
   <td><strong><code>legacy_link_flags</code></strong>
   </td>
   <td>link</td>
   <td>Linker flags coming from the legacy <code>CROSSTOOL</code> fields.
   </td>
  </tr>
  <tr>
   <td><strong><code>user_link_flags</code></strong>
   </td>
   <td>link</td>
   <td>Linker flags coming from the <code>--linkopt</code>
       or <code>linkopts</code> attribute.
   </td>
  </tr>
  <tr>
   <td><strong><code>linkstamp_paths</code></strong>
   </td>
   <td>link</td>
   <td>A build variable giving linkstamp paths.
   </td>
  </tr>
  <tr>
   <td><strong><code>force_pic</code></strong>
   </td>
   <td>link</td>
   <td>Presence of this variable indicates that PIC/PIE code should
     be generated (Bazel option `--force_pic` was passed).
   </td>
  </tr>
  <tr>
   <td><strong><code>strip_debug_symbols</code></strong>
   </td>
   <td>link</td>
   <td>Presence of this variable indicates that the debug
       symbols should be stripped.
   </td>
  </tr>
  <tr>
   <td><strong><code>is_cc_test</code></strong>
   </td>
   <td>link</td>
   <td>Truthy when current action is a <code>cc_test</code>
       linking action, false otherwise.
   </td>
  </tr>
  <tr>
   <td><strong><code>is_using_fission</code></strong>
   </td>
   <td>compile, link</td>
   <td>Presence of this variable indicates that fission (per-object debug info)
     is activated. Debug info will be in <code>.dwo</code> files instead
       of <code>.o</code> files and the compiler and linker need to know this.
   </td>
  </tr>
  <tr>
   <td><strong><code>fdo_instrument_path</code></strong>
   </td>
   <td>compile, link</td>
   <td> Path to the directory that stores FDO instrumentation profile.
   </td>
  </tr>
  <tr>
   <td><strong><code>fdo_profile_path</code></strong>
   </td>
   <td>compile</td>
   <td> Path to FDO profile.
   </td>
  </tr>
  <tr>
   <td><strong><code>fdo_prefetch_hints_path</code></strong>
   </td>
   <td>compile</td>
   <td> Path to the cache prefetch profile.
   </td>
  </tr>
  <tr>
   <td><strong><code>cs_fdo_instrument_path</code></strong>
   </td>
   <td>compile, link</td>
   <td> Path to the directory that stores context sensitive FDO
        instrumentation profile.
   </td>
  </tr>
</table>

### Well-known features {:#wellknown-features}

The following is a reference of features and their activation
conditions.

<table>
  <col width="300">
  <col width="600">
  <tr>
   <td><strong>Feature</strong>
   </td>
   <td><strong>Documentation</strong>
   </td>
  </tr>
  <tr>
   <td><strong><code>opt | dbg | fastbuild</code></strong>
   </td>
   <td>Enabled by default based on compilation mode.
   </td>
  </tr>
  <tr>
   <td><strong><code>static_linking_mode | dynamic_linking_mode</code></strong>
   </td>
   <td>Enabled by default based on linking mode.
   </td>
  </tr>
  <tr>
   <td><strong><code>per_object_debug_info</code></strong>
   </td>
    <td>Enabled if the <code>supports_fission</code> feature is specified and
        enabled and the current compilation mode is specified in the
        <code>--fission</code> flag.
   </td>
  </tr>
  <tr>
   <td><strong><code>supports_start_end_lib</code></strong>
   </td>
   <td>If enabled (and the option <code>--start_end_lib</code> is set), Bazel
     will not link against static libraries but instead use the
     <code>--start-lib/--end-lib</code> linker options to link against objects
     directly. This speeds up the build since Bazel doesn't have to build
     static libraries.
   </td>
  </tr>
  <tr>
   <td><strong><code>supports_interface_shared_libraries</code></strong>
   </td>
   <td>If enabled (and the option <code>--interface_shared_objects</code> is
     set), Bazel will link targets that have <code>linkstatic</code> set to
     False (<code>cc_test</code>s by default) against interface shared
     libraries. This makes incremental relinking faster.
   </td>
  </tr>
  <tr>
   <td><strong><code>supports_dynamic_linker</code></strong>
   </td>
   <td>If enabled, C++ rules will know the toolchain can produce shared
     libraries.
   </td>
  </tr>
  <tr>
   <td><strong><code>static_link_cpp_runtimes</code></strong>
   </td>
   <td>If enabled, Bazel will link the C++ runtime statically in static linking
     mode and dynamically in dynamic linking mode. Artifacts
     specified in the <code>cc_toolchain.static_runtime_lib</code> or
     <code>cc_toolchain.dynamic_runtime_lib</code> attribute (depending on the
     linking mode) will be added to the linking actions.
   </td>
  </tr>
  <tr>
   <td><strong><code>supports_pic</code></strong>
   </td>
   <td>If enabled, toolchain will know to use PIC objects for dynamic libraries.
     The `pic` variable is present whenever PIC compilation is needed. If not enabled
     by default, and `--force_pic` is passed, Bazel will request `supports_pic` and
     validate that the feature is enabled. If the feature is missing, or couldn't
      be enabled, `--force_pic` cannot be used.
   </td>
  </tr>
  <tr>
    <td>
      <strong><code>static_linking_mode | dynamic_linking_mode</code></strong>
    </td>
    <td>Enabled by default based on linking mode.</td>
  </tr>
  <tr>
     <td><strong><code>no_legacy_features</code></strong>
     </td>
     <td>
       Prevents Bazel from adding legacy features to
       the C++ configuration when present. See the complete list of
       features below.
     </td>
    </tr>
</table>

#### Legacy features patching logic {:#legacy-features-patching-logic}

<p>
  Bazel applies the following changes to the toolchain's features for backwards
  compatibility:

  <ul>
    <li>Moves <code>legacy_compile_flags</code> feature to the top of the toolchain</li>
    <li>Moves <code>default_compile_flags</code> feature to the top of the toolchain</li>
    <li>Adds <code>dependency_file</code> (if not present) feature to the top of the toolchain</li>
    <li>Adds <code>pic</code> (if not present) feature to the top of the toolchain</li>
    <li>Adds <code>per_object_debug_info</code> (if not present) feature to the top of the toolchain</li>
    <li>Adds <code>preprocessor_defines</code> (if not present) feature to the top of the toolchain</li>
    <li>Adds <code>includes</code> (if not present) feature to the top of the toolchain</li>
    <li>Adds <code>include_paths</code> (if not present) feature to the top of the toolchain</li>
    <li>Adds <code>fdo_instrument</code> (if not present) feature to the top of the toolchain</li>
    <li>Adds <code>fdo_optimize</code> (if not present) feature to the top of the toolchain</li>
    <li>Adds <code>cs_fdo_instrument</code> (if not present) feature to the top of the toolchain</li>
    <li>Adds <code>cs_fdo_optimize</code> (if not present) feature to the top of the toolchain</li>
    <li>Adds <code>fdo_prefetch_hints</code> (if not present) feature to the top of the toolchain</li>
    <li>Adds <code>autofdo</code> (if not present) feature to the top of the toolchain</li>
    <li>Adds <code>build_interface_libraries</code> (if not present) feature to the top of the toolchain</li>
    <li>Adds <code>dynamic_library_linker_tool</code> (if not present) feature to the top of the toolchain</li>
    <li>Adds <code>shared_flag</code> (if not present) feature to the top of the toolchain</li>
    <li>Adds <code>linkstamps</code> (if not present) feature to the top of the toolchain</li>
    <li>Adds <code>output_execpath_flags</code> (if not present) feature to the top of the toolchain</li>
    <li>Adds <code>runtime_library_search_directories</code> (if not present) feature to the top of the toolchain</li>
    <li>Adds <code>library_search_directories</code> (if not present) feature to the top of the toolchain</li>
    <li>Adds <code>archiver_flags</code> (if not present) feature to the top of the toolchain</li>
    <li>Adds <code>libraries_to_link</code> (if not present) feature to the top of the toolchain</li>
    <li>Adds <code>force_pic_flags</code> (if not present) feature to the top of the toolchain</li>
    <li>Adds <code>user_link_flags</code> (if not present) feature to the top of the toolchain</li>
    <li>Adds <code>legacy_link_flags</code> (if not present) feature to the top of the toolchain</li>
    <li>Adds <code>static_libgcc</code> (if not present) feature to the top of the toolchain</li>
    <li>Adds <code>fission_support</code> (if not present) feature to the top of the toolchain</li>
    <li>Adds <code>strip_debug_symbols</code> (if not present) feature to the top of the toolchain</li>
    <li>Adds <code>coverage</code> (if not present) feature to the top of the toolchain</li>
    <li>Adds <code>llvm_coverage_map_format</code> (if not present) feature to the top of the toolchain</li>
    <li>Adds <code>gcc_coverage_map_format</code> (if not present) feature to the top of the toolchain</li>
    <li>Adds <code>fully_static_link</code> (if not present) feature to the bottom of the toolchain</li>
    <li>Adds <code>user_compile_flags</code> (if not present) feature to the bottom of the toolchain</li>
    <li>Adds <code>sysroot</code> (if not present) feature to the bottom of the toolchain</li>
    <li>Adds <code>unfiltered_compile_flags</code> (if not present) feature to the bottom of the toolchain</li>
    <li>Adds <code>linker_param_file</code> (if not present) feature to the bottom of the toolchain</li>
    <li>Adds <code>compiler_input_flags</code> (if not present) feature to the bottom of the toolchain</li>
    <li>Adds <code>compiler_output_flags</code> (if not present) feature to the bottom of the toolchain</li>
  </ul>
</p>

This is a long list of features. The plan is to get rid of them once
[Crosstool in Starlark](https://github.com/bazelbuild/bazel/issues/5380){: .external} is
done. For the curious reader see the implementation in
[CppActionConfigs](https://source.bazel.build/bazel/+/master:src/main/java/com/google/devtools/build/lib/rules/cpp/CppActionConfigs.java?q=cppactionconfigs&ss=bazel),
and for production toolchains consider adding `no_legacy_features` to make
the toolchain more standalone.

