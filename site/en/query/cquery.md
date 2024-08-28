Project: /_project.yaml
Book: /_book.yaml

#  Configurable Query (cquery)

{% include "_buttons.html" %}

`cquery` is a variant of [`query`](/query/language) that correctly handles
[`select()`](/docs/configurable-attributes) and build options' effects on the build
graph.

It achieves this by running over the results of Bazel's [analysis
phase](/extending/concepts#evaluation-model),
which integrates these effects. `query`, by contrast, runs over the results of
Bazel's loading phase, before options are evaluated.

For example:

<pre>
$ cat > tree/BUILD &lt;&lt;EOF
sh_library(
    name = "ash",
    deps = select({
        ":excelsior": [":manna-ash"],
        ":americana": [":white-ash"],
        "//conditions:default": [":common-ash"],
    }),
)
sh_library(name = "manna-ash")
sh_library(name = "white-ash")
sh_library(name = "common-ash")
config_setting(
    name = "excelsior",
    values = {"define": "species=excelsior"},
)
config_setting(
    name = "americana",
    values = {"define": "species=americana"},
)
EOF
</pre>

<pre>
# Traditional query: query doesn't know which select() branch you will choose,
# so it conservatively lists all of possible choices, including all used config_settings.
$ bazel query "deps(//tree:ash)" --noimplicit_deps
//tree:americana
//tree:ash
//tree:common-ash
//tree:excelsior
//tree:manna-ash
//tree:white-ash

# cquery: cquery lets you set build options at the command line and chooses
# the exact dependencies that implies (and also the config_setting targets).
$ bazel cquery "deps(//tree:ash)" --define species=excelsior --noimplicit_deps
//tree:ash (9f87702)
//tree:manna-ash (9f87702)
//tree:americana (9f87702)
//tree:excelsior (9f87702)
</pre>

Each result includes a [unique identifier](#configurations) `(9f87702)` of
the [configuration](/reference/glossary#configuration) the
target is built with.

Since `cquery` runs over the configured target graph. it doesn't have insight
into artifacts like build actions nor access to [`test_suite`](/reference/be/general#test_suite)
rules as they are not configured targets. For the former, see [`aquery`](/query/aquery).

## Basic syntax {:#basic-syntax}

A simple `cquery` call looks like:

`bazel cquery "function(//target)"`

The query expression `"function(//target)"` consists of the following:

*   **`function(...)`** is the function to run on the target. `cquery`
    supports most
    of `query`'s [functions](/query/language#functions), plus a
    few new ones.
*   **`//target`** is the expression fed to the function. In this example, the
    expression is a simple target. But the query language also allows nesting of functions.
    See the [Query guide](/query/guide) for examples.


`cquery` requires a target to run through the [loading and analysis](/extending/concepts#evaluation-model)
phases. Unless otherwise specified, `cquery` parses the target(s) listed in the
query expression. See [`--universe_scope`](#universe-scope)
for querying dependencies of top-level build targets.

## Configurations {:#configurations}

The line:

<pre>
//tree:ash (9f87702)
</pre>

means `//tree:ash` was built in a configuration with ID `9f87702`. For most
targets, this is an opaque hash of the build option values defining the
configuration.

To see the configuration's complete contents, run:

<pre>
$ bazel config 9f87702
</pre>

`9f87702` is a prefix of the complete ID. This is because complete IDs are
SHA-256 hashes, which are long and hard to follow. `cquery` understands any valid
prefix of a complete ID, similar to
[Git short hashes](https://git-scm.com/book/en/v2/Git-Tools-Revision-Selection#_revision_selection){: .external}.
 To see complete IDs, run `$ bazel config`.

## Target pattern evaluation {:#target-pattern-evaluation}

`//foo` has a different meaning for `cquery` than for `query`. This is because
`cquery` evaluates _configured_ targets and the build graph may have multiple
configured versions of `//foo`.

For `cquery`, a target pattern in the query expression evaluates
to every configured target with a label that matches that pattern. Output is
deterministic, but `cquery` makes no ordering guarantee beyond the
[core query ordering contract](/query/language#graph-order).

This produces subtler results for query expressions than with `query`.
For example, the following can produce multiple results:

<pre>
# Analyzes //foo in the target configuration, but also analyzes
# //genrule_with_foo_as_tool which depends on an exec-configured
# //foo. So there are two configured target instances of //foo in
# the build graph.
$ bazel cquery //foo --universe_scope=//foo,//genrule_with_foo_as_tool
//foo (9f87702)
//foo (exec)
</pre>

If you want to precisely declare which instance to query over, use
the [`config`](#config) function.

See `query`'s [target pattern
documentation](/query/language#target-patterns) for more information on target patterns.

## Functions {:#functions}

Of the [set of functions](/query/language#functions "list of query functions")
supported by `query`, `cquery` supports all but
[`allrdeps`](/query/language#allrdeps),
[`buildfiles`](/query/language#buildfiles),
[`rbuildfiles`](/query/language#rbuildfiles),
[`siblings`](/query/language#siblings), [`tests`](/query/language#tests), and
[`visible`](/query/language#visible).

`cquery` also introduces the following new functions:

### config {:#config}

`expr ::= config(expr, word)`

The `config` operator attempts to find the configured target for
the label denoted by the first argument and configuration specified by the
second argument.

Valid values for the second argument are `null` or a
[custom configuration hash](#configurations). Hashes can be retrieved from `$
bazel config` or a previous `cquery`'s output.

Examples:

<pre>
$ bazel cquery "config(//bar, 3732cc8)" --universe_scope=//foo
</pre>

<pre>
$ bazel cquery "deps(//foo)"
//bar (exec)
//baz (exec)

$ bazel cquery "config(//baz, 3732cc8)"
</pre>

If not all results of the first argument can be found in the specified
configuration, only those that can be found are returned. If no results
can be found in the specified configuration, the query fails.

## Options {:#options}

### Build options {:#build-options}

`cquery` runs over a regular Bazel build and thus inherits the set of
[options](/reference/command-line-reference#build-options) available during a build.

###  Using cquery options {:#using-cquery-options}

#### `--universe_scope` (comma-separated list) {:#universe-scope}

Often, the dependencies of configured targets go through
[transitions](/extending/rules#configurations),
which causes their configuration to differ from their dependent. This flag
allows you to query a target as if it were built as a dependency or a transitive
dependency of another target. For example:

<pre>
# x/BUILD
genrule(
     name = "my_gen",
     srcs = ["x.in"],
     outs = ["x.cc"],
     cmd = "$(locations :tool) $&lt; >$@",
     tools = [":tool"],
)
cc_binary(
    name = "tool",
    srcs = ["tool.cpp"],
)
</pre>

Genrules configure their tools in the
[exec configuration](/extending/rules#configurations)
so the following queries would produce the following outputs:

<table class="table table-condensed table-bordered table-params">
  <thead>
    <tr>
      <th>Query</th>
      <th>Target Built</th>
      <th>Output</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>bazel cquery "//x:tool"</td>
      <td>//x:tool</td>
      <td>//x:tool(targetconfig)</td>
    </tr>
    <tr>
      <td>bazel cquery "//x:tool" --universe_scope="//x:my_gen"</td>
      <td>//x:my_gen</td>
      <td>//x:tool(execconfig)</td>
    </tr>
  </tbody>
</table>

If this flag is set, its contents are built. _If it's not set, all targets
mentioned in the query expression are built_ instead. The transitive closure of the
built targets are used as the universe of the query. Either way, the targets to
be built must be buildable at the top level (that is, compatible with top-level
options). `cquery` returns results in the transitive closure of these
top-level targets.

Even if it's possible to build all targets in a query expression at the top
level, it may be beneficial to not do so. For example, explicitly setting
`--universe_scope` could prevent building targets multiple times in
configurations you don't care about. It could also help specify which configuration version of a
target you're looking for (since it's not currently possible
to fully specify this any other way). You should set this flag
if your query expression is more complex than `deps(//foo)`.

#### `--implicit_deps` (boolean, default=True) {:#implicit-deps}

Setting this flag to false filters out all results that aren't explicitly set in
the BUILD file and instead set elsewhere by Bazel. This includes filtering resolved
toolchains.

#### `--tool_deps` (boolean, default=True) {:#tool-deps}

Setting this flag to false filters out all configured targets for which the
path from the queried target to them crosses a transition between the target
configuration and the
[non-target configurations](/extending/rules#configurations).
If the queried target is in the target configuration, setting `--notool_deps` will
only return targets that also are in the target configuration. If the queried
target is in a non-target configuration, setting `--notool_deps` will only return
targets also in non-target configurations. This setting generally does not affect filtering
of resolved toolchains.

#### `--include_aspects` (boolean, default=True) {:#include-aspects}

Include dependencies added by [aspects](/extending/aspects).

If this flag is disabled, `cquery somepath(X, Y)` and
`cquery deps(X) | grep 'Y'` omit Y if X only depends on it through an aspect.

## Output formats {:#output-formats}

By default, cquery outputs results in a dependency-ordered list of label and configuration pairs.
There are other options for exposing the results as well.

### Transitions {:#transitions}

<pre>
--transitions=lite
--transitions=full
</pre>

Configuration [transitions](/extending/rules#configurations)
are used to build targets underneath the top level targets in different
configurations than the top level targets.

For example, a target might impose a transition to the exec configuration on all
dependencies in its `tools` attribute. These are known as attribute
transitions. Rules can also impose transitions on their own configurations,
known as rule class transitions. This output format outputs information about
these transitions such as what type they are and the effect they have on build
options.

This output format is triggered by the `--transitions` flag which by default is
set to `NONE`. It can be set to `FULL` or `LITE` mode. `FULL` mode outputs
information about rule class transitions and attribute transitions including a
detailed diff of the options before and after the transition. `LITE` mode
outputs the same information without the options diff.

### Protocol message output {:#protocol-message-output}

<pre>
--output=proto
</pre>

This option causes the resulting targets to be printed in a binary protocol
buffer form. The definition of the protocol buffer can be found at
[src/main/protobuf/analysis_v2.proto](https://github.com/bazelbuild/bazel/blob/master/src/main/protobuf/analysis_v2.proto){: .external}.

`CqueryResult` is the top level message containing the results of the cquery. It
has a list of `ConfiguredTarget` messages and a list of `Configuration`
messages. Each `ConfiguredTarget` has a `configuration_id` whose value is equal
to that of the `id` field from the corresponding `Configuration` message.

#### --[no]proto:include_configurations {:#proto-include-configurations}

By default, cquery results return configuration information as part of each
configured target. If you'd like to omit this information and get proto output
that is formatted exactly like query's proto output, set this flag to false.

See [query's proto output documentation](/query/language#output-formats)
for more proto output-related options.

Note: While selects are resolved both at the top level of returned
targets and within attributes, all possible inputs for selects are still
included as `rule_input` fields.

### Graph output {:#graph-output}

<pre>
--output=graph
</pre>

This option generates output as a Graphviz-compatible .dot file. See `query`'s
[graph output documentation](/query/language#display-result-graph) for details. `cquery`
also supports [`--graph:node_limit`](/query/language#graph-nodelimit) and
[`--graph:factored`](/query/language#graph-factored).

### Files output {:#files-output}

<pre>
--output=files
</pre>

This option prints a list of the output files produced by each target matched
by the query similar to the list printed at the end of a `bazel build`
invocation. The output contains only the files advertised in the requested
output groups as determined by the
[`--output_groups`](/reference/command-line-reference#flag--output_groups) flag.
It does include source files.

All paths emitted by this output format are relative to the
[execroot](https://bazel.build/remote/output-directories), which can be obtained
via `bazel info execution_root`. If the `bazel-out` convenience symlink exists,
paths to files in the main repository also resolve relative to the workspace
directory.

Note: The output of `bazel cquery --output=files //pkg:foo` contains the output
files of `//pkg:foo` in *all* configurations that occur in the build (also see
the [section on target pattern evaluation](#target-pattern-evaluation)). If that
is not desired, wrap you query in [`config(..., target)`](#config).

### Defining the output format using Starlark {:#output-format-definition}

<pre>
--output=starlark
</pre>

This output format calls a [Starlark](/rules/language)
function for each configured target in the query result, and prints the value
returned by the call. The `--starlark:file` flag specifies the location of a
Starlark file that defines a function named `format` with a single parameter,
`target`. This function is called for each [Target](/rules/lib/builtins/Target)
in the query result. Alternatively, for convenience, you may specify just the
body of a function declared as `def format(target): return expr` by using the
`--starlark:expr` flag.

#### 'cquery' Starlark dialect {:#cquery-starlark}

The cquery Starlark environment differs from a BUILD or .bzl file. It includes
all core Starlark
[built-in constants and functions](https://github.com/bazelbuild/starlark/blob/master/spec.md#built-in-constants-and-functions){: .external},
plus a few cquery-specific ones described below, but not (for example) `glob`,
`native`, or `rule`, and it does not support load statements.

##### build_options(target) {:#build-options}

`build_options(target)` returns a map whose keys are build option identifiers (see
[Configurations](/extending/config))
and whose values are their Starlark values. Build options whose values are not legal Starlark
values are omitted from this map.

If the target is an input file, `build_options(target)` returns None, as input file
targets have a null configuration.

##### providers(target) {:#providers}

`providers(target)` returns a map whose keys are names of
[providers](/extending/rules#providers)
(for example, `"DefaultInfo"`) and whose values are their Starlark values. Providers
whose values are not legal Starlark values are omitted from this map.

#### Examples {:#output-format-definition-examples}

Print a space-separated list of the base names of all files produced by `//foo`:

<pre>
  bazel cquery //foo --output=starlark \
    --starlark:expr="' '.join([f.basename for f in target.files.to_list()])"
</pre>

Print a space-separated list of the paths of all files produced by **rule** targets in
`//bar` and its subpackages:

<pre>
  bazel cquery 'kind(rule, //bar/...)' --output=starlark \
    --starlark:expr="' '.join([f.path for f in target.files.to_list()])"
</pre>

Print a list of the mnemonics of all actions registered by `//foo`.

<pre>
  bazel cquery //foo --output=starlark \
    --starlark:expr="[a.mnemonic for a in target.actions]"
</pre>

Print a list of compilation outputs registered by a `cc_library` `//baz`.

<pre>
  bazel cquery //baz --output=starlark \
    --starlark:expr="[f.path for f in target.output_groups.compilation_outputs.to_list()]"
</pre>

Print the value of the command line option `--javacopt` when building `//foo`.

<pre>
  bazel cquery //foo --output=starlark \
    --starlark:expr="build_options(target)['//command_line_option:javacopt']"
</pre>

Print the label of each target with exactly one output. This example uses
Starlark functions defined in a file.

<pre>
  $ cat example.cquery

  def has_one_output(target):
    return len(target.files.to_list()) == 1

  def format(target):
    if has_one_output(target):
      return target.label
    else:
      return ""

  $ bazel cquery //baz --output=starlark --starlark:file=example.cquery
</pre>

Print the label of each target which is strictly Python 3. This example uses
Starlark functions defined in a file.

<pre>
  $ cat example.cquery

  def format(target):
    p = providers(target)
    py_info = p.get("PyInfo")
    if py_info and py_info.has_py3_only_sources:
      return target.label
    else:
      return ""

  $ bazel cquery //baz --output=starlark --starlark:file=example.cquery
</pre>

Extract a value from a user defined Provider.

<pre>
  $ cat some_package/my_rule.bzl

  MyRuleInfo = provider(fields={"color": "the name of a color"})

  def _my_rule_impl(ctx):
      ...
      return [MyRuleInfo(color="red")]

  my_rule = rule(
      implementation = _my_rule_impl,
      attrs = {...},
  )

  $ cat example.cquery

  def format(target):
    p = providers(target)
    my_rule_info = p.get("//some_package:my_rule.bzl%MyRuleInfo'")
    if my_rule_info:
      return my_rule_info.color
    return ""

  $ bazel cquery //baz --output=starlark --starlark:file=example.cquery
</pre>

## cquery vs. query {:#cquery-vs-query}

`cquery` and `query` complement each other and excel in
different niches. Consider the following to decide which is right for you:

*  `cquery` follows specific `select()` branches to
    model the exact graph you build. `query` doesn't know which
    branch the build chooses, so overapproximates by including all branches.
*   `cquery`'s precision requires building more of the graph than
    `query` does. Specifically, `cquery`
    evaluates _configured targets_ while `query` only
    evaluates _targets_. This takes more time and uses more memory.
*   `cquery`'s interpretation of
    the [query language](/query/language) introduces ambiguity
    that `query` avoids. For example,
    if `"//foo"` exists in two configurations, which one
    should `cquery "deps(//foo)"` use?
    The [`config`](#config) function can help with this.
*   As a newer tool, `cquery` lacks support for certain use
    cases. See [Known issues](#known-issues) for details.

## Known issues {:#known-issues}

**All targets that `cquery` "builds" must have the same configuration.**

Before evaluating queries, `cquery` triggers a build up to just
before the point where build actions would execute. The targets it
"builds" are by default selected from all labels that appear in the query
expression (this can be overridden
with [`--universe_scope`](#universe-scope)). These
must have the same configuration.

While these generally share the top-level "target" configuration,
rules can change their own configuration with
[incoming edge transitions](/extending/config#incoming-edge-transitions).
This is where `cquery` falls short.

Workaround: If possible, set `--universe_scope` to a stricter
scope. For example:

<pre>
# This command attempts to build the transitive closures of both //foo and
# //bar. //bar uses an incoming edge transition to change its --cpu flag.
$ bazel cquery 'somepath(//foo, //bar)'
ERROR: Error doing post analysis query: Top-level targets //foo and //bar
have different configurations (top-level targets with different
configurations is not supported)

# This command only builds the transitive closure of //foo, under which
# //bar should exist in the correct configuration.
$ bazel cquery 'somepath(//foo, //bar)' --universe_scope=//foo
</pre>

**No support for [`--output=xml`](/query/language#xml).**

**Non-deterministic output.**

`cquery` does not automatically wipe the build graph from
previous commands and is therefore prone to picking up results from past
queries. For example, `genrule` exerts an exec transition on
its `tools` attribute - that is, it configures its tools in the
[exec configuration](/extending/rules#configurations).

You can see the lingering effects of that transition below.

<pre>
$ cat > foo/BUILD &lt;&lt;&lt;EOF
genrule(
    name = "my_gen",
    srcs = ["x.in"],
    outs = ["x.cc"],
    cmd = "$(locations :tool) $&lt; >$@",
    tools = [":tool"],
)
cc_library(
    name = "tool",
)
EOF

    $ bazel cquery "//foo:tool"
tool(target_config)

    $ bazel cquery "deps(//foo:my_gen)"
my_gen (target_config)
tool (exec_config)
...

    $ bazel cquery "//foo:tool"
tool(exec_config)
</pre>

Workaround: change any startup option to force re-analysis of configured targets.
For example, add `--test_arg=<whatever>` to your build command.

## Troubleshooting {:#troubleshooting}

### Recursive target patterns (`/...`) {:#recursive-target-patterns}

If you encounter:

<pre>
$ bazel cquery --universe_scope=//foo:app "somepath(//foo:app, //foo/...)"
ERROR: Error doing post analysis query: Evaluation failed: Unable to load package '[foo]'
because package is not in scope. Check that all target patterns in query expression are within the
--universe_scope of this query.
</pre>

this incorrectly suggests package `//foo` isn't in scope even though
`--universe_scope=//foo:app` includes it. This is due to design limitations in
`cquery`. As a workaround, explicitly include `//foo/...` in the universe
scope:

<pre>
$ bazel cquery --universe_scope=//foo:app,//foo/... "somepath(//foo:app, //foo/...)"
</pre>

If that doesn't work (for example, because some target in `//foo/...` can't
build with the chosen build flags), manually unwrap the pattern into its
constituent packages with a pre-processing query:

<pre>
# Replace "//foo/..." with a subshell query call (not cquery!) outputting each package, piped into
# a sed call converting "&lt;pkg&gt;" to "//&lt;pkg&gt;:*", piped into a "+"-delimited line merge.
# Output looks like "//foo:*+//foo/bar:*+//foo/baz".
#
$  bazel cquery --universe_scope=//foo:app "somepath(//foo:app, $(bazel query //foo/...
--output=package | sed -e 's/^/\/\//' -e 's/$/:*/' | paste -sd "+" -))"
</pre>
