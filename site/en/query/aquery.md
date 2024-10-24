Project: /_project.yaml
Book: /_book.yaml

# Action Graph Query (aquery)

{% include "_buttons.html" %}

The `aquery` command allows you to query for actions in your build graph.
It operates on the post-analysis Configured Target Graph and exposes
information about **Actions, Artifacts and their relationships.**

`aquery` is useful when you are interested in the properties of the Actions/Artifacts
generated from the Configured Target Graph. For example, the actual commands run
and their inputs/outputs/mnemonics.

The tool accepts several command-line [options](#command-options).
Notably, the aquery command runs on top of a regular Bazel build and inherits
the set of options available during a build.

It supports the same set of functions that is also available to traditional
`query` but `siblings`, `buildfiles` and
`tests`.

An example `aquery` output (without specific details):

<pre>
$ bazel aquery 'deps(//some:label)'
action 'Writing file some_file_name'
  Mnemonic: ...
  Target: ...
  Configuration: ...
  ActionKey: ...
  Inputs: [...]
  Outputs: [...]
</pre>

## Basic syntax {:#basic-syntax}

A simple example of the syntax for `aquery` is as follows:

`bazel aquery "aquery_function(function(//target))"`

The query expression (in quotes) consists of the following:

*   `aquery_function(...)`: functions specific to `aquery`.
    More details [below](#using-aquery-functions).
*   `function(...)`: the standard [functions](/query/language#functions)
    as traditional `query`.
*   `//target` is the label to the interested target.

<pre>
# aquery examples:
# Get the action graph generated while building //src/target_a
$ bazel aquery '//src/target_a'

# Get the action graph generated while building all dependencies of //src/target_a
$ bazel aquery 'deps(//src/target_a)'

# Get the action graph generated while building all dependencies of //src/target_a
# whose inputs filenames match the regex ".*cpp".
$ bazel aquery 'inputs(".*cpp", deps(//src/target_a))'
</pre>

## Using aquery functions {:#using-aquery-functions}

There are three `aquery` functions:

*   `inputs`: filter actions by inputs.
*   `outputs`: filter actions by outputs
*   `mnemonic`: filter actions by mnemonic

`expr ::= inputs(word, expr)`

  The `inputs` operator returns the actions generated from building `expr`,
  whose input filenames match the regex provided by `word`.

`$ bazel aquery 'inputs(".*cpp", deps(//src/target_a))'`

`outputs` and `mnemonic` functions share a similar syntax.

You can also combine functions to achieve the AND operation. For example:

<pre>
  $ bazel aquery 'mnemonic("Cpp.*", (inputs(".*cpp", inputs("foo.*", //src/target_a))))'
</pre>

  The above command would find all actions involved in building `//src/target_a`,
  whose mnemonics match `"Cpp.*"` and inputs match the patterns
  `".*cpp"` and `"foo.*"`.

Important: aquery functions can't be nested inside non-aquery functions.
Conceptually, this makes sense since the output of aquery functions is Actions,
not Configured Targets.

An example of the syntax error produced:

<pre>
        $ bazel aquery 'deps(inputs(".*cpp", //src/target_a))'
        ERROR: aquery filter functions (inputs, outputs, mnemonic) produce actions,
        and therefore can't be the input of other function types: deps
        deps(inputs(".*cpp", //src/target_a))
</pre>

## Options {:#options}

### Build options {:#build-options}

`aquery` runs on top of a regular Bazel build and thus inherits the set of
[options](/reference/command-line-reference#build-options)
available during a build.

### Aquery options {:#aquery-options}

#### `--output=(text|summary|proto|jsonproto|textproto), default=text` {:#output}

The default output format (`text`) is human-readable,
use `proto`, `textproto`, or `jsonproto` for machine-readable format.
The proto message is `analysis.ActionGraphContainer`.

#### `--include_commandline, default=true` {:#include-commandline}

Includes the content of the action command lines in the output (potentially large).

#### `--include_artifacts, default=true` {:#include-artifacts}

Includes names of the action inputs and outputs in the output (potentially large).

#### `--include_aspects, default=true` {:#include-aspects}

Whether to include Aspect-generated actions in the output.

#### `--include_param_files, default=false` {:#include-param-files}

Include the content of the param files used in the command (potentially large).

Warning: Enabling this flag will automatically enable the `--include_commandline` flag.

#### `--include_file_write_contents, default=false` {:#include-file-write-contents}

Include file contents for the `actions.write()` action and the contents of the
manifest file for the `SourceSymlinkManifest` action The file contents is
returned in the `file_contents` field with `--output=`xxx`proto`.
With `--output=text`, the output has
```
FileWriteContents: [<base64-encoded file contents>]
```
line

#### `--skyframe_state, default=false` {:#skyframe-state}

Without performing extra analysis, dump the Action Graph from Skyframe.

Note: Specifying a target with `--skyframe_state` is currently not supported.
This flag is only available with `--output=proto` or `--output=textproto`.

## Other tools and features {:#other-tools-features}

### Querying against the state of Skyframe {:#querying-against-skyframe}

[Skyframe](/reference/skyframe) is the evaluation and
incrementality model of Bazel. On each instance of Bazel server, Skyframe stores the dependency graph
constructed from the previous runs of the [Analysis phase](/run/build#analysis).

In some cases, it is useful to query the Action Graph on Skyframe.
An example use case would be:

1.  Run `bazel build //target_a`
2.  Run `bazel build //target_b`
3.  File `foo.out` was generated.

_As a Bazel user, I want to determine if `foo.out` was generated from building
`//target_a` or `//target_b`_.

One could run `bazel aquery 'outputs("foo.out", //target_a)'` and
`bazel aquery 'outputs("foo.out", //target_b)'` to figure out the action responsible
for creating `foo.out`, and in turn the target. However, the number of different
targets previously built can be larger than 2, which makes running multiple `aquery`
commands a hassle.

As an alternative, the `--skyframe_state` flag can be used:

<pre>
  # List all actions on Skyframe's action graph
  $ bazel aquery --output=proto --skyframe_state

  # or

  # List all actions on Skyframe's action graph, whose output matches "foo.out"
  $ bazel aquery --output=proto --skyframe_state 'outputs("foo.out")'
</pre>

With `--skyframe_state` mode, `aquery` takes the content of the Action Graph
that Skyframe keeps on the instance of Bazel, (optionally) performs filtering on it and
outputs the content, without re-running the analysis phase.

#### Special considerations {:#special-considerations}

##### Output format {:#output-format}

`--skyframe_state` is currently only available for `--output=proto`
and `--output=textproto`

##### Non-inclusion of target labels in the query expression {:#target-labels-non-inclusion}

Currently, `--skyframe_state` queries the whole action graph that exists on Skyframe,
regardless of the targets. Having the target label specified in the query together with
`--skyframe_state` is considered a syntax error:

<pre>
  # WRONG: Target Included
  $ bazel aquery --output=proto --skyframe_state **//target_a**
  ERROR: Error while parsing '//target_a)': Specifying build target(s) [//target_a] with --skyframe_state is currently not supported.

  # WRONG: Target Included
  $ bazel aquery --output=proto --skyframe_state 'inputs(".*.java", **//target_a**)'
  ERROR: Error while parsing '//target_a)': Specifying build target(s) [//target_a] with --skyframe_state is currently not supported.

  # CORRECT: Without Target
  $ bazel aquery --output=proto --skyframe_state
  $ bazel aquery --output=proto --skyframe_state 'inputs(".*.java")'
</pre>

### Comparing aquery outputs {:#comparing-aquery-outputs}

You can compare the outputs of two different aquery invocations using the `aquery_differ` tool.
For instance: when you make some changes to your rule definition and want to verify that the
command lines being run did not change. `aquery_differ` is the tool for that.

The tool is available in the [bazelbuild/bazel](https://github.com/bazelbuild/bazel/tree/master/tools/aquery_differ){: .external} repository.
To use it, clone the repository to your local machine. An example usage:

<pre>
  $ bazel run //tools/aquery_differ -- \
  --before=/path/to/before.proto \
  --after=/path/to/after.proto \
  --input_type=proto \
  --attrs=cmdline \
  --attrs=inputs
</pre>

The above command returns the difference between the `before` and `after` aquery outputs:
which actions were present in one but not the other, which actions have different
command line/inputs in each aquery output, ...). The result of running the above command would be:

<pre>
  Aquery output 'after' change contains an action that generates the following outputs that aquery output 'before' change doesn't:
  ...
  /list of output files/
  ...

  [cmdline]
  Difference in the action that generates the following output(s):
    /path/to/abc.out
  --- /path/to/before.proto
  +++ /path/to/after.proto
  @@ -1,3 +1,3 @@
    ...
    /cmdline diff, in unified diff format/
    ...
</pre>

#### Command options {:#command-options}

`--before, --after`: The aquery output files to be compared

`--input_type=(proto|text_proto), default=proto`: the format of the input
files. Support is provided for `proto` and `textproto` aquery output.

`--attrs=(cmdline|inputs), default=cmdline`: the attributes of actions
to be compared.

### Aspect-on-aspect {:#aspect-on-aspect}

It is possible for [Aspects](/extending/aspects)
to be applied on top of each other. The aquery output of the action generated by
these Aspects would then include the _Aspect path_, which is the sequence of
Aspects applied to the target which generated the action.

An example of Aspect-on-Aspect:

<pre>
  t0
  ^
  | <- a1
  t1
  ^
  | <- a2
  t2
</pre>

Let t<sub>i</sub> be a target of rule r<sub>i</sub>, which applies an Aspect a<sub>i</sub>
to its dependencies.

Assume that a2 generates an action X when applied to target t0. The text output of
`bazel aquery --include_aspects 'deps(//t2)'` for action X would be:

<pre>
  action ...
  Mnemonic: ...
  Target: //my_pkg:t0
  Configuration: ...
  AspectDescriptors: [//my_pkg:rule.bzl%**a2**(foo=...)
    -> //my_pkg:rule.bzl%**a1**(bar=...)]
  ...
</pre>

This means that action `X` was generated by Aspect `a2` applied onto
`a1(t0)`, where `a1(t0)` is the result of Aspect `a1` applied
onto target `t0`.

Each `AspectDescriptor` has the following format:

<pre>
  AspectClass([param=value,...])
</pre>

`AspectClass` could be the name of the Aspect class (for native Aspects) or
`bzl_file%aspect_name` (for Starlark Aspects). `AspectDescriptor` are
sorted in topological order of the
[dependency graph](/extending/aspects#aspect_basics).

### Linking with the JSON profile {:#linking-with-json-profile}

While aquery provides information about the actions being run in a build (why they're being run,
their inputs/outputs), the [JSON profile](/rules/performance#performance-profiling)
tells us the timing and duration of their execution.
It is possible to combine these 2 sets of information via a common denominator: an action's primary output.

To include actions' outputs in the JSON profile, generate the profile with
`--experimental_include_primary_output --noslim_profile`.
Slim profiles are incompatible with the inclusion of primary outputs. An action's primary output
is included by default by aquery.

We don't currently provide a canonical tool to combine these 2 data sources, but you should be
able to build your own script with the above information.

## Known issues {:#known-issues}

### Handling shared actions {:#handling-shared-actions}

Sometimes actions are
[shared](https://source.bazel.build/bazel/+/master:src/main/java/com/google/devtools/build/lib/actions/Actions.java;l=59;drc=146d51aa1ec9dcb721a7483479ef0b1ac21d39f1){: .external}
between configured targets.

In the execution phase, those shared actions are
[simply considered as one](https://source.bazel.build/bazel/+/master:src/main/java/com/google/devtools/build/lib/actions/Actions.java;l=241;drc=003b8734036a07b496012730964ac220f486b61f){: .external} and only executed once.
However, aquery operates on the pre-execution, post-analysis action graph, and hence treats these
like separate actions whose output Artifacts have the exact same `execPath`. As a result,
equivalent Artifacts appear duplicated.

The list of aquery issues/planned features can be found on
[GitHub](https://github.com/bazelbuild/bazel/labels/team-Performance){: .external}.

## FAQs {:#faqs}

### The ActionKey remains the same even though the content of an input file changed. {:#actionkey-same}

In the context of aquery, the `ActionKey` refers to the `String` gotten from
[ActionAnalysisMetadata#getKey](https://source.bazel.build/bazel/+/master:src/main/java/com/google/devtools/build/lib/actions/ActionAnalysisMetadata.java;l=89;drc=8b856f5484f0117b2aebc302f849c2a15f273310){: .external}:

<pre>
  Returns a string encoding all of the significant behaviour of this Action that might affect the
  output. The general contract of `getKey` is this: if the work to be performed by the
  execution of this action changes, the key must change.

  ...

  Examples of changes that should affect the key are:

  - Changes to the BUILD file that materially affect the rule which gave rise to this Action.
  - Changes to the command-line options, environment, or other global configuration resources
      which affect the behaviour of this kind of Action (other than changes to the names of the
      input/output files, which are handled externally).
  - An upgrade to the build tools which changes the program logic of this kind of Action
      (typically this is achieved by incorporating a UUID into the key, which is changed each
      time the program logic of this action changes).
  Note the following exception: for actions that discover inputs, the key must change if any
  input names change or else action validation may falsely validate.
</pre>

This excludes the changes to the content of the input files, and is not to be confused with
[RemoteCacheClient#ActionKey](https://source.bazel.build/bazel/+/master:src/main/java/com/google/devtools/build/lib/remote/common/RemoteCacheClient.java;l=38;drc=21577f202eb90ce94a337ebd2ede824d609537b6){: .external}.

## Updates {:#updates}

For any issues/feature requests, please file an issue [here](https://github.com/bazelbuild/bazel/issues/new).
