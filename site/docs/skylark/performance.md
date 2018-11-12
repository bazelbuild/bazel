---
layout: documentation
title: Optimizing Performance
---

# Optimizing Performance

When writing rules, the most common performance pitfall is to traverse or copy
data that is accumulated from dependencies. When aggregated over the whole
build, these operations can easily take O(N^2) time or space. To avoid this, it
is crucial to understand how to use depsets effectively.

This can be hard to get right, so Bazel also provides a memory profiler that
assists you in finding spots where you might have made a mistake. Be warned:
The cost of writing an inefficient rule may not be evident until it is in
widespread use.

## Contents
{:.no_toc}

* ToC
{:toc}

## Use depsets

Whenever you are rolling up information from rule dependencies you should use
[depsets](lib/depset.html). Only use plain lists or dicts to publish information
local to the current rule.

A depset represents information as a nested graph which enables sharing.

Consider the following graph:

```
C -> B -> A
D ---^
```

Each node publishes a single string. With depsets the data looks like this:

```
a = depset(direct=['a'])
b = depset(direct=['b'], transitive=[a])
c = depset(direct=['c'], transitive=[b])
d = depset(direct=['d'], transitive=[b])
```

Note that each item is only mentioned once. With lists you would get this:

```
a = ['a']
b = ['b', 'a']
c = ['c', 'b', 'a']
d = ['d', 'b', 'a']
```

Note that in this case `'a'` is mentioned four times! With larger graphs this
problem will only get worse.

Here is an example of a rule implementation that uses depsets correctly to
publish transitive information. Note that it is OK to publish rule-local
information using lists if you want since this is not O(N^2).

```
MyProvider = provider()

def _impl(ctx):
  my_things = ctx.attr.things
  all_things = depset(
      direct=my_things,
      transitive=[dep[MyProvider].all_things for dep in ctx.attr.deps]
  )
  ...
  return [MyProvider(
    my_things=my_things,  # OK, a flat list of rule-local things only
    all_things=all_things,  # OK, a depset containing dependencies
  )]
```

See the [depset overview](depsets.md) page for more information.

### Avoid calling `depset.to_list()`

You can coerce a depset to a flat list using
[`to_list()`](lib/depset.html#to_list), but doing so usually results in O(N^2)
cost. If at all possible, avoid any flattening of depsets except for debugging
purposes.

A common misconception is that you can freely flatten depsets if you only do it
at top-level targets, such as an `<xx>_binary` rule, since then the cost is not
accumulated over each level of the build graph. But this is *still* O(N^2) when
you build a set of targets with overlapping dependencies. This happens when
building your tests `//foo/tests/...`, or when importing an IDE project.

**Note**: Today it is possible to flatten depsets implicitly by iterating over
the depset the way you would a list, tuple, or dictionary, or by taking the
depset's size via `len()`. This functionality is [deprecated](https://docs.bazel.build/versions/master/skylark/backward-compatibility.html#depset-is-no-longer-iterable).
and will be removed.

### Avoid calling `len(depset)`

It is O(N) to get the number of items in a depset. It is however
O(1) to check if a depset is empty. This includes checking the truthiness
of a depset:

```
def _impl(ctx):
  args = ctx.actions.args()
  files = depset(...)

  # Bad, has to iterate over entire depset to get length
  if len(files) != 0:
    args.add("--files")
    args.add_all(files)

  # Good, O(1)
  if files:
    args.add("--files")
    args.add_all(files)
```

As mentioned above, support for `len(<depset>)` is deprecated.

## Use `ctx.actions.args()` for command lines

When building command lines you should use [ctx.actions.args()](lib/Args.html).
This defers expansion of any depsets to the execution phase.

Apart from being strictly faster, this will reduce the memory consumption of
your rules -- sometimes by 90% or more.

Here are some tricks:

* Pass depsets and lists directly as arguments, instead of flattening them
yourself. They will get expanded by `ctx.actions.args()` for you.
If you need any transformations on the depset contents, look at
[ctx.actions.args#add](lib/Args.html#add) to see if anything fits the bill.

* Are you passing `File#path` as arguments? No need. Any
[File](lib/File.html) is automatically turned into its
[path](lib/File.html#path), deferred to expansion time.

* Avoid constructing strings by concatenating them together.
The best string argument is a constant as its memory will be shared between
all instances of your rule.

* If the args are too long for the command line an `ctx.actions.args()` object
can be conditionally or unconditionally written to a param file using
[`ctx.actions.args#use_param_file`](lib/Args.html#use_param_file). This is
done behind the scenes when the action is executed. If you need to explictly
control the params file you can write it manually using
[`ctx.actions.write`](lib/actions.html#write).

Example:

```
def _impl(ctx):
  ...
  args = ctx.actions.Args()
  file = ctx.declare_file(...)
  files = depset(...)

  # Bad, constructs a full string "--foo=<file path>" for each rule instance
  args.add("--foo=" + file.path)

  # Good, shares "-foo" among all rule instances, and defers file.path to later
  args.add("--foo")
  args.add(file)

  # Use format if you prefer ["--foo=<file path>"] to ["--foo", <file path>]
  args.add(format="--foo=%s", value=file)

  # Bad, makes a giant string of a whole depset
  args.add(" ".join(["-I%s" % file.short_path for file in files])

  # Good, only stores a reference to the depset
  args.add_all(files, format_each="-I%s", map_each=_to_short_path)

# Function passed to map_each above
def _to_short_path(f):
  return f.short_path
```

## Transitive action inputs should be depsets

When building an action using [ctx.actions.run](lib/actions.html?#run), do not
forget that the `inputs` field accepts a depset. Use this whenever inputs are
collected from dependencies transitively.

```
inputs = depset(...)
ctx.actions.run(
  inputs = inputs,  # Do *not* turn inputs into a list
  ...
)
```

## Performance profiling

To profile your code and analyze the performance, use the `--profile` flag:

```
$ bazel build --nobuild --profile=/tmp/prof //path/to:target
$ bazel analyze-profile /tmp/prof --html --html_details
```

Then, open the generated HTML file (`/tmp/prof.html` in the example).

## Memory Profiling

Bazel comes with a built-in memory profiler that can help you check your rule's
memory use. If there is a problem you can dump the heap to find the
exact line of code that is causing the problem.

### Enabling Memory Tracking

You must pass these two startup flags to *every* Bazel invocation:

  ```
  STARTUP_FLAGS=\
  --host_jvm_args=-javaagent:$(BAZEL)/third_party/allocation_instrumenter/java-allocation-instrumenter-3.0.1.jar \
  --host_jvm_args=-DRULE_MEMORY_TRACKER=1
  ```
  **NOTE**: The bazel repository comes with an allocation instrumenter.
  Make sure to adjust '$(BAZEL)' for your repository location.

These start the server in memory tracking mode. If you forget these for even
one Bazel invocation the server will restart and you will have to start over.

### Using the Memory Tracker

Let's have a look at the target `foo` and see what it's up to. We add
`--nobuild` since it doesn't matter to memory consumption if we actually build
or not, we just have to run the analysis phase.

```
$ bazel $(STARTUP_FLAGS) build --nobuild //foo:foo
```

Let's see how much memory the whole Bazel instance consumes:

```
$ bazel $(STARTUP_FLAGS) info used-heap-size-after-gc
> 2594MB
```

Let's break it down by rule class by using `bazel dump --rules`:

```
$ bazel $(STARTUP_FLAGS) dump --rules
>

RULE                                 COUNT     ACTIONS          BYTES         EACH
genrule                             33,762      33,801    291,538,824        8,635
config_setting                      25,374           0     24,897,336          981
filegroup                           25,369      25,369     97,496,272        3,843
cc_library                           5,372      73,235    182,214,456       33,919
proto_library                        4,140     110,409    186,776,864       45,115
android_library                      2,621      36,921    218,504,848       83,366
java_library                         2,371      12,459     38,841,000       16,381
_gen_source                            719       2,157      9,195,312       12,789
_check_proto_library_deps              719         668      1,835,288        2,552
... (more output)
```

And finally let's have a look at where the memory is going by producing a
`pprof` file using `bazel dump --skylark_memory`:

```
$ bazel $(STARTUP_FLAGS) dump --skylark_memory=$HOME/prof.gz
> Dumping skylark heap to: /usr/local/google/home/$USER/prof.gz
```

Next, we use the `pprof` tool to investigate the heap. A good starting point is
getting a flame graph by using `pprof -flame $HOME/prof.gz`.

  You can get `pprof` from https://github.com/google/pprof.

In this case we get a text dump of the hottest call sites annotated with lines:

```
$ pprof -text -lines $HOME/prof.gz
>
      flat  flat%   sum%        cum   cum%
  146.11MB 19.64% 19.64%   146.11MB 19.64%  android_library <native>:-1
  113.02MB 15.19% 34.83%   113.02MB 15.19%  genrule <native>:-1
   74.11MB  9.96% 44.80%    74.11MB  9.96%  glob <native>:-1
   55.98MB  7.53% 52.32%    55.98MB  7.53%  filegroup <native>:-1
   53.44MB  7.18% 59.51%    53.44MB  7.18%  sh_test <native>:-1
   26.55MB  3.57% 63.07%    26.55MB  3.57%  _generate_foo_files /foo/tc/tc.bzl:491
   26.01MB  3.50% 66.57%    26.01MB  3.50%  _build_foo_impl /foo/build_test.bzl:78
   22.01MB  2.96% 69.53%    22.01MB  2.96%  _build_foo_impl /foo/build_test.bzl:73
   ... (more output)
```
