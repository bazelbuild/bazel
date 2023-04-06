Project: /_project.yaml
Book: /_book.yaml

# Rules Tutorial

{% include "_buttons.html" %}

<!-- [TOC] -->

[Starlark](https://github.com/bazelbuild/starlark) is a Python-like
configuration language originally developed for use in Bazel and since adopted
by other tools. Bazel's `BUILD` and `.bzl` files are written in a dialect of
Starlark properly known as the "Build Language", though it is often simply
referred to as "Starlark", especially when emphasizing that a feature is
expressed in the Build Language as opposed to being a built-in or "native" part
of Bazel. Bazel augments the core language with numerous build-related functions
such as `glob`, `genrule`, `java_binary`, and so on.

See the
[Bazel](/start/) and [Starlark](/extending/concepts) documentation for
more details, and the
[Rules SIG template](https://github.com/bazel-contrib/rules-template) as a
starting point for new rulesets.

## The empty rule

To create your first rule, create the file `foo.bzl`:

```python
def _foo_binary_impl(ctx):
    pass

foo_binary = rule(
    implementation = _foo_binary_impl,
)
```

When you call the [`rule`](lib/globals#rule) function, you
must define a callback function. The logic will go there, but you
can leave the function empty for now. The [`ctx`](lib/ctx) argument
provides information about the target.

You can load the rule and use it from a `BUILD` file.

Create a `BUILD` file in the same directory:

```python
load(":foo.bzl", "foo_binary")

foo_binary(name = "bin")
```

Now, the target can be built:

```
$ bazel build bin
INFO: Analyzed target //:bin (2 packages loaded, 17 targets configured).
INFO: Found 1 target...
Target //:bin up-to-date (nothing to build)
```

Even though the rule does nothing, it already behaves like other rules: it has a
mandatory name, it supports common attributes like `visibility`, `testonly`, and
`tags`.

## Evaluation model

Before going further, it's important to understand how the code is evaluated.

Update `foo.bzl` with some print statements:

```python
def _foo_binary_impl(ctx):
    print("analyzing", ctx.label)

foo_binary = rule(
    implementation = _foo_binary_impl,
)

print("bzl file evaluation")
```

and BUILD:

```python
load(":foo.bzl", "foo_binary")

print("BUILD file")
foo_binary(name = "bin1")
foo_binary(name = "bin2")
```

[`ctx.label`](lib/ctx#label)
corresponds to the label of the target being analyzed. The `ctx` object has
many useful fields and methods; you can find an exhaustive list in the
[API reference](lib/ctx).

Query the code:

```
$ bazel query :all
DEBUG: /usr/home/bazel-codelab/foo.bzl:8:1: bzl file evaluation
DEBUG: /usr/home/bazel-codelab/BUILD:2:1: BUILD file
//:bin2
//:bin1
```

Make a few observations:

* "bzl file evaluation" is printed first. Before evaluating the `BUILD` file,
  Bazel evaluates all the files it loads. If multiple `BUILD` files are loading
  foo.bzl, you would see only one occurrence of "bzl file evaluation" because
  Bazel caches the result of the evaluation.
* The callback function `_foo_binary_impl` is not called. Bazel query loads
  `BUILD` files, but doesn't analyze targets.

To analyze the targets, use the [`cquery`](/query/cquery) ("configured
query") or the `build` command:

```
$ bazel build :all
DEBUG: /usr/home/bazel-codelab/foo.bzl:2:5: analyzing //:bin1
DEBUG: /usr/home/bazel-codelab/foo.bzl:2:5: analyzing //:bin2
INFO: Analyzed 2 targets (0 packages loaded, 0 targets configured).
INFO: Found 2 targets...
```

As you can see, `_foo_binary_impl` is now called twice - once for each target.

Notice that neither "bzl file evaluation" nor "BUILD file" are printed again,
because the evaluation of `foo.bzl` is cached after the call to `bazel query`.
Bazel only emits `print` statements when they are actually executed.

## Creating a file

To make your rule more useful, update it to generate a file. First, declare the
file and give it a name. In this example, create a file with the same name as
the target:

```python
ctx.actions.declare_file(ctx.label.name)
```

If you run `bazel build :all` now, you will get an error:

```
The following files have no generating action:
bin2
```

Whenever you declare a file, you have to tell Bazel how to generate it by
creating an action. Use [`ctx.actions.write`](lib/actions#write),
to create a file with the given content.

```python
def _foo_binary_impl(ctx):
    out = ctx.actions.declare_file(ctx.label.name)
    ctx.actions.write(
        output = out,
        content = "Hello\n",
    )
```

The code is valid, but it won't do anything:

```
$ bazel build bin1
Target //:bin1 up-to-date (nothing to build)
```

The `ctx.actions.write` function registered an action, which taught Bazel
how to generate the file. But Bazel won't create the file until it is
actually requested. So the last thing to do is tell Bazel that the file
is an output of the rule, and not a temporary file used within the rule
implementation.

```python
def _foo_binary_impl(ctx):
    out = ctx.actions.declare_file(ctx.label.name)
    ctx.actions.write(
        output = out,
        content = "Hello!\n",
    )
    return [DefaultInfo(files = depset([out]))]
```

Look at the `DefaultInfo` and `depset` functions later. For now,
assume that the last line is the way to choose the outputs of a rule.

Now, run Bazel:

```
$ bazel build bin1
INFO: Found 1 target...
Target //:bin1 up-to-date:
  bazel-bin/bin1

$ cat bazel-bin/bin1
Hello!
```

You have successfully generated a file!

## Attributes

To make the rule more useful, add new attributes using
[the `attr` module](lib/attr) and update the rule definition.

Add a string attribute called `username`:

```python
foo_binary = rule(
    implementation = _foo_binary_impl,
    attrs = {
        "username": attr.string(),
    },
)
```

Next, set it in the `BUILD` file:

```python
foo_binary(
    name = "bin",
    username = "Alice",
)
```

To access the value in the callback function, use `ctx.attr.username`. For
example:

```python
def _foo_binary_impl(ctx):
    out = ctx.actions.declare_file(ctx.label.name)
    ctx.actions.write(
        output = out,
        content = "Hello {}!\n".format(ctx.attr.username),
    )
    return [DefaultInfo(files = depset([out]))]
```

Note that you can make the attribute mandatory or set a default value. Look at
the documentation of [`attr.string`](lib/attr#string).
You may also use other types of attributes, such as [boolean](lib/attr#bool)
or [list of integers](lib/attr#int_list).

## Dependencies

Dependency attributes, such as [`attr.label`](lib/attr#label)
and [`attr.label_list`](lib/attr#label_list),
declare a dependency from the target that owns the attribute to the target whose
label appears in the attribute's value. This kind of attribute forms the basis
of the target graph.

In the `BUILD` file, the target label appears as a string object, such as
`//pkg:name`. In the implementation function, the target will be accessible as a
[`Target`](lib/Target) object. For example, view the files returned
by the target using [`Target.files`](lib/Target#modules.Target.files).

### Multiple files

By default, only targets created by rules may appear as dependencies (such as a
`foo_library()` target). If you want the attribute to accept targets that are
input files (such as source files in the repository), you can do it with
`allow_files` and specify the list of accepted file extensions (or `True` to
allow any file extension):

```python
"srcs": attr.label_list(allow_files = [".java"]),
```

The list of files can be accessed with `ctx.files.<attribute name>`. For
example, the list of files in the `srcs` attribute can be accessed through

```python
ctx.files.srcs
```

### Single file

If you need only one file, use `allow_single_file`:

```python
"src": attr.label(allow_single_file = [".java"])
```

This file is then accessible under `ctx.file.<attribute name>`:

```python
ctx.file.src
```

## Create a file with a template

You can create a rule that generates a .cc file based on a template. Also, you
can use `ctx.actions.write` to output a string constructed in the rule
implementation function, but this has two problems. First, as the template gets
bigger, it becomes more memory efficient to put it in a separate file and avoid
constructing large strings during the analysis phase. Second, using a separate
file is more convenient for the user. Instead, use
[`ctx.actions.expand_template`](lib/actions#expand_template),
which performs substitutions on a template file.

Create a `template` attribute to declare a dependency on the template
file:

```python
def _hello_world_impl(ctx):
    out = ctx.actions.declare_file(ctx.label.name + ".cc")
    ctx.actions.expand_template(
        output = out,
        template = ctx.file.template,
        substitutions = {"{NAME}": ctx.attr.username},
    )
    return [DefaultInfo(files = depset([out]))]

hello_world = rule(
    implementation = _hello_world_impl,
    attrs = {
        "username": attr.string(default = "unknown person"),
        "template": attr.label(
            allow_single_file = [".cc.tpl"],
            mandatory = True,
        ),
    },
)
```

Users can use the rule like this:

```python
hello_world(
    name = "hello",
    username = "Alice",
    template = "file.cc.tpl",
)

cc_binary(
    name = "hello_bin",
    srcs = [":hello"],
)
```

If you don't want to expose the template to the end-user and always use the
same one, you can set a default value and make the attribute private:

```python
    "_template": attr.label(
        allow_single_file = True,
        default = "file.cc.tpl",
    ),
```

Attributes that start with an underscore are private and cannot be set in a
`BUILD` file. The template is now an _implicit dependency_: Every `hello_world`
target has a dependency on this file. Don't forget to make this file visible
to other packages by updating the `BUILD` file and using
[`exports_files`](/reference/be/functions#exports_files):

```python
exports_files(["file.cc.tpl"])
```

## Going further

*   Take a look at the [reference documentation for rules](/extending/rules#contents).
*   Get familiar with [depsets](/extending/depsets).
*   Check out the [examples repository](https://github.com/bazelbuild/examples/tree/master/rules)
    which includes additional examples of rules.
