---
layout: documentation
title: Extensions examples
---
# Extensions examples

## <a name="macro"></a>Macro creating a rule

An example of a macro creating a rule.

`empty.bzl`:

```python
def _impl(ctx):
  print("This rule does nothing")

empty = rule(implementation=_impl)
```

`extension.bzl`:

```python
# Loading the rule. The rule doesn't have to be in a separate file.
load("//pkg:empty.bzl", "empty")

def macro(name, visibility=None):
  # Creating the rule.
  empty(name = name, visibility = visibility)
```

`BUILD`:

```python
load("//pkg:extension.bzl", "macro")

macro(name = "myrule")
```

## <a name="macro_native"></a>Macro creating a native rule

An example of a macro creating a native rule. Native rules are special rules
that are automatically available (without <code>load</code>). They are
accessed using the <a href="lib/native.html">native</a> module.

`extension.bzl`:

```python
def macro(name, visibility=None):
  # Creating a native genrule.
  native.genrule(
      name = name,
      outs = [name + ".txt"],
      cmd = "echo hello > $@",
      visibility = visibility,
  )
```

`BUILD`:

```python
load("//pkg:extension.bzl", "macro")

macro(name = "myrule")
```

## <a name="macro_compound"></a>Macro multiple rules

There's currently no easy way to create a rule that directly uses the
action of a native rule. You can work around this using macros:

```python
def _impl(ctx):
  return struct([...],
                # When instrumenting this rule, again hide implementation from
                # users.
                instrumented_files(
                  source_attributes = ["srcs", "csrcs"],
                  dependency_attributes = ["deps", "cdeps"]))

# This rule is private and can only be accessed from the current file.
_cc_and_something_else_binary = rule(implementation=_impl)


# This macro is public, it's the public interface to instantiate the rule.
def cc_and_something_else_binary(name, srcs, deps, csrcs, cdeps):
   cc_binary_name = "%s.cc_binary" % name

   native.cc_binary(
      name = cc_binary_name,
      srcs = csrcs,
      deps = cdeps,
      visibility = ["//visibility:private"]
  )

  _cc_and_something_else_binary(
    name = name,
    srcs = srcs,
    deps = deps,
    # A label attribute so that this depends on the internal rule.
    cc_binary = cc_binary_name,
    # Redundant labels attributes so that the rule with this target name knows
    # about everything it would know about if cc_and_something_else_binary
    # were an actual rule instead of a macro.
    csrcs = csrcs,
    cdeps = cdeps)
```


## <a name="conditional-instantiation"></a>Conditional instantiation

Macros can look at previously instantiated rules. This is done with
`native.existing_rule`, which returns information on a single rule defined in
the same `BUILD` file, eg.,

```python
native.existing_rule("descriptor_proto")
```

This is useful to avoid instantiating the same rule twice, which is an
error. For example, the following macro will simulate a test suite,
instantiating tests for diverse flavors of the same test.

`extension.bzl`:

```python
def system_test(name, test_file, flavor):
  n = "system_test_%s_%s_test" % (test_file, flavor)
  if native.existing_rule(n) == None:
    native.py_test(
        name = n,
        srcs = [
            "test_driver.py",
            test_file,
        ],
        args = ["--flavor=" + flavor],
    )
  return n

def system_test_suite(name, flavors=["default"], test_files=[]):
  ts = []
  for flavor in flavors:
    for test in test_files:
      ts.append(system_test(name, test, flavor))
  native.test_suite(name = name, tests = ts)
```

In the following BUILD file, note how `(basic_test.py, fast)` is emitted for
both the `smoke` test suite and the `thorough` test suite.

`BUILD`:

```python
load("//pkg:extension.bzl", "system_test_suite")

# Run all files through the 'fast' flavor.
system_test_suite(
    name = "smoke",
    flavors = ["fast"],
    test_files = glob(["*_test.py"]),
)

# Run the basic test through all flavors.
system_test_suite(
    name = "thorough",
    flavors = [
        "fast",
        "debug",
        "opt",
    ],
    test_files = ["basic_test.py"],
)
```


## <a name="aggregation"></a>Aggregating over the BUILD file

Macros can collect information from the BUILD file as processed so far.  We call
this aggregation. The typical example is collecting data from all rules of a
certain kind.  This is done by calling
<a href="lib/native.html#existing_rules">native.existing\_rules</a>, which
returns a dictionary representing all rules defined so far in the current BUILD
file. The dictionary has entries of the form `name` => `rule`, with the values
using the same format as `native.existing_rule`.

```python
def archive_cc_src_files(tag):
  """Create an archive of all C++ sources that have the given tag."""
  all_src = []
  for r in native.existing_rules().values():
    if tag in r["tags"] and r["kind"] == "cc_library":
      all_src.append(r["srcs"])
  native.genrule(cmd = "zip $@ $^", srcs = all_src, outs = ["out.zip"])
```

Since `native.existing_rules` constructs a potentially large dictionary, you
should avoid calling it repeatedly within BUILD file.

## <a name="empty"></a>Empty rule

Minimalist example of a rule that does nothing. If you build it, the target will
succeed (with no generated file).

`empty.bzl`:

```python
def _impl(ctx):
  # You may use print for debugging.
  print("This rule does nothing")

empty = rule(implementation=_impl)
```

`BUILD`:

```python
load("//pkg:empty.bzl", "empty")

empty(name = "nothing")
```

## <a name="attr"></a>Rule with attributes

Example of a rule that shows how to declare attributes and access them.

`printer.bzl`:

```python
def _impl(ctx):
  # You may use print for debugging.
  print("Rule name = %s, package = %s" % (ctx.label.name, ctx.label.package))

  # This prints the labels of the deps attribute.
  print("There are %d deps" % len(ctx.attr.deps))
  for i in ctx.attr.deps:
    print("- %s" % i.label)
    # A label can represent any number of files (possibly 0).
    print("  files = %s" % [f.path for f in i.files])

printer = rule(
    implementation=_impl,
    attrs={
      # Do not declare "name": It is added automatically.
      "number": attr.int(default = 1),
      "deps": attr.label_list(allow_files=True),
    })
```

`BUILD`:

```python
load("//pkg:printer.bzl", "printer")

printer(
    name = "nothing",
    deps = [
        "BUILD",
        ":other",
    ],
)

printer(name = "other")
```

If you execute this file, some information is printed as a warning by the
rule. No file is generated.

## <a name="shell"></a>Simple shell command

Example of a rule that runs a shell command on an input file specified by
the user. The output has the same name as the rule, with a `.size` suffix.

While convenient, Shell commands should be used carefully. Generating the
command-line can lead to escaping and injection issues. It can also create
portability problems. It is often better to declare a binary target in a
BUILD file and execute it. See the example [executing a binary](#execute-bin).

`size.bzl`:

```python
def _impl(ctx):
  output = ctx.outputs.out
  input = ctx.file.file
  # The command may only access files declared in inputs.
  ctx.actions.run_shell(
      inputs=[input],
      outputs=[output],
      progress_message="Getting size of %s" % input.short_path,
      command="stat -L -c%%s %s > %s" % (input.path, output.path))

size = rule(
    implementation=_impl,
    attrs={"file": attr.label(mandatory=True, allow_files=True, single_file=True)},
    outputs={"out": "%{name}.size"},
)
```

`foo.txt`:

```
Hello
```

`BUILD`:

```python
load("//pkg:size.bzl", "size")

size(
    name = "foo_size",
    file = "foo.txt",
)
```

## <a name="file"></a>Write string to a file

Example of a rule that writes a string to a file.

`file.bzl`:

```python
def _impl(ctx):
  output = ctx.outputs.out
  ctx.file_action(output=output, content=ctx.attr.content)

file = rule(
    implementation=_impl,
    attrs={"content": attr.string()},
    outputs={"out": "%{name}.txt"},
)
```

`BUILD`:

```python
load("//pkg:file.bzl", "file")

file(
    name = "hello",
    content = "Hello world",
)
```


## <a name="execute-bin"></a>Execute a binary

This rule executes an existing binary. In this particular example, the
binary is a tool that merges files. During the analysis phase, we cannot
access any arbitrary label: the dependency must have been previously
declared. To do so, the rule needs a label attribute. In this example, we
will give the label a default value and make it private (so that it is not
visible to end users). Keeping the label private can simplify maintenance,
since you can easily change the arguments and flags you pass to the tool.

`execute.bzl`:

```python
def _impl(ctx):
  # The list of arguments we pass to the script.
  args = [ctx.outputs.out.path] + [f.path for f in ctx.files.srcs]
  # Action to call the script.
  ctx.actions.run(
      inputs=ctx.files.srcs,
      outputs=[ctx.outputs.out],
      arguments=args,
      progress_message="Merging into %s" % ctx.outputs.out.short_path,
      executable=ctx.executable._merge_tool)

concat = rule(
  implementation=_impl,
  attrs={
      "srcs": attr.label_list(allow_files=True),
      "out": attr.output(mandatory=True),
      "_merge_tool": attr.label(executable=True, cfg="host", allow_files=True,
                                default=Label("//pkg:merge"))
  }
)
```

Any executable target can be used. In this example, we will use a
`sh_binary` rule that concatenates all the inputs.

`BUILD`:

```
load("execute", "concat")

concat(
    name = "sh",
    srcs = [
        "header.html",
        "body.html",
        "footer.html",
    ],
    out = "page.html",
)

# This target is used by the shell rule.
sh_binary(
    name = "merge",
    srcs = ["merge.sh"],
)
```

`merge.sh`:

```python
#!/bin/bash

out=$1
shift
cat $* > $out
```

`header.html`:

```
<html><body>
```

`body.html`:

```
content
```

`footer.html`:

```
</body></html>
```

## <a name="execute"></a>Execute an input binary

This rule has a mandatory `binary` attribute. It is a label that can refer
only to executable rules or files.

`execute.bzl`:

```python
def _impl(ctx):
  # ctx.actions.declare_file is used for temporary files.
  f = ctx.actions.declare_file(ctx.configuration.bin_dir, "hello")
  # As with outputs, each time you declare a file,
  # you need an action to generate it.
  ctx.actions.write(output=f, content=ctx.attr.input_content)

  ctx.actions.run(
      inputs=[f],
      outputs=[ctx.outputs.out],
      executable=ctx.executable.binary,
      progress_message="Executing %s" % ctx.executable.binary.short_path,
      arguments=[
          f.path,
          ctx.outputs.out.path,  # Access the output file using
                                 # ctx.outputs.<attribute name>
      ]
  )

execute = rule(
  implementation=_impl,
  attrs={
      "binary": attr.label(cfg="host", mandatory=True, allow_files=True,
                           executable=True),
      "input_content": attr.string(),
      "out": attr.output(mandatory=True),
      },
)
```

`a.sh`:

```bash
#!/bin/bash

tr 'a-z' 'A-Z' < $1 > $2
```

`BUILD`:

```python
load("//pkg:execute.bzl", "execute")

execute(
    name = "e",
    input_content = "some text",
    binary = "a.sh",
    out = "foo",
)
```

## <a name="runfiles"></a>Runfiles and location substitution

`execute.bzl`:

```python
def _impl(ctx):
  executable = ctx.outputs.executable
  command = ctx.attr.command
  # Expand the label in the command string to a runfiles-relative path.
  # The second arg is the list of labels that may be expanded.
  command = ctx.expand_location(command, ctx.attr.data)
  # Create the output executable file with command as its content.
  ctx.file_action(
      output=executable,
      content=command,
      executable=True)

  return [DefaultInfo(
      # Create runfiles from the files specified in the data attribute.
      # The shell executable - the output of this rule - can use them at
      #  runtime. It is also possible to define data_runfiles and
      # default_runfiles. However if runfiles is specified it's not possible to
      # define the above ones since runfiles sets them both.
      # Remember, that the struct returned by the implementation function needs
      # to have a field named "runfiles" in order to create the actual runfiles
      # symlink tree.
      runfiles=ctx.runfiles(files=ctx.files.data)
  )]

execute = rule(
  implementation=_impl,
  executable=True,
  attrs={
      "command": attr.string(),
      "data": attr.label_list(cfg="data", allow_files=True),
      },
)
```

`data.txt`:

```
Hello World!
```

`BUILD`:

```python
load("//pkg:execute.bzl", "execute")

execute(
    name = "e",
    # The location will be expanded to "pkg/data.txt", and it will reference
    # the data.txt file in runfiles when this target is invoked as
    # "bazel run //pkg:e".
    command = "cat $(location :data.txt)",
    data = [":data.txt"]
)
```

## <a name="late-bound"></a>Computed dependencies

Bazel needs to know about all dependencies before doing the analysis phase and
calling the implementation function. Dependencies can be computed based on the
rule attributes: to do so, use a function as the default
value of an attribute (the attribute must be private and have type `label` or
`list of labels`). The parameters of this function must correspond to the
attributes that are accessed in the function body.

Note: For legacy reasons, the function takes the configuration as an additional
parameter. Please do not rely on the configuration since it will be removed in
the future.

The example below computes the md5 sum of a file. The file can be preprocessed
using a filter. The exact dependencies depend on the filter chosen by the user.

`hash.bzl`:

```python
_filters = {
  "comments": Label("//pkg:comments"),
  "spaces": Label("//pkg:spaces"),
  "none": None,
}

def _get_filter(filter, cfg=None): # requires attribute "filter"
  # Return the value for the attribute "_filter_bin"
  # It can be a label or None.
  return _filters[filter]

def _impl(ctx):
  src = ctx.file.src

  if not ctx.attr._filter_bin:
    # Skip the processing
    processed = src
  else:
    processed = ctx.actions.declare_file(ctx.label.name + "_processed")
    # Run the selected binary
    ctx.actions.run(
        outputs = [processed],
        inputs = [ctx.file.src],
        progress_message="Apply filter '%s'" % ctx.attr.filter,
        arguments = [ctx.file.src.path, processed.path],
        executable = ctx.executable._filter_bin)

  # Compute the hash
  out = ctx.outputs.text
  ctx.actions.run(
      outputs = [out],
      inputs = [processed],
      command = "md5sum < %s > %s" % (processed.path, out.path))

md5_sum = rule(
  implementation=_impl,
  attrs={
      "filter": attr.string(values=_filters.keys(), default="none"),
      "src": attr.label(mandatory=True, single_file=True, allow_files=True),
      "_filter_bin": attr.label(default=_get_filter, executable=True),
  },
  outputs = {"text": "%{name}.txt"})
```

`BUILD`:

```python
load("//pkg:hash.bzl", "md5_sum")

md5_sum(
    name = "hash",
    src = "hello.txt",
    filter = "spaces",
)

sh_binary(
    name = "comments",
    srcs = ["comments.sh"],
)

sh_binary(
    name = "spaces",
    srcs = ["spaces.sh"],
)
```

`hello.txt`:

```
Hello World!
```

`comments.sh`:

```
#!/bin/bash
grep -v '^ *#' $1 > $2  # Remove lines with only a Python-style comment
```

`spaces.sh`:

```
#!/bin/bash
tr -d ' ' < $1 > $2  # Remove spaces
```

## <a name="mandatory-providers"></a>Mandatory providers

In this example, rules have a `number` attribute. Each rule adds its
number with the numbers of its transitive dependencies, and write the
result in a file. This shows how to transfer information from a dependency
to its dependents.

`sum.bzl`:

```python
NumberInfo = provider()

def _impl(ctx):
  result = ctx.attr.number
  for dep in ctx.attr.deps:
    result += dep[NumberInfo].number
  ctx.file_action(output=ctx.outputs.out, content=str(result))

  # Return the provider with result, visible to other rules.
  return [NumberInfo(number=result)]

sum = rule(
  implementation=_impl,
  attrs={
      "number": attr.int(default=1),
      # All deps must provide all listed providers.
      "deps": attr.label_list(providers=[NumberInfo]),
  },
  outputs = {"out": "%{name}.sum"}
)
```

`BUILD`:

```python
load("//pkg:sum.bzl", "sum")

sum(
  name = "n",
  deps = ["n2", "n5"],
)

sum(
  name = "n2",
  number = 2,
)

sum(
  name = "n5",
  number = 5,
)
```

## <a name="optional-providers"></a>Optional providers

This is a similar example, but dependencies may not provide a number.

`sum.bzl`:

```python
NumberInfo = provider()

def _impl(ctx):
  result = ctx.attr.number
  for dep in ctx.attr.deps:
    if NumberInfo in dep:
      result += dep[NumberInfo].number
  ctx.file_action(output=ctx.outputs.out, content=str(result))

  # Return the provider with result, visible to other rules.
  return [NumberInfo(number=result)]

sum = rule(
  implementation=_impl,
  attrs={
      "number": attr.int(default=1),
      "deps": attr.label_list(),
  },
  outputs = {"out": "%{name}.sum"}
)
```

`BUILD`:

```python
load("//pkg:sum.bzl", "sum")

sum(
  name = "n",
  deps = ["n2", "n5"],
)

sum(
  name = "n2",
  number = 2,
)

sum(
  name = "n5",
  number = 5,
)
```

## <a name="outputs-executable"></a>Default executable output

This example shows how to create a default executable output.

`extension.bzl`:

```python
def _impl(ctx):
  ctx.file_action(
      # Access the executable output file using ctx.outputs.executable.
      output=ctx.outputs.executable,
      content="#!/bin/bash\necho Hello!",
      executable=True
  )
  # The executable output is added automatically to this target.

executable_rule = rule(
    implementation=_impl,
    executable=True
)
```

`BUILD`:

```python
load("//pkg:extension.bzl", "executable_rule")

executable_rule(name = "my_rule")
```

## <a name="outputs-default"></a>Default outputs

This example shows how to create default outputs for a rule.

`extension.bzl`:

```python
def _impl(ctx):
  ctx.file_action(
      # Access the default outputs using ctx.outputs.<output name>.
      output=ctx.outputs.my_output,
      content="Hello World!"
  )
  # The default outputs are added automatically to this target.

rule_with_outputs = rule(
    implementation=_impl,
    outputs = {
        # %{name} is substituted with the rule's name
        "my_output": "%{name}.txt"
    }
)
```

`BUILD`:

```python
load("//pkg:extension.bzl", "rule_with_outputs")

rule_with_outputs(name = "my_rule")
```

## <a name="outputs-custom"></a>Custom outputs

This example shows how to create custom (user defined) outputs for a rule.
This rule takes a list of output file name templates from the user and
creates each of them containing a "Hello World!" message.

`extension.bzl`:

```python
def _impl(ctx):
  # Access the custom outputs using ctx.outputs.<attribute name>.
  for output in ctx.outputs.outs:
    ctx.file_action(
        output=output,
        content="Hello World!"
    )
  # The custom outputs are added automatically to this target.

rule_with_outputs = rule(
    implementation=_impl,
    attrs={
        "outs": attr.output_list()
    }
)
```

`BUILD`:

```python
load("//pkg:extension.bzl", "rule_with_outputs")

rule_with_outputs(
    name = "my_rule",
    outs = ["my_output.txt"]
)
```

## <a name="master-rule"></a>Master rules

This example shows how to create master rules to bind other rules together. The
code below uses genrules for simplicity, but this technique is more useful with
other rules. For example, if you need to compile C++ files, you can reuse
`cc_library`.

`extension.bzl`:

```python
def _impl(ctx):
  # Aggregate the output files from the depending rules
  files = depset()
  files += ctx.attr.dep_rule_1.files
  files += ctx.attr.dep_rule_2.files
  return [DefaultInfo(files=files)]

# This rule binds the depending rules together
master_rule = rule(
    implementation=_impl,
    attrs={
        "dep_rule_1": attr.label(),
        "dep_rule_2": attr.label()
    }
)

def macro(name, cmd, input):
  # Create the depending rules
  name_1 = name + "_dep_1"
  name_2 = name + "_dep_2"
  native.genrule(
      name = name_1,
      cmd = cmd,
      outs = [name_1 + ".txt"]
  )
  native.genrule(
      name = name_2,
      cmd = "echo " + input + " >$@",
      outs = [name_2 + ".txt"]
  )
  # Create the master rule
  master_rule(
      name = name,
      dep_rule_1 = ":" + name_1,
      dep_rule_2 = ":" + name_2
  )
```

`BUILD`:

```python
load("//pkg:extension.bzl", "macro")

# This creates the target :my_rule
macro(
    name = "my_rule",
    cmd = "echo something > $@",
    input = "Hello World"
)
```

## <a name="debugging-tips"></a>Debugging tips

Here are some examples on how to debug macros and rules using
<a href="lib/globals.html#print">print</a>.

`debug.bzl`:

```python
print("print something when the module is loaded")

def _impl(ctx):
  print("print something when the rule implementation is executed")
  print(type("abc"))     # prints string, the type of "abc"
  print(dir(ctx))        # prints all the fields and methods of ctx
  print(dir(ctx.attr))   # prints all the attributes of the rule
  # prints the objects each separated with new line
  print("object1", "object2", sep="\n")

debug = rule(implementation=_impl)
```

`BUILD`:

```python
load("//pkg:debug.bzl", "debug")

debug(
  name = "printing_rule"
)
```

