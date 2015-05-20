Skylark cookbook
================

## <a name="macro_native"></a>Macro creating a native rule

An example of a macro creating a native rule. Native rules are accessed using
the `native` module.

`extension.bzl`:

```python
def macro(name, visibility=None):
  # Creating a native genrule.
  native.genrule(
      name = name,
      outs = [name + '.txt'],
      cmd = 'echo hello > $@',
      visibility = visibility,
  )
```

`BUILD`:

```python
load("/pkg/extension", "macro")

macro(name = "myrule")
```

## <a name="macro_skylark"></a>Macro creating a Skylark rule

An example of a macro creating a Skylark rule.

`empty.bzl`:

```python
def _impl(ctx):
  print("This rule does nothing")

empty = rule(implementation=_impl)
```

`extension.bzl`:

```python
# Loading the Skylark rule. The rule doesn't have to be in a separate file.
load("/pkg/empty", "empty")

def macro(name, visibility=None):
  # Creating the Skylark rule.
  empty(name = name, visibility=visibility)
```

`BUILD`:

```python
load("/pkg/extension", "macro")

macro(name = "myrule")
```

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
load("/pkg/empty", "empty")

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
load("/pkg/printer", "printer")

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
the user. The output has the same name as the input, with a `.txt` suffix.

While convenient, Shell commands should be used carefully. Generating the
command-line can lead to escaping and injection issues. It can also create
portability problems. It is often better to declare a binary target in a
BUILD file and execute it. See the example "<a href="#execute-bin">
executing a binary</a>".

`size.bzl`:

```python
def _impl(ctx):
  output = ctx.outputs.out
  input = ctx.file.file
  # The command may only access files declared in inputs.
  ctx.action(
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
load("/pkg/size", "size")

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
load("/pkg/file", "file")

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
  ctx.action(
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
      "_merge_tool": attr.label(executable=True, default=Label("//pkg:merge"))
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
#! /bin/bash

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
  # ctx.new_file is used for temporary files.
  # If it should be visible for user, declare it in rule.outputs instead.
  f = ctx.new_file(ctx.configuration.bin_dir, "hello")
  # As with outputs, each time you declare a file,
  # you need an action to generate it.
  ctx.file_action(output=f, content=ctx.attr.input_content)

  ctx.action(
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
      "binary": attr.label(cfg=HOST_CFG, mandatory=True, allow_files=True,
                           executable=True),
      "input_content": attr.string(),
      "out": attr.output(mandatory=True),
      },
)
```

`a.sh`:

```bash
#! /bin/bash

tr 'a-z' 'A-Z' < $1 > $2
```

`BUILD`:

```python
load("/pkg/execute", "execute")

execute(
    name = "e",
    input_content = "some text",
    binary = "a.sh",
    out = "foo",
)
```

## <a name="runfiles"></a>Define simple runfiles

`execute.bzl`:

```python
def _impl(ctx):
  executable = ctx.outputs.executable
  # Create the output executable file with command as its content.
  ctx.file_action(
      output=executable,
      content=ctx.attr.command,
      executable=True)

  return struct(
      # Create runfiles from the files specified in the data attribute.
      # The shell executable - the output of this rule - can use them at runtime.
      # It is also possible to define data_runfiles and default_runfiles.
      # However if runfiles is specified it's not possible to define the above
      # ones since runfiles sets them both.
      # Remember, that the struct returned by the implementation function needs
      # to have a field named "runfiles" in order to create the actual runfiles
      # symlink tree.
      runfiles=ctx.runfiles(files=ctx.files.data)
  )

execute = rule(
  implementation=_impl,
  executable=True,
  attrs={
      "command": attr.string(),
      "data": attr.label_list(cfg=DATA_CFG, allow_files=True),
      },
)
```

`data.txt`:

```
Hello World!
```

`BUILD`:

```python
load("/pkg/execute", "execute")

execute(
    name = "e",
    # The path to data.txt has to include the package directories as well. I.e.
    # if the BUILD file is under foo/BUILD and the data file is foo/data.txt
    # then it needs to be referred as foo/data.txt in the command.
    command = "cat data.txt",
    data = [':data.txt']
)
```


## <a name="mandatory-providers"></a>Mandatory providers

In this example, rules have a `number` attribute. Each rule adds its
number with the numbers of its transitive dependencies, and write the
result in a file. This shows how to transfer information from a dependency
to its dependents.

`sum.bzl`:

```python
def _impl(ctx):
  result = ctx.attr.number
  for i in ctx.attr.deps:
    result += i.number
  ctx.file_action(output=ctx.outputs.out, content=str(result))

  # Fields in the struct will be visible by other rules.
  return struct(number=result)

sum = rule(
  implementation=_impl,
  attrs={
      "number": attr.int(default=1),
      # All deps must provide all listed providers.
      "deps": attr.label_list(providers=["number"]),
  },
  outputs = {"out": "%{name}.sum"}
)
```

`BUILD`:

```python
load("/pkg/sum", "sum")

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
def _impl(ctx):
  result = ctx.attr.number
  for i in ctx.attr.deps:
    if hasattr(i, "number"):
      result += i.number
  ctx.file_action(output=ctx.outputs.out, content=str(result))

  # Fields in the struct will be visible by other rules.
  return struct(number=result)

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
load("/pkg/sum", "sum")

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
load("/pkg/extension", "executable_rule")

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
load("/pkg/extension", "rule_with_outputs")

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
load("/pkg/extension", "rule_with_outputs")

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
  files = set()
  files += ctx.attr.dep_rule_1.files
  files += ctx.attr.dep_rule_2.files
  return struct(files=files)

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
load("/pkg/extension", "macro")

# This creates the target :my_rule
macro(
    name = "my_rule",
    cmd = "echo something > $@",
    input = "Hello World"
)
```

## <a name="debugging-tips"></a>Debugging tips

Here are some examples on how to do debug Skylark macros and rules.

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
load("/pkg/debug", "debug")

debug(
  name = "printing_rule"
)
```
