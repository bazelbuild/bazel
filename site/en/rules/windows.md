Project: /_project.yaml
Book: /_book.yaml

# Writing Rules on Windows

{% include "_buttons.html" %}

This page focuses on writing Windows-compatible rules, common problems of
writing portable rules, and some solutions.

## Paths

Problems:

- **Length limit**: maximum path length is 259 characters.

  Though Windows also supports longer paths (up to 32767 characters), many programs are built with
  the lower limit.

  Be aware of this about programs you run in the actions.

- **Working directory**: is also limited to 259 characters.

  Processes cannot `cd` into a directory longer than 259 characters.

- **Case-sensitivity**: Windows paths are case-insensitive, Unix paths are case-sensitive.

  Be aware of this when creating command lines for actions.

- **Path separators**: are backslash (`\`), not forward slash (`/`).

  Bazel stores paths Unix-style with `/` separators. Though some Windows programs support
  Unix-style paths, others don't. Some built-in commands in cmd.exe support them, some don't.

  It's best to always use `\` separators on Windows: replace `/` with `\` when you create command
  lines and environment variables for actions.

- **Absolute paths**: don't start with slash (`/`).

  Absolute paths on Windows start with a drive letter, such as `C:\foo\bar.txt`. There's no single
  filesystem root.

  Be aware of this if your rule checks if a path is absolute. Absolute paths
  should be avoided since they are often non-portable.

Solutions:

- **Keep paths short.**

  Avoid long directory names, deeply nested directory structures, long file names, long workspace
  names, long target names.

  All of these may become path components of actions' input files, and may exhaust the path length
  limit.

- **Use a short output root.**

  Use the `--output_user_root=<path>` flag to specify a short path for Bazel outputs. A good idea
  is to have a drive (or virtual drive) just for Bazel outputs (such as `D:\`), and adding this line
  to your `.bazelrc` file:

  ```
  build --output_user_root=D:/
  ```

  or

  ```
  build --output_user_root=C:/_bzl
  ```

- **Use junctions.**

  Junctions are, loosely speaking<sup>[1]</sup>, directory symlinks. Junctions are easy to create
  and can point to directories (on the same computer) with long paths. If a build action creates a
  junction whose path is short but whose target is long, then tools with short path limit can access
  the files in the junction'ed directory.

  In `.bat` files or in cmd.exe you can create junctions like so:

  ```
  mklink /J c:\path\to\junction c:\path\to\very\long\target\path
  ```

  <sup>[1]</sup>: Strictly speaking
  [Junctions are not Symbolic Links](https://superuser.com/a/343079), but for
  the sake of build actions you may regard Junctions as Directory Symlinks.

- **Replace `/` with `\` in paths in actions / envvars.**

  When you create the command line or environment variables for an action, make the paths
  Windows-style. Example:

  ```python
  def as_path(p, is_windows):
      if is_windows:
          return p.replace("/", "\\")
      else:
          return p
  ```

## Environment variables

Problems:

- **Case-sensitivity**: Windows environment variable names are case-insensitive.

  For example, in Java `System.getenv("SystemRoot")` and `System.getenv("SYSTEMROOT")` yields the
  same result. (This applies to other languages too.)

- **Hermeticity**: actions should use as few custom environment variables as possible.

  Environment variables are part of the action's cache key. If an action uses environment variables
  that change often, or are custom to users, that makes the rule less cache-able.

Solutions:

- **Only use upper-case environment variable names.**

  This works on Windows, macOS, and Linux.

- **Minimize action environments.**

  When using `ctx.actions.run`, set the environment to `ctx.configuration.default_shell_env`. If the
  action needs more environment variables, put them all in a dictionary and pass that to the action.
  Example:

  ```python
  load("@bazel_skylib//lib:dicts.bzl", "dicts")

  def _make_env(ctx, output_file, is_windows):
      out_path = output_file.path
      if is_windows:
          out_path = out_path.replace("/", "\\")
      return dicts.add(ctx.configuration.default_shell_env, {"MY_OUTPUT": out_path})
  ```

## Actions

Problems:

- **Executable outputs**: Every executable file must have an executable extension.

  The most common extensions are `.exe` (binary files) and `.bat` (Batch scripts).

  Be aware that shell scripts (`.sh`) are NOT executable on Windows; you cannot specify them as
  `ctx.actions.run`'s `executable`. There's also no `+x` permission that files can have, so you
  can't execute arbitrary files like on Linux.

- **Bash commands**: For sake of portability, avoid running Bash commands directly in actions.

  Bash is widespread on Unix-like systems, but it's often unavailable on Windows. Bazel itself is
  relying less and less on Bash (MSYS2), so in the future users would be less likely to have MSYS2
  installed along with Bazel. To make rules easier to use on Windows, avoid running Bash commands in
  actions.

- **Line endings**: Windows uses CRLF (`\r\n`), Unix-like systems uses LF (`\n`).

  Be aware of this when comparing text files. Be mindful of your Git settings, especially of line
  endings when checking out or committing. (See Git's `core.autocrlf` setting.)

Solutions:

- **Use a Bash-less purpose-made rule.**

  `native.genrule()` is a wrapper for Bash commands, and it's often used to solve simple problems
  like copying a file or writing a text file. You can avoid relying on Bash (and reinventing the
  wheel): see if bazel-skylib has a purpose-made rule for your needs. None of them depends on Bash
  when built/tested on Windows.

  Build rule examples:

  - `copy_file()`
    ([source](https://github.com/bazelbuild/bazel-skylib/blob/main/rules/copy_file.bzl),
    [documentation](https://github.com/bazelbuild/bazel-skylib/blob/main/docs/copy_file_doc.md)):
    copies a file somewhere else, optionally making it executable

  - `write_file()`
    ([source](https://github.com/bazelbuild/bazel-skylib/blob/main/rules/write_file.bzl),
    [documentation](https://github.com/bazelbuild/bazel-skylib/blob/main/docs/write_file_doc.md)):
    writes a text file, with the desired line endings (`auto`, `unix`, or `windows`), optionally
    making it executable (if it's a script)

  - `run_binary()`
    ([source](https://github.com/bazelbuild/bazel-skylib/blob/main/rules/run_binary.bzl),
    [documentation](https://github.com/bazelbuild/bazel-skylib/blob/main/docs/run_binary_doc.md)):
    runs a binary (or `*_binary` rule) with given inputs and expected outputs as a build action
    (this is a build rule wrapper for `ctx.actions.run`)

  - `native_binary()`
    ([source](https://github.com/bazelbuild/bazel-skylib/blob/main/rules/native_binary.bzl),
    [documentation](https://github.com/bazelbuild/bazel-skylib/blob/main/docs/native_binary_doc.md#native_binary)):
    wraps a native binary in a `*_binary` rule, which you can `bazel run` or use in `run_binary()`'s
    `tool` attribute or `native.genrule()`'s `tools` attribute

  Test rule examples:

  - `diff_test()`
    ([source](https://github.com/bazelbuild/bazel-skylib/blob/main/rules/diff_test.bzl),
    [documentation](https://github.com/bazelbuild/bazel-skylib/blob/main/docs/diff_test_doc.md)):
    test that compares contents of two files

  - `native_test()`
    ([source](https://github.com/bazelbuild/bazel-skylib/blob/main/rules/native_binary.bzl),
    [documentation](https://github.com/bazelbuild/bazel-skylib/blob/main/docs/native_binary_doc.md#native_test)):
    wraps a native binary in a `*_test` rule, which you can `bazel test`

- **On Windows, consider using `.bat` scripts for trivial things.**

  Instead of `.sh` scripts, you can solve trivial tasks with `.bat` scripts.

  For example, if you need a script that does nothing, or prints a message, or exits with a fixed
  error code, then a simple `.bat` file will suffice. If your rule returns a `DefaultInfo()`
  provider, the `executable` field may refer to that `.bat` file on Windows.

  And since file extensions don't matter on macOS and Linux, you can always use `.bat` as the
  extension, even for shell scripts.

  Be aware that empty `.bat` files cannot be executed. If you need an empty script, write one space
  in it.

- **Use Bash in a principled way.**

  In Starlark build and test rules, use `ctx.actions.run_shell` to run Bash scripts and Bash
  commands as actions.

  In Starlark macros, wrap Bash scripts and commands in a `native.sh_binary()` or
  `native.genrule()`. Bazel will check if Bash is available and run the script or command through
  Bash.

  In Starlark repository rules, try avoiding Bash altogether. Bazel currently offers no way to run
  Bash commands in a principled way in repository rules.

## Deleting files

Problems:

- **Files cannot be deleted while open.**

  Open files cannot be deleted (by default), attempts result in "Access Denied"
  errors. If you cannot delete a file, maybe a running process still holds it
  open.

- **Working directory of a running process cannot be deleted.**

  Processes have an open handle to their working directory, and the directory cannot be deleted
  until the process terminates.

Solutions:

- **In your code, try to close files eagerly.**

  In Java, use `try-with-resources`. In Python, use `with open(...) as f:`. In principle, try
  closing handles as soon as possible.

<!--
TODO:
- runfiles, runfiles libraries, -nolegacy_external_runfiles
- runfiles envvars, runfiles manifest structure
- avoid using runfiles for things that could be inputs
- whether to use runfiles manifest on non-windows
- how to patch tools that expect to read from the filesystem to do a lookup through the manifest file instead (including helpers in many languages)
- how this applies in tests as well that rely on $TEST_SRCDIR
- unzip is slow
- cmd.exe has 8k command length limit
- put paths in envvars instead of args
- put cmd.exe commands in .bat files
- use ctx.resolve_tools instead of ctx.resolve_command (Bash dep)
- how to run cmd.exe actions (maybe I should write a genrule-like rule for these)

-->
