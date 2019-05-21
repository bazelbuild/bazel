---
layout: documentation
title: Writing rules on Windows
---

# Writing rules on Windows

Common problems of writing platform-independent rules and ideas to overcome them.

This document focuses on writing rules that work on Windows.

## Paths

Problems:

- **Length limit**: maximum path length is 259 characters.

  While Windows also supports paths up to 32767 characters ("long paths"), many programs are built
  with the lower limits. The action your rule executes may not handle long paths either.

- **Working directory**: is also limited to 259 characters.

  Processes cannot `cd` into a directory longer than 259 characters.

- **Case-sensitivity**: Windows paths are case-insensitive, Unix paths are case-sensitive.

  Be aware of this when creating command lines for actions.

- **Path separators**: are backslash (`\`), not forward slash (`/`).

  Bazel uses paths with `/` separators, so on Windows you have to replace those with `\` when
  creating comamnd lines and environment variables for actions.

- **Absolute paths**: don't start with slash (`/`).

  Absolute paths on Windows start with a drive letter, e.g. `C:\foo\bar.txt`. There's no single
  filesystem root.

  Be aware of this if your rule checks if a path is absolute.

Solutions:

- **Keep paths short.**

  Avoid long directory names, deeply nested directory structures, long file names, long workspace
  names, long target names.

  All of these may show up in paths that actions need to consume, and may exhaust the path length
  limit.

- **Use junctions.**

  Junctions are directory symlinks. They are easy to create and can point to directories in long
  paths. If a build action create a junction whose path is short but whose target is long, even
  tools with lower path limits can access the files in the junction'ed directory.

  In `.bat` files or in cmd.exe you can create junctions like so:

  ```
  mklink /J c:\path\to\junction c:\path\to\very\long\target\path
  ```

- **Replace `/` with `\` in paths in actions / envvars.**

  When you create the command line or environment variables for an action, make the paths
  Windows-style. Example:

  ```
  def as_path(p, is_windows):
      if is_windows:
          return p.replace("/", "\\")
      else:
          return p
  ```

- **Use a short output root.**

  Use the `--output_user_root=<path>` flag to specify a short path for Bazel outputs. A good idea
  is to have a drive (or virtual drive) just for Bazel outputs (e.g. `D:\`), and adding this line
  to your `.bazelrc` file:

  ```
  build --output_user_root=D:/
  ```

  or

  ```
  build --output_user_root=C:/_bzl
  ```

## Environment Variables

Problems:

- **Case-sensitivity**: Windows environment variable names are case-insensitive.

  For example in Java `System.getenv("SystemRoot")` and `System.getenv("SYSTEMROOT")` yields the
  same result. (This applies to other languages too.)

- **Hermeticity**: actions should use as few custom environment variables as possible.

  Environment variables are part of the action's cache key. If an action uses environment varaibles
  that change often, or are custom to users, that makes the rule less cache-able.

Solutions:

- **Only use upper-case environment variable names.**

  This works on Windows, macOS, and Linux.

- **Minimize action environments.**

  When using `ctx.actions.run`, set the environment to `ctx.configuration.default_shell_env`. If the
  action needs more environment variables, put them all in a dictionary and pass that to the action.
  Example:

  ```
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

  Be aware that shell scripts (`.sh`) are NOT executable on Windows, i.e. you cannot specify them as
  `ctx.actions.run`'s `executable`. There's also no `+x` permission that files can have, so you
  can't execute arbitrary files like on e.g. Linux.

- **Bash commands**: For sake of portability, avoid running Bash commands directly in actions.

  Bash is convenient and widely present on Unix-like systems, but it's often unavailable on Windows.
  Bazel itself is relying less and less on Bash (MSYS2), so in the future, users would be less
  likely to have MSYS2 installed along with Bazel. To make rules easier to use on Windows, avoid
  running Bash commands in actions.

- **Line endings**: Windows uses CRLF (`\r\n`), Unix uses LF (`\n`).

  Be aware of this when comparing text files. Be mindful of your Git settings, especially of line
  endings when checking out / committing. (See Git's `core.autocrlf` setting.)

Solutions:

- **On Windows, consider using `.bat` scripts for trivial things.**

  Instead of relying on `.sh` scripts, for trivial things you can write simple `.bat` scripts.

  For example if you need a script that does nothing, or prints some message, or exits with a fixed
  error code, a simple `.bat` file will often suffice. And if your rule returns a `DefaultInfo()`
  provider, the `executable` field may refer to that `.bat` file on Windows.

  And since file extensions don't matter on macOS / Linux, you can always use `.bat` as the
  extension.

  Be aware that **empty `.bat` files cannot be executed**. If you need an empty script, write one
  space in it.

- **Use a Bash-less purpose-made rule.**

  `native.genrule()` and Bash commands are often used to solve simple problems, like copying files
  or writing text files. You can avoid relying on Bash (and reinventing the wheel): see if
  bazel-skylib has a purpose-made rule for your need. None of them depends on Bash when built/tested
  on Windows.

  Build rule examples:

  - `copy_file`
    ([source](https://github.com/bazelbuild/bazel-skylib/blob/master/rules/copy_file.bzl),
    [documentation](https://github.com/bazelbuild/bazel-skylib/blob/master/docs/copy_file_doc.md)):
    copies a file somewhere else, optionally making it executable

  - `write_file`
    ([source](https://github.com/bazelbuild/bazel-skylib/blob/master/rules/write_file.bzl),
    [documentation](https://github.com/bazelbuild/bazel-skylib/blob/master/docs/write_file_doc.md)):
    writes a text file, with the desired line endings (`auto`, `unix`, or `windows`), optionally
    making it executable (if it's a script)

  - `native_binary`
    ([source](https://github.com/bazelbuild/bazel-skylib/blob/master/rules/native_binary.bzl),
    [documentation](https://github.com/bazelbuild/bazel-skylib/blob/master/docs/native_binary_doc.md#native_binary)):
    wraps a native binary in a `*_binary` rule, which you can `bazel run` or use in
    `genrule.tools`

  - `run_binary`
    ([source](https://github.com/bazelbuild/bazel-skylib/blob/master/rules/run_binary.bzl),
    [documentation](https://github.com/bazelbuild/bazel-skylib/blob/master/docs/run_binary_doc.md)):
    runs a binary (or `*_binary` rule) with given inputs and expected outputs as a build action
    (this is a build rule wrapper for `ctx.actions.run`)

  Test rule examples:

  - `diff_test`
    ([source](https://github.com/bazelbuild/bazel-skylib/blob/master/rules/diff_test.bzl),
    [documentation](https://github.com/bazelbuild/bazel-skylib/blob/master/docs/diff_test_doc.md)):
    test that compares contents of two files

  - `native_test`
    ([source](https://github.com/bazelbuild/bazel-skylib/blob/master/rules/native_binary.bzl),
    [documentation](https://github.com/bazelbuild/bazel-skylib/blob/master/docs/native_binary_doc.md#native_test)):
    wraps a native binary in a `*_test` rule, which you can `bazel test`

- **Use Bash in a principled way.**

  You can still run Bash commands if you need to, just let Bazel know that it's a Bash command.

  In Starlark rules:

  - Use `ctx.actions.run_shell` if you need to run Bash commands.

  - **Do not** use `ctx.actions.run` with Bash commands -- this won't work on Windows because Bazel
    doesn't know that it needs to run the command through Bash.

  In Starlark macros:

  - Wrap the Bash commands in a `native.sh_binary()` or `native.genrule()`. Bazel will check if Bash
    is available and run the script / command through Bash.

## Deleting files

Problems:

- **Files cannot be deleted while open.**

  Open files cannot be deleted (by default), attempting to do so gives an "Access Denied" error.
  If you cannot delete a file, maybe a running process still holds it open.

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
- unzip is slow
- cmd.exe has 8k command length limit
- put paths in envvars instead of args
- put cmd.exe commands in .bat files
-->
