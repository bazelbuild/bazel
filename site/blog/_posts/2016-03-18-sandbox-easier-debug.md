---
layout: posts
title: Easier Debugging of Sandbox Failures
---

We have often heard that debugging failed execution due to issues with the sandbox is difficult and requires knowledge of the sandboxing code of Bazel to actually understand what’s happening. With [these changes](https://github.com/bazelbuild/bazel/commit/40ee9de052e3bb8cf5a59eeff3936148e1f55e69), we hope that you will be able to solve common issues easier on your own and make debugging easier.


If you don’t know much about Bazel sandbox, you might want to read [this blog post](http://bazel.build/blog/2015/09/11/sandboxing.html)


## What we did:

- When using `--verbose_failures` and `--sandbox_debug`, Bazel now shows the detailed command that it ran when your build failed, including the part that sets up the sandbox.
- When you copy & paste the shown command into a terminal, the failed command is rerun - but when it fails this time, we provide access to a shell inside a new sandbox which is the same as the old sandbox we made before, so that you can explore the sandbox from the inside and find out why the command failed.

## How to use it:
Let’s say you wrote a Skylark rule and forgot to add your compiler to the input files. Before this change, when you ran bazel build, you would get several error messages like this:

```
ERROR: path/to/your/project/BUILD:1:1: compilation of rule '//path/to/your/project:all' failed: No such file or directory.
ERROR: /your/project/BUILD:x:1: Executing genrule //project/dir:genrule failed: bash failed: error executing command /path/to/your/compiler some command
```

But you probably have no idea what to do, because the error message is not detailed enough and you have everything you needed in your system.

With this new feature, you will get an error message like this instead:

```
ERROR: path/to/your/project/BUILD:1:1: compilation of rule '//path/to/your/project:all' failed:

Sandboxed execution failed, which may be legitimate (e.g. a compiler error), or due to missing dependencies. To enter the sandbox environment for easier debugging, run the following command in parentheses. On command failure, a bash shell running inside the sandbox will then automatically be spawned

namespace-sandbox failed: error executing command
  (cd /some/path && \
  exec env - \
    LANG=en_US \
    PATH=/some/path/bin:/bin:/usr/bin \
    PYTHONPATH=/usr/local/some/path \
  /some/path/namespace-sandbox @/sandbox/root/path/this-sandbox-name.params -- /some/path/to/your/some-compiler --some-params some-target)
```

Then you can simply copy & paste the command above in parentheses into a new terminal:

```
(cd /some/path && \
  exec env - \
    LANG=en_US \
    PATH=/some/path/bin:/bin:/usr/bin \
    PYTHONPATH=/usr/local/some/path \
  /some/path/namespace-sandbox @/sandbox/root/path/this-sandbox-name.params -- /some/path/to/your/some-compiler --some-params some-target)
```

There will be the same error message about not finding your compiler, but after that error message, you will find yourself in a bash shell inside a new sandbox. You can now debug the failure, e.g. you can explore the sandbox: look for any missing file, check for possible errors in your BUILD files, run your compiler again manually, or even use strace.

For this example, we run our compiler in the sandbox again manually and the error message shows `No command ‘some-compiler’ found` - looking around, you notice that the compiler binary is missing. This means it was not part of the action inputs, because Bazel always mounts all action inputs into the sandbox - so you check out your Skylark rule and notice that this is indeed the case. Adding your compiler to the input files in your Skylark rule should thus fix the error.

Next time you run bazel build, it should mount your compiler into the sandbox and thus find it correctly. If you get a different error, you could repeat the steps above.
