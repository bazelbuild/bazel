---
layout: documentation
---

# Getting Started with Bazel

## Setup

Clone the Bazel [Github repo](https://github.com/google/bazel) and run the
provided compile script. Make sure that you are running Bazel on a supported
platform and that you have installed other required software as described in the
[installation guide](install.html).

{% highlight bash %}
$ git clone https://github.com/google/bazel.git
$ cd bazel
$ ./compile.sh
{% endhighlight %}

`./compile.sh` creates the `bazel` executable in `output/bazel`.

_**Note:** Bazel may support a binary installation at a later time._

## Using a workspace

A *workspace* is a directory on your filesystem that contains source code for
the software you want to build, as well symbolic links to directories that
contain the build outputs (for example, `bazel-bin` and `bazel-out`). The
location of the workspace directory is not significant, but it must contain an
empty file called `WORKSPACE` in the top-level directory. This file marks the
directory as the workspace root.

One workspace can be shared among multiple projects if desired.  To get
started, we'll focus on a simple example with one project.

Suppose that you have an existing project in a directory, say,
`~/gitroot/my-project/`. Create an empty file at
`~/gitroot/my-project/WORKSPACE` to show Bazel where your project's root is.

## Sanity Check: Building an Example

To make sure everything is set up correctly in your build root, build one of the
examples from the `examples/` directory.

{% highlight bash %}
$ cd ~/gitroot/my-project
$ bazel fetch //...
$ bazel build examples/java-native/src/main/java/com/example/myproject:hello-world
Extracting Bazel installation...
...........
INFO: Found 1 target...
Target //examples/java-native/src/main/java/com/example/myproject:hello-world up-to-date:
  bazel-bin/examples/java-native/src/main/java/com/example/myproject/hello-world.jar
  bazel-bin/examples/java-native/src/main/java/com/example/myproject/hello-world
INFO: Elapsed time: 3.040s, Critical Path: 1.14s
$ bazel-bin/examples/java-native/src/main/java/com/example/myproject/hello-world
Hello world
{% endhighlight %}

Bazel puts binaries it has built under `bazel-bin/`.  Note that you can
always look at the `build` command's output to find output file paths.

## Creating Your Own Build File

Now you can create your own BUILD file and start adding build rules. This
example assumes that `my-project/` is a Java project.  See the
[build encyclopedia](build-encyclopedia.html)
for advice on adding build rules for other languages.

Note that when we ran "bazel build" above, the third argument started with a
filesystem path ("examples/java"), followed by a colon. When you run
`bazel build examples/java-native/src/main/java/com/example/myproject:hello-world`,
Bazel will look for a special file named BUILD in the
`examples/java-native/src/main/java/com/example/myproject/` subdirectory. This
BUILD file defines rules about how Bazel should build things in this
subdirectory.

Thus, to add build rules to my-project, create a file named `BUILD` in the
`my-project/` directory.  Add the following lines to this BUILD file:

{% highlight python %}
# ~/gitroot/base_workspace/my-project/BUILD
java_binary(
    name = "my-runner",
    srcs = glob(["**/*.java"]),
    main_class = "com.example.ProjectRunner",
)
{% endhighlight %}

BUILD files are Python-like scripts. BUILD files cannot contain arbitrary
Python, but each build rule looks like a Python function call and you can use
"#" to start a single-line comment.

`java_binary` is the type of thing this rule will build.
`name` is how you'll refer to the rule when you run "bazel build"
(in the "examples/java:hello-world" build above the `name` was
"hello-world"). `srcs` lists the Java source files Bazel should
compile into a Java binary.  `glob(["**/*.java"])` is a handy
shorthand for "recursively include every file that ends with .java" (see the
[user manual](bazel-user-manual.html)
for more information about globbing). Replace `com.example.ProjectRunner` with
the class that contains the main method.

If you have no actual Java project you're using, you can use the following
commands to make a fake project for this example:

{% highlight bash %}
$ # If you're not already there, move to your build root directory.
$ cd ~/gitroot/base_workspace
$ mkdir -p my-project/java/com/example
$ cat > my-project/java/com/example/ProjectRunner.java <<EOF
package com.example;

public class ProjectRunner {
    public static void main(String args[]) {
        Greeting.sayHi();
    }
}
EOF
$ cat > my-project/java/com/example/Greeting.java <<EOF
package com.example;

public class Greeting {
    public static void sayHi() {
        System.out.println("Hi!");
    }
}
EOF
{% endhighlight %}

Now build your project:

{% highlight bash %}
$ bazel fetch my-project:my-runner
$ bazel build my-project:my-runner
INFO: Found 1 target...
Target //my-project:my-runner up-to-date:
  bazel-bin/my-project/my-runner.jar
  bazel-bin/my-project/my-runner
INFO: Elapsed time: 1.021s, Critical Path: 0.83s
$ bazel-bin/my-project/my-runner
Hi!
{% endhighlight %}

Congratulations, you've created your first Bazel BUILD file!

## Adding Dependencies

Creating one rule to build your entire project may be sufficient for small
projects, but as projects get larger it's important to break up the build into
self-contained libraries that can be assembled into a final product.  This way
the entire world doesn't need to be rebuilt on small changes and Bazel can
parallelize more of the build steps.

To break up a project, create separate rules for each subcomponent and then
make them depend on each other. For the example above, add the following rules
to the `my-project/BUILD` file:

{% highlight python %}
java_binary(
    name = "my-other-runner",
    srcs = ["java/com/example/ProjectRunner.java"],
    main_class = "com.example.ProjectRunner",
    deps = [":greeter"],
)

java_library(
    name = "greeter",
    srcs = ["java/com/example/Greeting.java"],
)
{% endhighlight %}

Now you can build and run `my-project:my-other-runner`:

{% highlight bash %}
$ bazel run my-project:my-other-runner
INFO: Found 1 target...
Target //my-project:my-other-runner up-to-date:
  bazel-bin/my-project/my-other-runner.jar
  bazel-bin/my-project/my-other-runner
INFO: Elapsed time: 2.454s, Critical Path: 1.58s

INFO: Running command line: bazel-bin/my-project/my-other-runner
Hi!
{% endhighlight %}

If you edit _ProjectRunner.java_ and rebuild `my-other-runner`, only
`ProjectRunner.java` needs to be rebuilt (<code>greeter</code> is unchanged).

## Using Multiple Packages

For larger projects, you will often be dealing with several directories. You
can refer to targets defined in other BUILD files using the syntax
`//package-name:target-name`.  For example, suppose
`my-project/java/com/example/` has a `cmdline/` subdirectory with the following
file:

{% highlight bash %}
$ mkdir my-project/java/com/example/cmdline
$ cat > my-project/java/com/example/cmdline/Runner.java <<EOF
package com.example.cmdline;

import com.example.Greeting;

public class Runner {
    public static void main(String args[]) {
        Greeting.sayHi();
    }
}
EOF
{% endhighlight %}

We could add a `BUILD` file at `my-project/java/com/example/cmdline/BUILD`
that contained the following rule:

{% highlight python %}
# ~/gitroot/base_workspace/my-project/java/com/example/cmdline/BUILD
java_binary(
    name = "runner",
    srcs = ["Runner.java"],
    main_class = "com.example.cmdline.Runner",
    deps = ["//my-project:greeter"]
)
{% endhighlight %}

However, by default, build rules are _private_. This means that they can only be
referred to by rules in the same BUILD file. This prevents libraries that are
implementation details from leaking into public APIs, but it also means that you
must explicitly allow `runner` to depend on `my-project:greeter`. As is, if we
build `runner` we'll get a permissions error:

{% highlight bash %}
$ bazel build my-project/java/com/example/cmdline:runner
ERROR: /usr/local/google/home/kchodorow/gitroot/base_workspace/my-project/java/com/example/cmdline/BUILD:2:1:
  Target '//my-project:greeter' is not visible from target '//my-project/java/com/example/cmdline:runner'.
  Check the visibility declaration of the former target if you think the dependency is legitimate.
ERROR: Analysis of target '//my-project/java/com/example/cmdline:runner' failed; build aborted.
INFO: Elapsed time: 0.091s
{% endhighlight %}

You can make a rule visibile to rules in other BUILD files by adding a
`visibility = level` attribute.  Change the `greeter` rule in
_my-project/BUILD_ to be visible to our new rule:

{% highlight python %}
java_library(
    name = "greeter",
    srcs = ["java/com/example/Greeting.java"],
    visibility = ["//my-project/java/com/example/cmdline:__pkg__"],
)
{% endhighlight %}

This makes `//my-project:greeter` visible to any rule in the
`//my-project/java/com/example/cmdline` package. Now we can build and
run the binary:

{% highlight bash %}
$ bazel run my-project/java/com/example/cmdline:runner
INFO: Found 1 target...
Target //my-project/java/com/example/cmdline:runner up-to-date:
  bazel-bin/my-project/java/com/example/cmdline/runner.jar
  bazel-bin/my-project/java/com/example/cmdline/runner
INFO: Elapsed time: 1.576s, Critical Path: 0.81s

INFO: Running command line: bazel-bin/my-project/java/com/example/cmdline/runner
Hi!
{% endhighlight %}

See the [build encyclopedia](build-encyclopedia.html) for more visibility options.

## Deploying

If you look at the contents of
_bazel-bin/my-project/java/com/example/cmdline/runner.jar_, you can see that it
only contains `Runner.class`, not its dependencies (`Greeting.class`):

{% highlight bash %}
$ jar tf bazel-bin/my-project/java/com/example/cmdline/runner.jar
META-INF/
META-INF/MANIFEST.MF
com/
com/example/
com/example/cmdline/
com/example/cmdline/Runner.class
{% endhighlight %}

To deploy a `runner` binary, we need a self-contained jar. To build this, build
runner_deploy.jar (or, more generally, _&lt;target-name&gt;_deploy.jar_):

{% highlight bash %}
$ bazel build my-project/java/com/example/cmdline:runner_deploy.jar
INFO: Found 1 target...
Target //my-project/java/com/example/cmdline:runner_deploy.jar up-to-date:
  bazel-bin/my-project/java/com/example/cmdline/runner_deploy.jar
INFO: Elapsed time: 1.700s, Critical Path: 0.23s
{% endhighlight %}

`runner_deploy.jar` will contain all of its dependencies.

## Next Steps

You can now create your own targets and compose them.  See the [build
encyclopedia](build-encyclopedia.html)
and Bazel
[user manual](bazel-user-manual.html)
for more information.
[Let us know](https://groups.google.com/forum/#!forum/bazel-discuss)
if you have any questions!
