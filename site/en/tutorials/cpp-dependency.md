Project: /_project.yaml
Book: /_book.yaml

# Review the dependency graph

{% include "_buttons.html" %}

A successful build has all of its dependencies explicitly stated in the `BUILD`
file. Bazel uses those statements to create the project's dependency graph,
which enables accurate incremental builds.

To visualize the sample project's dependencies, you can generate a text
representation of the dependency graph by running this command at the
workspace root:

```
bazel query --notool_deps --noimplicit_deps "deps(//main:hello-world)" \
  --output graph
```

The above command tells Bazel to look for all dependencies for the target
`//main:hello-world` (excluding host and implicit dependencies) and format the
output as a graph.

Then, paste the text into [GraphViz](http://www.webgraphviz.com/).

On Ubuntu, you can view the graph locally by installing GraphViz and the xdot
Dot Viewer:

```
sudo apt update && sudo apt install graphviz xdot
```

Then you can generate and view the graph by piping the text output above
straight to xdot:

```
xdot <(bazel query --notool_deps --noimplicit_deps "deps(//main:hello-world)" \
  --output graph)
```

As you can see, the first stage of the sample project has a single target
that builds a single source file with no additional dependencies:

![Dependency graph for 'hello-world'](/docs/images/cpp-tutorial-stage1.png "Dependency graph")

**Figure 1.** Dependency graph for `hello-world` displays a single target with a single
source file.

After you set up your workspace, build your project, and examine its
dependencies, then you can add some complexity.
