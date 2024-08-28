Project: /_project.yaml
Book: /_book.yaml

# Use labels to reference targets

{% include "_buttons.html" %}

In `BUILD` files and at the command line, Bazel uses *labels* to reference
targets - for example, `//main:hello-world` or `//lib:hello-time`. Their syntax
is:

```
//path/to/package:target-name
```

If the target is a rule target, then `path/to/package` is the path from the
workspace root (the directory containing the `MODULE.bazel` file) to the directory
containing the `BUILD` file, and `target-name` is what you named the target
in the `BUILD` file (the `name` attribute). If the target is a file target,
then `path/to/package` is the path to the root of the package, and
`target-name` is the name of the target file, including its full
path relative to the root of the package (the directory containing the
package's `BUILD` file).

When referencing targets at the repository root, the package path is empty,
just use `//:target-name`. When referencing targets within the same `BUILD`
file, you can even skip the `//` workspace root identifier and just use
`:target-name`.
