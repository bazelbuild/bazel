Project: /_project.yaml
Book: /_book.yaml

# Labels

{% include "_buttons.html" %}

A **label** is an identifier for a target. A typical label in its full canonical
form looks like:

```none
@@myrepo//my/app/main:app_binary
```

The first part of the label is the repository name, `@@myrepo`. The double-`@`
syntax signifies that this is a [*canonical* repo
name](/external/overview#canonical-repo-name), which is unique within
the workspace. Labels with canonical repo names unambiguously identify a target
no matter which context they appear in.

Often the canonical repo name is an arcane string that looks like
`@@rules_java~7.1.0~toolchains~local_jdk`. What is much more commonly seen is
labels with an [*apparent* repo name](/external/overview#apparent-repo-name),
which looks like:

```
@myrepo//my/app/main:app_binary
```

The only difference is the repo name being prefixed with one `@` instead of two.
This refers to a repo with the apparent name `myrepo`, which could be different
based on the context this label appears in.

In the typical case that a label refers to the same repository from which
it is used, the repo name part may be omitted.  So, inside `@@myrepo` the first
label is usually written as

```
//my/app/main:app_binary
```

The second part of the label is the un-qualified package name
`my/app/main`, the path to the package
relative to the repository root.  Together, the repository name and the
un-qualified package name form the fully-qualified package name
`@@myrepo//my/app/main`. When the label refers to the same
package it is used in, the package name (and optionally, the colon)
may be omitted.  So, inside `@@myrepo//my/app/main`,
this label may be written either of the following ways:

```
app_binary
:app_binary
```

It is a matter of convention that the colon is omitted for files,
but retained for rules, but it is not otherwise significant.

The part of the label after the colon, `app_binary` is the un-qualified target
name. When it matches the last component of the package path, it, and the
colon, may be omitted.  So, these two labels are equivalent:

```
//my/app/lib
//my/app/lib:lib
```

The name of a file target in a subdirectory of the package is the file's path
relative to the package root (the directory containing the `BUILD` file). So,
this file is in the `my/app/main/testdata` subdirectory of the repository:

```
//my/app/main:testdata/input.txt
```

Strings like `//my/app` and `@@some_repo//my/app` have two meanings depending on
the context in which they are used: when Bazel expects a label, they mean
`//my/app:app` and `@@some_repo//my/app:app`, respectively. But, when Bazel
expects a package (e.g. in `package_group` specifications), they reference the
package that contains that label.

A common mistake in `BUILD` files is using `//my/app` to refer to a package, or
to *all* targets in a package--it does not.  Remember, it is
equivalent to `//my/app:app`, so it names the `app` target in the `my/app`
package of the current repository.

However, the use of `//my/app` to refer to a package is encouraged in the
specification of a `package_group` or in `.bzl` files, because it clearly
communicates that the package name is absolute and rooted in the top-level
directory of the workspace.

Relative labels cannot be used to refer to targets in other packages; the
repository identifier and package name must always be specified in this case.
For example, if the source tree contains both the package `my/app` and the
package `my/app/testdata` (each of these two directories has its own
`BUILD` file), the latter package contains a file named `testdepot.zip`. Here
are two ways (one wrong, one correct) to refer to this file within
`//my/app:BUILD`:

<p><span class="compare-worse">Wrong</span> — <code>testdata</code> is a different package, so you can't use a relative path</p>
<pre class="prettyprint">testdata/testdepot.zip</pre>

<p><span class="compare-better">Correct</span> — refer to <code>testdata</code> with its full path</p>

<pre class="prettyprint">//my/app/testdata:testdepot.zip</pre>



Labels starting with `@@//` are references to the main
repository, which will still work even from external repositories.
Therefore `@@//a/b/c` is different from
`//a/b/c` when referenced from an external repository.
The former refers back to the main repository, while the latter
looks for `//a/b/c` in the external repository itself.
This is especially relevant when writing rules in the main
repository that refer to targets in the main repository, and will be
used from external repositories.

For information about the different ways you can refer to targets, see
[target patterns](/run/build#specifying-build-targets).

### Lexical specification of a label {:#labels-lexical-specification}

Label syntax discourages use of metacharacters that have special meaning to the
shell. This helps to avoid inadvertent quoting problems, and makes it easier to
construct tools and scripts that manipulate labels, such as the
[Bazel Query Language](/query/language).

The precise details of allowed target names are below.

### Target names — `{{ "<var>" }}package-name{{ "</var>" }}:target-name` {:#target-names}

`target-name` is the name of the target within the package. The name of a rule
is the value of the `name` attribute in the rule's declaration in a `BUILD`
file; the name of a file is its pathname relative to the directory containing
the `BUILD` file.

Target names must be composed entirely of characters drawn from the set `a`–`z`,
`A`–`Z`, `0`–`9`, and the punctuation symbols `!%-@^_"#$&'()*-+,;<=>?[]{|}~/.`.

Filenames must be relative pathnames in normal form, which means they must
neither start nor end with a slash (for example, `/foo` and `foo/` are
forbidden) nor contain multiple consecutive slashes as path separators
(for example, `foo//bar`). Similarly, up-level references (`..`) and
current-directory references (`./`) are forbidden.

<p><span class="compare-worse">Wrong</span> — Do not use <code>..</code> to refer to files in other packages</p>

<p><span class="compare-better">Correct</span> — Use
  <code>//{{ "<var>" }}package-name{{ "</var>" }}:{{ "<var>" }}filename{{ "</var>" }}</code></p>


While it is common to use `/` in the name of a file target, avoid the use of
`/` in the names of rules. Especially when the shorthand form of a label is
used, it may confuse the reader. The label `//foo/bar/wiz` is always a shorthand
for `//foo/bar/wiz:wiz`, even if there is no such package `foo/bar/wiz`; it
never refers to `//foo:bar/wiz`, even if that target exists.

However, there are some situations where use of a slash is convenient, or
sometimes even necessary. For example, the name of certain rules must match
their principal source file, which may reside in a subdirectory of the package.

### Package names — `//package-name:{{ "<var>" }}target-name{{ "</var>" }}` {:#package-names}

The name of a package is the name of the directory containing its `BUILD` file,
relative to the top-level directory of the containing repository.
For example: `my/app`.

Package names must be composed entirely of characters drawn from the set
`A`-`Z`, `a`–`z`, `0`–`9`, '`/`', '`-`', '`.`', '`@`', and '`_`', and cannot
start with a slash.

For a language with a directory structure that is significant to its module
system (for example, Java), it's important to choose directory names that are
valid identifiers in the language.

Although Bazel supports targets in the workspace's root package (for example,
`//:foo`), it's best to leave that package empty so all meaningful packages
have descriptive names.

Package names may not contain the substring `//`, nor end with a slash.

## Rules {:#rules}

A rule specifies the relationship between inputs and outputs, and the
steps to build the outputs. Rules can be of one of many different
kinds (sometimes called the _rule class_), which produce compiled
executables and libraries, test executables and other supported
outputs as described in the [Build Encyclopedia](/reference/be/overview).

`BUILD` files declare _targets_ by invoking _rules_.

In the example below, we see the declaration of the target `my_app`
using the `cc_binary` rule.

```python
cc_binary(
    name = "my_app",
    srcs = ["my_app.cc"],
    deps = [
        "//absl/base",
        "//absl/strings",
    ],
)
```

Every rule invocation has a `name` attribute (which must be a valid
[target name](#target-names)), that declares a target within the package
of the `BUILD` file.

Every rule has a set of _attributes_; the applicable attributes for a given
rule, and the significance and semantics of each attribute are a function of
the rule's kind; see the [Build Encyclopedia](/reference/be/overview) for a
list of rules and their corresponding attributes. Each attribute has a name and
a type. Some of the common types an attribute can have are integer, label, list
of labels, string, list of strings, output label, list of output labels. Not
all attributes need to be specified in every rule. Attributes thus form a
dictionary from keys (names) to optional, typed values.

The `srcs` attribute present in many rules has type "list of labels"; its
value, if present, is a list of labels, each being the name of a target that is
an input to this rule.

In some cases, the name of the rule kind is somewhat arbitrary, and more
interesting are the names of the files generated by the rule, and this is true
of genrules. For more information, see
[General Rules: genrule](/reference/be/general#genrule).

In other cases, the name is significant: for `*_binary` and `*_test` rules,
for example, the rule name determines the name of the executable produced by
the build.

This directed acyclic graph over targets is called the _target graph_ or
_build dependency graph_, and is the domain over which the
[Bazel Query tool](/query/guide) operates.

<table class="columns">
  <tr>
    <td><a class="button button-with-icon button-primary"
           href="/concepts/build-ref">
        <span class="material-icons" aria-hidden="true">arrow_back</span>Targets</a>
    </td>
    <td><a class="button button-with-icon button-primary"
           href="/concepts/build-files">
        BUILD files<span class="material-icons icon-after" aria-hidden="true">arrow_forward</span></a>
    </td>
  </tr>
</table>
