Project: /_project.yaml
Book: /_book.yaml

# Naming a Bazel related project

{% include "_buttons.html" %}

First, thank you for contributing to the Bazel ecosystem! Please reach out to
the Bazel community on the
[bazel-discuss mailing list](https://groups.google.com/forum/#!forum/bazel-discuss
){: .external} to share your project and its suggested name.

If you are building a Bazel related tool or sharing your Skylark rules,
we recommend following these guidelines for the name of your project:

## Naming Starlark rules {:#name-starlark-rules}

See [Deploying new Starlark rules](/rules/deploying)
in the docs.

## Naming other Bazel related tools {:#name-related-tools}

This section applies if you are building a tool to enrich the Bazel ecosystem.
For example, a new IDE plugin or a new build system migrator.

Picking a good name for your tool can be hard. If we’re not careful and use too
many codenames, the Bazel ecosystem could become very difficult to understand
for newcomers.

Follow these guidelines for naming Bazel tools:

1. Prefer **not introducing a new brand name**: "*Bazel*" is already a new brand
for our users, we should avoid confusing them with too many new names.

2. Prefer **using a name that includes "Bazel"**: This helps to express that it
is a Bazel related tool, it also helps people find it with a search engine.

3. Prefer **using names that are descriptive about what the tool is doing**:
Ideally, the name should not need a subtitle for users to have a first good
guess at what the tool does. Using english words separated by spaces is a good
way to achieve this.

4. **It is not a requirement to use a floral or food theme**: Bazel evokes
[basil](https://en.wikipedia.org/wiki/Basil), the plant. You do not need to
look for a name that is a plant, food or that relates to "basil."

5. **If your tool relates to another third party brand, use it only as a
descriptor**: For example, use "Bazel migrator for Cmake" instead of
"Cmake Bazel migrator".

These guidelines also apply to the GitHub repository URL. Reading the repository
URL should help people understand what the tool does. Of course, the repository
name can be shorter and must use dashes instead of spaces and lower case letters.


Examples of good names:

* *Bazel for Eclipse*: Users will understand that if they want to use Bazel
  with Eclipse, this is where they should be looking. It uses a third party brand
  as a descriptor.
* *Bazel buildfarm*: A "buildfarm" is a
  [compile farm](https://en.wikipedia.org/wiki/Compile_farm){: .external}. Users
  will understand that this project relates to building on servers.

Examples of names to avoid:

* *Ocimum*: The [scientific name of basil](https://en.wikipedia.org/wiki/Ocimum){: .external}
  does not relate enough to the Bazel project.
* *Bazelizer*: The tool behind this name could do a lot of things, this name is
   not descriptive enough.

Note that these recommendations are aligned with the
[guidelines](https://opensource.google.com/docs/releasing/preparing/#name){: .external}
Google uses when open sourcing a project.
