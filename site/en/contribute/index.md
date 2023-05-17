Project: /_project.yaml
Book: /_book.yaml

# Contributing to Bazel

{% dynamic setvar source_file "site/en/contribute/index.md" %}
{% include "_buttons.html" %}

There are many ways to help the Bazel project and ecosystem.

## Provide feedback {:#feedback}

As you use Bazel, you may find things that can be improved.
You can help by [reporting issues](http://github.com/bazelbuild/bazel/issues){: .external}
when:

   - Bazel crashes or you encounter a bug that can [only be resolved using `bazel
     clean`](/run/build#correct-incremental-rebuilds).
   - The documentation is incomplete or unclear. You can also report issues
     from the page you are viewing by using the "Create issue"
     link at the top right corner of the page.
   - An error message could be improved.

## Participate in the community {:#community}

You can engage with the Bazel community by:

   - Answering questions [on Stack Overflow](
     https://stackoverflow.com/questions/tagged/bazel){: .external}.
   - Helping other users [on Slack](https://slack.bazel.build){: .external}.
   - Improving documentation or [contributing examples](
     https://github.com/bazelbuild/examples){: .external}.
   - Sharing your experience or your tips, for example, on a blog or social media.

## Contribute code {:#contribute-code}

Bazel is a large project and making a change to the Bazel source code
can be difficult.

You can contribute to the Bazel ecosystem by:

   - Helping rules maintainers by contributing pull requests.
   - Creating new rules and open-sourcing them.
   - Contributing to Bazel-related tools, for example, migration tools.
   - Improving Bazel integration with other IDEs and tools.

Before making a change, [create a GitHub
issue](http://github.com/bazelbuild/bazel/issues){: .external}
or email [bazel-discuss@](mailto:bazel-discuss@googlegroups.com){: .external}.

The most helpful contributions fix bugs or add features (as opposed
to stylistic, refactoring, or "cleanup" changes). Your change should
include tests and documentation, keeping in mind backward-compatibility,
portability, and the impact on memory usage and performance.

To learn about how to submit a change, see the
[patch acceptance process](/contribute/patch-acceptance).

## Bazel's code description {:#code-description}

Bazel has a large codebase with code in multiple locations. See the [codebase guide](/contribute/codebase) for more details.

Bazel is organized as follows:

*  Client code is in `src/main/cpp` and provides the command-line interface.
*  Protocol buffers are in `src/main/protobuf`.
*  Server code is in `src/main/java` and `src/test/java`.
   *  Core code which is mostly composed of [SkyFrame](/reference/skyframe)
      and some utilities.
   *  Built-in rules are in `com.google.devtools.build.lib.rules` and in
     `com.google.devtools.build.lib.bazel.rules`. You might want to read about
     the [Challenges of Writing Rules](/rules/challenges) first.
*  Java native interfaces are in `src/main/native`.
*  Various tooling for language support are described in the list in the
   [compiling Bazel](/install/compile-source) section.


### Searching Bazel's source code {:#search-code}

To quickly search through Bazel's source code, use
[Bazel Code Search](https://source.bazel.build/). You can navigate Bazel's
repositories, branches, and files. You can also view history, diffs, and blame
information. To learn more, see the
[Bazel Code Search User Guide](/contribute/search).
