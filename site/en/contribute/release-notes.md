Project: /_project.yaml
Book: /_book.yaml

# Writing release notes

{% include "_buttons.html" %}

This document is targeted at Bazel contributors.

Commit descriptions in Bazel include a `RELNOTES:` tag followed by a release
note. This is used by the Bazel team to track changes in each release and write
the release announcement.

## Overview {:#overview}

* Is your change a bugfix? In that case, you don't need a release note. Please
  include a reference to the GitHub issue.

* If the change adds / removes / changes Bazel in a user-visible way, then it
  may be advantageous to mention it.

If the change is significant, follow the [design document
policy](/contribute/design-documents) first.

## Guidelines {:#guidelines}

The release notes will be read by our users, so it should be short (ideally one
sentence), avoid jargon (Bazel-internal terminology), should focus on what the
change is about.

* Include a link to the relevant documentation. Almost any release note should
  contain a link. If the description mentions a flag, a feature, a command name,
  users will probably want to know more about it.

* Use backquotes around code, symbols, flags, or any word containing an
  underscore.

* Do not just copy and paste bug descriptions. They are often cryptic and only
  make sense to us and leave the user scratching their head. Release notes are
  meant to explain what has changed and why in user-understandable language.

* Always use present tense and the format "Bazel now supports Y" or "X now does
  Z." We don't want our release notes to sound like bug entries. All release
  note entries should be informative and use a consistent style and language.

* If something has been deprecated or removed, use "X has been deprecated" or "X
  has been removed." Not "is removed" or "was removed."

* If Bazel now does something differently, use "X now $newBehavior instead of
  $oldBehavior" in present tense. This lets the user know in detail what to
  expect when they use the new release.

* If Bazel now supports or no longer supports something, use "Bazel now supports
  / no longer supports X".

* Explain why something has been removed / deprecated / changed. One sentence is
  enough but we want the user to be able to evaluate impact on their builds.

* Do NOT make any promises about future functionality. Avoid "this flag will be
  removed" or "this will be changed." It introduces uncertainty. The first thing
  the user will wonder is "when?" and we don't want them to start worrying about
  their current builds breaking at some unknown time.

## Process {:#process}

As part of the [release
process](https://github.com/bazelbuild/continuous-integration/blob/master/docs/release-playbook.md){: .external},
we collect the `RELNOTES` tags of every commit. We copy everything in a [Google
Doc](https://docs.google.com/document/d/1wDvulLlj4NAlPZamdlEVFORks3YXJonCjyuQMUQEmB0/edit){: .external}
where we review, edit, and organize the notes.

The release manager sends an email to the
[bazel-dev](https://groups.google.com/forum/#!forum/bazel-dev) mailing-list.
Bazel contributors are invited to contribute to the document and make sure
their changes are correctly reflected in the announcement.

Later, the announcement will be submitted to the [Bazel
blog](https://blog.bazel.build/), using the [bazel-blog
repository](https://github.com/bazelbuild/bazel-blog/tree/master/_posts){: .external}.
