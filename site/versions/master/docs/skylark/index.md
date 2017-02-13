---
layout: documentation
title: Extensions
---

# Extensions
Skylark is the name of the extension mechanism in Bazel. It lets you add support
for new languages and tools by writing [custom build rules](rules.md). You can
also compose existing rules into [macros](macros.md).

## Getting started

Read the [concepts](concepts.md) behind Skylark and try the
[cookbook examples](cookbook.md). To go further, read about the
[standard library](lib/globals.html).
-->


## How can I profile my code?

```shell
$ bazel build --nobuild --profile=/tmp/prof //path/to:target
$ bazel analyze-profile /tmp/prof --html --html_details
```

Then, open the generated HTML file (`/tmp/prof.html` in the example).
