---
layout: documentation
title: Build Event Protocol Examples
---

# Build Event Protocol Examples

The full specification of the Build Event Protocol can be found in its protocol
buffer definition. However, it might be helpful to build up some intuition
before looking at the specification.

Consider a simple Bazel workspace that consists of two empty shell scripts
`foo.sh` and `foo_test.sh` and the following BUILD file:

```bash
sh_binary(
    name = "foo",
    srcs = ["foo.sh"],
)

sh_library(
    name = "foo_lib",
    data = [":foo"],
)

sh_test(
    name = "foo_test",
    srcs = ["foo_test.sh"],
    deps = [":foo_lib"],
)
```

When running `bazel test ...` on this project the build graph of the generated
build events will resemble the graph below. The arrows indicate the
aforementioned parent and child relationship. Note that some build events and
most fields have been omitted for brevity.

![bep-graph](/assets/bep-graph.svg)

Initially, a `BuildStarted` event is published. The event informs us that the
build was invoked through the `bazel test` command and it also announces five
child events: `OptionsParsed`, `WorkspaceStatus`, `CommandLine`,
`PatternExpanded` and `Progress`. The first three events provide information
about how Bazel was invoked. The `PatternExpanded` build event provides insight
into which specific targets the `...` pattern expanded to: `//:foo`,
`//:foo_lib` and `//:foo_test`. It does so by declaring three `TargetConfigured`
events as children.

Note that the `TargetConfigured` event declares the `Configuration` event as a
child event, even though `Configuration` has been posted before the
`TargetConfigured` event.

Besides the parent and child relationship, events may also refer to each other
using their build event identifiers. For example, in the above graph the
`TargetComplete` event refers to the `NamedSetOfFiles` event in its `fileSets`
field.

Build events that refer to files (i.e. outputs) usually don’t embed the file
names and paths in the event. Instead, they contain the build event identifier
of a `NamedSetOfFiles` event, which will then contain the actual file names and
paths. The `NamedSetOfFiles` event allows a set of files to be reported once and
referred to by many targets. This structure is necessary because otherwise in
some cases the Build Event Protocol output size would grow quadratically with
the number of files. A `NamedSetOfFiles` event may also not have all its files
embedded, but instead refer to other `NamedSetOfFiles` events through their
build event identifiers.

Below is an instance of the `TargetComplete` event for the `//:foo_lib` target
from the above graph, printed in protocol buffer’s JSON representation. The
build event identifier contains the target as an opaque string and refers to the
`Configuration` event using its build event identifier. The event does not
announce any child events. The payload contains information about whether the
target was built successfully, the set of output files, and the kind of target
built.

```json
{
  "id": {
    "targetCompleted": {
      "label": "//:foo_lib",
      "configuration": {
        "id": "544e39a7f0abdb3efdd29d675a48bc6a"
      }
    }
  },
  "completed": {
    "success": true,
    "outputGroup": [{
      "name": "default",
      "fileSets": [{
        "id": "0"
      }]
    }],
    "targetKind": "sh_library rule"
  }
}
```
