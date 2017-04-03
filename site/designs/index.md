---
layout: contribute
title: Design Documents
---

# Design Documents

The design documents published here are part of a design process:

1. Design is reviewed
2. Design document is published here
3. Design is implemented

It's possible that designs change as they are implemented in practice. The
design documents published here capture the initial design, and _not_ the
ongoing changes as designs are implemented.

Always go to the [documentation](/versions/master/docs/bazel-overview.md) for
descriptions of current Bazel functionality.

## Bazel Design Documents

<ul>
{% for doc in site.categories.designs %}
  <li><a href="{{ doc.url }}">{{ doc.title }}</a>
      {{ doc.date | date_to_long_string }}</a></li>
{% endfor %}
</ul>



## Skylark Design Documents

Changes to the Bazel build and extension language (Skylark) should go
through the [Skylark Design Process](/designs/skylark/skylark-design-process.html).

1. [Parameterized Skylark Aspects](/designs/skylark/parameterized-aspects.html).
2. [Declared Providers](/designs/skylark/declared-providers.html).
3. [Saner Skylark Sets](/designs/skylark/saner-skylark-sets.html).
