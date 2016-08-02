---
layout: contribute
title: Design Documents
---

# Bazel Design Documents

<ul>
{% for doc in site.categories.designs %}
  <li><a href="{{ doc.url }}">{{ doc.title }}</a>
      {{ doc.date | date_to_long_string }}</a></li>
{% endfor %}
</ul>



## Skylark Design Documents

Changes to the Bazel build and extension language (Skylark) should go
through the [Skylark Design Process](/designs/skylark-design-process.md).

1. [Parameterized Skylark Aspects](/designs/skylark-parameterized-aspects.md).
