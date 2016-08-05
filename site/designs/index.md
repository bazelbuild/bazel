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
through the [Skylark Design Process](/designs/skylark/skylark-design-process.html).

1. [Parameterized Skylark Aspects](/designs/skylark/parameterized-aspects.html).
2. [Declared Providers](/designs/skylark/declared-providers.html).
3. [Saner Skylark Sets](/designs/skylark/saner-skylark-sets.html).
