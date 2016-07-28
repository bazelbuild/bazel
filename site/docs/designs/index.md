---
layout: documentation
title: Design Documents
---

# Bazel Design Documents

<ul>
{% for doc in site.categories.designs %}
  <li><a href="{{ doc.url }}">{{ doc.title }}</a>
      {{ doc.date | date_to_long_string }}</a></li>
{% endfor %}
</ul>
