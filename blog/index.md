---
layout: blog
---

{% for post in site.categories.blog %}

<div class="blog-post">
  <h1 class="blog-post-title"><a href="{{ post.url }}">{{ post.title }}</a></h1>
  <div class="blog-post-meta">
    <span class="text-muted">{{ post.date | date_to_long_string }}</span>
  </div>
  {{ post.content }}
</div>

{% endfor %}
