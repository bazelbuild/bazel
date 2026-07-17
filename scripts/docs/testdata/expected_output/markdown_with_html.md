Project: /_project.yaml
Book: /versions/6.6.6/_book.yaml

{% dynamic setvar version "6.6.6" %}
{% dynamic setvar original_path "/markdown_with_html" %}
{% include "_buttons.html" %}

Lorem ipsum [short link](/versions/6.6.6/foo/bar). Or rather a [long link](https://bazel.build/versions/6.6.6/foo/bar)?

![Scalability graph](/versions/6.6.6/rules/scalability-graph.png "Scalability graph")

**Figure 1.** Scalability graph.

Please ignore this [relative link](relative/link).

This might be a <a href="/versions/6.6.6/foo/bar">test</a>,

<img src="https://bazel.build/versions/6.6.6/images/test.jpg">
