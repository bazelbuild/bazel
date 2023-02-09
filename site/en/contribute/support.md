Project: /_project.yaml
Book: /_book.yaml

# Support Policy

{% include "_buttons.html" %}

The Bazel team generally avoids making backwards-incompatible changes. However,
these changes are sometimes necessary to fix bugs, make improvements (such as
improving performance or usability) to the system, or to lock down APIs that
are known to be brittle.

Major changes are announced in advance on the
[bazel-discuss](https://groups.google.com/forum/#!forum/bazel-discuss){: .external} mailing
list. Both undocumented features (attributes, rules, "Make" variables, and
flags) and documented features that are marked *experimental* are subject to
change at any time without prior notice.

Report any bugs or regressions you find on
[GitHub](https://github.com/bazelbuild/bazel/issues){: .external}. The repository maintainers
make an effort to triage reported issues within 2 business days.
