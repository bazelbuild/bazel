---
name: 'Release issue (For release managers only)'
about: Communicate the progress of a release
title: 'Release X.Y - $MONTH $YEAR'
labels: ['release','team-OSS','P1','type: process']

---

# Status of Bazel X.Y

<!-- The first item is only needed for major releases (X.0.0) -->
-   Target baseline: [date]
-   Expected release date: [date]
-   [List of release blockers](link-to-milestone)

To report a release-blocking bug, please add a comment with the text `@bazel-io flag` to the issue. A release manager will triage it and add it to the milestone.

To cherry-pick a mainline commit into X.Y, simply send a PR against the `release-X.Y.0` branch.

Task list:

<!-- The first three items are only needed for major releases (X.0.0) -->

-   [ ] Pick release baseline:
-   [ ] Create release candidate:
-   [ ] Check downstream projects:
-   [ ] [Create draft release announcement](https://docs.google.com/document/d/1wDvulLlj4NAlPZamdlEVFORks3YXJonCjyuQMUQEmB0/edit)
-   [ ] Send for review the release announcement PR:
-   [ ] Push the release, notify package maintainers:
-   [ ] Update the documentation
-   [ ] Push the blog post
-   [ ] Update the [release page](https://github.com/bazelbuild/bazel/releases/)
