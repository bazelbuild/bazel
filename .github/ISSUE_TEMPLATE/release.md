---
name: 'Release issue (For release managers only)'
about: Communicate the progress of a release
title: 'Release X.Y.Z - $MONTH $YEAR'
labels: ['release','team-OSS','P1','type: process']
assignees:
  - iancha1992

---

# Status of Bazel X.Y.Z

-   Expected first release candidate date: [date]
-   Expected release date: [date]
-   [List of release blockers](link-to-milestone)

To report a release-blocking bug, please add a comment with the text `@bazel-io flag` to the issue. A release manager will triage it and add it to the milestone.

To cherry-pick a mainline commit into X.Y.Z, simply send a PR against the `release-X.Y.Z` branch.

**Task list:**

<!-- The first item is only needed for major releases (X.0.0) -->

-   [ ] Pick release baseline: [link to base commit]
-   [ ] Create release candidate: X.Y.Zrc1
-   [ ] Check downstream projects
-   [ ] Create [draft release announcement](https://docs.google.com/document/d/1pu2ARPweOCTxPsRR8snoDtkC9R51XWRyBXeiC6Ql5so/edit) <!-- Note that there should be a new Bazel Release Announcement document for every major release. For minor and patch releases, use the latest open doc. -->
-   [ ] Send the release announcement PR for review: [link to bazel-blog PR] <!-- Only for major releases. -->
-   [ ] Push the release and notify package maintainers: [link to comment notifying package maintainers]
-   [ ] Update the documentation
-   [ ] Push the blog post: [link to blog post] <!-- Only for major releases. -->
-   [ ] Update the [release page](https://github.com/bazelbuild/bazel/releases/)
