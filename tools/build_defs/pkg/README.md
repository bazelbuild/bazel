# Packaging for Bazel

## Deprecated

These rules have been extracted from the Bazel sources and are now available at
[bazelbuild/rules_pkg](https://github.com/bazelbuild/rules_pkg/releases)
 [(docs)](https://github.com/bazelbuild/rules_pkg/tree/main/pkg).

Issues and PRs against the built-in versions of these rules will no longer be
addressed. This page will exist for reference until the code is removed from
Bazel.

For more information, follow [issue 8857](https://github.com/bazelbuild/bazel/issues/8857)

## rules_pkg

<div class="toc">
  <h2>Rules</h2>
  <ul>
    <li><a href="#pkg_tar">pkg_tar</a></li>
  </ul>
</div>

## Overview

`pkg_tar()` is available for building a .tar file without depending
on anything besides Bazel. Since this feature is deprecated and will
eventually be removed from Bazel, you should migrate to `@rules_pkg`.

