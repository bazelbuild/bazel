// Copyright 2026 The Bazel Authors. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package com.google.devtools.build.lib.pkgcache;

import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.cmdline.LabelSyntaxException;
import com.google.devtools.build.lib.cmdline.PackageIdentifier;
import com.google.devtools.build.lib.cmdline.RepositoryName;
import com.google.devtools.build.lib.vfs.PathFragment;

/**
 * The set of packages marked as deleted via {@code --deleted_packages}.
 *
 * <p>Two flavors are supported: exact match and subtree match.
 */
public record DeletedPackages(
    ImmutableSet<PackageIdentifier> exact, ImmutableSet<PackageIdentifier> subtrees) {

  public static final DeletedPackages EMPTY =
      new DeletedPackages(ImmutableSet.of(), ImmutableSet.of());

  /** A single {@code --deleted_packages} entry. */
  public record Pattern(PackageIdentifier id, boolean subtree) {

    /**
     * Parses a single {@code --deleted_packages} entry.
     *
     * <p>An entry of the form {@code pkg/...} (or {@code //...}, {@code @repo//...}, etc.) denotes
     * a subtree match — the named package and every package below it. Otherwise the entry denotes
     * an exact match.
     */
    public static Pattern parse(String input) throws LabelSyntaxException {
      // Support ... and @repo//...; even though they don't make sense, it's consistent.
      boolean subtree = false;
      String pkg = input;
      if (pkg.equals("...")) {
        subtree = true;
        pkg = "";
      } else if (pkg.endsWith("/...")) {
        subtree = true;
        boolean isRepoRoot = 5 <= pkg.length() && pkg.charAt(pkg.length() - 5) == '/';
        pkg = pkg.substring(0, pkg.length() - (isRepoRoot ? 3 : 4));
      }
      return new Pattern(PackageIdentifier.parse(pkg), subtree);
    }
  }

  public static DeletedPackages exact(Iterable<PackageIdentifier> exact) {
    return of(ImmutableSet.copyOf(exact), ImmutableSet.of());
  }

  public static DeletedPackages of(
      Iterable<PackageIdentifier> exact, Iterable<PackageIdentifier> subtrees) {
    return new DeletedPackages(ImmutableSet.copyOf(exact), ImmutableSet.copyOf(subtrees));
  }

  public static DeletedPackages fromPatterns(Iterable<Pattern> patterns) {
    ImmutableSet.Builder<PackageIdentifier> exact = ImmutableSet.builder();
    ImmutableSet.Builder<PackageIdentifier> subtrees = ImmutableSet.builder();
    for (Pattern pattern : patterns) {
      (pattern.subtree() ? subtrees : exact).add(pattern.id());
    }
    return of(exact.build(), subtrees.build());
  }

  /** Returns whether {@code id} is considered deleted. */
  public boolean matches(PackageIdentifier id) {
    if (exact.contains(id)) {
      return true;
    }
    if (subtrees.isEmpty()) {
      return false;
    }
    RepositoryName repo = id.getRepository();
    for (PathFragment ancestor = id.getPackageFragment();
        ancestor != null;
        ancestor = ancestor.getParentDirectory()) {
      if (subtrees.contains(PackageIdentifier.create(repo, ancestor))) {
        return true;
      }
    }
    return false;
  }
}
