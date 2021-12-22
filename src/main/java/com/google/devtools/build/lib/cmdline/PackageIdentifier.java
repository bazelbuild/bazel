// Copyright 2015 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.cmdline;

import com.google.common.base.Preconditions;
import com.google.common.collect.ComparisonChain;
import com.google.common.collect.Interner;
import com.google.devtools.build.lib.concurrent.BlazeInterners;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec;
import com.google.devtools.build.lib.vfs.PathFragment;
import java.util.Objects;
import javax.annotation.concurrent.Immutable;

/**
 * Uniquely identifies a package. Contains the (canonical) name of the repository this package lives
 * in, and the package's path fragment.
 */
@AutoCodec
@Immutable
public final class PackageIdentifier implements Comparable<PackageIdentifier> {
  private static final Interner<PackageIdentifier> INTERNER = BlazeInterners.newWeakInterner();

  public static PackageIdentifier create(String repository, PathFragment pkgName)
      throws LabelSyntaxException {
    return create(RepositoryName.create(repository), pkgName);
  }

  @AutoCodec.Instantiator
  public static PackageIdentifier create(RepositoryName repository, PathFragment pkgName) {
    // Note: We rely on these being (weakly) interned to fast-path Label#equals.
    return INTERNER.intern(new PackageIdentifier(repository, pkgName));
  }

  public static final PackageIdentifier EMPTY_PACKAGE_ID =
      createInMainRepo(PathFragment.EMPTY_FRAGMENT);

  public static PackageIdentifier createInMainRepo(String name) {
    return createInMainRepo(PathFragment.create(name));
  }

  public static PackageIdentifier createInMainRepo(PathFragment name) {
    return create(RepositoryName.MAIN, name);
  }

  /**
   * Tries to infer the package identifier from the given exec path. This method does not perform
   * any I/O, but looks solely at the structure of the exec path. The resulting identifier may
   * actually be a subdirectory of a package rather than a package, e.g.:
   *
   * <pre><code>
   * + WORKSPACE
   * + foo/BUILD
   * + foo/bar/bar.java
   * </code></pre>
   *
   * In this case, this method returns a package identifier for foo/bar, even though that is not a
   * package. Callers need to look up the actual package if needed.
   */
  public static PackageIdentifier discoverFromExecPath(
      PathFragment execPath, boolean forFiles, boolean siblingRepositoryLayout) {
    Preconditions.checkArgument(!execPath.isAbsolute(), execPath);
    PathFragment tofind =
        forFiles
            ? Preconditions.checkNotNull(
                execPath.getParentDirectory(), "Must pass in files, not root directory")
            : execPath;
    PathFragment prefix =
        siblingRepositoryLayout
            ? LabelConstants.EXPERIMENTAL_EXTERNAL_PATH_PREFIX
            : LabelConstants.EXTERNAL_PATH_PREFIX;
    if (tofind.startsWith(prefix)) {
      // Using the path prefix can be either "external" or "..", depending on whether the sibling
      // repository layout is used.
      RepositoryName repository = RepositoryName.createFromValidStrippedName(tofind.getSegment(1));
      return PackageIdentifier.create(repository, tofind.subFragment(2));
    } else {
      return PackageIdentifier.createInMainRepo(tofind);
    }
  }

  /**
   * The identifier for this repository. This is either "" or prefixed with an "@", e.g., "@myrepo".
   */
  private final RepositoryName repository;

  /** The name of the package. */
  private final PathFragment pkgName;

  /**
   * Precomputed hash code. Hash/equality is based on repository and pkgName. Note that due to weak
   * interning, x.equals(y) usually implies x==y.
   */
  private final int hashCode;

  private PackageIdentifier(RepositoryName repository, PathFragment pkgName) {
    this.repository = Preconditions.checkNotNull(repository);
    this.pkgName = Preconditions.checkNotNull(pkgName);
    this.hashCode = Objects.hash(repository, pkgName);
  }

  public static PackageIdentifier parse(String input) throws LabelSyntaxException {
    if (input.contains(":")) {
      throw LabelParser.syntaxErrorf("invalid package identifier '%s': contains ':'", input);
    }
    LabelParser.Parts parts = LabelParser.Parts.parse(input + ":dummy_target");
    RepositoryName repoName =
        parts.repo == null
            ? RepositoryName.MAIN
            : RepositoryName.createFromValidStrippedName(parts.repo);
    return create(repoName, PathFragment.create(parts.pkg));
  }

  public RepositoryName getRepository() {
    return repository;
  }

  public PathFragment getPackageFragment() {
    return pkgName;
  }

  /**
   * Returns a path to the source code for this package relative to the corresponding source root.
   * Returns pkgName for all repositories.
   */
  public PathFragment getSourceRoot() {
    return pkgName;
  }

  /**
   * Returns the package path fragment to derived artifacts for this package. Returns pkgName if
   * this is in the main repository or siblingRepositoryLayout is true. Otherwise, returns
   * external/[repository name]/[pkgName].
   */
  // TODO(bazel-team): Rename getDerivedArtifactPath or similar.
  public PathFragment getPackagePath(boolean siblingRepositoryLayout) {
    return repository.isMain() || siblingRepositoryLayout
        ? pkgName
        : LabelConstants.EXTERNAL_PATH_PREFIX
            .getRelative(repository.strippedName())
            .getRelative(pkgName);
  }

  public PathFragment getExecPath(boolean siblingRepositoryLayout) {
    return repository.getExecPath(siblingRepositoryLayout).getRelative(pkgName);
  }

  /**
   * Returns the runfiles/execRoot path for this repository (relative to the x.runfiles/main-repo/
   * directory).
   */
  public PathFragment getRunfilesPath() {
    return repository.getRunfilesPath().getRelative(pkgName);
  }

  /**
   * Returns the package in label syntax format.
   *
   * <p>Packages in the main repo are formatted without a repo qualifier.
   */
  // TODO(bazel-team): Maybe rename to "getDefaultForm"?
  public String getCanonicalForm() {
    String repository = getRepository().getCanonicalForm();
    return repository + "//" + getPackageFragment();
  }

  /**
   * Returns the package path, possibly qualified with a repository name.
   *
   * <p>Packages that live in the main repo are stringified without a "@" qualifier or "//"
   * separator (e.g. "foo/bar"). All other packages include these (e.g. "@repo//foo/bar").
   */
  // TODO(bazel-team): The absence of "//" for the main repo seems strange. Can we eliminate
  // that disparity?
  @Override
  public String toString() {
    return (repository.isMain() ? "" : repository + "//") + pkgName;
  }

  @Override
  public boolean equals(Object object) {
    if (this == object) {
      return true;
    }
    if (!(object instanceof PackageIdentifier)) {
      return false;
    }
    PackageIdentifier that = (PackageIdentifier) object;
    return this.hashCode == that.hashCode
        && pkgName.equals(that.pkgName)
        && repository.equals(that.repository);
  }

  @Override
  public int hashCode() {
    return this.hashCode;
  }

  @Override
  @SuppressWarnings("ReferenceEquality") // Performance optimization.
  public int compareTo(PackageIdentifier that) {
    // Fast-paths for the common case of the same package or a package in the same repository.
    if (this == that) {
      return 0;
    }
    if (repository == that.repository) {
      return pkgName.compareTo(that.pkgName);
    }
    return ComparisonChain.start()
        .compare(repository.toString(), that.repository.toString())
        .compare(pkgName, that.pkgName)
        .result();
  }
}
