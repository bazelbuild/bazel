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
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.Interner;
import com.google.devtools.build.lib.concurrent.BlazeInterners;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec;
import com.google.devtools.build.lib.vfs.PathFragment;
import java.io.Serializable;
import java.util.Objects;
import javax.annotation.concurrent.Immutable;

/**
 * Uniquely identifies a package, given a repository name and a package's path fragment.
 *
 * <p>The repository the build is happening in is the <i>default workspace</i>, and is identified by
 * the workspace name "". Other repositories can be named in the WORKSPACE file. These workspaces
 * are prefixed by {@literal @}.
 */
@AutoCodec
@Immutable
public final class PackageIdentifier implements Comparable<PackageIdentifier>, Serializable {
  private static final Interner<PackageIdentifier> INTERNER = BlazeInterners.newWeakInterner();

  public static PackageIdentifier create(String repository, PathFragment pkgName)
      throws LabelSyntaxException {
    return create(RepositoryName.create(repository), pkgName);
  }

  @AutoCodec.Instantiator
  public static PackageIdentifier create(RepositoryName repository, PathFragment pkgName) {
    // Note: We rely on these being interned to fast-path Label#equals.
    return INTERNER.intern(new PackageIdentifier(repository, pkgName));
  }

  public static final PackageIdentifier EMPTY_PACKAGE_ID = createInMainRepo(
      PathFragment.EMPTY_FRAGMENT);

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
   *
   * @throws LabelSyntaxException if the exec path seems to be for an external repository that does
   *     not have a valid repository name (see {@link RepositoryName#create})
   */
  public static PackageIdentifier discoverFromExecPath(
      PathFragment execPath, boolean forFiles, boolean siblingRepositoryLayout) {
    Preconditions.checkArgument(!execPath.isAbsolute(), execPath);
    PathFragment tofind = forFiles
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
   * The identifier for this repository. This is either "" or prefixed with an "@",
   * e.g., "@myrepo".
   */
  private final RepositoryName repository;

  /** The name of the package. */
  private final PathFragment pkgName;

  /**
   * Precomputed hash code. Hash/equality is based on repository and pkgName. Note that due to
   * interning, x.equals(y) <=> x==y.
   **/
  private final int hashCode;

  private PackageIdentifier(RepositoryName repository, PathFragment pkgName) {
    this.repository = Preconditions.checkNotNull(repository);
    this.pkgName = Preconditions.checkNotNull(pkgName);
    this.hashCode = Objects.hash(repository, pkgName);
  }

  public static PackageIdentifier parse(String input) throws LabelSyntaxException {
    return parse(input, /* repo= */ null, /* repositoryMapping= */ null);
  }

  public static PackageIdentifier parse(
      String input, String repo, ImmutableMap<RepositoryName, RepositoryName> repositoryMapping)
      throws LabelSyntaxException {
    String packageName;
    int packageStartPos = input.indexOf("//");
    if (repo != null) {
      packageName = input;
    } else if (input.startsWith("@") && packageStartPos > 0) {
      repo = input.substring(0, packageStartPos);
      packageName = input.substring(packageStartPos + 2);
    } else if (input.startsWith("@")) {
      throw new LabelSyntaxException("starts with a '@' but does not contain '//'");
    } else if (packageStartPos == 0) {
      repo = RepositoryName.DEFAULT_REPOSITORY;
      packageName = input.substring(2);
    } else {
      repo = RepositoryName.DEFAULT_REPOSITORY;
      packageName = input;
    }

    String error = RepositoryName.validate(repo);
    if (error != null) {
      throw new LabelSyntaxException(error);
    }

    error = LabelValidator.validatePackageName(packageName);
    if (error != null) {
      throw new LabelSyntaxException(error);
    }

    if (repositoryMapping != null) {
      RepositoryName repositoryName = RepositoryName.create(repo);
      repositoryName = repositoryMapping.getOrDefault(repositoryName, repositoryName);
      return create(repositoryName, PathFragment.create(packageName));
    } else {
      return create(repo, PathFragment.create(packageName));
    }
  }

  public RepositoryName getRepository() {
    return repository;
  }

  public PathFragment getPackageFragment() {
    return pkgName;
  }

  /**
   * Returns a path to the source code for this package relative to the corresponding source root.
   * Returns pkgName if this is in the main repository or [repository name]/[pkgName] if not.
   */
  public PathFragment getSourceRoot() {
    return repository.getSourceRoot().getRelative(pkgName);
  }

  /**
   * Returns the package path to the source code for this package. Returns pkgName if this is in the
   * main repository or external/[repository name]/[pkgName] if not.
   */
  public PathFragment getPackagePath(boolean siblingRepositoryLayout) {
    return repository.getPackagePath().getRelative(pkgName);
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

  public PackageIdentifier makeAbsolute() {
    if (!repository.isDefault()) {
      return this;
    }

    return create(RepositoryName.MAIN, pkgName);
  }

  /**
   * Returns the name of this package.
   *
   * <p>There are certain places that expect the path fragment as the package name ('foo/bar') as a
   * package identifier. This isn't specific enough for packages in other repositories, so their
   * stringified version is '@baz//foo/bar'.</p>
   */
  @Override
  public String toString() {
    return (repository.isDefault() || repository.isMain() ? "" : repository + "//") + pkgName;
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
    return this.hashCode == that.hashCode && pkgName.equals(that.pkgName)
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
