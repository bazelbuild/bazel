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

import com.google.common.collect.ComparisonChain;
import com.google.common.collect.Interner;
import com.google.common.collect.Interners;
import com.google.devtools.build.lib.util.Preconditions;
import com.google.devtools.build.lib.vfs.Canonicalizer;
import com.google.devtools.build.lib.vfs.PathFragment;

import java.io.Serializable;

import javax.annotation.concurrent.Immutable;

/**
 * Uniquely identifies a package, given a repository name and a package's path fragment.
 *
 * <p>The repository the build is happening in is the <i>default workspace</i>, and is identified
 * by the workspace name "". Other repositories can be named in the WORKSPACE file.  These
 * workspaces are prefixed by {@literal @}.</p>
 */
@Immutable
public final class PackageIdentifier implements Comparable<PackageIdentifier>, Serializable {

  private static final Interner<PackageIdentifier> INTERNER = Interners.newWeakInterner();


  public static PackageIdentifier create(String repository, PathFragment pkgName)
      throws LabelSyntaxException {
    return create(RepositoryName.create(repository), pkgName);
  }

  public static PackageIdentifier create(RepositoryName repository, PathFragment pkgName) {
    return INTERNER.intern(new PackageIdentifier(repository, pkgName));
  }

  public static final String DEFAULT_REPOSITORY = "";
  public static final RepositoryName DEFAULT_REPOSITORY_NAME;
  public static final RepositoryName MAIN_REPOSITORY_NAME;

  static {
    try {
      DEFAULT_REPOSITORY_NAME = RepositoryName.create(DEFAULT_REPOSITORY);
      MAIN_REPOSITORY_NAME = RepositoryName.create("@");
    } catch (LabelSyntaxException e) {
      throw new IllegalStateException(e);
    }
  }

  // Temporary factory for identifiers without explicit repositories.
  // TODO(bazel-team): remove all usages of this.
  public static PackageIdentifier createInDefaultRepo(String name) {
    return createInDefaultRepo(new PathFragment(name));
  }

  public static PackageIdentifier createInDefaultRepo(PathFragment name) {
    try {
      return create(DEFAULT_REPOSITORY, name);
    } catch (LabelSyntaxException e) {
      throw new IllegalArgumentException("could not create package identifier for " + name
          + ": " + e.getMessage());
    }
  }

  /**
   * The identifier for this repository. This is either "" or prefixed with an "@",
   * e.g., "@myrepo".
   */
  private final RepositoryName repository;

  /** The name of the package. Canonical (i.e. x.equals(y) <=> x==y). */
  private final PathFragment pkgName;

  private PackageIdentifier(RepositoryName repository, PathFragment pkgName) {
    this.repository = Preconditions.checkNotNull(repository);
    this.pkgName = Canonicalizer.fragments().intern(
            Preconditions.checkNotNull(pkgName).normalize());
  }

  public static PackageIdentifier parse(String input) throws LabelSyntaxException {
    String repo;
    String packageName;
    int packageStartPos = input.indexOf("//");
    if (input.startsWith("@") && packageStartPos > 0) {
      repo = input.substring(0, packageStartPos);
      packageName = input.substring(packageStartPos + 2);
    } else if (input.startsWith("@")) {
      throw new LabelSyntaxException("starts with a '@' but does not contain '//'");
    } else if (packageStartPos == 0) {
      repo = PackageIdentifier.DEFAULT_REPOSITORY;
      packageName = input.substring(2);
    } else {
      repo = PackageIdentifier.DEFAULT_REPOSITORY;
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

    return create(repo, new PathFragment(packageName));
  }

  public RepositoryName getRepository() {
    return repository;
  }

  public PathFragment getPackageFragment() {
    return pkgName;
  }

  /**
   * Returns a relative path that should be unique across all remote and packages, based on the
   * repository and package names.
   */
  public PathFragment getPathFragment() {
    return repository.getPathFragment().getRelative(pkgName);
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
    return (repository.isDefault() ? "" : repository + "//") + pkgName;
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
    return pkgName.equals(that.pkgName) && repository.equals(that.repository);
  }

  @Override
  public int hashCode() {
    return 31 * repository.hashCode() + pkgName.hashCode();
  }

  @Override
  public int compareTo(PackageIdentifier that) {
    return ComparisonChain.start()
        .compare(repository.toString(), that.repository.toString())
        .compare(pkgName, that.pkgName)
        .result();
  }
}
