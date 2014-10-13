// Copyright 2014 Google Inc. All rights reserved.
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

package com.google.devtools.build.lib.packages;

import com.google.common.base.Objects;
import com.google.common.base.Preconditions;
import com.google.common.collect.ComparisonChain;
import com.google.devtools.build.lib.cmdline.LabelValidator;
import com.google.devtools.build.lib.syntax.Label.SyntaxException;
import com.google.devtools.build.lib.util.StringCanonicalizer;
import com.google.devtools.build.lib.util.StringUtilities;
import com.google.devtools.build.lib.vfs.PathFragment;

import javax.annotation.concurrent.Immutable;

/**
 * Uniquely identifies a package, given a repository name and a package's path fragment.
 *
 * <p>The repository the build is happening in is the <i>default workspace</i>, and is identified
 * by the workspace name "". Other repositories can be named in the WORKSPACE file.  These
 * workspaces are prefixed by {@literal @}.</p>
 */
@Immutable
public final class PackageIdentifier implements Comparable<PackageIdentifier> {
  public static final String DEFAULT_REPOSITORY = "";

  /**
   * Validates the given repository name and returns a canonical String instance if it is valid.
   * Otherwise throws a SyntaxException.
   * @throws SyntaxException
   */
  private static String canonicalizeRepositoryName(String repositoryName) throws SyntaxException {
    String error = LabelValidator.validateWorkspaceName(repositoryName);
    if (error != null) {
      error = "invalid repository name '" + StringUtilities.sanitizeControlChars(repositoryName)
          + "': " + error;
      throw new SyntaxException(error);
    }

    return StringCanonicalizer.intern(repositoryName);
  }

  // Temporary factory for identifiers without explicit repositories.
  // TODO(bazel-team): remove all usages of this.
  public static PackageIdentifier createInDefaultRepo(String name) {
    return createInDefaultRepo(new PathFragment(name));
  }

  public static PackageIdentifier createInDefaultRepo(PathFragment name) {
    try {
      return new PackageIdentifier(DEFAULT_REPOSITORY, name);
    } catch (SyntaxException e) {
      throw new IllegalArgumentException("could not create package identifier for " + name
          + ": " + e.getMessage());
    }
  }


  /**
   * The identifier for this repository. This is either "" or prefixed with an "@",
   * e.g., "@myrepo".
   */
  private final String repository;

  /** The name of the package. Canonical (i.e. x.equals(y) <=> x==y). */
  private final PathFragment pkgName;

  public PackageIdentifier(String repository, PathFragment pkgName) throws SyntaxException {
    Preconditions.checkNotNull(repository);
    Preconditions.checkNotNull(pkgName);
    this.repository = canonicalizeRepositoryName(repository);
    this.pkgName = pkgName;
  }

  public String getRepository() {
    return repository;
  }

  public PathFragment getPackageFragment() {
    return pkgName;
  }

  @Override
  public String toString() {
    return repository + "//" + pkgName;
  }

  @Override
  public boolean equals(Object object) {
    if (this == object) {
      return true;
    }
    if (object instanceof PackageIdentifier) {
      PackageIdentifier that = (PackageIdentifier) object;
      return repository.equals(that.repository) && pkgName.equals(that.pkgName);
    }
    return false;
  }

  @Override
  public int hashCode() {
    return Objects.hashCode(repository, pkgName);
  }

  @Override
  public int compareTo(PackageIdentifier that) {
    return ComparisonChain.start()
        .compare(repository, that.repository)
        .compare(pkgName, that.pkgName)
        .result();
  }
}