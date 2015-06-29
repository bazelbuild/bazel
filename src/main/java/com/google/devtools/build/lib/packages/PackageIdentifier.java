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

import com.google.common.base.Preconditions;
import com.google.common.collect.ComparisonChain;
import com.google.devtools.build.lib.syntax.Label.SyntaxException;
import com.google.devtools.build.lib.util.StringCanonicalizer;
import com.google.devtools.build.lib.util.StringUtilities;
import com.google.devtools.build.lib.vfs.Canonicalizer;
import com.google.devtools.build.lib.vfs.PathFragment;

import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.ObjectStreamException;
import java.io.Serializable;
import java.util.Objects;

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

  /**
   * A human-readable name for the repository.
   */
  public static final class RepositoryName {
    private final String name;

    /**
     * Makes sure that name is a valid repository name and creates a new RepositoryName using it.
     * @throws SyntaxException if the name is invalid.
     */
    public static RepositoryName create(String name) throws SyntaxException {
      String errorMessage = validate(name);
      if (errorMessage != null) {
        errorMessage = "invalid repository name '"
            + StringUtilities.sanitizeControlChars(name) + "': " + errorMessage;
        throw new SyntaxException(errorMessage);
      }
      return new RepositoryName(StringCanonicalizer.intern(name));
    }

    private RepositoryName(String name) {
      this.name = name;
    }

    /**
     * Performs validity checking.  Returns null on success, an error message otherwise.
     */
    private static String validate(String name) {
      if (name.isEmpty()) {
        return null;
      }

      if (!name.startsWith("@")) {
        return "workspace name must start with '@'";
      }

      // "@" isn't a valid workspace name.
      if (name.length() == 1) {
        return "empty workspace name";
      }

      // Check for any character outside of [/0-9A-Z_a-z-._]. Try to evaluate the
      // conditional quickly (by looking in decreasing order of character class
      // likelihood).
      if (name.startsWith("@/") || name.endsWith("/")) {
        return "workspace names cannot start nor end with '/'";
      } else if (name.contains("//")) {
        return "workspace names cannot contain multiple '/'s in a row";
      }

      for (int i = name.length() - 1; i >= 1; --i) {
        char c = name.charAt(i);
        if ((c < 'a' || c > 'z') && c != '_' && c != '-' && c != '/' && c != '.'
            && (c < '0' || c > '9') && (c < 'A' || c > 'Z')) {
          return "workspace names may contain only A-Z, a-z, 0-9, '-', '_', '.', and '/'";
        }
      }
      return null;
    }

    /**
     * Returns the repository name without the leading "{@literal @}".  For the default repository,
     * returns "".
     */
    public String strippedName() {
      if (name.isEmpty()) {
        return name;
      }
      return name.substring(1);
    }

    /**
     * Returns if this is the default repository, that is, {@link #name} is "".
     */
    public boolean isDefault() {
      return name.isEmpty();
    }

    /**
     * Returns the repository name, with leading "{@literal @}" (or "" for the default repository).
     */
    @Override
    public String toString() {
      return name;
    }

    @Override
    public boolean equals(Object object) {
      if (this == object) {
        return true;
      }
      if (!(object instanceof RepositoryName)) {
        return false;
      }
      return name.equals(((RepositoryName) object).name);
    }

    @Override
    public int hashCode() {
      return name.hashCode();
    }
  }

  public static final String DEFAULT_REPOSITORY = "";

  /**
   * Helper for serializing PackageIdentifiers.
   *
   * <p>PackageIdentifier's field should be final, but then it couldn't be deserialized. This
   * allows the fields to be deserialized and copied into a new PackageIdentifier.</p>
   */
  private static final class SerializationProxy implements Serializable {
    PackageIdentifier packageId;

    public SerializationProxy(PackageIdentifier packageId) {
      this.packageId = packageId;
    }

    private void writeObject(ObjectOutputStream out) throws IOException {
      out.writeObject(packageId.repository.toString());
      out.writeObject(packageId.pkgName);
    }

    private void readObject(ObjectInputStream in)
        throws IOException, ClassNotFoundException {
      try {
        packageId = new PackageIdentifier((String) in.readObject(), (PathFragment) in.readObject());
      } catch (SyntaxException e) {
        throw new IOException("Error serializing package identifier: " + e.getMessage());
      }
    }

    @SuppressWarnings("unused")
    private void readObjectNoData() throws ObjectStreamException {
    }

    private Object readResolve() {
      return packageId;
    }
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
  private final RepositoryName repository;

  /** The name of the package. Canonical (i.e. x.equals(y) <=> x==y). */
  private final PathFragment pkgName;

  public PackageIdentifier(String repository, PathFragment pkgName) throws SyntaxException {
    this(RepositoryName.create(repository), pkgName);
  }

  public PackageIdentifier(RepositoryName repository, PathFragment pkgName) {
    Preconditions.checkNotNull(repository);
    Preconditions.checkNotNull(pkgName);
    this.repository = repository;
    this.pkgName = Canonicalizer.fragments().intern(pkgName.normalize());
  }

  private Object writeReplace() throws ObjectStreamException {
    return new SerializationProxy(this);
  }

  private void readObject(ObjectInputStream in)
      throws IOException, ClassNotFoundException {
    throw new IOException("Serialization is allowed only by proxy");
  }

  @SuppressWarnings("unused")
  private void readObjectNoData() throws ObjectStreamException {
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
    return repository.isDefault() ? pkgName
        : new PathFragment(ExternalPackage.NAME).getRelative(repository.strippedName())
            .getRelative(pkgName);
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
    return Objects.hash(repository, pkgName);
  }

  @Override
  public int compareTo(PackageIdentifier that) {
    return ComparisonChain.start()
        .compare(repository.toString(), that.repository.toString())
        .compare(pkgName, that.pkgName)
        .result();
  }
}
