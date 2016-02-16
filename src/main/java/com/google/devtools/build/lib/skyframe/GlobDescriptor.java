// Copyright 2014 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.skyframe;

import com.google.devtools.build.lib.cmdline.PackageIdentifier;
import com.google.devtools.build.lib.concurrent.ThreadSafety.ThreadSafe;
import com.google.devtools.build.lib.util.Preconditions;
import com.google.devtools.build.lib.util.StringCanonicalizer;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;

import java.io.Serializable;
import java.util.Objects;

/**
 * A descriptor for a glob request.
 *
 * <p>{@code subdir} must be empty or point to an existing directory.</p>
 *
 * <p>{@code pattern} must be valid, as indicated by {@code UnixGlob#checkPatternForError}.
 */
@ThreadSafe
public final class GlobDescriptor implements Serializable {
  final PackageIdentifier packageId;
  final Path packageRoot;
  final PathFragment subdir;
  final String pattern;
  final boolean excludeDirs;

  /**
   * Constructs a GlobDescriptor.
   *
   * @param packageId the name of the owner package (must be an existing package)
   * @param packageRoot the package root of {@code packageId}
   * @param subdir the subdirectory being looked at (must exist and must be a directory. It's
   *               assumed that there are no other packages between {@code packageName} and
   *               {@code subdir}.
   * @param pattern a valid glob pattern
   * @param excludeDirs true if directories should be excluded from results
   */
  public GlobDescriptor(PackageIdentifier packageId, Path packageRoot, PathFragment subdir,
      String pattern, boolean excludeDirs) {
    this.packageId = Preconditions.checkNotNull(packageId);
    this.packageRoot = Preconditions.checkNotNull(packageRoot);
    this.subdir = Preconditions.checkNotNull(subdir);
    this.pattern = Preconditions.checkNotNull(StringCanonicalizer.intern(pattern));
    this.excludeDirs = excludeDirs;
  }

  @Override
  public String toString() {
    return String.format(
        "<GlobDescriptor packageName=%s packageRoot=%s subdir=%s pattern=%s excludeDirs=%s>",
        packageId, packageRoot, subdir, pattern, excludeDirs);
  }

  /**
   * Returns the package that "owns" this glob.
   *
   * <p>The glob evaluation code ensures that the boundaries of this package are not crossed.
   */
  public PackageIdentifier getPackageId() {
    return packageId;
  }

  /**
   * Returns the package root of {@code getPackageId()}.
   */
  public Path getPackageRoot() {
    return packageRoot;
  }

  /**
   * Returns the subdirectory of the package under consideration.
   */
  public PathFragment getSubdir() {
    return subdir;
  }

  /**
   * Returns the glob pattern under consideration. May contain wildcards.
   *
   * <p>As the glob evaluator traverses deeper into the file tree, components are added at the
   * beginning of {@code subdir} and removed from the beginning of {@code pattern}.
   */
  public String getPattern() {
    return pattern;
  }

  /**
   * Returns true if directories should be excluded from results.
   */
  public boolean excludeDirs() {
    return excludeDirs;
  }

  @Override
  public int hashCode() {
    return Objects.hash(packageId, subdir, pattern, excludeDirs);
  }

  @Override
  public boolean equals(Object obj) {
    if (this == obj) {
      return true;
    }
    if (!(obj instanceof GlobDescriptor)) {
      return false;
    }
    GlobDescriptor other = (GlobDescriptor) obj;
    return packageId.equals(other.packageId) && packageRoot.equals(other.packageRoot)
        && subdir.equals(other.subdir) && pattern.equals(other.pattern)
        && excludeDirs == other.excludeDirs;
  }
}