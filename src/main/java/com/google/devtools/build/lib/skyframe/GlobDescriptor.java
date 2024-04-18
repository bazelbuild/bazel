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

import com.google.common.base.Preconditions;
import com.google.devtools.build.lib.cmdline.PackageIdentifier;
import com.google.devtools.build.lib.concurrent.ThreadSafety.ThreadSafe;
import com.google.devtools.build.lib.packages.Globber;
import com.google.devtools.build.lib.skyframe.serialization.VisibleForSerialization;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec;
import com.google.devtools.build.lib.util.StringCanonicalizer;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.Root;
import com.google.devtools.build.skyframe.SkyFunctionName;
import com.google.devtools.build.skyframe.SkyKey;

/**
 * A descriptor for a glob request, used as the {@link SkyKey} for {@link GlobFunction}.
 *
 * <p>{@code subdir} must be empty or point to an existing directory.
 *
 * <p>{@code pattern} must be valid, as indicated by {@code UnixGlob#checkPatternForError}.
 */
@AutoCodec
@ThreadSafe
public final class GlobDescriptor implements SkyKey {

  private static final SkyKeyInterner<GlobDescriptor> interner = SkyKey.newInterner();

  /**
   * Returns interned instance based on the parameters.
   *
   * @param packageId the name of the owner package (must be an existing package)
   * @param packageRoot the package root of {@code packageId}
   * @param subdir the subdirectory being looked at (must exist and must be a directory. It's
   *     assumed that there are no other packages between {@code packageName} and {@code subdir}.
   * @param pattern a valid glob pattern
   * @param globberOperation type of Globber operation being tracked.
   */
  public static GlobDescriptor create(
      PackageIdentifier packageId,
      Root packageRoot,
      PathFragment subdir,
      String pattern,
      Globber.Operation globberOperation) {
    return interner.intern(
        new GlobDescriptor(packageId, packageRoot, subdir, pattern, globberOperation));
  }

  @VisibleForSerialization
  @AutoCodec.Interner
  static GlobDescriptor intern(GlobDescriptor globDescriptor) {
    return interner.intern(globDescriptor);
  }

  private final PackageIdentifier packageId;
  private final Root packageRoot;
  private final PathFragment subdir;
  private final String pattern;
  private final Globber.Operation globberOperation;

  private GlobDescriptor(
      PackageIdentifier packageId,
      Root packageRoot,
      PathFragment subdir,
      String pattern,
      Globber.Operation globberOperation) {
    this.packageId = Preconditions.checkNotNull(packageId);
    this.packageRoot = Preconditions.checkNotNull(packageRoot);
    this.subdir = Preconditions.checkNotNull(subdir);
    this.pattern = Preconditions.checkNotNull(StringCanonicalizer.intern(pattern));
    this.globberOperation = globberOperation;
  }

  @Override
  public String toString() {
    return String.format(
        "<GlobDescriptor packageName=%s packageRoot=%s subdir=%s pattern=%s globberOperation=%s>",
        packageId, packageRoot, subdir, pattern, globberOperation.name());
  }

  /**
   * Returns the package that "owns" this glob.
   *
   * <p>The glob evaluation code ensures that the boundaries of this package are not crossed.
   */
  public PackageIdentifier getPackageId() {
    return packageId;
  }

  /** Returns the package root of {@code getPackageId()}. */
  public Root getPackageRoot() {
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

  /** Returns the type of Globber operation that produced the results. */
  public Globber.Operation globberOperation() {
    return globberOperation;
  }

  @Override
  public boolean equals(Object obj) {
    if (this == obj) {
      return true;
    }
    if (!(obj instanceof GlobDescriptor other)) {
      return false;
    }
    return packageId.equals(other.packageId)
        && packageRoot.equals(other.packageRoot)
        && subdir.equals(other.subdir)
        && pattern.equals(other.pattern)
        && globberOperation == other.globberOperation;
  }

  @Override
  public int hashCode() {
    // Generated instead of Objects.hashCode to avoid intermediate array required for latter.
    final int prime = 31;
    int result = 1;
    result = prime * result + globberOperation.hashCode();
    result = prime * result + packageId.hashCode();
    result = prime * result + packageRoot.hashCode();
    result = prime * result + pattern.hashCode();
    result = prime * result + subdir.hashCode();
    return result;
  }

  @Override
  public SkyFunctionName functionName() {
    return SkyFunctions.GLOB;
  }

  @Override
  public SkyKeyInterner<GlobDescriptor> getSkyKeyInterner() {
    return interner;
  }
}
