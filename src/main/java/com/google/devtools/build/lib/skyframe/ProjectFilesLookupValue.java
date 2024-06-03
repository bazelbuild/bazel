// Copyright 2024 The Bazel Authors. All rights reserved.
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

import com.google.common.base.Objects;
import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.cmdline.PackageIdentifier;
import com.google.devtools.build.lib.skyframe.serialization.VisibleForSerialization;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec;
import com.google.devtools.build.skyframe.AbstractSkyKey;
import com.google.devtools.build.skyframe.SkyFunctionName;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;
import java.util.Collection;

/**
 * {@link SkyValue} for finding the PROJECT files associated with a package.
 *
 * <p>See {@link com.google.devtools.build.lib.analysis.Project}.
 */
public class ProjectFilesLookupValue implements SkyValue {
  private final ImmutableList<Label> projectFiles;

  /**
   * Returns the {@link com.google.devtools.build.lib.analysis.Project} files associated with the
   * corresponding {@link Key}'s package.
   *
   * <p>Given {@code a/b/c/d}, project resolution walks up the package path (i.e. walks up the
   * directory tree from {@code d} back to {@code a}, only counting directories with BUILD files).
   * Each directory with both a BUILD file and project file has a label reference to the project
   * file here.
   *
   * <p>Order is innermost to outermost: if both {@code a/PROJECT.scl} and {@code a/b/c/PROJECT.scl}
   * are included, {@code a/b/c/PROJECT.scl} appears first.
   */
  public ImmutableList<Label> getProjectFiles() {
    return projectFiles;
  }

  /**
   * Lookup key.
   *
   * @param id the package for which to find enclosing {@link
   *     com.google.devtools.build.lib.analysis.Project} files
   */
  public static Key key(PackageIdentifier id) {
    Preconditions.checkArgument(!id.getPackageFragment().isAbsolute(), id);
    return Key.create(id);
  }

  private static final ProjectFilesLookupValue NO_PROJECT_FILES =
      new ProjectFilesLookupValue(ImmutableList.of());

  static ProjectFilesLookupValue of(Collection<Label> projectFiles) {
    return projectFiles.isEmpty()
        ? NO_PROJECT_FILES
        : new ProjectFilesLookupValue(ImmutableList.copyOf(projectFiles));
  }

  private ProjectFilesLookupValue(ImmutableList<Label> projectFiles) {
    this.projectFiles = projectFiles;
  }

  @Override
  public boolean equals(Object o) {
    if (this == o) {
      return true;
    }
    if (!(o instanceof ProjectFilesLookupValue)) {
      return false;
    }
    return Objects.equal(projectFiles, ((ProjectFilesLookupValue) o).projectFiles);
  }

  @Override
  public int hashCode() {
    return java.util.Objects.hashCode(projectFiles);
  }

  /** {@link SkyKey} for {@code ProjectFilesLookupValue}. */
  @AutoCodec
  public static class Key extends AbstractSkyKey<PackageIdentifier> {
    private static final SkyKeyInterner<Key> interner = SkyKey.newInterner();

    private Key(PackageIdentifier arg) {
      super(arg);
    }

    private static Key create(PackageIdentifier arg) {
      return interner.intern(new Key(arg));
    }

    @VisibleForSerialization
    @AutoCodec.Interner
    static Key intern(Key key) {
      return interner.intern(key);
    }

    @Override
    public SkyFunctionName functionName() {
      return SkyFunctions.PROJECT_FILES_LOOKUP;
    }

    @Override
    public SkyKeyInterner<Key> getSkyKeyInterner() {
      return interner;
    }
  }
}
