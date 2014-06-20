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
package com.google.devtools.build.lib.skyframe;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.vfs.ModifiedFileSet;
import com.google.devtools.build.lib.vfs.Path;

import javax.annotation.Nullable;

/**
 * Interface for computing modifications of files under a package path entry.
 *
 * <p> Skyframe has a {@link DiffAwareness} instance per package-path entry, and each instance is
 * responsible for all files under its path entry. At the beginning of each incremental build,
 * skyframe queries for changes using {@link #getDiff}. Ideally, {@link #getDiff} should be
 * constant-time; if it were linear in the number of files of interest, we might as well just
 * detect modifications manually.
 */
public interface DiffAwareness {

  /** Factory for creating {@link DiffAwareness} instances. */
  public interface Factory {
    /**
     * Returns a {@link DiffAwareness} instance suitable for managing changes to files under the
     * given package path entry, or {@code null} if this factory cannot create such an instance.
     *
     * <p> Skyframe has a collection of factories, and will create a {@link DiffAwareness} instance
     * per package path entry using one of the factories that returns a non-null value.
     */
    @Nullable
    DiffAwareness maybeCreate(Path pathEntry, ImmutableList<Path> pathEntries);
  }

  /**
   * Returns details about modifications of files of interest since the last call to
   * {@link #getDiff}. If this is the first such call then
   * {@code ModifiedFileSet.EVERYTHING_MODIFIED} should be returned.
   *
   * The caller should either fully process these results, or conservatively throw away this
   * {@link DiffAwareness} instance and make another one so as to not forget the changes.
   */
  ModifiedFileSet getDiff();

  /**
   * Returns whether the {@link DiffAwareness} object is still appropriate for watching its files
   * of interest. If not, {@link BlindDiffAwareness} will be used (but the factory may be asked in
   * the future to potentially create a more sophisticated {@link DiffAwareness}).
   */
  boolean canStillBeUsed();
}
