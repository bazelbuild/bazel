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

import java.io.Closeable;

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
public interface DiffAwareness extends Closeable {

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
   * {@link #getDiff}
   *
   * <p> Throws a {@link BrokenDiffAwarenessException} to indicate that something is
   * wrong and the caller should throw away this {@link DiffAwareness} instance. The
   * {@link DiffAwareness} closes itself in this case.
   *
   * <p> If this is the first call to {@link #getDiff} then
   * {@code ModifiedFileSet.EVERYTHING_MODIFIED} should be returned.
   *
   * <p> The caller should either fully process these results, or conservatively call
   * {@link #close} and throw away this {@link DiffAwareness} instance. Otherwise the results of
   * the next {@link #getDiff} call won't make sense.
   */
  ModifiedFileSet getDiff() throws BrokenDiffAwarenessException;

  /**
   * Must be called whenever the {@link DiffAwareness} object is to be discarded. Using a
   * {@link DiffAwareness} instance after calling {@link #close} on it is unspecified behavior.
   */
  @Override
  void close();
}
