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

package com.google.devtools.build.lib.actions.cache;

import com.google.devtools.build.lib.concurrent.ThreadSafety.ThreadCompatible;
import com.google.devtools.build.lib.vfs.PathFragment;

import java.io.IOException;
import java.io.PrintStream;
import java.util.Collection;

/**
 * An interface defining a cache of already-executed Actions.
 *
 * Each action entry uses one of its output paths as a key (after conversion
 * to the string).
 *
 * TODO(bazel-team): (2010) Key should not be absolute output path but rather relative
 * path (relative to the, e.g., output directory).
 */
@ThreadCompatible
public interface ActionCache {

  /**
   * Updates the cache entry for the specified key.
   */
  void put(String key, ActionCache.Entry entry);

  /**
   * Returns the corresponding cache entry for the specified key, if any, or
   * null if not found.
   */
  ActionCache.Entry get(String key);

  /**
   * Removes entry from cache
   */
  void remove(String key);

  /**
   * Returns a new Entry instance. This method allows ActionCache subclasses to
   * define their own Entry implementation.
   */
  ActionCache.Entry createEntry(String key);

  /**
   * An entry in the ActionCache that contains all action input and output
   * artifact paths and their metadata plus action key itself.
   *
   * Cache entry operates under assumption that once it is fully initialized
   * and ensurePacked() method is called, it becomes logically immutable (all methods
   * will continue to return same result regardless of internal data transformations).
   */
  public interface Entry {

    /**
     * May compress cache entry data into the more compact representation. Called
     * when new action cache entry is created or when dependency checker
     * finished working with the particular entry instance.
     */
    public void ensurePacked();

    /**
     * @return action key string.
     */
    public String getActionKey();

    /**
     * Returns the combined digest of the action's inputs and outputs.
     */
    public Digest getFileDigest();

    /**
     * Adds the artifact, specified by the executable relative path and its
     * metadata into the cache entry.
     */
    public void addFile(PathFragment relativePath, Metadata metadata);

    /**
     * Returns true if this cache entry is corrupted and should be ignored.
     * It will do so by trying to unpack the cache entry and catching
     * the IllegalStateException that might be thrown by the unpack() method.
     */
    public boolean isCorrupted();

    /**
     * @return stored path strings.
     */
    public Collection<String> getPaths();

  }

  /**
   * Give persistent cache implementations a notification to write to disk.
   * @return size in bytes of the serialized cache.
   */
  long save() throws IOException;

  /**
   * Dumps action cache content into the given PrintStream.
   */
  void dump(PrintStream out);
}
