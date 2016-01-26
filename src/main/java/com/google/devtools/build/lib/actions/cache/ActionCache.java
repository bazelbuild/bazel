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

package com.google.devtools.build.lib.actions.cache;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.Lists;
import com.google.devtools.build.lib.concurrent.ThreadSafety.ThreadCompatible;
import com.google.devtools.build.lib.util.Preconditions;
import com.google.devtools.build.lib.vfs.PathFragment;

import java.io.IOException;
import java.io.PrintStream;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import javax.annotation.Nullable;

/**
 * An interface defining a cache of already-executed Actions.
 *
 * <p>This class' naming is misleading; it doesn't cache the actual actions, but it stores a
 * fingerprint of the action state (ie. a hash of the input and output files on disk), so
 * we can tell if we need to rerun an action given the state of the file system.
 *
 * <p>Each action entry uses one of its output paths as a key (after conversion
 * to the string).
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
  ActionCache.Entry createEntry(String key, boolean discoversInputs);

  /**
   * An entry in the ActionCache that contains all action input and output
   * artifact paths and their metadata plus action key itself.
   *
   * Cache entry operates under assumption that once it is fully initialized
   * and getFileDigest() method is called, it becomes logically immutable (all methods
   * will continue to return same result regardless of internal data transformations).
   */
  final class Entry {
    private final String actionKey;
    @Nullable
    // Null iff the corresponding action does not do input discovery.
    private final List<String> files;
    // If null, digest is non-null and the entry is immutable.
    private Map<String, Metadata> mdMap;
    private Digest digest;

    public Entry(String key, boolean discoversInputs) {
      actionKey = key;
      files = discoversInputs ? new ArrayList<String>() : null;
      mdMap = new HashMap<>();
    }

    public Entry(String key, @Nullable List<String> files, Digest digest) {
      actionKey = key;
      this.files = files;
      this.digest = digest;
      mdMap = null;
    }

    /**
     * Adds the artifact, specified by the executable relative path and its
     * metadata into the cache entry.
     */
    public void addFile(PathFragment relativePath, Metadata md) {
      Preconditions.checkState(mdMap != null);
      Preconditions.checkState(!isCorrupted());
      Preconditions.checkState(digest == null);

      String execPath = relativePath.getPathString();
      if (discoversInputs()) {
        files.add(execPath);
      }
      mdMap.put(execPath, md);
    }

    /**
     * @return action key string.
     */
    public String getActionKey() {
      return actionKey;
    }

    /**
     * Returns the combined digest of the action's inputs and outputs.
     *
     * This may compresses the data into a more compact representation, and
     * makes the object immutable.
     */
    public Digest getFileDigest() {
      if (digest == null) {
        digest = Digest.fromMetadata(mdMap);
        mdMap = null;
      }
      return digest;
    }

    /**
     * Returns true if this cache entry is corrupted and should be ignored.
     */
    public boolean isCorrupted() {
      return actionKey == null;
    }

    /**
     * @return stored path strings, or null if the corresponding action does not discover inputs.
     */
    public Collection<String> getPaths() {
      return discoversInputs() ? files : ImmutableList.<String>of();
    }

    /**
     * @return whether the corresponding action discovers input files dynamically.
     */
    public boolean discoversInputs() {
      return files != null;
    }

    @Override
    public String toString() {
      StringBuilder builder = new StringBuilder();
      builder.append("      actionKey = ").append(actionKey).append("\n");
      builder.append("      digestKey = ");
      if (digest == null) {
        builder.append(Digest.fromMetadata(mdMap)).append(" (from mdMap)\n");
      } else {
        builder.append(digest).append("\n");
      }

      if (discoversInputs()) {
        List<String> fileInfo = Lists.newArrayListWithCapacity(files.size());
        fileInfo.addAll(files);
        Collections.sort(fileInfo);
        for (String info : fileInfo) {
          builder.append("      ").append(info).append("\n");
        }
      }
      return builder.toString();
    }
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
