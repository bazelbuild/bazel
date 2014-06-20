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

import com.google.common.base.Preconditions;
import com.google.common.collect.Lists;
import com.google.common.collect.Maps;
import com.google.devtools.build.lib.concurrent.ThreadSafety.ThreadSafe;
import com.google.devtools.build.lib.vfs.PathFragment;

import java.io.PrintStream;
import java.util.Collection;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * An implementation of the ActionCache interface that saves cached actions
 * in memory; this is good as long as the build system is running, but not
 * persistent across executions.
 */
@ThreadSafe
public class InMemoryActionCache implements ActionCache {

  private final Map<String, Entry> actionCache = new HashMap<>();

  /**
   * Simple cache entry implementation.
   */
  public static class InMemoryEntry implements Entry {

    private final String actionKey;
    private final List<String> files;
    private boolean isPacked; // true iff ensurePacked() has been called.
    private Map<String, Metadata> mdMap;
    private Digest digest;

    public InMemoryEntry(String key) {
      actionKey = key;
      files = Lists.newArrayList();
      isPacked = false;
      mdMap = Maps.newHashMap();
    }

    @Override
    public void addFile(PathFragment relativePath, Metadata md) {
      Preconditions.checkState(!isPacked);
      files.add(relativePath.getPathString());
      mdMap.put(relativePath.getPathString(), md);
    }

    @Override
    public void ensurePacked() {
      getFileDigest();
      mdMap = null;
      isPacked = true;
    }

    @Override
    public String getActionKey() {
      return actionKey;
    }

    @Override
    public Digest getFileDigest() {
      if (digest == null) {
        digest = Digest.fromMetadata(mdMap);
      }
      return digest;
    }

    @Override
    public Collection<String> getPaths() { return files; }

    @Override
    public boolean isCorrupted() { return false; }
  }

  @Override
  public synchronized void put(String key, Entry entry) {
    actionCache.put(key, entry);
  }

  @Override
  public synchronized Entry get(String key) {
    return actionCache.get(key);
  }

  @Override
  public synchronized void remove(String key) {
    actionCache.remove(key);
  }

  @Override
  public Entry createEntry(String key) {
    return new InMemoryEntry(key);
  }

  public synchronized void reset() {
    actionCache.clear();
  }

  @Override
  public long save() {
    // safe to ignore
    return 0;
  }

  @Override
  public void dump(PrintStream out) {
    out.println("In-memory action cache has " + actionCache.size() + " records");
  }
}
