// Copyright 2015 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.android.ziputils;

import com.google.common.base.Preconditions;
import java.util.Collection;
import java.util.LinkedHashMap;
import java.util.Map;

class Splitter {

  private static final String ARCHIVE_FILE_SEPARATOR = "/";

  private final int numberOfShards;
  private final Map<String, Integer> assigned;
  private int size = 0; // Number of classes in the current shard
  private int shard = 0; // Current shard number
  private String prevPath = null;
  private int remaining; // Number of classes remaining to be assigned shards
  private int idealSize;
  private int almostFull;

  /**
   * Creates a splitter for splitting an expected number of entries into
   * a given number of shards. The current shard is shard 0.
   */
  public Splitter(int numberOfShards, int expectedSize) {
    this.numberOfShards = numberOfShards;
    this.remaining = expectedSize;
    this.assigned = new LinkedHashMap<>();
    idealSize = remaining / (numberOfShards - shard);
    // Before you change this, please do the math.
    // It's not always perfect, but designed to keep shards reasonably balanced in most cases.
    int limit = Math.min(Math.min(10, (idealSize + 3) / 4), (int) Math.log(numberOfShards));
    almostFull = idealSize - limit;
  }

  /**
   * Forces mapping of the given entries to be that of the current shard. The estimated number of
   * remaining entries to process is adjusted, by subtracting the number of as-of-yet unassigned
   * entries from the filter.
   */
  public void assignAllToCurrentShard(Collection<String> entries) {
    if (entries != null) {
      for (String e : entries) {
        if (!assigned.containsKey(e)) {
          remaining--;
        }
        assigned.put(e, shard);
      }
      size += entries.size();
    }
  }

  /** Assigns the given entry to an output file. */
  public int assign(String path) {
    Preconditions.checkState(shard < numberOfShards, "Too many shards!");
    Integer assignment = assigned.get(path);
    if (assignment != null) {
      return assignment;
    }
    remaining--;

    // last shard, no choice
    if (shard == numberOfShards - 1) {
      size++;
      assigned.put(path, shard);
      return shard;
    }

    // Forced split to try to avoid empty shards
    if (remaining < numberOfShards - shard - 1) {
      if (size > 0) {
        nextShard();
      }
      size++;
      assigned.put(path, shard);
      return shard;
    }

    // If current shard is "over-full", forcibly roll over. Otherwise, if current shard is
    // "almost full", check for package boundary. The forced rollover is in particular important
    // when all or almost all classes are in the default package, as Proguard likes to make it.
    if (size >= (idealSize + (idealSize - almostFull))) {
      nextShard();
    } else if (prevPath != null && size >= almostFull) {
      int i = path.lastIndexOf(ARCHIVE_FILE_SEPARATOR);
      String dir = i > 0 ? path.substring(0, i) : ".";
      i = prevPath.lastIndexOf(ARCHIVE_FILE_SEPARATOR);
      String prevDir = i > 0 ? prevPath.substring(0, i) : ".";
      if (!dir.equals(prevDir)) {
        nextShard();
      }
    }
    prevPath = path;
    assigned.put(path, shard);
    size++;
    return shard;
  }

  /**
   * Forces increment of the current shard. May be called externally.
   * Typically right after calling {@link #assign(java.util.Collection)}.
   */
  public void nextShard() {
    if (shard < numberOfShards - 1) {
      shard++;
      size = 0;
      addEntries(0);
    }
  }

  /**
   * Adjusts the number of estimated entries to be process by the given count.
   */
  public void addEntries(int count) {
    this.remaining += count;
    idealSize = numberOfShards > shard ? remaining / (numberOfShards - shard) : remaining;
    // Before you change this, please do the math.
    // It's not always perfect, but designed to keep shards reasonably balanced in most cases.
    int limit = Math.min(Math.min(10, (idealSize + 3) / 4), (int) Math.log(numberOfShards));
    almostFull = idealSize - limit;
  }
}
