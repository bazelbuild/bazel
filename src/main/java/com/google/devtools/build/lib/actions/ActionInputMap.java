// Copyright 2018 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.actions;

import static java.util.stream.Collectors.toList;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.MoreObjects;
import com.google.common.base.Objects;
import com.google.common.base.Preconditions;
import java.util.Arrays;
import javax.annotation.Nullable;

/**
 * Helper for {@link MetadataProvider} implementations.
 *
 * <p>Allows {@link FileArtifactValue} lookups by exec path or {@link ActionInput}. <i>Also</i>
 * allows {@link ActionInput} to be looked up by exec path.
 *
 * <p>This class implements a closed hash-map with the "links" of each bucket's linked list being
 * stored in a flat array to avoid memory allocations and garbage collection.
 *
 * <p>This class is thread-compatible.
 */
public final class ActionInputMap implements MetadataProvider, ActionInputMapSink {
  /** The number of elements contained in this map. */
  int size;

  /**
   * The hash buckets. Values are indexes into the four arrays. The number of buckets is always the
   * smallest power of 2 that is larger than the number of elements.
   */
  private int[] table;

  /** Flat array of the next pointers that make up the linked list behind each hash bucket. */
  private int[] next;

  /**
   * The {@link ActionInput} keys stored in this map. For performance reasons, they need to be
   * stored as {@link Object}s as otherwise, the JVM does not seem to be as good optimizing the
   * store operations (maybe it does checks on the type being stored?).
   */
  private Object[] keys;

  /**
   * Extra storage for the execPathStrings of the values in {@link #keys}. This extra storage is
   * necessary for speed as otherwise, we'd need to cast to {@link ActionInput}, which is slow.
   */
  private Object[] paths;

  /**
   * The {@link FileArtifactValue} data stored in this map. Same as the other arrays, this is stored
   * as {@link Object} for performance reasons.
   */
  private Object[] values;

  public ActionInputMap(int sizeHint) {
    sizeHint = Math.max(1, sizeHint);
    int tableSize = Integer.highestOneBit(sizeHint) << 1;
    size = 0;

    table = new int[tableSize];
    Arrays.fill(table, -1);

    next = new int[sizeHint];
    keys = new Object[sizeHint];
    paths = new Object[sizeHint];
    values = new Object[sizeHint];
  }

  private int getIndex(String execPathString) {
    int hashCode = execPathString.hashCode();
    int index = hashCode & (table.length - 1);
    if (table[index] == -1) {
      return -1;
    }
    index = table[index];
    while (index != -1) {
      if (hashCode == paths[index].hashCode() && execPathString.equals(paths[index])) {
        return index;
      }
      index = next[index];
    }
    return -1;
  }

  @Nullable
  @Override
  public FileArtifactValue getMetadata(ActionInput input) {
    return getMetadata(input.getExecPathString());
  }

  @Nullable
  public FileArtifactValue getMetadata(String execPathString) {
    int index = getIndex(execPathString);
    return index == -1 ? null : (FileArtifactValue) values[index];
  }

  @Nullable
  @Override
  public ActionInput getInput(String execPathString) {
    int index = getIndex(execPathString);
    return index == -1 ? null : (ActionInput) keys[index];
  }

  /** Count of contained entries. */
  public int size() {
    return size;
  }

  @Override
  public boolean put(ActionInput input, FileArtifactValue metadata, @Nullable Artifact depOwner) {
    return putWithNoDepOwner(input, metadata);
  }

  public boolean putWithNoDepOwner(ActionInput input, FileArtifactValue metadata) {
    Preconditions.checkNotNull(input);
    if (size >= keys.length) {
      resize();
    }
    String path = input.getExecPathString();
    int hashCode = path.hashCode();
    int index = hashCode & (table.length - 1);
    int nextIndex = table[index];
    if (nextIndex == -1) {
      table[index] = size;
    } else {
      do {
        index = nextIndex;
        if (hashCode == paths[index].hashCode() && Objects.equal(path, paths[index])) {
          return false;
        }
        nextIndex = next[index];
      } while (nextIndex != -1);
      next[index] = size;
    }
    next[size] = -1;
    keys[size] = input;
    paths[size] = input.getExecPathString();
    values[size] = metadata;
    size++;
    return true;
  }

  @VisibleForTesting
  void clear() {
    Arrays.fill(table, -1);
    Arrays.fill(next, -1);
    Arrays.fill(keys, null);
    Arrays.fill(paths, null);
    Arrays.fill(values, null);
    size = 0;
  }

  private void resize() {
    // Resize the data containers.
    keys = Arrays.copyOf(keys, size * 2);
    paths = Arrays.copyOf(paths, size * 2);
    values = Arrays.copyOf(values, size * 2);
    next = Arrays.copyOf(next, size * 2);

    // Resize and recreate the table and links if necessary. We can take shortcuts here as we know
    // there are no duplicate keys.
    if (table.length < size * 2) {
      table = new int[table.length * 2];
      next = new int[size * 2];
      Arrays.fill(table, -1);
      for (int i = 0; i < size; i++) {
        int index = paths[i].hashCode() & (table.length - 1);
        next[i] = table[index];
        table[index] = i;
      }
    }
  }

  @Override
  public String toString() {
    return MoreObjects.toStringHelper(this)
        .add("size", size())
        .add("first-fifty-keys", Arrays.stream(keys).limit(50).collect(toList()))
        .add("first-fifty-values", Arrays.stream(values).limit(50).collect(toList()))
        .add("first-fifty-paths", Arrays.stream(paths).limit(50).collect(toList()))
        .toString();
  }
}
