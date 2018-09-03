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

import com.google.common.base.Preconditions;
import java.util.Arrays;
import javax.annotation.Nullable;

/**
 * Helper for {@link MetadataProvider} implementations.
 *
 * <p>Allows {@link FileArtifactValue} lookups by exec path or {@link ActionInput}. <i>Also</i>
 * allows {@link ActionInput} to be looked up by exec path.
 *
 * <p>This class is thread-compatible.
 */
public final class ActionInputMap implements MetadataProvider {
  /**
   * {@link ActionInput} keys stored in even indices
   *
   * <p>{@link FileArtifactValue} values stored in odd indices
   */
  private Object[] data;

  /** Number of contained elements. */
  private int size;

  /** Length of data = 2^(numBits+1) */
  private int numBits;

  /** Mask to use to perform the modulo operation since the table size is a power of 2. */
  private int mask;

  public ActionInputMap(int sizeHint) {
    this.numBits = 1;
    while ((1 << numBits) <= sizeHint) {
      ++numBits;
    }
    this.mask = (1 << numBits) - 1;
    this.data = new Object[1 << (numBits + 1)];
    this.size = 0;
  }

  @Nullable
  @Override
  public FileArtifactValue getMetadata(ActionInput input) {
    return getMetadata(input.getExecPathString());
  }

  @Nullable
  public FileArtifactValue getMetadata(String execPathString) {
    int hashCode = execPathString.hashCode();
    int probe = getProbe(hashCode);
    ActionInput nextKey;
    while ((nextKey = (ActionInput) data[probe]) != null) {
      if (hashCode == nextKey.getExecPathString().hashCode()
          && nextKey.getExecPathString().equals(execPathString)) {
        return (FileArtifactValue) data[probe + 1];
      }
      probe = incProbe(probe);
    }
    return null;
  }

  @Nullable
  @Override
  public ActionInput getInput(String execPathString) {
    int hashCode = execPathString.hashCode();
    int probe = getProbe(hashCode);
    ActionInput nextKey;
    while ((nextKey = (ActionInput) data[probe]) != null) {
      if (hashCode == nextKey.getExecPathString().hashCode()
          && nextKey.getExecPathString().equals(execPathString)) {
        return nextKey;
      }
      probe = incProbe(probe);
    }
    return null;
  }

  /** Count of contained entries. */
  public int size() {
    return size;
  }

  /** @return true if an entry was added, false if the map already contains {@code input} */
  public boolean put(ActionInput input, FileArtifactValue metadata) {
    Preconditions.checkNotNull(input);
    if (size * 4 >= data.length) {
      resize();
    }
    return putImpl(input, metadata);
  }

  public void clear() {
    Arrays.fill(data, null);
    size = 0;
  }

  private void resize() {
    Object[] oldData = data;
    data = new Object[data.length * 2];
    ++numBits;
    mask = (1 << numBits) - 1;
    for (int i = 0; i < oldData.length; i += 2) {
      ActionInput key = (ActionInput) oldData[i];
      if (key == null) {
        continue;
      }
      int hashCode = key.getExecPathString().hashCode();
      int probe = getProbe(hashCode);
      while (true) {
        // Only checks for empty slots because all map keys are known to be unique.
        if (data[probe] == null) {
          data[probe] = key;
          data[probe + 1] = oldData[i + 1];
          break;
        }
        probe = incProbe(probe);
      }
    }
  }

  /**
   * Unlike the public version, this doesn't resize.
   *
   * <p>REQUIRES: there are free positions in {@link data}.
   */
  private boolean putImpl(ActionInput key, FileArtifactValue value) {
    int hashCode = key.getExecPathString().hashCode();
    int probe = getProbe(hashCode);
    while (true) {
      ActionInput next = (ActionInput) data[probe];
      if (next == null) {
        data[probe] = key;
        data[probe + 1] = value;
        ++size;
        return true;
      }
      if (hashCode == next.getExecPathString().hashCode()
          && next.getExecPathString().equals(key.getExecPathString())) {
        return false; // already present
      }
      probe = incProbe(probe);
    }
  }

  private int getProbe(int hashCode) {
    return (hashCode & mask) << 1;
  }

  private int incProbe(int probe) {
    probe += 2;
    if (probe >= data.length) {
      probe -= data.length;
    }
    return probe;
  }
}
