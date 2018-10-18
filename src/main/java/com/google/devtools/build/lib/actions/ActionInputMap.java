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
  private ActionInput[] keys;
  private FileArtifactValue[] values;

  /** Number of contained elements. */
  private int size;

  /** Length of keys = 2^(numBits+1) */
  private int numBits;

  /** Mask to use to perform the modulo operation since the table size is a power of 2. */
  private int mask;

  public ActionInputMap(int sizeHint) {
    this.numBits = 1;
    while ((1 << numBits) <= sizeHint) {
      ++numBits;
    }
    this.mask = (1 << numBits) - 1;
    this.keys = new ActionInput[1 << numBits];
    this.values = new FileArtifactValue[1 << numBits];
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
    while (keys[probe] != null) {
      String execPath = keys[probe].getExecPathString();
      if (hashCode == execPath.hashCode() && execPath.equals(execPathString)) {
        return values[probe];
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
    while (keys[probe] != null) {
      String execPath = keys[probe].getExecPathString();
      if (hashCode == execPath.hashCode() && execPath.equals(execPathString)) {
        return keys[probe];
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
    if (size * 2 >= keys.length) {
      resize();
    }
    return putImpl(input, metadata);
  }

  public void clear() {
    Arrays.fill(keys, null);
    Arrays.fill(values, null);
    size = 0;
  }

  private void resize() {
    ActionInput[] oldKeys = keys;
    FileArtifactValue[] oldValues = values;
    keys = new ActionInput[keys.length * 2];
    values = new FileArtifactValue[values.length * 2];
    ++numBits;
    mask = (1 << numBits) - 1;
    for (int i = 0; i < oldKeys.length; i++) {
      ActionInput key = oldKeys[i];
      if (key == null) {
        continue;
      }
      int hashCode = key.getExecPathString().hashCode();
      int probe = getProbe(hashCode);
      while (true) {
        // Only checks for empty slots because all map keys are known to be unique.
        if (keys[probe] == null) {
          keys[probe] = key;
          values[probe] = oldValues[i];
          break;
        }
        probe = incProbe(probe);
      }
    }
  }

  /**
   * Unlike the public version, this doesn't resize.
   *
   * <p>REQUIRES: there are free positions in {@link keys}.
   */
  private boolean putImpl(ActionInput key, FileArtifactValue value) {
    int hashCode = key.getExecPathString().hashCode();
    int probe = getProbe(hashCode);
    while (true) {
      ActionInput next = keys[probe];
      if (next == null) {
        keys[probe] = key;
        values[probe] = value;
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
    return hashCode & mask;
  }

  private int incProbe(int probe) {
    return (probe + 1) & mask;
  }
}
