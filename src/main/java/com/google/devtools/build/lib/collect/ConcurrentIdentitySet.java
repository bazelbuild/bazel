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

package com.google.devtools.build.lib.collect;

import com.google.common.base.Preconditions;
import java.util.Arrays;

/**
 * A set that performs identity-based deduplication.
 *
 * <p>This class is thread-safe except as noted.
 *
 * <p>It optimistically performs lookups without locking and locks only when mutations are needed,
 * possibly causing some operations to be retried.
 *
 * <p>This class uses closed hashing and thus does not need entry wrappers, making it
 * memory-efficient. The memory savings compared to {@link
 * com.google.common.collect.Sets#newConcurrentHashSet} is approximately 32 fewer bytes per entry.
 */
// TODO(b/17553173): Can this (perhaps with value equality) be used more widely to save memory?
public final class ConcurrentIdentitySet {
  private volatile Object[] data;
  private int size = 0;

  /**
   * @param sizeHint how many elements to store without resizing
   */
  public ConcurrentIdentitySet(int sizeHint) {
    int size = 1;
    while (size <= sizeHint) {
      size *= 2;
    }
    this.data = new Object[size * 2];
  }

  /**
   * Tries to add {@code obj} to the set of tracked identities.
   *
   * @return true if {@code obj} was absent and added to the set
   */
  public boolean add(Object obj) {
    Preconditions.checkNotNull(obj);
    int hashCode = System.identityHashCode(obj);
    while (true) {
      Object[] snapshot = data;
      int probe = hash(/* hashCode= */ hashCode, /* length= */ snapshot.length);
      Object probedValue = snapshot[probe];
      while (true) {
        if (probedValue != null) {
          if (probedValue == obj) {
            return false; // Duplicate found.
          }
          if (++probe == snapshot.length) {
            probe = 0;
          }
          probedValue = snapshot[probe];
          continue;
        }
        // probe points to a likely empty slot
        synchronized (this) {
          if (snapshot != data) {
            break; // Another thread updated the snapshot.
          }
          // Re-reads the probed value under lock. It's possible another thread updated it.
          probedValue = snapshot[probe];
          if (probedValue != null) {
            continue;
          }
          snapshot[probe] = obj;
          if (++size * 2 >= snapshot.length) {
            resize();
          }
        }
        return true;
      }
    }
  }

  /** Not thread safe. */
  public void clear() {
    Arrays.fill(data, null);
    size = 0;
  }

  /** Requires synchronized (this). */
  private void resize() {
    Object[] newData = new Object[data.length * 2];
    for (Object obj : data) {
      if (obj == null) {
        continue;
      }
      int probe = hash(/*hashCode=*/ System.identityHashCode(obj), /*length=*/ newData.length);
      while (newData[probe] != null) {
        if (++probe == newData.length) {
          probe = 0;
        }
      }
      // No need to check for equality because all values are unique.
      newData[probe] = obj;
    }
    data = newData;
  }

  /** Copied from {@link java.util.IdentityHashMap}. */
  private static int hash(int hashCode, int length) {
    return ((hashCode << 1) - (hashCode << 8)) & (length - 1);
  }
}
