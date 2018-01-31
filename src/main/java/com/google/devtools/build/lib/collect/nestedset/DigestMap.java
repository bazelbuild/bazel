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
package com.google.devtools.build.lib.collect.nestedset;

import com.google.common.base.Preconditions;
import com.google.devtools.build.lib.util.Fingerprint;
import java.util.concurrent.locks.StampedLock;

/**
 * Map of key -> [digest bytes].
 *
 * <p>This class uses a single array of keys and a big single block of bytes. To read/store digests
 * we index straight into the byte array. This is more memory-efficient and uses less GC than a
 * corresponding Map<Object, byte[]>.
 *
 * <p>Keys use reference equality.
 */
final class DigestMap {
  private final int digestSize;
  private final StampedLock readWriteLock = new StampedLock();
  private Object[] keys;
  private byte[] bytes;
  private int tableSize;
  private int nextResize;
  private int size;

  DigestMap(int digestSize, int initialSize) {
    Preconditions.checkArgument(
        initialSize > 0 && (initialSize & (initialSize - 1)) == 0,
        "initialSize must be a power of 2 greater than 0");
    this.digestSize = digestSize;
    this.keys = new Object[initialSize];
    this.bytes = new byte[initialSize * digestSize];
    this.tableSize = initialSize;
    this.nextResize = getNextResize(initialSize);
  }

  /** Finds the digest for the corresponding key and adds it to the passed fingerprint. */
  boolean readDigest(Object key, Fingerprint fingerprint) {
    long stamp = readWriteLock.readLock();
    try {
      int index = findKeyReadLocked(key);
      if (index >= 0) {
        fingerprint.addBytes(bytes, index * digestSize, digestSize);
        return true;
      } else {
        return false;
      }
    } finally {
      readWriteLock.unlockRead(stamp);
    }
  }

  // Finds the key in the array. Must be called under read lock.
  private int findKeyReadLocked(Object key) {
    int hash = hash(key);
    int index = hash & (tableSize - 1);
    while (true) {
      Object currentKey = keys[index];
      if (currentKey == key) {
        return index;
      } else if (currentKey == null) {
        return -1;
      }
      index = probe(index, tableSize);
    }
  }

  /**
   * Inserts a digest for the corresponding key, then immediately reads it into another fingerprint.
   *
   * <p>This is equivalent to <code>
   * digestMap.insertDigest(key, digest.digestAndReset(); digestMap.readDigest(key, readTo); </code>
   * but it will be faster.
   *
   * @param key The key to insert.
   * @param digest The fingerprint to insert. This will reset the fingerprint instance.
   * @param readTo A fingerprint to read the just-added fingerprint into.
   */
  void insertAndReadDigest(Object key, Fingerprint digest, Fingerprint readTo) {
    long stamp = readWriteLock.writeLock();
    try {
      int index = insertKeyWriteLocked(key);
      digest.digestAndReset(bytes, index * digestSize, digestSize);
      readTo.addBytes(bytes, index * digestSize, digestSize);
    } finally {
      readWriteLock.unlockWrite(stamp);
    }
  }

  // Inserts a key into the array and returns the index. Must be called under write lock.
  private int insertKeyWriteLocked(Object key) {
    if (size >= nextResize) {
      resizeTableWriteLocked();
    }
    int hash = hash(key);
    int index = hash & (tableSize - 1);
    while (true) {
      Object currentKey = keys[index];
      if (currentKey == null) {
        keys[index] = key;
        ++size;
        return index;
      } else if (currentKey == key) {
        // Key is already present
        return index;
      }
      index = probe(index, tableSize);
    }
  }

  private void resizeTableWriteLocked() {
    int digestSize = this.digestSize;
    int tableSize = this.tableSize;
    Object[] keys = this.keys;
    byte[] bytes = this.bytes;
    int newTableSize = this.tableSize * 2;
    Object[] newKeys = new Object[newTableSize];
    byte[] newBytes = new byte[newTableSize * digestSize];
    for (int i = 0; i < tableSize; ++i) {
      Object key = keys[i];
      if (key != null) {
        int newIndex = firstFreeIndex(newKeys, newTableSize, key);
        newKeys[newIndex] = key;
        System.arraycopy(bytes, i * digestSize, newBytes, newIndex * digestSize, digestSize);
      }
    }
    this.tableSize = newTableSize;
    this.keys = newKeys;
    this.bytes = newBytes;
    this.nextResize = getNextResize(newTableSize);
  }

  private static int firstFreeIndex(Object[] keys, int tableSize, Object key) {
    int hash = hash(key);
    int index = hash & (tableSize - 1);
    while (true) {
      Object currentKey = keys[index];
      if (currentKey == null) {
        return index;
      }
      index = probe(index, tableSize);
    }
  }

  private static int hash(Object key) {
    return smear(System.identityHashCode(key));
  }

  private static int probe(int index, int tableSize) {
    return (index + 1) & (tableSize - 1);
  }

  private static int getNextResize(int newTableSize) {
    // 75% load
    return (newTableSize * 3) / 4;
  }

  /*
   * This method was rewritten in Java from an intermediate step of the Murmur hash function in
   * http://code.google.com/p/smhasher/source/browse/trunk/MurmurHash3.cpp, which contained the
   * following header:
   *
   * MurmurHash3 was written by Austin Appleby, and is placed in the public domain. The author
   * hereby disclaims copyright to this source code.
   */
  private static int smear(int hashCode) {
    return 0x1b873593 * Integer.rotateLeft(hashCode * 0xcc9e2d51, 15);
  }
}
