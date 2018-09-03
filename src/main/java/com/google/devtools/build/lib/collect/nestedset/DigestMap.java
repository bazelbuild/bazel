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
import com.google.devtools.build.lib.vfs.DigestHashFunction;
import com.google.devtools.build.lib.vfs.DigestHashFunction.DigestLength;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicReferenceArray;
import java.util.concurrent.locks.StampedLock;

/**
 * Map of key -> [digest bytes].
 *
 * <p>This class uses a single array of keys and a big single block of bytes. To read/store digests
 * we index straight into the byte array. This is more memory-efficient and uses less GC than a
 * corresponding Map<Object, byte[]>.
 *
 * <p>Keys use reference equality.
 *
 * <p>Reading is lock free. During writes a read lock is taken. If we need to resize the table, a
 * write lock is taken to flush all the readers and writers before the table is resized.
 */
final class DigestMap {
  private static final Object INSERTION_IN_PROGRESS = new Object();
  private final DigestLength digestLength;
  private final StampedLock readWriteLock = new StampedLock();

  static class Table {
    final int tableSize;
    final int nextResize;
    final AtomicReferenceArray<Object> keys;
    final byte[] bytes;

    Table(int tableSize, int digestLength) {
      this.tableSize = tableSize;
      this.nextResize = getNextResize(tableSize);
      this.keys = new AtomicReferenceArray<>(tableSize);
      this.bytes = new byte[tableSize * digestLength];
    }
  }

  private volatile Table table;
  private final AtomicInteger allocatedSlots;

  DigestMap(DigestHashFunction digestHashFunction, int initialSize) {
    Preconditions.checkArgument(
        initialSize > 0 && (initialSize & (initialSize - 1)) == 0,
        "initialSize must be a power of 2 greater than 0");
    this.digestLength = digestHashFunction.getDigestLength();
    this.table = new Table(initialSize, digestLength.getDigestMaximumLength());
    this.allocatedSlots = new AtomicInteger();
  }

  /** Finds the digest for the corresponding key and adds it to the passed fingerprint. */
  boolean readDigest(Object key, Fingerprint fingerprint) {
    Table table = this.table; // Read once for duration of method
    int index = findKey(table, key);
    if (index >= 0) {
      int offset = index * this.digestLength.getDigestMaximumLength();
      int digestLength = this.digestLength.getDigestLength(table.bytes, offset);
      fingerprint.addBytes(table.bytes, offset, digestLength);
      return true;
    }
    return false;
  }

  private static int findKey(Table table, Object key) {
    int hash = hash(key);
    int index = hash & (table.tableSize - 1);
    while (true) {
      Object currentKey = table.keys.get(index);
      if (currentKey == key) {
        return index;
      } else if (currentKey == null) {
        return -1;
      }
      index = probe(index, table.tableSize);
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
    // Check if we have to resize the table first and do that under write lock
    // We assume that we are going to insert an item. If we do not do this, multiple
    // threads could race and all think they do not need to resize, then some get stuck
    // trying to insert the item.
    Table table = this.table;
    if (allocatedSlots.incrementAndGet() >= table.nextResize) {
      long resizeLock = readWriteLock.writeLock();
      try {
        // Guard against race to make sure only one thread resizes
        if (table == this.table) {
          resizeTableWriteLocked();
        }
      } finally {
        readWriteLock.unlockWrite(resizeLock);
      }
    }
    final int index;
    long stamp = readWriteLock.readLock();
    try {
      table = this.table; // Grab the table again under read lock
      index = insertKey(table, key, digest);
    } finally {
      readWriteLock.unlockRead(stamp);
    }
    // This can be done outside of the read lock since the slot is immutable once inserted
    int offset = index * this.digestLength.getDigestMaximumLength();
    int digestLength = this.digestLength.getDigestLength(table.bytes, offset);
    readTo.addBytes(table.bytes, offset, digestLength);
  }

  // Inserts a key into the passed table and returns the index.
  @SuppressWarnings("ThreadPriorityCheck") // We're not relying on thread scheduler for correctness
  private int insertKey(Table table, Object key, Fingerprint digest) {
    int hash = hash(key);
    int index = hash & (table.tableSize - 1);
    while (true) {
      Object currentKey = table.keys.get(index);
      if (currentKey == null) {
        if (!table.keys.compareAndSet(index, null, INSERTION_IN_PROGRESS)) {
          // We raced to insert a key in a free slot, retry this slot in case it's the same key.
          // Failure to do so could lead to a double insertion.
          continue;
        }
        digest.digestAndReset(
            table.bytes,
            index * digestLength.getDigestMaximumLength(),
            digestLength.getDigestMaximumLength());
        table.keys.set(index, key);
        return index;
      } else if (currentKey == key) {
        // Key is already present, give back the slot allocation
        allocatedSlots.decrementAndGet();
        return index;
      } else if (currentKey == INSERTION_IN_PROGRESS) {
        // We are in the progress of inserting an item in this slot, but we don't yet know
        // what the item is. Since it could be an insertion of ourselves we need to wait
        // until done to avoid double insertion. We yield the thread in case the other
        // thread is stuck between insertion and completion.
        Thread.yield();
        continue;
      }
      index = probe(index, table.tableSize);
    }
  }

  private void resizeTableWriteLocked() {
    int digestSize = this.digestLength.getDigestMaximumLength();
    Table oldTable = this.table;
    Table newTable = new Table(oldTable.tableSize * 2, digestSize);
    for (int i = 0; i < oldTable.tableSize; ++i) {
      Object key = oldTable.keys.get(i);
      if (key != null) {
        int newIndex = firstFreeIndex(newTable.keys, newTable.tableSize, key);
        newTable.keys.set(newIndex, key);
        System.arraycopy(
            oldTable.bytes, i * digestSize, newTable.bytes, newIndex * digestSize, digestSize);
      }
    }
    this.table = newTable;
  }

  private static int firstFreeIndex(AtomicReferenceArray<Object> keys, int tableSize, Object key) {
    int hash = hash(key);
    int index = hash & (tableSize - 1);
    while (true) {
      Object currentKey = keys.get(index);
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
