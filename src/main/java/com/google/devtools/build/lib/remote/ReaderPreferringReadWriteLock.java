// Copyright 2026 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.remote;

import java.lang.invoke.MethodHandles;
import java.lang.invoke.VarHandle;

/**
 * A non-reentrant read-write lock that admits new readers even while a writer is waiting.
 *
 * <p>A reader only ever waits for a writer that holds the lock, never for other readers or for a
 * waiting writer. A writer waits until the lock is free and may thus be starved by a steady stream
 * of readers. Writers exclude each other.
 *
 * <p>This class is optimized for a low memory footprint and for not inflating the object's monitor
 * under contention between readers. The monitor is only entered by writers and by the last reader
 * releasing the lock while a writer is waiting for it. It is thus only inflated while a writer
 * holds or waits for the lock.
 */
final class ReaderPreferringReadWriteLock {
  private static final VarHandle HOLDS;

  static {
    try {
      HOLDS =
          MethodHandles.lookup()
              .findVarHandle(ReaderPreferringReadWriteLock.class, "holds", int.class);
    } catch (ReflectiveOperationException e) {
      throw new ExceptionInInitializerError(e);
    }
  }

  private static final int WRITER = -1;
  private static final int WRITER_WAITING = 1 << 30;
  private static final int READER_COUNT_MASK = WRITER_WAITING - 1;

  // WRITER means that the writer holds the lock. Otherwise, the low bits count the readers holding
  // it and WRITER_WAITING is set while a writer waits for them to release it. The lock is free at
  // zero.
  private volatile int holds;

  void lockReadInterruptibly() throws InterruptedException {
    while (true) {
      if (Thread.interrupted()) {
        throw new InterruptedException();
      }
      int currentHolds = holds;
      if (currentHolds != WRITER) {
        // Readers are admitted even if a writer is waiting.
        if (HOLDS.compareAndSet(this, currentHolds, currentHolds + 1)) {
          return;
        }
      } else {
        synchronized (this) {
          // Rechecked under the monitor so that the notification in unlockWrite can't be missed.
          while (holds == WRITER) {
            wait();
          }
        }
      }
    }
  }

  void unlockRead() {
    int currentHolds;
    int newHolds;
    do {
      currentHolds = holds;
      if (currentHolds == WRITER || (currentHolds & READER_COUNT_MASK) == 0) {
        throw new IllegalMonitorStateException("holds: " + currentHolds);
      }
      // Clear the record of a waiting writer when the last reader releases the lock, even if all
      // waiting writers have been interrupted. Notified writers recheck the state and record
      // themselves again if needed.
      newHolds = currentHolds == WRITER_WAITING + 1 ? 0 : currentHolds - 1;
    } while (!HOLDS.compareAndSet(this, currentHolds, newHolds));
    if (currentHolds == WRITER_WAITING + 1) {
      // This was the last reader and a writer may still be waiting for it.
      synchronized (this) {
        notifyAll();
      }
    }
  }

  void lockWriteInterruptibly() throws InterruptedException {
    synchronized (this) {
      while (true) {
        if (Thread.interrupted()) {
          throw new InterruptedException();
        }
        int currentHolds = holds;
        if (currentHolds == 0) {
          // Barging ahead of another waiting writer is safe since it will be woken up and recheck
          // the state after the next unlock.
          if (HOLDS.compareAndSet(this, currentHolds, WRITER)) {
            return;
          }
        } else if (currentHolds == WRITER || (currentHolds & WRITER_WAITING) != 0) {
          wait();
        } else {
          // Readers hold the lock. Record the waiting writer so that the last reader notifies it.
          HOLDS.compareAndSet(this, currentHolds, currentHolds | WRITER_WAITING);
        }
      }
    }
  }

  void unlockWrite() {
    if (!HOLDS.compareAndSet(this, WRITER, 0)) {
      throw new IllegalMonitorStateException("holds: " + holds);
    }
    synchronized (this) {
      notifyAll();
    }
  }

  @Override
  public String toString() {
    int currentHolds = holds;
    boolean writer = currentHolds == WRITER;
    return "ReaderPreferringReadWriteLock[readers=%d, writer=%b, writerWaiting=%b]"
        .formatted(
            writer ? 0 : currentHolds & READER_COUNT_MASK,
            writer,
            !writer && (currentHolds & WRITER_WAITING) != 0);
  }
}
