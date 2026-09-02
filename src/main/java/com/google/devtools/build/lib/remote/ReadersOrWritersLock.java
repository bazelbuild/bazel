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

/**
 * A lock that admits any number of "readers" or any number of "writers", but never both at the same
 * time.
 *
 * <p>A thread waiting to lock as a member of one group is only ever waiting for the members of the
 * other group to release the lock, never for members of its own group. As long as members of the
 * other group keep acquiring the lock, that thread may be starved.
 *
 * <p>Both groups are reentrant by counting and do not track thread ownership.
 */
final class ReadersOrWritersLock {
  // Positive values count read holds, negative values count write holds; the lock is free at zero.
  private int holds;

  synchronized void lockReadInterruptibly() throws InterruptedException {
    if (Thread.interrupted()) {
      throw new InterruptedException();
    }
    while (holds < 0) {
      wait();
    }
    holds++;
  }

  synchronized void unlockRead() {
    if (holds <= 0) {
      throw new IllegalMonitorStateException();
    }
    if (--holds == 0) {
      notifyAll();
    }
  }

  synchronized void lockWriteInterruptibly() throws InterruptedException {
    if (Thread.interrupted()) {
      throw new InterruptedException();
    }
    while (holds > 0) {
      wait();
    }
    holds--;
  }

  synchronized void unlockWrite() {
    if (holds >= 0) {
      throw new IllegalMonitorStateException();
    }
    if (++holds == 0) {
      notifyAll();
    }
  }

  @Override
  public synchronized String toString() {
    return "ReadersOrWritersLock[readers=%d, writers=%d]"
        .formatted(Math.max(holds, 0), Math.max(-holds, 0));
  }
}
