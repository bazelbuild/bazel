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

import static com.google.common.truth.Truth.assertThat;
import static org.junit.Assert.assertThrows;

import com.google.devtools.build.lib.testutil.TestThread;
import com.google.devtools.build.lib.testutil.TestUtils;
import com.google.testing.junit.testparameterinjector.TestParameter;
import com.google.testing.junit.testparameterinjector.TestParameterInjector;
import java.util.Random;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.Executors;
import java.util.concurrent.atomic.AtomicInteger;
import org.junit.Test;
import org.junit.runner.RunWith;

/** Tests for {@link ReadersOrWritersLock}. */
@RunWith(TestParameterInjector.class)
public class ReadersOrWritersLockTest {

  public enum Group {
    READ {
      @Override
      void lock(ReadersOrWritersLock lock) throws InterruptedException {
        lock.lockReadInterruptibly();
      }

      @Override
      void unlock(ReadersOrWritersLock lock) {
        lock.unlockRead();
      }

      @Override
      Group otherGroup() {
        return WRITE;
      }
    },
    WRITE {
      @Override
      void lock(ReadersOrWritersLock lock) throws InterruptedException {
        lock.lockWriteInterruptibly();
      }

      @Override
      void unlock(ReadersOrWritersLock lock) {
        lock.unlockWrite();
      }

      @Override
      Group otherGroup() {
        return READ;
      }
    };

    abstract void lock(ReadersOrWritersLock lock) throws InterruptedException;

    abstract void unlock(ReadersOrWritersLock lock);

    abstract Group otherGroup();
  }

  @Test
  public void lockAndUnlock_sameGroup_succeeds(@TestParameter Group group) throws Exception {
    var lock = new ReadersOrWritersLock();
    for (int i = 0; i < 10; i++) {
      group.lock(lock);
    }
    for (int i = 0; i < 10; i++) {
      group.unlock(lock);
    }
    // The lock has been fully released: the other group can acquire it.
    group.otherGroup().lock(lock);
    group.otherGroup().unlock(lock);
  }

  @Test
  public void unlock_lockNotHeld_throws(@TestParameter Group group) {
    var lock = new ReadersOrWritersLock();
    assertThrows(IllegalMonitorStateException.class, () -> group.unlock(lock));
  }

  @Test
  public void unlock_lockHeldByOtherGroup_throws(@TestParameter Group group) throws Exception {
    var lock = new ReadersOrWritersLock();
    group.lock(lock);
    assertThrows(IllegalMonitorStateException.class, () -> group.otherGroup().unlock(lock));
    group.unlock(lock);
  }

  @Test
  public void lock_lockHeldByOtherGroup_blocksUntilFullyReleased(@TestParameter Group group)
      throws Exception {
    var otherGroup = group.otherGroup();
    var lock = new ReadersOrWritersLock();
    group.lock(lock);
    // otherGroup acquires block.
    var otherGroupAcquisitions = new AtomicInteger();
    var otherGroupThread =
        new TestThread(
            () -> {
              otherGroup.lock(lock);
              otherGroupAcquisitions.incrementAndGet();
              otherGroup.unlock(lock);
            });
    otherGroupThread.start();
    waitUntilBlocked(otherGroupThread);
    assertThat(otherGroupAcquisitions.get()).isEqualTo(0);
    // group acquires succeed immediately.
    group.lock(lock);
    group.unlock(lock);
    assertThat(otherGroupAcquisitions.get()).isEqualTo(0);
    // otherGroup runs last.
    group.unlock(lock);
    otherGroupThread.joinAndAssertState(TestUtils.WAIT_TIMEOUT_MILLISECONDS);
    assertThat(otherGroupAcquisitions.get()).isEqualTo(1);
    // group can acquire again after otherGroup has released the lock.
    group.lock(lock);
    group.unlock(lock);
  }

  @Test
  public void lock_interruptedWhileBlocked_throwsAndKeepsLockUsable(@TestParameter Group group)
      throws Exception {
    var otherGroup = group.otherGroup();
    var lock = new ReadersOrWritersLock();
    group.lock(lock);

    var interrupted = new CountDownLatch(1);
    var otherGroupThread =
        new TestThread(
            () -> {
              try {
                otherGroup.lock(lock);
              } catch (InterruptedException e) {
                interrupted.countDown();
              }
            });
    otherGroupThread.start();
    waitUntilBlocked(otherGroupThread);
    otherGroupThread.interrupt();
    otherGroupThread.joinAndAssertState(TestUtils.WAIT_TIMEOUT_MILLISECONDS);
    assertThat(interrupted.getCount()).isEqualTo(0);

    // The interrupted acquisition must not have taken or corrupted the lock: group still holds it
    // and after releasing it, otherGroup can acquire it.
    group.unlock(lock);
    otherGroup.lock(lock);
    otherGroup.unlock(lock);
  }

  @Test
  public void lock_interruptedBeforeAcquisition_throwsAndKeepsLockUsable(
      @TestParameter Group group) throws Exception {
    var lock = new ReadersOrWritersLock();

    try {
      Thread.currentThread().interrupt();
      assertThrows(InterruptedException.class, () -> group.lock(lock));
    } finally {
      // Avoid leaking the interrupt into other tests if the assertion fails.
      Thread.interrupted();
    }

    // The interrupted acquisition must not have taken or corrupted the lock.
    group.otherGroup().lock(lock);
    group.otherGroup().unlock(lock);
  }

  @Test
  public void lockAndUnlock_concurrentGroups_neverOverlap() {
    var lock = new ReadersOrWritersLock();
    var writers = new AtomicInteger();
    var readers = new AtomicInteger();
    var violations = new AtomicInteger();

    try (var executor = Executors.newVirtualThreadPerTaskExecutor()) {
      for (int i = 0; i < 16; i++) {
        int seed = i;
        executor.execute(
            () -> {
              try {
                var random = new Random(seed);
                for (int j = 0; j < 20_000; j++) {
                  if (random.nextInt(4) == 0) {
                    lock.lockWriteInterruptibly();
                    try {
                      writers.incrementAndGet();
                      if (readers.get() != 0) {
                        violations.incrementAndGet();
                      }
                      writers.decrementAndGet();
                    } finally {
                      lock.unlockWrite();
                    }
                  } else {
                    lock.lockReadInterruptibly();
                    try {
                      readers.incrementAndGet();
                      if (writers.get() != 0) {
                        violations.incrementAndGet();
                      }
                      readers.decrementAndGet();
                    } finally {
                      lock.unlockRead();
                    }
                  }
                }
              } catch (InterruptedException e) {
                // Interruptions are not expected.
                violations.incrementAndGet();
              }
            });
      }
    }

    assertThat(violations.get()).isEqualTo(0);
    assertThat(readers.get()).isEqualTo(0);
    assertThat(writers.get()).isEqualTo(0);
  }

  private static void waitUntilBlocked(Thread thread) {
    Thread.State state;
    while ((state = thread.getState()) != Thread.State.WAITING) {
      assertThat(state).isNotEqualTo(Thread.State.TERMINATED);
      Thread.yield();
    }
  }
}
