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

/** Tests for {@link ReaderPreferringReadWriteLock}. */
@RunWith(TestParameterInjector.class)
public class ReaderPreferringReadWriteLockTest {

  public enum Group {
    READ {
      @Override
      void lock(ReaderPreferringReadWriteLock lock) throws InterruptedException {
        lock.lockReadInterruptibly();
      }

      @Override
      void unlock(ReaderPreferringReadWriteLock lock) {
        lock.unlockRead();
      }

      @Override
      Group otherGroup() {
        return WRITE;
      }
    },
    WRITE {
      @Override
      void lock(ReaderPreferringReadWriteLock lock) throws InterruptedException {
        lock.lockWriteInterruptibly();
      }

      @Override
      void unlock(ReaderPreferringReadWriteLock lock) {
        lock.unlockWrite();
      }

      @Override
      Group otherGroup() {
        return READ;
      }
    };

    abstract void lock(ReaderPreferringReadWriteLock lock) throws InterruptedException;

    abstract void unlock(ReaderPreferringReadWriteLock lock);

    abstract Group otherGroup();
  }

  @Test
  public void lockRead_multipleReaders_succeeds() throws Exception {
    var lock = new ReaderPreferringReadWriteLock();
    for (int i = 0; i < 10; i++) {
      lock.lockReadInterruptibly();
    }
    for (int i = 0; i < 10; i++) {
      lock.unlockRead();
    }
    // The lock has been fully released: a writer can acquire it.
    lock.lockWriteInterruptibly();
    lock.unlockWrite();
  }

  @Test
  public void unlock_lockNotHeld_throws(@TestParameter Group group) {
    var lock = new ReaderPreferringReadWriteLock();
    assertThrows(IllegalMonitorStateException.class, () -> group.unlock(lock));
  }

  @Test
  public void unlock_lockHeldByOtherGroup_throws(@TestParameter Group group) throws Exception {
    var lock = new ReaderPreferringReadWriteLock();
    group.lock(lock);
    assertThrows(IllegalMonitorStateException.class, () -> group.otherGroup().unlock(lock));
    group.unlock(lock);
  }

  @Test
  public void lock_lockHeldByOtherGroup_blocksUntilReleased(@TestParameter Group group)
      throws Exception {
    var otherGroup = group.otherGroup();
    var lock = new ReaderPreferringReadWriteLock();
    group.lock(lock);
    var otherGroupAcquisitions = new AtomicInteger();
    var otherGroupThread =
        new TestThread(
            () -> {
              otherGroup.lock(lock);
              otherGroupAcquisitions.incrementAndGet();
              otherGroup.unlock(lock);
            });
    otherGroupThread.start();
    waitUntilState(otherGroupThread, Thread.State.WAITING);
    assertThat(otherGroupAcquisitions.get()).isEqualTo(0);
    group.unlock(lock);
    otherGroupThread.joinAndAssertState(TestUtils.WAIT_TIMEOUT_MILLISECONDS);
    assertThat(otherGroupAcquisitions.get()).isEqualTo(1);
    // group can acquire again after otherGroup has released the lock.
    group.lock(lock);
    group.unlock(lock);
  }

  @Test
  public void lockWrite_lockHeldByWriter_blocksUntilReleased() throws Exception {
    var lock = new ReaderPreferringReadWriteLock();
    lock.lockWriteInterruptibly();
    var otherWriterAcquisitions = new AtomicInteger();
    var otherWriter =
        new TestThread(
            () -> {
              lock.lockWriteInterruptibly();
              otherWriterAcquisitions.incrementAndGet();
              lock.unlockWrite();
            });
    otherWriter.start();
    waitUntilState(otherWriter, Thread.State.WAITING);
    assertThat(otherWriterAcquisitions.get()).isEqualTo(0);
    lock.unlockWrite();
    otherWriter.joinAndAssertState(TestUtils.WAIT_TIMEOUT_MILLISECONDS);
    assertThat(otherWriterAcquisitions.get()).isEqualTo(1);
  }

  @Test
  public void lockRead_writerWaiting_isAdmitted() throws Exception {
    var lock = new ReaderPreferringReadWriteLock();
    lock.lockReadInterruptibly();
    var writerAcquisitions = new AtomicInteger();
    var writer =
        new TestThread(
            () -> {
              lock.lockWriteInterruptibly();
              writerAcquisitions.incrementAndGet();
              lock.unlockWrite();
            });
    writer.start();
    waitUntilState(writer, Thread.State.WAITING);
    // A new reader is admitted while the writer waits.
    lock.lockReadInterruptibly();
    assertThat(writerAcquisitions.get()).isEqualTo(0);
    // The writer only acquires the lock once all readers have released it.
    lock.unlockRead();
    assertThat(writerAcquisitions.get()).isEqualTo(0);
    lock.unlockRead();
    writer.joinAndAssertState(TestUtils.WAIT_TIMEOUT_MILLISECONDS);
    assertThat(writerAcquisitions.get()).isEqualTo(1);
    // Both groups can acquire the lock without blocking after the writer has released it.
    lock.lockReadInterruptibly();
    lock.unlockRead();
    lock.lockWriteInterruptibly();
    lock.unlockWrite();
  }

  @Test
  public void unlockRead_newReaderBeforeNotification_writerIsNotStranded() throws Exception {
    var lock = new ReaderPreferringReadWriteLock();
    lock.lockReadInterruptibly();
    var writerAcquisitions = new AtomicInteger();
    var writer =
        new TestThread(
            () -> {
              lock.lockWriteInterruptibly();
              writerAcquisitions.incrementAndGet();
              lock.unlockWrite();
            });
    writer.start();
    waitUntilState(writer, Thread.State.WAITING);

    var releasingReader = new TestThread(lock::unlockRead);
    synchronized (lock) {
      releasingReader.start();
      // Pause the last reader after it releases the lock but before it notifies the writer.
      waitUntilState(releasingReader, Thread.State.BLOCKED);
      lock.lockReadInterruptibly();
    }
    releasingReader.joinAndAssertState(TestUtils.WAIT_TIMEOUT_MILLISECONDS);
    // The notified writer must record itself as waiting again while this new reader holds the lock.
    waitUntilState(writer, Thread.State.WAITING);
    assertThat(writerAcquisitions.get()).isEqualTo(0);
    lock.unlockRead();
    writer.joinAndAssertState(TestUtils.WAIT_TIMEOUT_MILLISECONDS);
    assertThat(writerAcquisitions.get()).isEqualTo(1);
  }

  @Test
  public void lock_interruptedWhileBlocked_throwsAndKeepsLockUsable(@TestParameter Group group)
      throws Exception {
    var otherGroup = group.otherGroup();
    var lock = new ReaderPreferringReadWriteLock();
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
    waitUntilState(otherGroupThread, Thread.State.WAITING);
    otherGroupThread.interrupt();
    otherGroupThread.joinAndAssertState(TestUtils.WAIT_TIMEOUT_MILLISECONDS);
    assertThat(interrupted.getCount()).isEqualTo(0);

    // The interrupted acquisition must not have taken or corrupted the lock: group still holds it
    // and after releasing it, both groups can acquire it.
    group.unlock(lock);
    otherGroup.lock(lock);
    otherGroup.unlock(lock);
    group.lock(lock);
    group.unlock(lock);
  }

  @Test
  @SuppressWarnings("ThreadPriorityCheck")
  public void lockRead_interruptedWriter_doesNotEnterMonitor() throws Exception {
    var lock = new ReaderPreferringReadWriteLock();
    lock.lockReadInterruptibly();
    var writer =
        new TestThread(
            () -> assertThrows(InterruptedException.class, lock::lockWriteInterruptibly));
    writer.start();
    waitUntilState(writer, Thread.State.WAITING);
    writer.interrupt();
    writer.joinAndAssertState(TestUtils.WAIT_TIMEOUT_MILLISECONDS);
    lock.unlockRead();

    // Cancelling the only writer must leave subsequent readers independent of the monitor.
    var reader =
        new TestThread(
            () -> {
              for (int i = 0; i < 3; i++) {
                lock.lockReadInterruptibly();
                lock.unlockRead();
              }
            });
    try {
      synchronized (lock) {
        reader.start();
        Thread.State state;
        while ((state = reader.getState()) != Thread.State.TERMINATED) {
          assertThat(state).isNotEqualTo(Thread.State.BLOCKED);
          Thread.yield();
        }
      }
    } finally {
      reader.joinAndAssertState(TestUtils.WAIT_TIMEOUT_MILLISECONDS);
    }
  }

  @Test
  public void lock_interruptedBeforeAcquisition_throwsAndKeepsLockUsable(@TestParameter Group group)
      throws Exception {
    var lock = new ReaderPreferringReadWriteLock();

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
  public void lockAndUnlock_concurrentThreads_readersAndWritersNeverOverlap() {
    var lock = new ReaderPreferringReadWriteLock();
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
                      // The writer is exclusive: no other writer and no reader may be active.
                      if (writers.incrementAndGet() != 1 || readers.get() != 0) {
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

  @SuppressWarnings("ThreadPriorityCheck")
  private static void waitUntilState(Thread thread, Thread.State expectedState) {
    Thread.State state;
    while ((state = thread.getState()) != expectedState) {
      assertThat(state).isNotEqualTo(Thread.State.TERMINATED);
      Thread.yield();
    }
  }
}
