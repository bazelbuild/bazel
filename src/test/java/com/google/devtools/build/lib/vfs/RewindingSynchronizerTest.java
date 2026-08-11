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

package com.google.devtools.build.lib.vfs;

import static com.google.common.truth.Truth.assertThat;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.fail;

import com.google.devtools.build.lib.profiler.SilentCloseable;
import com.google.devtools.build.lib.testutil.TestThread;
import com.google.devtools.build.lib.vfs.RewindingSynchronizer.TransferableWriteLock;
import java.util.concurrent.atomic.AtomicBoolean;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link RewindingSynchronizer}. */
@RunWith(JUnit4.class)
public final class RewindingSynchronizerTest {
  private static final long TIMEOUT_MILLIS = 10_000;
  private static final Object KEY = "producer";
  private static final Object OTHER_KEY = "other producer";

  private final RewindingSynchronizer synchronizer = new RewindingSynchronizer();

  @Test
  public void acquireReadLock_replacementsDisabled_acquiresNothing() throws Exception {
    synchronizer.reset(/* replacementsEnabled= */ false);

    try (SilentCloseable unused =
        synchronizer.acquireReadLock(RewindingSynchronizerTest::failKeyEvaluation)) {
      assertThat(synchronizer.hasBlockingReadLockForTesting(KEY)).isFalse();
    }
  }

  @Test
  public void acquireReadLock_noReplacementYet_holdsSharedLockWithoutEvaluatingKey()
      throws Exception {
    synchronizer.reset(/* replacementsEnabled= */ true);

    try (SilentCloseable unused =
        synchronizer.acquireReadLock(RewindingSynchronizerTest::failKeyEvaluation)) {
      assertThat(synchronizer.hasBlockingReadLockForTesting(KEY)).isTrue();
      assertThat(synchronizer.hasBlockingReadLockForTesting(OTHER_KEY)).isTrue();
    }
    assertThat(synchronizer.hasBlockingReadLockForTesting(KEY)).isFalse();
  }

  @Test
  public void acquireReadLock_afterReplacement_holdsOnlyItsKeysLock() throws Exception {
    switchToPerKeyLocks();

    try (SilentCloseable unused = synchronizer.acquireReadLock(() -> KEY)) {
      assertThat(synchronizer.hasBlockingReadLockForTesting(KEY)).isTrue();
      assertThat(synchronizer.hasBlockingReadLockForTesting(OTHER_KEY)).isFalse();
    }
    assertThat(synchronizer.hasBlockingReadLockForTesting(KEY)).isFalse();
  }

  @Test
  public void acquireReadLock_replacementInProgress_waitsForItsEnd() throws Exception {
    synchronizer.reset(/* replacementsEnabled= */ true);
    synchronizer.markReplacementsPossible();
    TransferableWriteLock writeLock = synchronizer.acquireWriteLock(KEY);
    AtomicBoolean acquired = new AtomicBoolean();
    TestThread reader = readerThread(KEY, acquired);

    reader.start();
    awaitState(reader, Thread.State.WAITING);
    assertThat(acquired.get()).isFalse();
    writeLock.close();
    reader.joinAndAssertState(TIMEOUT_MILLIS);

    assertThat(acquired.get()).isTrue();
  }

  @Test
  public void acquireReadLock_otherKeyReplacementInProgress_doesNotWait() throws Exception {
    synchronizer.reset(/* replacementsEnabled= */ true);
    synchronizer.markReplacementsPossible();

    try (TransferableWriteLock unused = synchronizer.acquireWriteLock(OTHER_KEY)) {
      try (SilentCloseable unusedReadLock = synchronizer.acquireReadLock(() -> KEY)) {
        assertThat(synchronizer.hasBlockingReadLockForTesting(KEY)).isTrue();
      }
    }
  }

  @Test
  public void acquireWriteLock_waitsForSharedReadLock() throws Exception {
    synchronizer.reset(/* replacementsEnabled= */ true);
    synchronizer.markReplacementsPossible();
    SilentCloseable readLock =
        synchronizer.acquireReadLock(RewindingSynchronizerTest::failKeyEvaluation);
    AtomicBoolean acquired = new AtomicBoolean();
    TestThread producer = producerThread(KEY, acquired);

    producer.start();
    awaitState(producer, Thread.State.WAITING);
    assertThat(acquired.get()).isFalse();
    readLock.close();
    producer.joinAndAssertState(TIMEOUT_MILLIS);

    assertThat(acquired.get()).isTrue();
  }

  @Test
  public void acquireWriteLock_waitsForPerKeyReadLock() throws Exception {
    switchToPerKeyLocks();
    SilentCloseable readLock = synchronizer.acquireReadLock(() -> KEY);
    AtomicBoolean acquired = new AtomicBoolean();
    TestThread producer = producerThread(KEY, acquired);

    producer.start();
    awaitState(producer, Thread.State.WAITING);
    assertThat(acquired.get()).isFalse();
    readLock.close();
    producer.joinAndAssertState(TIMEOUT_MILLIS);

    assertThat(acquired.get()).isTrue();
  }

  @Test
  public void acquireReadLock_replacementKeptForRestart_waitsUntilReleased() throws Exception {
    synchronizer.reset(/* replacementsEnabled= */ true);
    synchronizer.markReplacementsPossible();
    TransferableWriteLock writeLock = synchronizer.acquireWriteLock(KEY);
    writeLock.keepLockedForRestart();
    AtomicBoolean acquired = new AtomicBoolean();
    TestThread reader = readerThread(KEY, acquired);

    reader.start();
    awaitState(reader, Thread.State.WAITING);
    assertThat(acquired.get()).isFalse();
    synchronizer.releaseWriteLocksKeptForRestart();
    reader.joinAndAssertState(TIMEOUT_MILLIS);

    assertThat(acquired.get()).isTrue();
  }

  @Test
  public void acquireReadLock_replacementResumedAfterRestart_waitsForItsEnd() throws Exception {
    synchronizer.reset(/* replacementsEnabled= */ true);
    synchronizer.markReplacementsPossible();
    TransferableWriteLock writeLock = synchronizer.acquireWriteLock(KEY);
    writeLock.keepLockedForRestart();
    AtomicBoolean acquired = new AtomicBoolean();
    TestThread reader = readerThread(KEY, acquired);

    reader.start();
    awaitState(reader, Thread.State.WAITING);
    // The restarted evaluation takes over the lock and finishes the replacement.
    TransferableWriteLock resumedWriteLock = synchronizer.acquireWriteLock(KEY);
    assertThat(acquired.get()).isFalse();
    resumedWriteLock.close();
    reader.joinAndAssertState(TIMEOUT_MILLIS);

    assertThat(acquired.get()).isTrue();
  }

  @Test
  public void acquireReadLock_interruptedWhileWaiting_throws() throws Exception {
    synchronizer.reset(/* replacementsEnabled= */ true);
    synchronizer.markReplacementsPossible();

    try (TransferableWriteLock unused = synchronizer.acquireWriteLock(KEY)) {
      TestThread reader =
          new TestThread(
              () ->
                  assertThrows(
                      InterruptedException.class, () -> synchronizer.acquireReadLock(() -> KEY)));
      reader.start();
      awaitState(reader, Thread.State.WAITING);
      reader.interrupt();
      reader.joinAndAssertState(TIMEOUT_MILLIS);
    }
  }

  /** Makes the first replacement, which switches all later consumers to per-key locks. */
  private void switchToPerKeyLocks() throws InterruptedException {
    synchronizer.reset(/* replacementsEnabled= */ true);
    synchronizer.markReplacementsPossible();
    synchronizer.acquireWriteLock(OTHER_KEY).close();
  }

  private TestThread readerThread(Object key, AtomicBoolean acquired) {
    return new TestThread(
        () -> {
          try (SilentCloseable unused = synchronizer.acquireReadLock(() -> key)) {
            acquired.set(true);
          }
        });
  }

  private TestThread producerThread(Object key, AtomicBoolean acquired) {
    return new TestThread(
        () -> {
          try (TransferableWriteLock unused = synchronizer.acquireWriteLock(key)) {
            acquired.set(true);
          }
        });
  }

  private static Object failKeyEvaluation() {
    throw new AssertionError("key evaluated while the shared lock stands in for all keys");
  }

  private static void awaitState(Thread thread, Thread.State state) throws InterruptedException {
    long deadline = System.currentTimeMillis() + TIMEOUT_MILLIS;
    while (thread.getState() != state) {
      if (!thread.isAlive()) {
        fail("thread " + thread.getName() + " ended before reaching " + state);
      }
      if (System.currentTimeMillis() > deadline) {
        fail("thread " + thread.getName() + " did not reach " + state + " within the timeout");
      }
      Thread.sleep(10);
    }
  }
}
