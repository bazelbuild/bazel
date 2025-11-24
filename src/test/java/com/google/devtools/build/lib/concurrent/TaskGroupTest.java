// Copyright 2025 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.concurrent;

import static com.google.common.truth.Truth.assertThat;
import static org.junit.Assert.assertThrows;

import com.google.devtools.build.lib.concurrent.TaskGroup.Joiners;
import com.google.devtools.build.lib.concurrent.TaskGroup.Policies;
import com.google.devtools.build.lib.concurrent.TaskGroup.Subtask;
import java.util.NoSuchElementException;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicReference;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

@RunWith(JUnit4.class)
public final class TaskGroupTest {

  @Test
  public void allSuccessful_waitsForAllSubtasks() throws Exception {
    var group = TaskGroup.open(Policies.allSuccessful(), Joiners.voidOrThrow());
    try (group) {
      var subtask1 =
          group.fork(
              () -> {
                Thread.sleep(100);
                return 1;
              });
      var subtask2 =
          group.fork(
              () -> {
                Thread.sleep(200);
                return 2;
              });

      group.join();

      assertThat(subtask1.state()).isEqualTo(TaskGroup.Subtask.State.SUCCESS);
      assertThat(subtask1.get()).isEqualTo(1);
      assertThat(subtask2.state()).isEqualTo(TaskGroup.Subtask.State.SUCCESS);
      assertThat(subtask2.get()).isEqualTo(2);
    }
    assertThat(group.isCancelled()).isFalse();
  }

  @Test
  public void allSuccessful_anySubtaskFails_setErrorBeforeCanceling() throws Throwable {
    var subtask2Ready = new CountDownLatch(1);
    var letSubtask1Fail = new CountDownLatch(1);
    var latch = new CountDownLatch(1);
    var joiner = new Joiners.VoidOrThrow<>();
    AtomicReference<Throwable> assertErrorRef = new AtomicReference<>(null);
    var policy =
        new TaskGroup.Policy<Object>() {
          @Override
          public boolean onComplete(Subtask<? extends Object> subtask) {
            if (subtask.state() == Subtask.State.FAILED) {
              // Assert that the joiner has the error from subtask1 before we decide to cancel the
              // group.
              try {
                assertThat(joiner.getError()).isInstanceOf(RuntimeException.class);
                assertThat(joiner.getError()).hasMessageThat().isEqualTo("test");
              } catch (Throwable e2) {
                assertErrorRef.set(e2);
              }
              return true;
            }
            return false;
          }
        };
    try (var group = TaskGroup.open(policy, joiner)) {
      group.fork(
          () -> {
            letSubtask1Fail.await();
            throw new RuntimeException("test");
          });
      group.fork(
          () -> {
            subtask2Ready.countDown();
            latch.await();
            return 2;
          });

      subtask2Ready.await();
      letSubtask1Fail.countDown();

      assertThrows(ExecutionException.class, () -> group.join());
    }
    var assertError = assertErrorRef.get();
    if (assertError != null) {
      throw assertError;
    }
  }

  @Test
  public void allSuccessful_anySubtaskFails_cancelsOthersAndThrows() throws Exception {
    var subtask2Ready = new CountDownLatch(1);
    var letSubtask1Fail = new CountDownLatch(1);
    var latch = new CountDownLatch(1);
    try (var group = TaskGroup.open(Policies.allSuccessful(), Joiners.voidOrThrow())) {
      var subtask1 =
          group.fork(
              () -> {
                letSubtask1Fail.await();
                throw new RuntimeException("test");
              });
      var subtask2 =
          group.fork(
              () -> {
                subtask2Ready.countDown();
                latch.await();
                return 2;
              });

      subtask2Ready.await();
      letSubtask1Fail.countDown();
      var e = assertThrows(ExecutionException.class, () -> group.join());

      assertThat(group.isCancelled()).isTrue();
      assertThat(subtask1.state()).isEqualTo(TaskGroup.Subtask.State.FAILED);
      assertThat(subtask1.exception()).isInstanceOf(RuntimeException.class);
      assertThat(subtask1.exception()).hasMessageThat().isEqualTo("test");
      assertThat(subtask2.state()).isEqualTo(TaskGroup.Subtask.State.FAILED);
      assertThat(subtask2.exception()).isInstanceOf(InterruptedException.class);
      assertThat(e).hasCauseThat().isInstanceOf(RuntimeException.class);
      assertThat(e).hasCauseThat().hasMessageThat().isEqualTo("test");
    }
  }

  @Test
  public void allSuccessful_interrupted_cancelsRunningSubtasks() throws Exception {
    var latch = new CountDownLatch(1);
    var subtask1Done = new CountDownLatch(1);
    var subtask2Ready = new CountDownLatch(1);
    var subtask3Ready = new CountDownLatch(1);
    AtomicReference<TaskGroup<Object, Void>> groupRef = new AtomicReference<>(null);
    AtomicReference<Subtask<Integer>> subtask1Ref = new AtomicReference<>(null);
    AtomicReference<Subtask<Integer>> subtask2Ref = new AtomicReference<>(null);
    AtomicReference<Subtask<Integer>> subtask3Ref = new AtomicReference<>(null);
    AtomicBoolean interrupted = new AtomicBoolean(false);
    AtomicReference<Throwable> errorRef = new AtomicReference<>(null);
    var thread =
        Thread.ofPlatform()
            .start(
                () -> {
                  var group = TaskGroup.open(Policies.allSuccessful(), Joiners.voidOrThrow());
                  groupRef.set(group);
                  try (group) {
                    var subtask1 =
                        group.fork(
                            () -> {
                              subtask1Done.countDown();
                              return 1;
                            });
                    subtask1Ref.set(subtask1);
                    var subtask2 =
                        group.fork(
                            () -> {
                              subtask2Ready.countDown();
                              latch.await();
                              return 2;
                            });
                    subtask2Ref.set(subtask2);
                    var subtask3 =
                        group.fork(
                            () -> {
                              subtask3Ready.countDown();
                              latch.await();
                              return 3;
                            });
                    subtask3Ref.set(subtask3);

                    try {
                      group.join();
                    } catch (InterruptedException e) {
                      interrupted.set(true);
                    }
                  } catch (Throwable e) {
                    errorRef.set(e);
                  }
                });

    subtask1Done.await();
    subtask2Ready.await();
    subtask3Ready.await();
    thread.interrupt();
    thread.join();

    assertThat(interrupted.get()).isTrue();
    assertThat(groupRef.get().isCancelled()).isTrue();
    assertThat(subtask1Ref.get().state()).isEqualTo(Subtask.State.SUCCESS);
    assertThat(subtask1Ref.get().get()).isEqualTo(1);
    assertThat(subtask2Ref.get().state()).isEqualTo(Subtask.State.FAILED);
    assertThat(subtask2Ref.get().exception()).isInstanceOf(InterruptedException.class);
    assertThat(subtask3Ref.get().state()).isEqualTo(Subtask.State.FAILED);
    assertThat(subtask3Ref.get().exception()).isInstanceOf(InterruptedException.class);
    assertThat(errorRef.get()).isNull();
  }

  @Test
  public void anySuccessful_returnsFirstSuccessfulAndCancelsOthers() throws Exception {
    var latch = new CountDownLatch(1);
    try (var group = TaskGroup.open(Policies.anySuccessful(), Joiners.anySuccessfulOrThrow())) {
      var subtask1 =
          group.fork(
              () -> {
                Thread.sleep(100);
                return 1;
              });
      var subtask2 =
          group.fork(
              () -> {
                latch.await();
                return 2;
              });
      var subtask3 =
          group.fork(
              () -> {
                latch.await();
                return 3;
              });

      var result = group.join();

      assertThat(group.isCancelled()).isTrue();
      assertThat(result).isEqualTo(1);
      assertThat(subtask1.state()).isEqualTo(TaskGroup.Subtask.State.SUCCESS);
      assertThat(subtask1.get()).isEqualTo(1);
      assertThat(subtask2.state()).isEqualTo(TaskGroup.Subtask.State.FAILED);
      assertThat(subtask2.exception()).isInstanceOf(InterruptedException.class);
      assertThat(subtask3.state()).isEqualTo(TaskGroup.Subtask.State.FAILED);
      assertThat(subtask3.exception()).isInstanceOf(InterruptedException.class);
    }
  }

  @Test
  public void anySuccessful_allSubtaskFails_throws() {
    try (var group = TaskGroup.open(Policies.anySuccessful(), Joiners.voidOrThrow())) {
      var subtask1 =
          group.fork(
              () -> {
                Thread.sleep(100);
                throw new RuntimeException("test1");
              });
      var subtask2 =
          group.fork(
              () -> {
                Thread.sleep(200);
                throw new RuntimeException("test2");
              });

      var e = assertThrows(ExecutionException.class, () -> group.join());

      assertThat(group.isCancelled()).isFalse();
      assertThat(subtask1.state()).isEqualTo(TaskGroup.Subtask.State.FAILED);
      assertThat(subtask1.exception()).isInstanceOf(RuntimeException.class);
      assertThat(subtask1.exception()).hasMessageThat().isEqualTo("test1");
      assertThat(subtask2.state()).isEqualTo(TaskGroup.Subtask.State.FAILED);
      assertThat(subtask2.exception()).isInstanceOf(RuntimeException.class);
      assertThat(subtask2.exception()).hasMessageThat().isEqualTo("test2");
      assertThat(e).hasCauseThat().isInstanceOf(RuntimeException.class);
      assertThat(e).hasCauseThat().hasMessageThat().contains("test");
    }
  }

  @Test
  public void anySuccessfulOrThrow_notForked_throws() {
    try (var group = TaskGroup.open(Policies.anySuccessful(), Joiners.anySuccessfulOrThrow())) {
      var e = assertThrows(ExecutionException.class, () -> group.join());
      assertThat(e).hasCauseThat().isInstanceOf(NoSuchElementException.class);
      assertThat(e).hasCauseThat().hasMessageThat().isEqualTo("No subtasks completed");
    }
  }

  @Test
  public void fork_afterJoined_throws() throws Exception {
    try (var group = TaskGroup.open(Policies.allSuccessful(), Joiners.voidOrThrow())) {
      group.join();
      var e = assertThrows(IllegalStateException.class, () -> group.fork(() -> {}));
      assertThat(e).hasMessageThat().contains("Already joined or task group is closed");
    }
  }

  @Test
  public void fork_fromDifferentThread_throws() throws Exception {
    try (var group = TaskGroup.open(Policies.allSuccessful(), Joiners.voidOrThrow())) {
      AtomicReference<Throwable> errorRef = new AtomicReference<>(null);
      var thread =
          Thread.ofPlatform()
              .start(
                  () -> {
                    try {
                      group.fork(() -> {});
                    } catch (Throwable e) {
                      errorRef.set(e);
                    }
                  });
      thread.join();
      var e = errorRef.get();
      assertThat(e).isNotNull();
      assertThat(e).hasMessageThat().contains("Current thread not owner");
    }
  }

  @Test
  public void join_afterJoined_throws() throws Exception {
    try (var group = TaskGroup.open(Policies.allSuccessful(), Joiners.voidOrThrow())) {
      group.join();
      var e = assertThrows(IllegalStateException.class, () -> group.join());
      assertThat(e).hasMessageThat().contains("Already joined or task group is closed");
    }
  }

  @Test
  public void join_fromDifferentThread_throws() throws Exception {
    try (var group = TaskGroup.open(Policies.allSuccessful(), Joiners.voidOrThrow())) {
      AtomicReference<Throwable> errorRef = new AtomicReference<>(null);
      var thread =
          Thread.ofPlatform()
              .start(
                  () -> {
                    try {
                      group.join();
                    } catch (Throwable e) {
                      errorRef.set(e);
                    }
                  });
      thread.join();
      var e = errorRef.get();
      assertThat(e).isNotNull();
      assertThat(e).hasMessageThat().contains("Current thread not owner");
    }
  }

  @Test
  public void close_notForkedAndNotJoined_doesNotThrow() {
    try (var group = TaskGroup.open(Policies.allSuccessful(), Joiners.voidOrThrow())) {}
  }

  @Test
  public void close_forkedButNotJoined_throws() {
    var e =
        assertThrows(
            IllegalStateException.class,
            () -> {
              try (var group = TaskGroup.open(Policies.allSuccessful(), Joiners.voidOrThrow())) {
                group.fork(
                    () -> {
                      Thread.sleep(1);
                      return 1;
                    });
              }
            });
    assertThat(e).hasMessageThat().contains("Owner did not join after forking");
  }

  @Test
  public void afterSubtaskCompleted_removesThreadFromSet() throws Exception {
    var subtask1Ready = new CountDownLatch(1);
    var letSubtask1Complete = new CountDownLatch(1);
    var subtask2Ready = new CountDownLatch(1);
    var letSubtask2Complete = new CountDownLatch(1);
    try (var group = TaskGroup.open(Policies.allSuccessful(), Joiners.voidOrThrow())) {
      group.fork(
          () -> {
            subtask1Ready.countDown();
            letSubtask1Complete.await();
            return 1;
          });

      subtask1Ready.await();
      assertThat(group.getThreads()).hasSize(1);

      group.fork(
          () -> {
            subtask2Ready.countDown();
            letSubtask2Complete.await();
            return 2;
          });
      subtask2Ready.await();
      assertThat(group.getThreads()).hasSize(2);

      letSubtask1Complete.countDown();
      letSubtask2Complete.countDown();

      group.join();

      assertThat(group.getThreads()).hasSize(0);
    }
  }

  @Test
  public void joinOrThrow_checkedException_throwsIt() throws Exception {
    try (var group = TaskGroup.open(Policies.allSuccessful(), Joiners.voidOrThrow())) {
      group.fork(
          () -> {
            throw new Exception("test");
          });
      var e = assertThrows(Exception.class, () -> group.joinOrThrow(Exception.class));
      assertThat(e).hasMessageThat().isEqualTo("test");
    }
  }

  @Test
  public void joinOrThrow_runtimeException_throwsIt() throws Exception {
    try (var group = TaskGroup.open(Policies.allSuccessful(), Joiners.voidOrThrow())) {
      group.fork(
          () -> {
            throw new RuntimeException("test");
          });
      var e = assertThrows(RuntimeException.class, () -> group.joinOrThrow(Exception.class));
      assertThat(e).hasMessageThat().isEqualTo("test");
    }
  }

  @Test
  public void joinOrThrow_unexpectedCheckedException_throwsIllegalStateException()
      throws Exception {
    class MyException1 extends Exception {}
    class MyException2 extends Exception {}

    try (var group = TaskGroup.open(Policies.allSuccessful(), Joiners.voidOrThrow())) {
      group.fork(
          () -> {
            throw new MyException1();
          });
      var e =
          assertThrows(IllegalStateException.class, () -> group.joinOrThrow(MyException2.class));
      assertThat(e).hasCauseThat().isInstanceOf(MyException1.class);
    }
  }
}
