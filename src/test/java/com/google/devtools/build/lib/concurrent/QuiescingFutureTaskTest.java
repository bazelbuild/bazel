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
package com.google.devtools.build.lib.concurrent;

import static com.google.common.truth.Truth.assertThat;
import static com.google.common.util.concurrent.MoreExecutors.directExecutor;
import static org.junit.Assert.assertThrows;

import java.util.concurrent.ExecutionException;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

@RunWith(JUnit4.class)
public final class QuiescingFutureTaskTest {

  @Test
  public void runOnce() throws Exception {
    AtomicInteger callCount = new AtomicInteger(0);
    var task =
        new QuiescingFutureTask<String>(directExecutor()) {
          @Override
          protected void arrangeSubtasks() {
            callCount.incrementAndGet();
          }

          @Override
          protected String getValue() {
            return "result";
          }
        };

    assertThat(task.isDone()).isFalse();
    task.run();
    assertThat(task.isDone()).isTrue();
    assertThat(task.get()).isEqualTo("result");
    assertThat(callCount.get()).isEqualTo(1);

    // Running again should not call arrangeSubtasks but still result in the same value (already
    // done)
    task.run();
    assertThat(callCount.get()).isEqualTo(1);
  }

  @Test
  public void subtasksCompletion() throws Exception {
    AtomicInteger subtaskCallCount = new AtomicInteger(0);
    var task =
        new QuiescingFutureTask<String>(directExecutor()) {
          @Override
          protected void arrangeSubtasks() {
            increment();
            subtaskCallCount.incrementAndGet();
          }

          @Override
          protected String getValue() {
            return "result";
          }
        };

    task.run();
    assertThat(task.isDone()).isFalse();
    assertThat(subtaskCallCount.get()).isEqualTo(1);

    task.decrement();
    assertThat(task.isDone()).isTrue();
    assertThat(task.get()).isEqualTo("result");
  }

  @Test
  public void exceptionInArrangeSubtasks() throws Exception {
    var error = new RuntimeException("oops");
    var task =
        new QuiescingFutureTask<String>(directExecutor()) {
          @Override
          protected void arrangeSubtasks() {
            throw error;
          }

          @Override
          protected String getValue() {
            return "result";
          }
        };

    task.run();
    assertThat(task.isDone()).isTrue();
    var thrown = assertThrows(ExecutionException.class, task::get);
    assertThat(thrown).hasCauseThat().isSameInstanceAs(error);
  }

  @Test
  public void doneWithErrorCalled() throws Exception {
    AtomicBoolean doneWithErrorCalled = new AtomicBoolean(false);
    var task =
        new QuiescingFutureTask<String>(directExecutor()) {
          @Override
          protected void arrangeSubtasks() {
            notifyException(new RuntimeException("error"));
          }

          @Override
          protected String getValue() {
            return "result";
          }

          @Override
          protected void doneWithError() {
            doneWithErrorCalled.set(true);
          }
        };

    task.run();
    assertThat(task.isDone()).isTrue();
    assertThat(doneWithErrorCalled.get()).isTrue();
  }
}
