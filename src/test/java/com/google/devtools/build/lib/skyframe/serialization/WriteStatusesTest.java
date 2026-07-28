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
package com.google.devtools.build.lib.skyframe.serialization;

import static com.google.common.truth.Truth.assertThat;
import static com.google.common.util.concurrent.MoreExecutors.directExecutor;
import static com.google.devtools.build.lib.skyframe.serialization.WriteStatuses.aggregateWriteStatuses;
import static com.google.devtools.build.lib.skyframe.serialization.WriteStatuses.immediateFailedWriteStatus;
import static com.google.devtools.build.lib.skyframe.serialization.WriteStatuses.immediateWriteStatus;
import static java.util.concurrent.TimeUnit.SECONDS;
import static org.junit.Assert.assertThrows;

import com.google.common.collect.ImmutableList;
import com.google.common.util.concurrent.SettableFuture;
import com.google.devtools.build.lib.skyframe.serialization.WriteStatuses.SettableWriteStatus;
import java.util.concurrent.CancellationException;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.Executor;
import java.util.concurrent.atomic.AtomicReference;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

@RunWith(JUnit4.class)
public final class WriteStatusesTest {

  @Test
  public void immediateWriteStatus_isDone() throws Exception {
    WriteStatus status = immediateWriteStatus();

    assertThat(status.isDone()).isTrue();
    assertThat(status.isCancelled()).isFalse();
    assertThat(status.cancel(true)).isFalse();
    assertThat(status.cancel(false)).isFalse();

    assertThat(status.get()).isTrue();
    assertThat(status.get(0, SECONDS)).isTrue();

    assertListenerExecutesImmediately(status);
  }

  @Test
  public void failedWriteStatus_isDone() throws Exception {
    var exception = new Exception();
    WriteStatus status = immediateFailedWriteStatus(exception);

    assertThat(status.isDone()).isTrue();
    assertThat(status.isCancelled()).isFalse();
    assertThat(status.cancel(true)).isFalse();
    assertThat(status.cancel(false)).isFalse();

    var thrown = assertThrows(ExecutionException.class, status::get);
    assertThat(thrown).hasCauseThat().isSameInstanceAs(exception);

    thrown = assertThrows(ExecutionException.class, () -> status.get(0, SECONDS));
    assertThat(thrown).hasCauseThat().isSameInstanceAs(exception);

    assertListenerExecutesImmediately(status);
  }

  @Test
  public void settableWriteStatus_markSuccess_isDone() throws Exception {
    var status = new SettableWriteStatus();

    assertThat(status.isDone()).isFalse();
    var setOnRun = new SetOnRun();
    status.addListener(setOnRun, directExecutor());
    assertThat(setOnRun.isSet).isFalse();

    status.markSuccess();
    assertThat(setOnRun.isSet).isTrue();

    assertThat(status.isDone()).isTrue();
    assertThat(status.get()).isTrue();

    assertListenerExecutesImmediately(status);
  }

  @Test
  public void settableWriteStatus_failWith() throws Exception {
    var status = new SettableWriteStatus();

    assertThat(status.isDone()).isFalse();
    var setOnRun = new SetOnRun();
    status.addListener(setOnRun, directExecutor());
    assertThat(setOnRun.isSet).isFalse();

    var exception = new Exception();
    status.failWith(exception);
    assertThat(setOnRun.isSet).isTrue();
    assertThat(status.isDone()).isTrue();

    var thrown = assertThrows(ExecutionException.class, status::get);
    assertThat(thrown).hasCauseThat().isSameInstanceAs(exception);

    thrown = assertThrows(ExecutionException.class, () -> status.get(0, SECONDS));
    assertThat(thrown).hasCauseThat().isSameInstanceAs(exception);

    assertListenerExecutesImmediately(status);
  }

  @Test
  public void settableWriteStatus_failWith_cancellationException() throws Exception {
    var status = new SettableWriteStatus();

    assertThat(status.isDone()).isFalse();
    var setOnRun = new SetOnRun();
    status.addListener(setOnRun, directExecutor());
    assertThat(setOnRun.isSet).isFalse();

    status.failWith(new CancellationException());

    assertThat(setOnRun.isSet).isTrue();
    assertThat(status.isDone()).isTrue();
    assertThat(status.isCancelled()).isTrue();

    assertThrows(CancellationException.class, status::get);
    assertThrows(CancellationException.class, () -> status.get(0, SECONDS));

    assertListenerExecutesImmediately(status);
  }

  @Test
  public void settableWriteStatus_completeWith_successfulFuture() throws Exception {
    var status = new SettableWriteStatus();

    assertThat(status.isDone()).isFalse();
    var setOnRun = new SetOnRun();
    status.addListener(setOnRun, directExecutor());
    assertThat(setOnRun.isSet).isFalse();

    status.completeWith(immediateWriteStatus());
    assertThat(setOnRun.isSet).isTrue();

    assertThat(status.isDone()).isTrue();
    assertThat(status.get()).isTrue();
    assertThat(status.get(0, SECONDS)).isTrue();

    assertListenerExecutesImmediately(status);
  }

  @Test
  public void settableWriteStatus_completeWith_failingFuture() throws Exception {
    var status = new SettableWriteStatus();

    assertThat(status.isDone()).isFalse();
    var setOnRun = new SetOnRun();
    status.addListener(setOnRun, directExecutor());
    assertThat(setOnRun.isSet).isFalse();

    var exception = new Exception();
    status.completeWith(immediateFailedWriteStatus(exception));

    assertThat(status.isDone()).isTrue();
    assertThat(setOnRun.isSet).isTrue();

    var thrown = assertThrows(ExecutionException.class, status::get);
    assertThat(thrown).hasCauseThat().isSameInstanceAs(exception);

    thrown = assertThrows(ExecutionException.class, () -> status.get(0, SECONDS));
    assertThat(thrown).hasCauseThat().isSameInstanceAs(exception);

    assertListenerExecutesImmediately(status);
  }

  @Test
  public void aggregateWriteStatus_waitsForInputs() throws Exception {
    var input = new SettableWriteStatus();

    var status = aggregateWriteStatuses(ImmutableList.of(input, immediateWriteStatus()));

    assertThat(status.isDone()).isFalse();
    var setOnRun = new SetOnRun();
    status.addListener(setOnRun, directExecutor());
    assertThat(setOnRun.isSet).isFalse();

    input.markSuccess();
    assertThat(setOnRun.isSet).isTrue();
    assertThat(status.isDone()).isTrue();

    assertThat(status.get()).isTrue();
    assertThat(status.get(0, SECONDS)).isTrue();

    assertListenerExecutesImmediately(status);
  }

  @Test
  public void aggregateWriteStatus_failsOnFailedInput() throws Exception {
    var input = new SettableWriteStatus();

    var status = aggregateWriteStatuses(ImmutableList.of(input, immediateWriteStatus()));

    assertThat(status.isDone()).isFalse();
    var setOnRun = new SetOnRun();
    status.addListener(setOnRun, directExecutor());
    assertThat(setOnRun.isSet).isFalse();

    var exception = new Exception();
    input.failWith(exception);
    assertThat(setOnRun.isSet).isTrue();

    assertThat(status.isDone()).isTrue();
    assertListenerExecutesImmediately(status);

    var thrown = assertThrows(ExecutionException.class, status::get);
    assertThat(thrown).hasCauseThat().isSameInstanceAs(exception);

    thrown = assertThrows(ExecutionException.class, () -> status.get(0, SECONDS));
    assertThat(thrown).hasCauseThat().isSameInstanceAs(exception);
  }

  @Test
  public void aggregateWriteStatus_multipleFailedInputs() throws Exception {
    var input = new SettableWriteStatus();

    var exception1 = new Exception();
    var status =
        aggregateWriteStatuses(ImmutableList.of(input, immediateFailedWriteStatus(exception1)));

    assertThat(status.isDone()).isTrue();
    assertListenerExecutesImmediately(status);

    var thrown = assertThrows(ExecutionException.class, status::get);
    assertThat(thrown).hasCauseThat().isSameInstanceAs(exception1);

    thrown = assertThrows(ExecutionException.class, () -> status.get(0, SECONDS));
    assertThat(thrown).hasCauseThat().isSameInstanceAs(exception1);

    Exception exception2 = new Exception();
    input.failWith(exception2);

    thrown = assertThrows(ExecutionException.class, status::get);
    assertThat(thrown).hasCauseThat().isSameInstanceAs(exception1);
    thrown = assertThrows(ExecutionException.class, () -> status.get(0, SECONDS));
    assertThat(thrown).hasCauseThat().isSameInstanceAs(exception1);
  }

  @Test
  public void aggregateWriteStatus_alreadyCancelledInput_propagates() throws Exception {
    var input = new SettableWriteStatus();
    input.cancel(/* mayInterruptIfRunning= */ false);
    assertThat(input.isCancelled()).isTrue();
    var unused = assertThrows(CancellationException.class, input::get);

    var status = aggregateWriteStatuses(ImmutableList.of(input, immediateWriteStatus()));

    assertThat(status.isDone()).isTrue();
    assertThat(status.isCancelled()).isTrue();
    unused = assertThrows(CancellationException.class, status::get);
  }

  @Test
  public void aggregateWriteStatus_cancellingInput_propagates() throws Exception {
    var input = new SettableWriteStatus();

    var status = aggregateWriteStatuses(ImmutableList.of(input, immediateWriteStatus()));

    assertThat(status.isDone()).isFalse();

    input.cancel(/* mayInterruptIfRunning= */ false);

    assertThat(status.isDone()).isTrue();
    assertThat(status.isCancelled()).isTrue();
    var unused = assertThrows(CancellationException.class, status::get);
  }


  private static void assertListenerExecutesImmediately(WriteStatus status) {
    var captured = new AtomicReference<Runnable>();
    Executor capturingExecutor =
        command -> assertThat(captured.compareAndSet(null, command)).isTrue();

    Runnable runnable = () -> {};
    status.addListener(runnable, capturingExecutor);
    assertThat(captured.get()).isSameInstanceAs(runnable);
  }

  private static class SetOnRun implements Runnable {
    private boolean isSet = false;

    @Override
    public void run() {
      isSet = true;
    }
  }

  @Test
  public void writeStatusBuilder_empty_isImmediate() throws Exception {
    var builder = new WriteStatuses.WriteStatusBuilder();
    assertThat(builder.build()).isSameInstanceAs(immediateWriteStatus());
  }

  @Test
  public void writeStatusBuilder_singleton_forwards() throws Exception {
    var builder = new WriteStatuses.WriteStatusBuilder();
    var inner = new SettableWriteStatus();
    builder.add(inner);
    assertThat(builder.build()).isSameInstanceAs(inner);
  }

  @Test
  public void writeStatusBuilder_singletonPlainFuture_wraps() throws Exception {
    var builder = new WriteStatuses.WriteStatusBuilder();
    var inner = SettableFuture.<Boolean>create();
    builder.add(inner);
    WriteStatus status = builder.build();
    assertThat(status.isDone()).isFalse();
    inner.set(true);
    assertThat(status.isDone()).isTrue();
    assertThat(status.get()).isTrue();
  }

  @Test
  public void writeStatusBuilder_addDone() throws Exception {
    var builder = new WriteStatuses.WriteStatusBuilder();
    builder.add(immediateWriteStatus());
    var exception = new Exception("test");
    builder.add(immediateFailedWriteStatus(exception));

    WriteStatus status = builder.build();
    assertThat(status.isDone()).isTrue();
    var thrown = assertThrows(ExecutionException.class, status::get);
    assertThat(thrown).hasCauseThat().isSameInstanceAs(exception);
  }

  @Test
  public void writeStatusBuilder_addPending() throws Exception {
    var builder = new WriteStatuses.WriteStatusBuilder();
    var pending1 = new SettableWriteStatus();
    var pending2 = new SettableWriteStatus();
    builder.add(pending1);
    builder.add(pending2);

    WriteStatus status = builder.build();
    assertThat(status.isDone()).isFalse();

    pending1.markSuccess();
    assertThat(status.isDone()).isFalse();

    pending2.markSuccess();
    assertThat(status.isDone()).isTrue();
    assertThat(status.get()).isTrue();
  }

  @Test
  public void writeStatusBuilder_addAll() throws Exception {
    var builder = new WriteStatuses.WriteStatusBuilder();
    var pending1 = new SettableWriteStatus();
    var exception = new Exception("test");
    builder.addAll(
        ImmutableList.of(pending1, immediateWriteStatus(), immediateFailedWriteStatus(exception)));

    WriteStatus status = builder.build();
    assertThat(status.isDone()).isTrue(); // Fails fast
    var thrown = assertThrows(ExecutionException.class, status::get);
    assertThat(thrown).hasCauseThat().isSameInstanceAs(exception);
  }

  @Test
  public void writeStatusBuilder_buildTwice_isIdempotent() throws Exception {
    // Zero dependencies
    var emptyBuilder = new WriteStatuses.WriteStatusBuilder();
    WriteStatus emptyFirstBuild = emptyBuilder.build();
    assertThat(emptyBuilder.build()).isSameInstanceAs(emptyFirstBuild);

    // One dependency (already WriteStatus)
    var singleWriteStatusBuilder = new WriteStatuses.WriteStatusBuilder();
    singleWriteStatusBuilder.add(immediateWriteStatus());
    WriteStatus singleWriteStatusFirstBuild = singleWriteStatusBuilder.build();
    assertThat(singleWriteStatusBuilder.build()).isSameInstanceAs(singleWriteStatusFirstBuild);

    // One dependency (wrapped future)
    var singleFutureBuilder = new WriteStatuses.WriteStatusBuilder();
    singleFutureBuilder.add(SettableFuture.<Boolean>create());
    WriteStatus singleFutureFirstBuild = singleFutureBuilder.build();
    assertThat(singleFutureBuilder.build()).isSameInstanceAs(singleFutureFirstBuild);

    // Multiple dependencies (AggregateWriteStatus)
    var aggregateBuilder = new WriteStatuses.WriteStatusBuilder();
    aggregateBuilder.add(new SettableWriteStatus());
    aggregateBuilder.add(new SettableWriteStatus());
    WriteStatus aggregateFirstBuild = aggregateBuilder.build();
    assertThat(aggregateBuilder.build()).isSameInstanceAs(aggregateFirstBuild);
  }

  @Test
  public void writeStatusBuilder_addAfterBuild_throwsException() throws Exception {
    var builder = new WriteStatuses.WriteStatusBuilder();
    var unused = builder.build();
    WriteStatus status = immediateWriteStatus();
    IllegalStateException thrown =
        assertThrows(IllegalStateException.class, () -> builder.add(status));
    assertThat(thrown).hasMessageThat().contains("cannot add to WriteStatusBuilder after build()");
  }



  @Test
  public void aggregate_propagatesNovelty() throws Exception {
    var n1 = new SettableWriteStatus();
    var n2 = new SettableWriteStatus();
    var aggregate = aggregateWriteStatuses(ImmutableList.of(n1, n2));

    assertThat(aggregate.isDone()).isFalse();

    n1.markSuccess(true);
    assertThat(aggregate.isDone()).isFalse();

    n2.markSuccess(false);
    assertThat(aggregate.isDone()).isTrue();
    assertThat(aggregate.get()).isTrue();
  }

  @Test
  public void aggregate_allFalse_isFalse() throws Exception {
    var n1 = new SettableWriteStatus();
    var n2 = new SettableWriteStatus();
    var aggregate = aggregateWriteStatuses(ImmutableList.of(n1, n2));

    n1.markSuccess(false);
    n2.markSuccess(false);
    assertThat(aggregate.get()).isFalse();
  }



  @Test
  public void aggregateWriteStatus_mixedPendingAndImmediate_trueThenFalse() throws Exception {
    var pending = new SettableWriteStatus();
    var status = aggregateWriteStatuses(ImmutableList.of(pending, immediateWriteStatus()));

    assertThat(status.isDone()).isFalse();

    pending.markSuccess(false);

    assertThat(status.isDone()).isTrue();
    // immediateWriteStatus() is true. (true OR false) is true.
    assertThat(status.get()).isTrue();
  }

  @Test
  public void aggregateWriteStatus_mixedPendingAndImmediate_falseThenTrue() throws Exception {
    var pending = new SettableWriteStatus();
    var immediateFalse = new SettableWriteStatus();
    immediateFalse.markSuccess(false);
    var status = aggregateWriteStatuses(ImmutableList.of(pending, immediateFalse));

    assertThat(status.isDone()).isFalse();

    pending.markSuccess(true);

    assertThat(status.isDone()).isTrue();
    assertThat(status.get()).isTrue();
  }


}
