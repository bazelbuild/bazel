// Copyright 2019 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.cmdline;

import static com.google.common.truth.Truth.assertThat;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMultimap;
import com.google.common.collect.Iterables;
import com.google.common.collect.Sets;
import com.google.devtools.build.lib.cmdline.BatchCallback.SafeBatchCallback;
import com.google.devtools.build.lib.concurrent.ThreadSafety.ThreadSafe;
import com.google.devtools.build.lib.testutil.TestThread;
import java.util.ArrayList;
import java.util.Set;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.Executors;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Unit tests for {@link ParallelVisitor}. */
@RunWith(JUnit4.class)
public class ParallelVisitorTest {

  private static final int BATCH_CALLBACK_SIZE = 10_000;
  private static final long MIN_PENDING_TASKS = 3L;

  @ThreadSafe
  private interface TestCallback<T> extends SafeBatchCallback<T> {
    @Override
    void process(Iterable<T> partialResult);
  }

  /**
   * A dummy {@link ParallelVisitor} which waits for signal from a {@link CountDownLatch} when
   * {@link #getVisitResult} is invoked. It allows us to test interruptibility.
   */
  private static class DelayGettingVisitResultParallelVisitor
      extends ParallelVisitor<
          String,
          String,
          String,
          String,
          QueryExceptionMarkerInterface.MarkerRuntimeException,
          TestCallback<String>> {
    private final CountDownLatch invocationLatch;
    private final CountDownLatch delayLatch;

    private DelayGettingVisitResultParallelVisitor(
        CountDownLatch invocationLatch, CountDownLatch delayLatch) {
      super(
          targets -> {},
          QueryExceptionMarkerInterface.MarkerRuntimeException.class,
          /*visitBatchSize=*/ 1,
          /*processResultsBatchSize=*/ 1,
          /*minPendingTasks=*/ MIN_PENDING_TASKS,
          /*batchCallbackSize=*/ BATCH_CALLBACK_SIZE,
          Executors.newFixedThreadPool(3),
          VisitTaskStatusCallback.NULL_INSTANCE);
      this.invocationLatch = invocationLatch;
      this.delayLatch = delayLatch;
    }

    @Override
    protected Visit getVisitResult(Iterable<String> values) throws InterruptedException {
      invocationLatch.countDown();
      delayLatch.await();
      return new Visit(ImmutableList.of(), values);
    }

    @Override
    protected Iterable<String> preprocessInitialVisit(Iterable<String> visitationKeys) {
      return visitationKeys;
    }

    @Override
    protected Iterable<String> outputKeysToOutputValues(Iterable<String> targetKeys) {
      return ImmutableList.of();
    }

    @Override
    protected Iterable<String> noteAndReturnUniqueVisitationKeys(
        Iterable<String> prospectiveVisitationKeys) {
      return ImmutableList.copyOf(prospectiveVisitationKeys);
    }
  }

  @Test
  public void testInterrupt() throws Exception {
    // This test verifies that visitations by ParallelVisitor can be interrupted. It also serves as
    // a regression test of b/62221332.
    CountDownLatch invocationLatch = new CountDownLatch(1);
    CountDownLatch delayLatch = new CountDownLatch(1);
    DelayGettingVisitResultParallelVisitor visitor =
        new DelayGettingVisitResultParallelVisitor(invocationLatch, delayLatch);
    ImmutableList<String> keysToVisit = ImmutableList.of("for_testing");

    TestThread testThread = new TestThread(() -> visitor.visitAndWaitForCompletion(keysToVisit));
    testThread.start();

    // Send an interrupt signal to the visitor after #visitAndWaitForCompletion is invoked.
    invocationLatch.await();
    testThread.interrupt();

    // Verify that the thread is interruptible (unit test will time out if it's not interruptible).
    testThread.join();
  }

  private static class RecordingParallelVisitor
      extends ParallelVisitor<
          InputKey,
          String,
          String,
          String,
          QueryExceptionMarkerInterface.MarkerRuntimeException,
          TestCallback<String>> {
    private final ArrayList<Iterable<String>> visits = new ArrayList<>();
    private final ImmutableMultimap<String, String> successorMap;
    private final Set<String> visited = Sets.newConcurrentHashSet();

    private RecordingParallelVisitor(
        ImmutableMultimap<String, String> successors,
        RecordingCallback recordingCallback,
        int visitBatchSize,
        int processResultsBatchSize) {
      super(
          recordingCallback,
          QueryExceptionMarkerInterface.MarkerRuntimeException.class,
          visitBatchSize,
          processResultsBatchSize,
          MIN_PENDING_TASKS,
          BATCH_CALLBACK_SIZE,
          Executors.newFixedThreadPool(3),
          VisitTaskStatusCallback.NULL_INSTANCE);
      this.successorMap = successors;
    }

    @Override
    protected Visit getVisitResult(Iterable<String> values) {
      synchronized (this) {
        visits.add(values);
      }
      return new Visit(values, Iterables.concat(Iterables.transform(values, successorMap::get)));
    }

    @Override
    protected Iterable<String> noteAndReturnUniqueVisitationKeys(
        Iterable<String> prospectiveVisitationKeys) {
      return Iterables.filter(prospectiveVisitationKeys, visited::add);
    }

    @Override
    protected Iterable<String> outputKeysToOutputValues(Iterable<String> targetKeys) {
      return targetKeys;
    }

    @Override
    protected Iterable<String> preprocessInitialVisit(Iterable<InputKey> visitationKeys) {
      return Iterables.transform(visitationKeys, InputKey::extract);
    }
  }

  private static class RecordingCallback implements TestCallback<String> {
    private final ArrayList<Iterable<String>> results = new ArrayList<>();

    @Override
    public synchronized void process(Iterable<String> partialResult) {
      results.add(partialResult);
    }
  }

  private static class InputKey {
    private final String str;

    private InputKey(String str) {
      this.str = str;
    }

    private static String extract(InputKey key) {
      return key.str;
    }

    @Override
    public boolean equals(Object obj) {
      if (!(obj instanceof InputKey other)) {
        return false;
      }
      return this.str.equals(other.str);
    }

    @Override
    public int hashCode() {
      return str.hashCode();
    }
  }

  @Test
  public void testRespectsBatchSizes() throws Exception {
    int visitBatchSize = 2;
    int processResultsBatchSize = 1;
    RecordingCallback callback = new RecordingCallback();
    RecordingParallelVisitor visitor =
        new RecordingParallelVisitor(
            ImmutableMultimap.<String, String>builder()
                .putAll("k1", ImmutableList.of("k2", "k3", "k4", "k5"))
                .putAll("k2", ImmutableList.of("k6", "k7", "k8", "k9"))
                .putAll("k3", ImmutableList.of("k4", "k5", "k6", "k7", "k8", "k9"))
                .build(),
            callback,
            visitBatchSize,
            processResultsBatchSize);
    visitor.visitAndWaitForCompletion(ImmutableList.of(new InputKey("k1")));

    for (Iterable<String> visitBatch : visitor.visits) {
      assertThat(Iterables.size(visitBatch)).isAtMost(visitBatchSize);
    }
    for (Iterable<String> resultBatch : callback.results) {
      assertThat(Iterables.size(resultBatch)).isAtMost(processResultsBatchSize);
    }
  }
}
