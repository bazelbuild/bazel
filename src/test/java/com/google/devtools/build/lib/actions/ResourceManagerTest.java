// Copyright 2015 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.actions;

import static com.google.common.truth.Truth.assertThat;
import static org.junit.Assert.assertThrows;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.spy;
import static org.mockito.Mockito.times;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.ImmutableSortedMap;
import com.google.common.hash.HashCode;
import com.google.devtools.build.lib.actions.ExecutionRequirements.WorkerProtocolFormat;
import com.google.devtools.build.lib.actions.ResourceManager.ResourceHandle;
import com.google.devtools.build.lib.actions.ResourceManager.ResourcePriority;
import com.google.devtools.build.lib.analysis.platform.PlatformInfo;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.collect.nestedset.Order;
import com.google.devtools.build.lib.server.FailureDetails;
import com.google.devtools.build.lib.server.FailureDetails.FailureDetail;
import com.google.devtools.build.lib.server.FailureDetails.Worker.Code;
import com.google.devtools.build.lib.testutil.TestThread;
import com.google.devtools.build.lib.testutil.TestUtils;
import com.google.devtools.build.lib.vfs.DigestHashFunction;
import com.google.devtools.build.lib.vfs.FileSystem;
import com.google.devtools.build.lib.vfs.inmemoryfs.InMemoryFileSystem;
import com.google.devtools.build.lib.worker.Worker;
import com.google.devtools.build.lib.worker.WorkerKey;
import com.google.devtools.build.lib.worker.WorkerProcessStatus;
import com.google.devtools.build.lib.worker.WorkerProcessStatus.Status;
import com.google.devtools.build.lib.worker.WorkerTestUtils;
import com.google.errorprone.annotations.CanIgnoreReturnValue;
import java.io.IOException;
import java.util.Collection;
import java.util.NoSuchElementException;
import java.util.concurrent.CyclicBarrier;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;
import javax.annotation.Nullable;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link ResourceManager}. */
@RunWith(JUnit4.class)
public final class ResourceManagerTest {

  private final FileSystem fs = new InMemoryFileSystem(DigestHashFunction.SHA256);
  private final ActionExecutionMetadata resourceOwner = new ResourceOwnerStub();
  private final ResourceManager rm = ResourceManager.instanceForTestingOnly();
  private Worker worker;
  private WorkerProcessStatus workerStatus;
  private AtomicInteger counter;
  CyclicBarrier sync;
  CyclicBarrier sync2;

  @Before
  public void configureResourceManager() throws Exception {
    rm.setAvailableResources(
        ResourceSet.create(
            ImmutableMap.of(
                ResourceSet.MEMORY, 1000.0, ResourceSet.CPU, 1.0, "gpu", 2.0, "fancyresource", 1.5),
            /* localTestCount= */ 2));
    counter = new AtomicInteger(0);
    sync = new CyclicBarrier(2);
    sync2 = new CyclicBarrier(2);
    rm.resetResourceUsage();
    worker = mock(Worker.class);
    workerStatus = spy(new WorkerProcessStatus());
    when(worker.getStatus()).thenReturn(workerStatus);
    rm.setWorkerPool(WorkerTestUtils.createTestWorkerPool(worker));
  }

  private ResourceHandle acquire(double ram, double cpu, int tests, ResourcePriority priority)
      throws InterruptedException, IOException {
    return rm.acquireResources(resourceOwner, ResourceSet.create(ram, cpu, tests), priority);
  }

  private ResourceHandle acquire(double ram, double cpu, int tests)
      throws InterruptedException, IOException {
    return acquire(ram, cpu, tests, ResourcePriority.LOCAL);
  }

  private ResourceHandle acquire(double ram, double cpu, int tests, String mnemonic)
      throws InterruptedException, IOException {

    return rm.acquireResources(
        resourceOwner,
        ResourceSet.create(
            ImmutableMap.of(ResourceSet.MEMORY, ram, ResourceSet.CPU, cpu),
            tests,
            createWorkerKey(mnemonic)),
        ResourcePriority.LOCAL);
  }

  @CanIgnoreReturnValue
  private ResourceHandle acquire(
      double ram,
      double cpu,
      ImmutableMap<String, Double> extraResources,
      int tests,
      ResourcePriority priority)
      throws InterruptedException, IOException, NoSuchElementException {
    ImmutableMap.Builder<String, Double> resources = ImmutableMap.builder();
    resources.putAll(extraResources).put(ResourceSet.MEMORY, ram).put(ResourceSet.CPU, cpu);
    return rm.acquireResources(
        resourceOwner, ResourceSet.create(resources.buildOrThrow(), tests), priority);
  }

  @CanIgnoreReturnValue
  private ResourceHandle acquire(
      double ram, double cpu, ImmutableMap<String, Double> extraResources, int tests)
      throws InterruptedException, IOException, NoSuchElementException {
    return acquire(ram, cpu, extraResources, tests, ResourcePriority.LOCAL);
  }

  private void release(double ram, double cpu, int tests) throws IOException, InterruptedException {
    rm.releaseResources(resourceOwner, ResourceSet.create(ram, cpu, tests), /* worker= */ null);
  }

  private void release(
      double ram, double cpu, ImmutableMap<String, Double> extraResources, int tests)
      throws InterruptedException, IOException {
    ImmutableMap.Builder<String, Double> resources = ImmutableMap.builder();
    resources.putAll(extraResources).put(ResourceSet.MEMORY, ram).put(ResourceSet.CPU, cpu);
    rm.releaseResources(
        resourceOwner, ResourceSet.create(resources.buildOrThrow(), tests), /* worker= */ null);
  }

  private void validate(int count) {
    assertThat(counter.incrementAndGet()).isEqualTo(count);
  }

  private WorkerKey createWorkerKey(String mnemonic) {
    return new WorkerKey(
        /* args= */ ImmutableList.of(),
        /* env= */ ImmutableMap.of(),
        /* execRoot= */ fs.getPath("/outputbase/execroot/workspace"),
        /* mnemonic= */ mnemonic,
        /* workerFilesCombinedHash= */ HashCode.fromInt(0),
        /* workerFilesWithDigests= */ ImmutableSortedMap.of(),
        /* sandboxed= */ false,
        /* multiplex= */ false,
        /* cancellable= */ false,
        WorkerProtocolFormat.PROTO);
  }

  @Test
  public void testOverBudgetRequests() throws Exception {
    assertThat(rm.inUse()).isFalse();

    // When nothing is consuming RAM,
    // Then Resource Manager will successfully acquire an over-budget request for RAM:
    double bigRam = 10000.0;
    acquire(bigRam, 0, 0);
    // When RAM is consumed,
    // Then Resource Manager will be "in use":
    assertThat(rm.inUse()).isTrue();
    release(bigRam, 0, 0);
    // When that RAM is released,
    // Then Resource Manager will not be "in use":
    assertThat(rm.inUse()).isFalse();

    // Ditto, for CPU:
    double bigCpu = 10.0;
    acquire(0, bigCpu, 0);
    assertThat(rm.inUse()).isTrue();
    release(0, bigCpu, 0);
    assertThat(rm.inUse()).isFalse();

    // Ditto, for tests:
    int bigTests = 10;
    acquire(0, 0, bigTests);
    assertThat(rm.inUse()).isTrue();
    release(0, 0, bigTests);
    assertThat(rm.inUse()).isFalse();

    // Ditto, for extra resources:
    ImmutableMap<String, Double> bigExtraResources =
        ImmutableMap.of("gpu", 10.0, "fancyresource", 10.0);
    acquire(0, 0, bigExtraResources, 0);
    assertThat(rm.inUse()).isTrue();
    release(0, 0, bigExtraResources, 0);
    assertThat(rm.inUse()).isFalse();
  }

  @Test
  public void testThatCpuCanBeOverallocated() throws Exception {
    assertThat(rm.inUse()).isFalse();

    // Given CPU is partially acquired:
    acquire(0, 0.5, 0);

    // When a request for CPU is made that would slightly overallocate CPU,
    // Then the request succeeds:
    TestThread thread1 = new TestThread(() -> assertThat(acquire(0, 0.6, 0)).isNotNull());
    thread1.start();
    thread1.joinAndAssertState(10000);
  }

  @Test
  public void testThatCpuAllocationIsNoncommutative() throws Exception {
    assertThat(rm.inUse()).isFalse();

    // Given that CPU has a small initial allocation:
    acquire(0, 0.099, 0);

    // When a request for a large CPU allocation is made,
    // Then the request succeeds:
    TestThread thread1 =
        new TestThread(
            () -> {
              acquire(0, 0.99, 0);
              // Cleanup
              release(0, 0.99, 0);
            });
    thread1.start();
    thread1.joinAndAssertState(10000);

    // Cleanup
    release(0, 0.099, 0);
    assertThat(rm.inUse()).isFalse();

    // Given that CPU has a large initial allocation:
    acquire(0, 0.99, 0);

    // When a request for a small CPU allocation is made,
    // Then the request fails:
    TestThread thread2 = new TestThread(() -> acquire(0, 0.099, 0));
    thread2.start();
    AssertionError e = assertThrows(AssertionError.class, () -> thread2.joinAndAssertState(1000));
    assertThat(e).hasCauseThat().hasMessageThat().contains("is still alive");
    // Note that this behavior is surprising and probably not intended.
  }

  @Test
  public void testThatRamCannotBeOverallocated() throws Exception {
    assertThat(rm.inUse()).isFalse();

    // Given RAM is partially acquired:
    acquire(500, 0, 0);

    // When a request for RAM is made that would slightly overallocate RAM,
    // Then the request fails (got timeout):
    TestThread thread1 = new TestThread(() -> acquire(600, 0, 0));
    thread1.start();
    AssertionError e = assertThrows(AssertionError.class, () -> thread1.joinAndAssertState(1000));
    assertThat(e).hasCauseThat().hasMessageThat().contains("is still alive");
  }

  @Test
  public void testThatTestsCannotBeOverallocated() throws Exception {
    assertThat(rm.inUse()).isFalse();

    // Given test count is partially acquired:
    acquire(0, 0, 1);

    // When a request for tests is made that would slightly overallocate tests,
    // Then the request fails:
    TestThread thread1 = new TestThread(() -> acquire(0, 0, 2));
    thread1.start();
    AssertionError e = assertThrows(AssertionError.class, () -> thread1.joinAndAssertState(1000));
    assertThat(e).hasCauseThat().hasMessageThat().contains("is still alive");
  }

  @Test
  public void testThatExtraResourcesCannotBeOverallocated() throws Exception {
    assertThat(rm.inUse()).isFalse();

    // Given a partially acquired extra resources:
    acquire(0, 0, ImmutableMap.of("gpu", 1.0), 1);

    // When a request for extra resources is made that would overallocate,
    // Then the request fails:
    TestThread thread1 = new TestThread(() -> acquire(0, 0, ImmutableMap.of("gpu", 1.1), 0));
    thread1.start();
    AssertionError e = assertThrows(AssertionError.class, () -> thread1.joinAndAssertState(1000));
    assertThat(e).hasCauseThat().hasMessageThat().contains("is still alive");
  }

  @Test
  public void testHasResources() throws Exception {
    assertThat(rm.inUse()).isFalse();
    assertThat(rm.threadHasResources()).isFalse();
    acquire(1, 0.1, ImmutableMap.of("gpu", 1.0), 1);
    assertThat(rm.threadHasResources()).isTrue();

    // We have resources in this thread - make sure other threads
    // are not affected.
    TestThread thread1 =
        new TestThread(
            () -> {
              assertThat(rm.threadHasResources()).isFalse();
              acquire(1, 0, 0);
              assertThat(rm.threadHasResources()).isTrue();
              release(1, 0, 0);
              assertThat(rm.threadHasResources()).isFalse();
              acquire(0, 0.1, 0);
              assertThat(rm.threadHasResources()).isTrue();
              release(0, 0.1, 0);
              assertThat(rm.threadHasResources()).isFalse();
              acquire(0, 0, 1);
              assertThat(rm.threadHasResources()).isTrue();
              release(0, 0, 1);
              assertThat(rm.threadHasResources()).isFalse();
              acquire(0, 0, ImmutableMap.of("gpu", 1.0), 0);
              assertThat(rm.threadHasResources()).isTrue();
              release(0, 0, ImmutableMap.of("gpu", 1.0), 0);
              assertThat(rm.threadHasResources()).isFalse();
            });
    thread1.start();
    thread1.joinAndAssertState(10000);

    release(1, 0.1, ImmutableMap.of("gpu", 1.0), 1);
    assertThat(rm.threadHasResources()).isFalse();
    assertThat(rm.inUse()).isFalse();
  }

  @Test
  @SuppressWarnings("ThreadPriorityCheck")
  public void testConcurrentLargeRequests() throws Exception {
    assertThat(rm.inUse()).isFalse();
    TestThread thread1 =
        new TestThread(
            () -> {
              acquire(2000, 2, 0);
              sync.await();
              validate(1);
              sync.await();
              // Wait till other thread will be locked.
              while (rm.getWaitCount() == 0) {
                Thread.yield();
              }
              release(2000, 2, 0);
              assertThat(rm.getWaitCount()).isEqualTo(0);
              acquire(2000, 2, 0); // Will be blocked by the thread2.
              validate(3);
              release(2000, 2, 0);
            });
    TestThread thread2 =
        new TestThread(
            () -> {
              sync2.await();
              assertThat(isAvailable(rm, 2000, 2, 0)).isFalse();
              acquire(2000, 2, 0); // Will be blocked by the thread1.
              validate(2);
              sync2.await();
              // Wait till other thread will be locked.
              while (rm.getWaitCount() == 0) {
                Thread.yield();
              }
              release(2000, 2, 0);
            });

    thread1.start();
    thread2.start();
    sync.await(1, TimeUnit.SECONDS);
    assertThat(rm.inUse()).isTrue();
    assertThat(rm.getWaitCount()).isEqualTo(0);
    sync2.await(1, TimeUnit.SECONDS);
    sync.await(1, TimeUnit.SECONDS);
    sync2.await(1, TimeUnit.SECONDS);
    thread1.joinAndAssertState(1000);
    thread2.joinAndAssertState(1000);
    assertThat(rm.inUse()).isFalse();
  }

  @Test
  public void testInterruptedAcquisitionClearsResources() throws Exception {
    assertThat(rm.inUse()).isFalse();
    // Acquire a small amount of resources so that future requests can block (the initial request
    // always succeeds even if it's for too much).
    TestThread smallThread = new TestThread(() -> acquire(1, 0, 0));
    smallThread.start();
    smallThread.joinAndAssertState(TestUtils.WAIT_TIMEOUT_MILLISECONDS);
    TestThread thread1 =
        new TestThread(
            () -> {
              Thread.currentThread().interrupt();
              assertThrows(InterruptedException.class, () -> acquire(1999, 0, 0));
            });
    thread1.start();
    thread1.joinAndAssertState(TestUtils.WAIT_TIMEOUT_MILLISECONDS);
    // This should process the queue. If the request from above is still present, it will take all
    // the available memory. But it shouldn't.
    rm.setAvailableResources(
        ResourceSet.create(/* memoryMb= */ 2000, /* cpu= */ 1, /* localTestCount= */ 2));
    TestThread thread2 =
        new TestThread(
            () -> {
              acquire(1999, 0, 0);
              release(1999, 0, 0);
            });
    thread2.start();
    thread2.joinAndAssertState(TestUtils.WAIT_TIMEOUT_MILLISECONDS);
  }

  @Test
  @SuppressWarnings("ThreadPriorityCheck")
  public void testOutOfOrderAllocation() throws Exception {
    final CyclicBarrier sync3 = new CyclicBarrier(2);
    final CyclicBarrier sync4 = new CyclicBarrier(2);

    assertThat(rm.inUse()).isFalse();

    TestThread thread1 =
        new TestThread(
            () -> {
              sync.await();
              acquire(900, 0.5, 0); // Will be blocked by the main thread.
              validate(5);
              release(900, 0.5, 0);
              sync.await();
            });

    TestThread thread2 =
        new TestThread(
            () -> {
              // Wait till other thread will be locked
              while (rm.getWaitCount() == 0) {
                Thread.yield();
              }
              acquire(100, 0.1, 0);
              validate(2);
              release(100, 0.1, 0);
              sync2.await();
              acquire(200, 0.5, 0);
              validate(4);
              sync2.await();
              release(200, 0.5, 0);
            });

    TestThread thread3 =
        new TestThread(
            () -> {
              acquire(100, 0.4, 0);
              sync3.await();
              sync3.await();
              release(100, 0.4, 0);
            });

    TestThread thread4 =
        new TestThread(
            () -> {
              acquire(750, 0.3, 0);
              sync4.await();
              sync4.await();
              release(750, 0.3, 0);
            });

    // Lock 900 MB, 0.9 CPU in total (spread over three threads so that we can individually release
    // parts of it).
    acquire(50, 0.2, 0);
    thread3.start();
    thread4.start();
    sync3.await(1, TimeUnit.SECONDS);
    sync4.await(1, TimeUnit.SECONDS);
    validate(1);

    // Start thread1, which will try to acquire 900 MB, 0.5 CPU, but can't, so it has to wait.
    thread1.start();
    sync.await(1, TimeUnit.SECONDS);

    // Start thread2, which will successfully acquire and release 100 MB, 0.1 CPU.
    thread2.start();
    // Signal thread2 to acquire 200 MB and 0.5 CPU, which will block.
    sync2.await(1, TimeUnit.SECONDS);

    // Waiting till both threads are locked.
    while (rm.getWaitCount() < 2) {
      Thread.yield();
    }

    validate(3); // Thread1 is now first in the queue and Thread2 is second.

    // Release 100 MB, 0.4 CPU. This allows Thread2 to continue out of order.
    sync3.await(1, TimeUnit.SECONDS);
    sync2.await(1, TimeUnit.SECONDS);

    // Release 750 MB, 0.3 CPU. At this point thread1 will finally acquire resources.
    sync4.await(1, TimeUnit.SECONDS);
    sync.await(1, TimeUnit.SECONDS);

    // Release all remaining resources.
    release(50, 0.2, 0);
    thread1.join();
    thread2.join();
    thread3.join();
    thread4.join();

    assertThat(rm.inUse()).isFalse();
  }

  @Test
  @SuppressWarnings("ThreadPriorityCheck")
  public void testRelease_highPriorityFirst() throws Exception {
    assertThat(rm.inUse()).isFalse();

    TestThread thread1 =
        new TestThread(
            () -> {
              acquire(700, 0, 0);
              sync.await();
              sync2.await();
              release(700, 0, 0);
            });
    thread1.start();
    // Wait for thread1 to have acquired its RAM
    sync.await(1, TimeUnit.SECONDS);

    // Set up threads that compete for resources
    CyclicBarrier syncDynamicStandalone =
        startAcquireReleaseThread(ResourcePriority.DYNAMIC_STANDALONE);
    while (rm.getWaitCount() < 1) {
      Thread.yield();
    }
    CyclicBarrier syncDynamicWorker = startAcquireReleaseThread(ResourcePriority.DYNAMIC_WORKER);
    while (rm.getWaitCount() < 2) {
      Thread.yield();
    }
    CyclicBarrier syncLocal = startAcquireReleaseThread(ResourcePriority.LOCAL);
    while (rm.getWaitCount() < 3) {
      Thread.yield();
    }

    sync2.await();

    while (syncLocal.getNumberWaiting()
            + syncDynamicWorker.getNumberWaiting()
            + syncDynamicStandalone.getNumberWaiting()
        == 0) {
      Thread.yield();
    }
    assertThat(rm.getWaitCount()).isEqualTo(2);
    assertThat(syncLocal.getNumberWaiting()).isEqualTo(1);
    syncLocal.await(1, TimeUnit.SECONDS);

    while (syncDynamicWorker.getNumberWaiting() + syncDynamicStandalone.getNumberWaiting() == 0) {
      Thread.yield();
    }
    assertThat(syncDynamicWorker.getNumberWaiting()).isEqualTo(1);
    assertThat(rm.getWaitCount()).isEqualTo(1);

    syncDynamicWorker.await(1, TimeUnit.SECONDS);
    while (syncDynamicStandalone.getNumberWaiting() == 0) {
      Thread.yield();
    }
    assertThat(syncDynamicStandalone.getNumberWaiting()).isEqualTo(1);
    assertThat(rm.getWaitCount()).isEqualTo(0);
    syncDynamicStandalone.await(1, TimeUnit.SECONDS);
  }

  @Test
  @SuppressWarnings("ThreadPriorityCheck")
  public void testRelease_dynamicLifo() throws Exception {
    assertThat(rm.inUse()).isFalse();

    TestThread thread1 =
        new TestThread(
            () -> {
              acquire(700, 0, 0);
              sync.await();
              sync2.await();
              release(700, 0, 0);
            });
    thread1.start();
    // Wait for thread1 to have acquired enough RAM to block the other threads.
    sync.await(1, TimeUnit.SECONDS);

    // Set up threads that compete for resources
    final CyclicBarrier syncDynamicStandalone1 =
        startAcquireReleaseThread(ResourcePriority.DYNAMIC_STANDALONE);
    while (rm.getWaitCount() < 1) {
      Thread.yield();
    }
    final CyclicBarrier syncDynamicWorker1 =
        startAcquireReleaseThread(ResourcePriority.DYNAMIC_WORKER);
    while (rm.getWaitCount() < 2) {
      Thread.yield();
    }
    final CyclicBarrier syncDynamicStandalone2 =
        startAcquireReleaseThread(ResourcePriority.DYNAMIC_STANDALONE);
    while (rm.getWaitCount() < 3) {
      Thread.yield();
    }
    final CyclicBarrier syncDynamicWorker2 =
        startAcquireReleaseThread(ResourcePriority.DYNAMIC_WORKER);
    while (rm.getWaitCount() < 4) {
      Thread.yield();
    }

    // Wewease the kwaken!
    sync2.await();

    while (syncDynamicStandalone1.getNumberWaiting()
            + syncDynamicStandalone2.getNumberWaiting()
            + syncDynamicWorker1.getNumberWaiting()
            + syncDynamicWorker2.getNumberWaiting()
        == 0) {
      Thread.yield();
    }
    assertThat(rm.getWaitCount()).isEqualTo(3);
    assertThat(syncDynamicWorker2.getNumberWaiting()).isEqualTo(1);
    syncDynamicWorker2.await(1, TimeUnit.SECONDS);

    while (syncDynamicStandalone1.getNumberWaiting()
            + syncDynamicStandalone2.getNumberWaiting()
            + syncDynamicWorker1.getNumberWaiting()
        == 0) {
      Thread.yield();
    }
    assertThat(rm.getWaitCount()).isEqualTo(2);
    assertThat(syncDynamicWorker1.getNumberWaiting()).isEqualTo(1);
    syncDynamicWorker1.await(1, TimeUnit.SECONDS);

    while (syncDynamicStandalone1.getNumberWaiting() + syncDynamicStandalone2.getNumberWaiting()
        == 0) {
      Thread.yield();
    }
    assertThat(rm.getWaitCount()).isEqualTo(1);
    assertThat(syncDynamicStandalone2.getNumberWaiting()).isEqualTo(1);
    syncDynamicStandalone2.await(1, TimeUnit.SECONDS);

    while (syncDynamicStandalone1.getNumberWaiting() == 0) {
      Thread.yield();
    }
    assertThat(rm.getWaitCount()).isEqualTo(0);
    assertThat(syncDynamicStandalone1.getNumberWaiting()).isEqualTo(1);
    syncDynamicStandalone1.await(1, TimeUnit.SECONDS);
  }

  private CyclicBarrier startAcquireReleaseThread(ResourcePriority priority) {
    final CyclicBarrier sync = new CyclicBarrier(2);
    TestThread thread =
        new TestThread(
            () -> {
              acquire(700, 0, 0, priority);
              sync.await();
              release(700, 0, 0);
            });
    thread.start();
    return sync;
  }

  @Test
  public void testNonexistingResource() throws Exception {
    // If we try to use nonexisting resource we should return an error
    TestThread thread1 =
        new TestThread(
            () ->
                assertThrows(
                    NoSuchElementException.class,
                    () -> acquire(0, 0, ImmutableMap.of("nonexisting", 1.0), 0)));
    thread1.start();
    thread1.joinAndAssertState(1000);
  }

  @Test
  public void testAcquireWithWorker_acquireAndRelease() throws Exception {
    int memory = 100;
    when(worker.getWorkerKey()).thenReturn(createWorkerKey("dummy"));

    assertThat(rm.inUse()).isFalse();
    ResourceHandle handle = acquire(memory, 1, 0, "dummy");
    assertThat(rm.inUse()).isTrue();

    assertThat(handle.getWorker().getWorkerKey().getMnemonic()).isEqualTo("dummy");
    release(memory, 1, 0);
    // When that RAM is released,
    // Then Resource Manager will not be "in use":
    assertThat(rm.inUse()).isFalse();
  }

  @Test
  public void testInvalidateAndClose() throws IOException, InterruptedException {
    ResourceHandle handle;
    verify(workerStatus, times(0)).maybeUpdateStatus(any());

    handle = acquire(0, 0, 0, "dummy");
    handle.invalidateAndClose(new InterruptedException());
    verify(workerStatus).maybeUpdateStatus(Status.PENDING_KILL_DUE_TO_INTERRUPTED_EXCEPTION);

    handle = acquire(0, 0, 0, "dummy");
    handle.invalidateAndClose(new IOException());
    verify(workerStatus).maybeUpdateStatus(Status.PENDING_KILL_DUE_TO_IO_EXCEPTION);

    handle = acquire(0, 0, 0, "dummy");
    handle.invalidateAndClose(
        new UserExecException(
            FailureDetail.newBuilder()
                .setWorker(FailureDetails.Worker.newBuilder().setCode(Code.NO_RESPONSE))
                .build()));
    verify(workerStatus)
        .maybeUpdateStatus(Status.PENDING_KILL_DUE_TO_USER_EXEC_EXCEPTION, Code.NO_RESPONSE);

    handle = acquire(0, 0, 0, "dummy");
    handle.invalidateAndClose(null);
    verify(workerStatus).maybeUpdateStatus(Status.PENDING_KILL_DUE_TO_UNKNOWN);
  }

  synchronized boolean isAvailable(ResourceManager rm, double ram, double cpu, int localTestCount) {
    return rm.areResourcesAvailable(ResourceSet.create(ram, cpu, localTestCount));
  }

  private static class ResourceOwnerStub implements ActionExecutionMetadata {

    @Override
    @Nullable
    public String getProgressMessage() {
      throw new IllegalStateException();
    }

    @Override
    public ActionOwner getOwner() {
      throw new IllegalStateException();
    }

    @Override
    public boolean isShareable() {
      throw new IllegalStateException();
    }

    @Override
    public String prettyPrint() {
      throw new IllegalStateException();
    }

    @Override
    public String getMnemonic() {
      throw new IllegalStateException();
    }

    @Override
    public boolean inputsKnown() {
      throw new IllegalStateException();
    }

    @Override
    public boolean discoversInputs() {
      throw new IllegalStateException();
    }

    @Override
    public NestedSet<Artifact> getTools() {
      throw new IllegalStateException();
    }

    @Override
    public NestedSet<Artifact> getInputs() {
      throw new IllegalStateException();
    }

    @Override
    public NestedSet<Artifact> getSchedulingDependencies() {
      throw new IllegalStateException();
    }

    @Override
    public Collection<String> getClientEnvironmentVariables() {
      throw new IllegalStateException();
    }

    @Override
    public ImmutableSet<Artifact> getOutputs() {
      throw new IllegalStateException();
    }

    @Override
    public Artifact getPrimaryInput() {
      throw new IllegalStateException();
    }

    @Override
    public Artifact getPrimaryOutput() {
      throw new IllegalStateException();
    }

    @Override
    public NestedSet<Artifact> getMandatoryInputs() {
      throw new IllegalStateException();
    }

    @Override
    public NestedSet<Artifact> getInputFilesForExtraAction(
        ActionExecutionContext actionExecutionContext) {
      return NestedSetBuilder.emptySet(Order.STABLE_ORDER);
    }

    @Override
    public String getKey(
        ActionKeyContext actionKeyContext, @Nullable Artifact.ArtifactExpander artifactExpander) {
      throw new IllegalStateException();
    }

    @Override
    @Nullable
    public String describeKey() {
      throw new IllegalStateException();
    }

    @Override
    public String describe() {
      return "ResourceOwnerStubAction";
    }

    @Override
    public ImmutableSet<Artifact> getMandatoryOutputs() {
      return ImmutableSet.of();
    }

    @Override
    public MiddlemanType getActionType() {
      throw new IllegalStateException();
    }

    @Override
    public ImmutableMap<String, String> getExecProperties() {
      throw new IllegalStateException();
    }

    @Nullable
    @Override
    public PlatformInfo getExecutionPlatform() {
      throw new IllegalStateException();
    }
  }
}
