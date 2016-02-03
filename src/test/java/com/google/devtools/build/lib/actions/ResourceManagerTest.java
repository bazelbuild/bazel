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


import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

import com.google.common.collect.ImmutableSet;
import com.google.common.eventbus.EventBus;
import com.google.devtools.build.lib.testutil.TestThread;

import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

import java.util.concurrent.CyclicBarrier;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;

import javax.annotation.Nullable;

/**
 *
 * Tests for @{link ResourceManager}.
 */
@RunWith(JUnit4.class)
public class ResourceManagerTest {

  private final ActionMetadata resourceOwner = new ResourceOwnerStub();
  private final ResourceManager rm = ResourceManager.instanceForTestingOnly();
  private AtomicInteger counter;
  CyclicBarrier sync;
  CyclicBarrier sync2;

  @Before
  public final void configureResourceManager() throws Exception  {
    rm.setRamUtilizationPercentage(100);
    rm.setAvailableResources(
        ResourceSet.create(/*memoryMb=*/1000.0, /*cpuUsage=*/1.0, /*ioUsage=*/1.0,
        /*testCount=*/2));
    rm.setEventBus(new EventBus());
    counter = new AtomicInteger(0);
    sync = new CyclicBarrier(2);
    sync2 = new CyclicBarrier(2);
    rm.resetResourceUsage();
  }

  private void acquire(double ram, double cpu, double io, int tests)
      throws InterruptedException {
    rm.acquireResources(resourceOwner, ResourceSet.create(ram, cpu, io, tests));
  }

  private boolean acquireNonblocking(double ram, double cpu, double io, int tests) {
    return rm.tryAcquire(resourceOwner, ResourceSet.create(ram, cpu, io, tests));
  }

  private void release(double ram, double cpu, double io, int tests) {
    rm.releaseResources(resourceOwner, ResourceSet.create(ram, cpu, io, tests));
  }

  private void validate (int count) {
    assertEquals(count, counter.incrementAndGet());
  }

  @Test
  public void testOverBudgetRequests() throws Exception {
    assertFalse(rm.inUse());

    // When nothing is consuming RAM,
    // Then Resource Manager will successfully acquire an over-budget request for RAM:
    double bigRam = 10000.0;
    acquire(bigRam, 0, 0, 0);
    // When RAM is consumed,
    // Then Resource Manager will be "in use":
    assertTrue(rm.inUse());
    release(bigRam, 0, 0, 0);
    // When that RAM is released,
    // Then Resource Manager will not be "in use":
    assertFalse(rm.inUse());

    // Ditto, for CPU:
    double bigCpu = 10.0;
    acquire(0, bigCpu, 0, 0);
    assertTrue(rm.inUse());
    release(0, bigCpu, 0, 0);
    assertFalse(rm.inUse());

    // Ditto, for IO:
    double bigIo = 10.0;
    acquire(0, 0, bigIo, 0);
    assertTrue(rm.inUse());
    release(0, 0, bigIo, 0);
    assertFalse(rm.inUse());

    // Ditto, for tests:
    int bigTests = 10;
    acquire(0, 0, 0, bigTests);
    assertTrue(rm.inUse());
    release(0, 0, 0, bigTests);
    assertFalse(rm.inUse());
  }

  @Test
  public void testThatCpuCanBeOverallocated() throws Exception {
    assertFalse(rm.inUse());

    // Given CPU is partially acquired:
    acquire(0, 0.5, 0, 0);

    // When a request for CPU is made that would slightly overallocate CPU,
    // Then the request succeeds:
    assertTrue(acquireNonblocking(0, 0.6, 0, 0));
  }

  @Test
  public void testThatCpuAllocationIsNoncommutative() throws Exception {
    assertFalse(rm.inUse());

    // Given that CPU has a small initial allocation:
    acquire(0, 0.099, 0, 0);

    // When a request for a large CPU allocation is made,
    // Then the request succeeds:
    assertTrue(acquireNonblocking(0, 0.99, 0, 0));

    // Cleanup
    release(0, 1.089, 0, 0);
    assertFalse(rm.inUse());


    // Given that CPU has a large initial allocation:
    acquire(0, 0.99, 0, 0);

    // When a request for a small CPU allocation is made,
    // Then the request fails:
    assertFalse(acquireNonblocking(0, 0.099, 0, 0));

    // Note that this behavior is surprising and probably not intended.
  }

  @Test
  public void testThatRamCannotBeOverallocated() throws Exception {
    assertFalse(rm.inUse());

    // Given RAM is partially acquired:
    acquire(500, 0, 0, 0);

    // When a request for RAM is made that would slightly overallocate RAM,
    // Then the request fails:
    assertFalse(acquireNonblocking(600, 0, 0, 0));
  }

  @Test
  public void testThatIOCannotBeOverallocated() throws Exception {
    assertFalse(rm.inUse());

    // Given IO is partially acquired:
    acquire(0, 0, 0.5, 0);

    // When a request for IO is made that would slightly overallocate IO,
    // Then the request fails:
    assertFalse(acquireNonblocking(0, 0, 0.6, 0));
  }

  @Test
  public void testThatTestsCannotBeOverallocated() throws Exception {
    assertFalse(rm.inUse());

    // Given test count is partially acquired:
    acquire(0, 0, 0, 1);

    // When a request for tests is made that would slightly overallocate tests,
    // Then the request fails:
    assertFalse(acquireNonblocking(0, 0, 0, 2));
  }

  @Test
  public void testHasResources() throws Exception {
    assertFalse(rm.inUse());
    assertFalse(rm.threadHasResources());
    acquire(1.0, 0.1, 0.1, 1);
    assertTrue(rm.threadHasResources());

    // We have resources in this thread - make sure other threads
    // are not affected.
    TestThread thread1 = new TestThread () {
      @Override public void runTest() throws Exception {
        assertFalse(rm.threadHasResources());
        acquire(1.0, 0, 0, 0);
        assertTrue(rm.threadHasResources());
        release(1.0, 0, 0, 0);
        assertFalse(rm.threadHasResources());
        acquire(0, 0.1, 0, 0);
        assertTrue(rm.threadHasResources());
        release(0, 0.1, 0, 0);
        assertFalse(rm.threadHasResources());
        acquire(0, 0, 0.1, 0);
        assertTrue(rm.threadHasResources());
        release(0, 0, 0.1, 0);
        assertFalse(rm.threadHasResources());
        acquire(0, 0, 0, 1);
        assertTrue(rm.threadHasResources());
        release(0, 0, 0, 1);
        assertFalse(rm.threadHasResources());
      }
    };
    thread1.start();
    thread1.joinAndAssertState(10000);

    release(1.0, 0.1, 0.1, 1);
    assertFalse(rm.threadHasResources());
    assertFalse(rm.inUse());
  }

  @Test
  public void testConcurrentLargeRequests() throws Exception {
    assertFalse(rm.inUse());
    TestThread thread1 = new TestThread () {
      @Override public void runTest() throws Exception {
        acquire(2000, 2, 0, 0);
        sync.await();
        validate(1);
        sync.await();
        // Wait till other thread will be locked.
        while (rm.getWaitCount() == 0) {
          Thread.yield();
        }
        release(2000, 2, 0, 0);
        assertEquals(0, rm.getWaitCount());
        acquire(2000, 2, 0, 0); // Will be blocked by the thread2.
        validate(3);
        release(2000, 2, 0, 0);
      }
    };
    TestThread thread2 = new TestThread () {
      @Override public void runTest() throws Exception {
        sync2.await();
        assertFalse(rm.isAvailable(2000, 2, 0, 0));
        acquire(2000, 2, 0, 0); // Will be blocked by the thread1.
        validate(2);
        sync2.await();
        // Wait till other thread will be locked.
        while (rm.getWaitCount() == 0) {
          Thread.yield();
        }
        release(2000, 2, 0, 0);
      }
    };

    thread1.start();
    thread2.start();
    sync.await(1, TimeUnit.SECONDS);
    assertTrue(rm.inUse());
    assertEquals(0, rm.getWaitCount());
    sync2.await(1, TimeUnit.SECONDS);
    sync.await(1, TimeUnit.SECONDS);
    sync2.await(1, TimeUnit.SECONDS);
    thread1.joinAndAssertState(1000);
    thread2.joinAndAssertState(1000);
    assertFalse(rm.inUse());
  }

  @Test
  public void testOutOfOrderAllocation() throws Exception {
    assertFalse(rm.inUse());
    TestThread thread1 = new TestThread () {
      @Override public void runTest() throws Exception {
        sync.await();
        acquire(900, 0.5, 0, 0); // Will be blocked by the main thread.
        validate(5);
        release(900, 0.5, 0, 0);
        sync.await();
      }
    };
    TestThread thread2 = new TestThread() {
      @Override public void runTest() throws Exception {
        // Wait till other thread will be locked
        while (rm.getWaitCount() == 0) {
          Thread.yield();
        }
        acquire(100, 0.1, 0, 0);
        validate(2);
        release(100, 0.1, 0, 0);
        sync2.await();
        acquire(200, 0.5, 0, 0);
        validate(4);
        sync2.await();
        release(200, 0.5, 0, 0);
      }
    };
    acquire(900, 0.9, 0, 0);
    validate(1);
    thread1.start();
    sync.await(1, TimeUnit.SECONDS);
    thread2.start();
    sync2.await(1, TimeUnit.SECONDS);
    //Waiting till both threads are locked.
    while (rm.getWaitCount() < 2) {
      Thread.yield();
    }
    validate(3); // Thread1 is now first in the queue and Thread2 is second.
    release(100, 0.4, 0, 0); // This allows Thread2 to continue out of order.
    sync2.await(1, TimeUnit.SECONDS);
    release(750, 0.3, 0, 0); // At this point thread1 will finally acquire resources.
    sync.await(1, TimeUnit.SECONDS);
    release(50, 0.2, 0, 0);
    thread1.join();
    thread2.join();
    assertFalse(rm.inUse());
  }

  private static class ResourceOwnerStub implements ActionMetadata {

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
    public Iterable<Artifact> getTools() {
      throw new IllegalStateException();
    }

    @Override
    public Iterable<Artifact> getInputs() {
      throw new IllegalStateException();
    }

    @Override
    public RunfilesSupplier getRunfilesSupplier() {
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
    public Iterable<Artifact> getMandatoryInputs() {
      throw new IllegalStateException();
    }

    @Override
    public String getKey() {
      throw new IllegalStateException();
    }

    @Override
    @Nullable
    public String describeKey() {
      throw new IllegalStateException();
    }

    @Override
    public ImmutableSet<Artifact> getMandatoryOutputs() {
      return ImmutableSet.of();
    }
  }
}
