// Copyright 2015 Google Inc. All rights reserved.
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
import static org.junit.Assert.fail;

import com.google.common.collect.ImmutableSet;
import com.google.common.eventbus.EventBus;
import com.google.devtools.build.lib.testutil.TestThread;
import com.google.devtools.common.options.OptionsParsingException;

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
  public void setUp() throws Exception {
    rm.setRamUtilizationPercentage(100);
    rm.setAvailableResources(ResourceSet.createWithRamCpuIo(1000, 1, 1));
    rm.setEventBus(new EventBus());
    counter = new AtomicInteger(0);
    sync = new CyclicBarrier(2);
    sync2 = new CyclicBarrier(2);
    rm.resetResourceUsage();
  }

  private void acquire(double ram, double cpu, double io) throws InterruptedException {
    rm.acquireResources(resourceOwner, ResourceSet.createWithRamCpuIo(ram, cpu, io));
  }

  private boolean acquireNonblocking(double ram, double cpu, double io) {
    return rm.tryAcquire(resourceOwner, ResourceSet.createWithRamCpuIo(ram, cpu, io));
  }

  private void release(double ram, double cpu, double io) {
    rm.releaseResources(resourceOwner, ResourceSet.createWithRamCpuIo(ram, cpu, io));
  }

  private void validate (int count) {
    assertEquals(count, counter.incrementAndGet());
  }

  @Test
  public void testIndependentLargeRequests() throws Exception {
    // Available: 1000 RAM and 1 CPU.
    assertFalse(rm.inUse());
    acquire(10000, 0, 0); // Available: 0 RAM 1 CPU 1 IO.
    acquire(0, 100, 0);   // Available: 0 RAM 0 CPU 1 IO.
    acquire(0, 0, 1);     // Available: 0 RAM 0 CPU 0 IO.
    assertTrue(rm.inUse());
    release(9500, 0, 0);  // Available: 500 RAM 0 CPU 0 IO.
    acquire(400, 0, 0);   // Available: 100 RAM 0 CPU 0 IO.
    release(0, 99.5, 0.6);  // Available: 100 RAM 0.5 CPU 0.4 IO.
    acquire(100, 0.5, 0.4); // Available: 0 RAM 0 CPU 0 IO.
    release(1000, 1, 1);  // Available: 1000 RAM 1 CPU 1 IO.
    assertFalse(rm.inUse());
  }

  @Test
  public void testOverallocation() throws Exception {
    // Since ResourceManager.MIN_NECESSARY_RAM_RATIO = 1.0, overallocation is
    // enabled only for the CPU resource.
    assertFalse(rm.inUse());
    acquire(900, 0.5, 0.1);  // Available: 100 RAM 0.5 CPU 0.9 IO.
    acquire(100, 0.6, 0.9);  // Available: 0 RAM 0 CPU 0 IO.
    release(100, 0.6, 0.9);  // Available: 100 RAM 0.5 CPU 0.9 IO.
    acquire(100, 0.1, 0.1);  // Available: 0 RAM 0.4 CPU 0.8 IO.
    acquire(0, 0.5, 0.8);    // Available: 0 RAM 0 CPU 0.8 IO.
    release(1020, 1.1, 1.05); // Available: 1000 RAM 1 CPU 1 IO.
    assertFalse(rm.inUse());
  }

  @Test
  public void testNonblocking() throws Exception {
    assertFalse(rm.inUse());
    assertTrue(acquireNonblocking(900, 0.5, 0));  // Available: 100 RAM 0.5 CPU 1 IO.
    assertTrue(acquireNonblocking(100, 0.5, 0.2));  // Available: 0 RAM 0 CPU 0.8 IO.
    assertFalse(acquireNonblocking(.1, .01, 0.0));
    assertFalse(acquireNonblocking(0, 0, 0.9));
    assertTrue(acquireNonblocking(0, 0, 0.8));  // Available: 0 RAM 0 CPU 0 IO.
    release(100, 0.5, 0.1);  // Available: 100 RAM 0.5 CPU 0.1 IO.
    assertTrue(acquireNonblocking(100, 0.1, 0.1));  // Available: 0 RAM 0.4 CPU 0 IO.
    assertFalse(acquireNonblocking(5, .5, 0));
    assertFalse(acquireNonblocking(0, .5, 0.1));
    assertTrue(acquireNonblocking(0, 0.4, 0));    // Available: 0 RAM 0 CPU 0 IO.
    release(1000, 1, 1); // Available: 1000 RAM 1 CPU 1 IO.
    assertFalse(rm.inUse());
  }

  @Test
  public void testHasResources() throws Exception {
    assertFalse(rm.inUse());
    assertFalse(rm.threadHasResources());
    acquire(1, .1, .1);
    assertTrue(rm.threadHasResources());

    // We have resources in this thread - make sure other threads
    // are not affected.
    TestThread thread1 = new TestThread () {
      @Override public void runTest() throws Exception {
        assertFalse(rm.threadHasResources());
        acquire(1, 0, 0);
        assertTrue(rm.threadHasResources());
        release(1, 0, 0);
        assertFalse(rm.threadHasResources());
        acquire(0, 0.1, 0);
        assertTrue(rm.threadHasResources());
        release(0, 0.1, 0);
        assertFalse(rm.threadHasResources());
        acquire(0, 0, 0.1);
        assertTrue(rm.threadHasResources());
        release(0, 0, 0.1);
        assertFalse(rm.threadHasResources());
      }
    };
    thread1.start();
    thread1.joinAndAssertState(10000);

    release(1, .1, .1);
    assertFalse(rm.threadHasResources());
    assertFalse(rm.inUse());
  }

  @Test
  public void testConcurrentLargeRequests() throws Exception {
    assertFalse(rm.inUse());
    TestThread thread1 = new TestThread () {
      @Override public void runTest() throws Exception {
        acquire(2000, 2, 0);
        sync.await();
        validate(1);
        sync.await();
        // Wait till other thread will be locked.
        while (rm.getWaitCount() == 0) {
          Thread.yield();
        }
        release(2000, 2, 0);
        assertEquals(0, rm.getWaitCount());
        acquire(2000, 2, 0); // Will be blocked by the thread2.
        validate(3);
        release(2000, 2, 0);
      }
    };
    TestThread thread2 = new TestThread () {
      @Override public void runTest() throws Exception {
        sync2.await();
        assertFalse(rm.isAvailable(2000, 2, 0));
        acquire(2000, 2, 0); // Will be blocked by the thread1.
        validate(2);
        sync2.await();
        // Wait till other thread will be locked.
        while (rm.getWaitCount() == 0) {
          Thread.yield();
        }
        release(2000, 2, 0);
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
        acquire(900, 0.5, 0); // Will be blocked by the main thread.
        validate(5);
        release(900, 0.5, 0);
        sync.await();
      }
    };
    TestThread thread2 = new TestThread() {
      @Override public void runTest() throws Exception {
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
      }
    };
    acquire(900, 0.9, 0);
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
    release(100, 0.4, 0); // This allows Thread2 to continue out of order.
    sync2.await(1, TimeUnit.SECONDS);
    release(750, 0.3, 0); // At this point thread1 will finally acquire resources.
    sync.await(1, TimeUnit.SECONDS);
    release(50, 0.2, 0);
    thread1.join();
    thread2.join();
    assertFalse(rm.inUse());
  }

  @Test
  public void testSingleton() throws Exception {
    ResourceManager.instance();
  }

  /**
   * Checks that that resource manager
   * can recover from LocalHostCapacity.getFreeResources() failure.
   */
  @Test
  public void testAutoSenseFailure() throws Exception {
    boolean isDisabled = LocalHostCapacity.isDisabled;
    assertFalse(rm.inUse());
    try {
      rm.setAutoSensing(true);
      // Resource manager autosense state should be enabled now if
      // LocalHostCapacity class supports it.
      assertEquals(rm.isAutoSensingEnabled(), !LocalHostCapacity.isDisabled);
      rm.setAutoSensing(false);
      assertFalse(rm.isAutoSensingEnabled());

      // Emulate failure to parse /proc/* filesystem.
      LocalHostCapacity.isDisabled = true;
      rm.setAutoSensing(true);
      assertFalse(rm.isAutoSensingEnabled());
      rm.setAutoSensing(false);
      assertFalse(rm.isAutoSensingEnabled());
    } finally {
      LocalHostCapacity.isDisabled = isDisabled;
      rm.setAutoSensing(false);
    }
    assertFalse(rm.inUse());
  }

  @Test
  public void testResourceSetConverter() throws Exception {
    ResourceSet.ResourceSetConverter converter = new ResourceSet.ResourceSetConverter();

    ResourceSet resources = converter.convert("1,0.5,2");
    assertEquals(1.0, resources.getMemoryMb(), 0.01);
    assertEquals(0.5, resources.getCpuUsage(), 0.01);
    assertEquals(2.0, resources.getIoUsage(), 0.01);

    try {
      converter.convert("0,0,");
      fail();
    } catch (OptionsParsingException ope) {
      // expected
    }

    try {
      converter.convert("0,0,0,0");
      fail();
    } catch (OptionsParsingException ope) {
      // expected
    }

    try {
      converter.convert("-1,0,0");
      fail();
    } catch (OptionsParsingException ope) {
      // expected
    }
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
    public String describeStrategy(Executor executor) {
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
    public Iterable<Artifact> getInputs() {
      throw new IllegalStateException();
    }

    @Override
    public int getInputCount() {
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
  }
}
