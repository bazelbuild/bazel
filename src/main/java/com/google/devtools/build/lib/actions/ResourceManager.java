// Copyright 2014 The Bazel Authors. All rights reserved.
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

import static com.google.devtools.build.lib.profiler.AutoProfiler.profiled;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.eventbus.EventBus;
import com.google.devtools.build.lib.concurrent.ThreadSafety.ThreadSafe;
import com.google.devtools.build.lib.profiler.AutoProfiler;
import com.google.devtools.build.lib.profiler.ProfilerTask;
import com.google.devtools.build.lib.util.Pair;
import com.google.devtools.build.lib.util.Preconditions;

import java.util.Iterator;
import java.util.LinkedList;
import java.util.List;
import java.util.concurrent.CountDownLatch;

/**
 * Used to keep track of resources consumed by the Blaze action execution threads and throttle them
 * when necessary.
 *
 * <p>Threads which are known to consume a significant amount of resources should call
 * {@link #acquireResources} method. This method will check whether requested resources are
 * available and will either mark them as used and allow the thread to proceed or will block the
 * thread until requested resources will become available. When the thread completes its task, it
 * must release allocated resources by calling {@link #releaseResources} method.
 *
 * <p>Available resources can be calculated using one of three ways:
 * <ol>
 * <li>They can be preset using {@link #setAvailableResources(ResourceSet)} method. This is used
 *     mainly by the unit tests (however it is possible to provide a future option that would
 *     artificially limit amount of CPU/RAM consumed by the Blaze).
 * <li>They can be preset based on the /proc/cpuinfo and /proc/meminfo information. Blaze will
 *     calculate amount of available CPU cores (adjusting for hyperthreading logical cores) and
 *     amount of the total available memory and will limit itself to the number of effective cores
 *     and 2/3 of the available memory. For details, please look at the {@link
 *     LocalHostCapacity#getLocalHostCapacity} method.
 * </ol>
 *
 * <p>The resource manager also allows a slight overallocation of the resources to account for the
 * fact that requested resources are usually estimated using a pessimistic approximation. It also
 * guarantees that at least one thread will always be able to acquire any amount of requested
 * resources (even if it is greater than amount of available resources). Therefore, assuming that
 * threads correctly release acquired resources, Blaze will never be fully blocked.
 */
@ThreadSafe
public class ResourceManager {

  private EventBus eventBus;

  private final ThreadLocal<Boolean> threadLocked = new ThreadLocal<Boolean>() {
    @Override
    protected Boolean initialValue() {
      return false;
    }
  };

  /**
   * Singleton reference defined in a separate class to ensure thread-safe lazy
   * initialization.
   */
  private static class Singleton {
    static ResourceManager instance = new ResourceManager();
  }

  /**
   * Returns singleton instance of the resource manager.
   */
  public static ResourceManager instance() {
    return Singleton.instance;
  }

  // Allocated resources are allowed to go "negative", but at least
  // MIN_AVAILABLE_CPU_RATIO portion of CPU and MIN_AVAILABLE_RAM_RATIO portion
  // of RAM should be available.
  // Please note that this value is purely empirical - we assume that generally
  // requested resources are somewhat pessimistic and thread would end up
  // using less than requested amount.
  private static final double MIN_NECESSARY_CPU_RATIO = 0.6;
  private static final double MIN_NECESSARY_RAM_RATIO = 1.0;
  private static final double MIN_NECESSARY_IO_RATIO = 1.0;

  // List of blocked threads. Associated CountDownLatch object will always
  // be initialized to 1 during creation in the acquire() method.
  private final List<Pair<ResourceSet, CountDownLatch>> requestList;

  // The total amount of resources on the local host. Must be set by
  // an explicit call to setAvailableResources(), often using
  // LocalHostCapacity.getLocalHostCapacity() as an argument.
  private ResourceSet staticResources = null;

  private ResourceSet availableResources = null;

  // Used amount of CPU capacity (where 1.0 corresponds to the one fully
  // occupied CPU core. Corresponds to the CPU resource definition in the
  // ResourceSet class.
  private double usedCpu;

  // Used amount of RAM capacity in MB. Corresponds to the RAM resource
  // definition in the ResourceSet class.
  private double usedRam;

  // Used amount of I/O resources. Corresponds to the I/O resource
  // definition in the ResourceSet class.
  private double usedIo;

  // Used local test count. Corresponds to the local test count definition in the ResourceSet class.
  private int usedLocalTestCount;

  // Specifies how much of the RAM in staticResources we should allow to be used.
  public static final int DEFAULT_RAM_UTILIZATION_PERCENTAGE = 67;
  private int ramUtilizationPercentage = DEFAULT_RAM_UTILIZATION_PERCENTAGE;

  private ResourceManager() {
    requestList = new LinkedList<>();
  }

  @VisibleForTesting public static ResourceManager instanceForTestingOnly() {
    return new ResourceManager();
  }

  /**
   * Resets resource manager state and releases all thread locks.
   * Note - it does not reset available resources. Use
   * separate call to setAvailableResoures().
   */
  public synchronized void resetResourceUsage() {
    usedCpu = 0;
    usedRam = 0;
    usedIo = 0;
    usedLocalTestCount = 0;
    for (Pair<ResourceSet, CountDownLatch> request : requestList) {
      // CountDownLatch can be set only to 0 or 1.
      request.second.countDown();
    }
    requestList.clear();
  }

  /**
   * Sets available resources using given resource set. Must be called
   * at least once before using resource manager.
   */
  public synchronized void setAvailableResources(ResourceSet resources) {
    Preconditions.checkNotNull(resources);
    staticResources = resources;
    availableResources = ResourceSet.create(
        staticResources.getMemoryMb() * this.ramUtilizationPercentage / 100.0,
        staticResources.getCpuUsage(),
        staticResources.getIoUsage(),
        staticResources.getLocalTestCount());
    processWaitingThreads();
  }

  /**
   * Specify how much of the available RAM we should allow to be used.
   * This has no effect if autosensing is enabled.
   */
  public synchronized void setRamUtilizationPercentage(int percentage) {
    ramUtilizationPercentage = percentage;
  }

  /**
   * Acquires requested resource set. Will block if resource is not available.
   * NB! This method must be thread-safe!
   */
  public void acquireResources(ActionMetadata owner, ResourceSet resources)
      throws InterruptedException {
    Preconditions.checkNotNull(resources);
    AutoProfiler p = profiled(owner, ProfilerTask.ACTION_LOCK);
    CountDownLatch latch = null;
    try {
      waiting(owner);
      latch = acquire(resources);
      if (latch != null) {
        latch.await();
      }
    } finally {
      threadLocked.set(resources.getCpuUsage() != 0 || resources.getMemoryMb() != 0
          || resources.getIoUsage() != 0 || resources.getLocalTestCount() != 0);
      acquired(owner);

      // Profile acquisition only if it waited for resource to become available.
      if (latch != null) {
        p.complete();
      }
    }
  }

  /**
   * Acquires the given resources if available immediately. Does not block.
   * @return true iff the given resources were locked (all or nothing).
   */
  public boolean tryAcquire(ActionMetadata owner, ResourceSet resources) {
    boolean acquired = false;
    synchronized (this) {
      if (areResourcesAvailable(resources)) {
        incrementResources(resources);
        acquired = true;
      }
    }

    if (acquired) {
      threadLocked.set(resources.getCpuUsage() != 0 || resources.getMemoryMb() != 0
          || resources.getIoUsage() != 0 || resources.getLocalTestCount() != 0);
      acquired(owner);
    }

    return acquired;
  }

  private void incrementResources(ResourceSet resources) {
    usedCpu += resources.getCpuUsage();
    usedRam += resources.getMemoryMb();
    usedIo += resources.getIoUsage();
    usedLocalTestCount += resources.getLocalTestCount();
  }

  /**
   * Return true if any resources have been claimed through this manager.
   */
  public synchronized boolean inUse() {
    return usedCpu != 0.0 || usedRam != 0.0 || usedIo != 0.0 || usedLocalTestCount != 0
        || !requestList.isEmpty();
  }


  /**
   * Return true iff this thread has a lock on non-zero resources.
   */
  public boolean threadHasResources() {
    return threadLocked.get();
  }

  public void setEventBus(EventBus eventBus) {
    Preconditions.checkState(this.eventBus == null);
    this.eventBus = Preconditions.checkNotNull(eventBus);
  }

  public void unsetEventBus() {
    Preconditions.checkState(this.eventBus != null);
    this.eventBus = null;
  }

  private void waiting(ActionMetadata owner) {
    if (eventBus != null) {
      // Null only in tests.
      eventBus.post(ActionStatusMessage.schedulingStrategy(owner));
    }
  }

  private void acquired(ActionMetadata owner) {
    if (eventBus != null) {
      // Null only in tests.
      eventBus.post(ActionStatusMessage.runningStrategy(owner, "unknown"));
    }
  }

  /**
   * Releases previously requested resource =.
   *
   * <p>NB! This method must be thread-safe!
   */
  public void releaseResources(ActionMetadata owner, ResourceSet resources) {
    boolean isConflict = false;
    AutoProfiler p = profiled(owner, ProfilerTask.ACTION_RELEASE);
    try {
      isConflict = release(resources);
    } finally {
      threadLocked.set(false);

      // Profile resource release only if it resolved at least one allocation request.
      if (isConflict) {
        p.complete();
      }
    }
  }

  private synchronized CountDownLatch acquire(ResourceSet resources) {
    if (areResourcesAvailable(resources)) {
      incrementResources(resources);
      return null;
    }
    Pair<ResourceSet, CountDownLatch> request =
      new Pair<>(resources, new CountDownLatch(1));
    requestList.add(request);
    return request.second;
  }

  private synchronized boolean release(ResourceSet resources) {
    usedCpu -= resources.getCpuUsage();
    usedRam -= resources.getMemoryMb();
    usedIo -= resources.getIoUsage();
    usedLocalTestCount -= resources.getLocalTestCount();

    // TODO(bazel-team): (2010) rounding error can accumulate and value below can end up being
    // e.g. 1E-15. So if it is small enough, we set it to 0. But maybe there is a better solution.
    double epsilon = 0.0001;
    if (usedCpu < epsilon) {
      usedCpu = 0;
    }
    if (usedRam < epsilon) {
      usedRam = 0;
    }
    if (usedIo < epsilon) {
      usedIo = 0;
    }
    if (!requestList.isEmpty()) {
      processWaitingThreads();
      return true;
    }
    return false;
  }


  /**
   * Tries to unblock one or more waiting threads if there are sufficient resources available.
   */
  private synchronized void processWaitingThreads() {
    Iterator<Pair<ResourceSet, CountDownLatch>> iterator = requestList.iterator();
    while (iterator.hasNext()) {
      Pair<ResourceSet, CountDownLatch> request = iterator.next();
      if (areResourcesAvailable(request.first)) {
        incrementResources(request.first);
        request.second.countDown();
        iterator.remove();
      }
    }
  }

  // Method will return true if all requested resources are considered to be available.
  private boolean areResourcesAvailable(ResourceSet resources) {
    Preconditions.checkNotNull(availableResources);
    // Comparison below is robust, since any calculation errors will be fixed
    // by the release() method.
    if (usedCpu == 0.0 && usedRam == 0.0 && usedIo == 0.0 && usedLocalTestCount == 0) {
      return true;
    }
    // Use only MIN_NECESSARY_???_RATIO of the resource value to check for
    // allocation. This is necessary to account for the fact that most of the
    // requested resource sets use pessimistic estimations. Note that this
    // ratio is used only during comparison - for tracking we will actually
    // mark whole requested amount as used.
    double cpu = resources.getCpuUsage() * MIN_NECESSARY_CPU_RATIO;
    double ram = resources.getMemoryMb() * MIN_NECESSARY_RAM_RATIO;
    double io = resources.getIoUsage() * MIN_NECESSARY_IO_RATIO;
    int localTestCount = resources.getLocalTestCount();

    double availableCpu = availableResources.getCpuUsage();
    double availableRam = availableResources.getMemoryMb();
    double availableIo = availableResources.getIoUsage();
    int availableLocalTestCount = availableResources.getLocalTestCount();

    // Resources are considered available if any one of the conditions below is true:
    // 1) If resource is not requested at all, it is available.
    // 2) If resource is not used at the moment, it is considered to be
    // available regardless of how much is requested. This is necessary to
    // ensure that at any given time, at least one thread is able to acquire
    // resources even if it requests more than available.
    // 3) If used resource amount is less than total available resource amount.
    boolean cpuIsAvailable = cpu == 0.0 || usedCpu == 0.0 || usedCpu + cpu <= availableCpu;
    boolean ramIsAvailable = ram == 0.0 || usedRam == 0.0 || usedRam + ram <= availableRam;
    boolean ioIsAvailable = io == 0.0 || usedIo == 0.0 || usedIo + io <= availableIo;
    boolean localTestCountIsAvailable = localTestCount == 0 || usedLocalTestCount == 0
        || usedLocalTestCount + localTestCount <= availableLocalTestCount;
    return cpuIsAvailable && ramIsAvailable && ioIsAvailable && localTestCountIsAvailable;
  }


  @VisibleForTesting
  synchronized int getWaitCount() {
    return requestList.size();
  }

  @VisibleForTesting
  synchronized boolean isAvailable(double ram, double cpu, double io, int localTestCount) {
    return areResourcesAvailable(ResourceSet.create(ram, cpu, io, localTestCount));
  }
}
