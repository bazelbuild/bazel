// Copyright 2021 The Bazel Authors. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
package com.google.devtools.build.lib.dynamic;

import static java.lang.Math.min;

import com.google.devtools.build.lib.concurrent.ThreadSafety.ThreadSafe;
import java.util.concurrent.Semaphore;
import java.util.concurrent.atomic.AtomicInteger;

/** A {@link Semaphore} that can be temporarily shrunk. */
@ThreadSafe
public class ShrinkableSemaphore extends Semaphore {
  /** How many permits the semaphore has in total. It cannot grow beyond this. */
  private final int totalPermits;
  /** How many units of load is considered 100% load. */
  private final int maxLoad;
  /**
   * The load factor that determines how we respond to high load. The closer to 0, the more we
   * shrink the number of permits under high load.
   */
  private final double loadFactor;
  /** How many permits the semaphore is currently shrunk by. */
  private AtomicInteger permitsPreviouslyShrunkBy = new AtomicInteger(0);

  /**
   * Create a new ShrinkableSemaphore.
   *
   * @param permits How many permits to allow maximally.
   * @param maxLoad How many units of load is considered 100% load.
   * @param loadFactor How much load to allow
   */
  public ShrinkableSemaphore(int permits, int maxLoad, double loadFactor) {
    super(permits);
    this.totalPermits = permits;
    this.maxLoad = maxLoad;
    this.loadFactor = loadFactor;
  }

  /**
   * Set how much to shrink the number of permits available from this semaphore, relative to the
   * total number of permits. The number of total available permits is rounded up and never goes
   * below 1. If given a higher {@code shrinkFactor} than previously, the number of permits may
   * increase again.
   */
  private void setShrinkFactor(double shrinkFactor) {
    int shrinkBy = min((int) (totalPermits * shrinkFactor), totalPermits - 1);
    int oldShrink = permitsPreviouslyShrunkBy.get();
    int reduction = shrinkBy - oldShrink;
    if (permitsPreviouslyShrunkBy.compareAndSet(oldShrink, shrinkBy)) {
      if (reduction > 0) {
        this.reducePermits(reduction);
      } else {
        this.release(-reduction);
      }
    }
  }

  /**
   * Calculates the "load factor" for adjusting the number of threads available for local execution.
   * When many actions are waiting to be locally executed, we (counter-intuitively) reduce the
   * number of threads we allow running locally. Having many actions waiting indicates a large
   * build, in which case remote execution/caching will handle the bulk of the actions, and using
   * more CPUs for local execution doesn't make much difference. So to make the machine more
   * responsive during large builds, we reduce the number of CPUs we allow using locally. This limit
   * does not take the number of local-only actions into account, those always need to be run
   * anyway.
   *
   * @param load The current load, relative to the {@code maxLoad} passed as constructor parameter.
   *     A load of >100% is allowed.
   */
  public void updateLoad(int load) {
    if (loadFactor > 0 && loadFactor < 1) {
      if (load < 0) {
        throw new IllegalArgumentException("Cannot have negative load");
      }
      setShrinkFactor((1 - loadFactor) * (((float) load) / maxLoad));
    }
  }
}
