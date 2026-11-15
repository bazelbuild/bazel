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
package com.google.devtools.build.lib.skyframe.serialization.analysis;

import com.google.devtools.build.lib.skybridge.SkybridgeInterface;
import java.util.concurrent.atomic.AtomicLong;

/**
 * Coordinates non-blocking load shedding and channel state awareness across Skycache stores based
 * on an aggregate in-flight request capacity threshold derived from the Bandwidth-Delay Product
 * (BDP).
 */
@SkybridgeInterface
public class SkycacheChannelStateAdvisor {
  public static final SkycacheChannelStateAdvisor DISABLED =
      new SkycacheChannelStateAdvisor(/* maxInFlightRequests= */ 0) {
        @Override
        public boolean isSaturated() {
          return false;
        }

        @Override
        public void incrementInFlightRequests() {}

        @Override
        public void decrementInFlightRequests(long delta) {}

        @Override
        public long getInFlightRequests() {
          return 0;
        }

        @Override
        public boolean isSchedulingHashesExhausted() {
          return false;
        }

        @Override
        public void markSchedulingHashesExhausted() {}
      };

  private final AtomicLong inFlightRequests = new AtomicLong(0);
  private final long maxInFlightRequests;
  private volatile boolean schedulingHashesExhausted = false;

  public SkycacheChannelStateAdvisor(long maxInFlightRequests) {
    this.maxInFlightRequests = maxInFlightRequests;
  }

  public boolean isSaturated() {
    if (maxInFlightRequests <= 0) {
      return false;
    }
    return inFlightRequests.get() >= maxInFlightRequests;
  }

  public void incrementInFlightRequests() {
    inFlightRequests.incrementAndGet();
  }

  public void decrementInFlightRequests(long delta) {
    inFlightRequests.addAndGet(-delta);
  }

  public long getInFlightRequests() {
    return inFlightRequests.get();
  }

  /**
   * Returns true if backend scheduling hashes have been exhausted, indicating that no active
   * backend storage proxy or ACS instance is reachable for remote analysis cache retrieval.
   */
  public boolean isSchedulingHashesExhausted() {
    return schedulingHashesExhausted;
  }

  /** Marks backend scheduling hashes as exhausted, shedding upstream retrieval attempts. */
  public void markSchedulingHashesExhausted() {
    this.schedulingHashesExhausted = true;
  }
}
