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

package com.google.devtools.build.lib.profiler.memory;

import com.google.devtools.build.lib.skybridge.ScOnly;

/** The {@link AllocationTrackerService} implementation. */
@ScOnly
public final class AllocationTrackerServiceImpl implements AllocationTrackerService {

  @Override
  public boolean isSupported() {
    try {
      Class.forName("com.google.monitoring.runtime.instrumentation.Sampler");
      return true;
    } catch (ClassNotFoundException e) {
      return false;
    }
  }

  @Override
  public void installAllocationTracker(AllocationSampler sampler) {
    if (!isSupported()) {
      throw new UnsupportedOperationException("allocation tracking is not supported");
    }
    AllocationTrackerInstaller.installAllocationTracker(sampler::sampleAllocation);
  }
}
