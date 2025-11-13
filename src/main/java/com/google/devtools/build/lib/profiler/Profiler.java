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
package com.google.devtools.build.lib.profiler;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.Preconditions;
import com.google.devtools.build.lib.util.TestType;

/**
 * Static accessor for the {@link TraceProfilerService} instance.
 *
 * <p>This class provides the global access point to the profiler. The actual implementation is set
 * during the server startup.
 */
public final class Profiler {
  private static volatile TraceProfilerService instance = null;

  /**
   * Sets the global profiler instance. This is expected to be called during server initialization.
   */
  public static void setInstance(TraceProfilerService profiler) {
    Preconditions.checkState(instance == null, "The profiler is only meant to be set once.");
    instance = profiler;
  }

  /** Sets the global profiler instance for testing only. */
  @VisibleForTesting
  public static void forceSetInstanceForTestingOnly(TraceProfilerService profiler) {
    Preconditions.checkState(TestType.isInTest());
    instance = profiler;
  }

  /**
   * Returns the profiler instance.
   *
   * <p>Do not store the returned instance in class fields. Instead, access the profiler through
   * this method when needed.
   *
   * <p>Non-Bazel code may depend on Bazel code that transitively depends on the profiler without
   * ever starting it. In such cases, a no-op profiler will be returned. This no-op profiler will
   * crash if profiling is started by calling {@link TraceProfilerService#start}. If your code,
   * including test code, requires a {@link BlazeRuntime}, make sure that the TraceProfilerService
   * instance is set by calling {@link #setInstance} (or {@link #forceSetInstanceForTestingOnly}
   * when required by a test) before instantiating it.
   */
  public static TraceProfilerService instance() {
    if (instance == null) {
      instance = new NoOpTraceProfilerService();
    }
    return instance;
  }

  private Profiler() {}
}
