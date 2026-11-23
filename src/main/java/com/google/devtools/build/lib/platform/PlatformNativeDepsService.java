// Copyright 2024 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.platform;

import com.google.devtools.build.lib.runtime.BlazeService;
import com.google.errorprone.annotations.CanIgnoreReturnValue;
import java.util.function.IntConsumer;

/** Service interface for platform-specific native dependencies. */
public interface PlatformNativeDepsService extends BlazeService {
  /**
   * Push a request to disable automatic sleep for hardware. Useful for making sure computers don't
   * go to sleep during long builds. Must be matched with a {@link #popDisableSleep} call.
   *
   * @return 0 on success, -1 if sleep is not supported.
   */
  @CanIgnoreReturnValue
  int pushDisableSleep();

  /**
   * Pop a request to disable automatic sleep for hardware. Useful for making sure computers don't
   * go to sleep during long builds. Must be matched with a previous {@link #pushDisableSleep} call.
   *
   * @return 0 on success, -1 if sleep is not supported.
   */
  @CanIgnoreReturnValue
  int popDisableSleep();

  /** Registers the JNI callbacks for the CPU speed module. */
  void registerCPUSpeedJni(IntConsumer callback);

  /**
   * Returns the current CPU speed as a percentage.
   *
   * @return 1-100 to represent CPU speed. Returns -1 in case of error.
   */
  int cpuSpeed();

  /** Registers the JNI callbacks for the disk space module. */
  void registerDiskSpaceJni(IntConsumer callback);

  /** Registers the JNI callbacks for the load advisory module. */
  void registerLoadAdvisoryJni(IntConsumer callback);

  /** Returns the system load advisory. */
  int systemLoadAdvisory();

  /** Registers the JNI callbacks for the memory pressure monitor. */
  void registerMemoryPressureJni(IntConsumer callback);

  /** Returns the current memory pressure. */
  int systemMemoryPressure();

  /** Registers the JNI callbacks for the suspension module. */
  void registerSuspensionJni(IntConsumer callback);

  /** Registers the JNI callbacks for the thermal module. */
  void registerThermalJni(IntConsumer callback);

  /** Returns the current thermal load. */
  int thermalLoad();
}
