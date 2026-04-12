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

import com.google.devtools.build.lib.jni.JniLoader;
import java.util.function.IntConsumer;

/** Implementation of {@link PlatformNativeDepsService}. */
public class PlatformNativeDepsServiceImpl implements PlatformNativeDepsService {
  static {
    JniLoader.loadJni();
  }

  @Override
  public int pushDisableSleep() {
    return pushDisableSleepNative();
  }

  private native int pushDisableSleepNative();

  @Override
  public int popDisableSleep() {
    return popDisableSleepNative();
  }

  private native int popDisableSleepNative();

  @Override
  public void registerCPUSpeedJni(IntConsumer callback) {
    registerCPUSpeedNative(callback);
  }

  private native void registerCPUSpeedNative(IntConsumer callback);

  @Override
  public int cpuSpeed() {
    return cpuSpeedNative();
  }

  private native int cpuSpeedNative();

  @Override
  public void registerDiskSpaceJni(IntConsumer callback) {
    registerDiskSpaceNative(callback);
  }

  private native void registerDiskSpaceNative(IntConsumer callback);

  @Override
  public void registerLoadAdvisoryJni(IntConsumer callback) {
    registerLoadAdvisoryNative(callback);
  }

  private native void registerLoadAdvisoryNative(IntConsumer callback);

  @Override
  public int systemLoadAdvisory() {
    return systemLoadAdvisoryNative();
  }

  private native int systemLoadAdvisoryNative();

  @Override
  public void registerMemoryPressureJni(IntConsumer callback) {
    registerMemoryPressureNative(callback);
  }

  private native void registerMemoryPressureNative(IntConsumer callback);

  @Override
  public int systemMemoryPressure() {
    return systemMemoryPressureNative();
  }

  private native int systemMemoryPressureNative();

  @Override
  public void registerSuspensionJni(IntConsumer callback) {
    registerSuspensionNative(callback);
  }

  private native void registerSuspensionNative(IntConsumer callback);

  @Override
  public void registerThermalJni(IntConsumer callback) {
    registerThermalNative(callback);
  }

  private native void registerThermalNative(IntConsumer callback);

  @Override
  public int thermalLoad() {
    return thermalLoadNative();
  }

  private native int thermalLoadNative();
}
