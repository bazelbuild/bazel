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
    if (JniLoader.isJniAvailable()) {
      return pushDisableSleepNative();
    }

    return -1;
  }

  private native int pushDisableSleepNative();

  @Override
  public int popDisableSleep() {
    if (JniLoader.isJniAvailable()) {
      return popDisableSleepNative();
    }
    return -1;
  }

  private native int popDisableSleepNative();

  @Override
  public void registerCPUSpeedJni(IntConsumer callback) {
    if (JniLoader.isJniAvailable()) {
      registerCPUSpeedNative(callback);
    }
  }

  private native void registerCPUSpeedNative(IntConsumer callback);

  @Override
  public int cpuSpeed() {
    if (JniLoader.isJniAvailable()) {
      return cpuSpeedNative();
    }
    return -1;
  }

  private native int cpuSpeedNative();

  @Override
  public void registerDiskSpaceJni(IntConsumer callback) {
    if (JniLoader.isJniAvailable()) {
      registerDiskSpaceNative(callback);
    }
  }

  private native void registerDiskSpaceNative(IntConsumer callback);

  @Override
  public void registerLoadAdvisoryJni(IntConsumer callback) {
    if (JniLoader.isJniAvailable()) {
      registerLoadAdvisoryNative(callback);
    }
  }

  private native void registerLoadAdvisoryNative(IntConsumer callback);

  @Override
  public int systemLoadAdvisory() {
    if (JniLoader.isJniAvailable()) {
      return systemLoadAdvisoryNative();
    }

    return -1;
  }

  private native int systemLoadAdvisoryNative();

  @Override
  public void registerMemoryPressureJni(IntConsumer callback) {
    if (JniLoader.isJniAvailable()) {
      registerMemoryPressureNative(callback);
    }
  }

  private native void registerMemoryPressureNative(IntConsumer callback);

  @Override
  public int systemMemoryPressure() {
    if (JniLoader.isJniAvailable()) {
      return systemMemoryPressureNative();
    }

    return -1;
  }

  private native int systemMemoryPressureNative();

  @Override
  public void registerSuspensionJni(IntConsumer callback) {
    if (JniLoader.isJniAvailable()) {
      registerSuspensionNative(callback);
    }
  }

  private native void registerSuspensionNative(IntConsumer callback);

  @Override
  public void registerThermalJni(IntConsumer callback) {
    if (JniLoader.isJniAvailable()) {
      registerThermalNative(callback);
    }
  }

  private native void registerThermalNative(IntConsumer callback);

  @Override
  public int thermalLoad() {
    if (JniLoader.isJniAvailable()) {
      return thermalLoadNative();
    }

    return -1;
  }

  private native int thermalLoadNative();
}
