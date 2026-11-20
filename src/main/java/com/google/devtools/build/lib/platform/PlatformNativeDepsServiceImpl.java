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
import com.google.devtools.build.lib.skybridge.ScOnly;
import com.google.devtools.build.lib.util.OS;
import java.util.function.IntConsumer;

/** Implementation of {@link PlatformNativeDepsService}. */
@ScOnly
public class PlatformNativeDepsServiceImpl implements PlatformNativeDepsService {
  private static final boolean IS_DARWIN = OS.getCurrent() == OS.DARWIN;
  private static final boolean SUPPORTS_SLEEP_PREVENTION =
      IS_DARWIN || OS.getCurrent() == OS.WINDOWS;

  static {
    JniLoader.loadJni();
  }

  @Override
  public int pushDisableSleep() {
    return SUPPORTS_SLEEP_PREVENTION ? pushDisableSleepNative() : -1;
  }

  private native int pushDisableSleepNative();

  @Override
  public int popDisableSleep() {
    return SUPPORTS_SLEEP_PREVENTION ? popDisableSleepNative() : -1;
  }

  private native int popDisableSleepNative();

  @Override
  public void registerCPUSpeedJni(IntConsumer callback) {
    if (IS_DARWIN) {
      registerCPUSpeedNative(callback);
    }
  }

  private native void registerCPUSpeedNative(IntConsumer callback);

  @Override
  public int cpuSpeed() {
    return IS_DARWIN ? cpuSpeedNative() : -1;
  }

  private native int cpuSpeedNative();

  @Override
  public void registerDiskSpaceJni(IntConsumer callback) {
    if (IS_DARWIN) {
      registerDiskSpaceNative(callback);
    }
  }

  private native void registerDiskSpaceNative(IntConsumer callback);

  @Override
  public void registerLoadAdvisoryJni(IntConsumer callback) {
    if (IS_DARWIN) {
      registerLoadAdvisoryNative(callback);
    }
  }

  private native void registerLoadAdvisoryNative(IntConsumer callback);

  @Override
  public int systemLoadAdvisory() {
    return IS_DARWIN ? systemLoadAdvisoryNative() : 0;
  }

  private native int systemLoadAdvisoryNative();

  @Override
  public void registerMemoryPressureJni(IntConsumer callback) {
    if (IS_DARWIN) {
      registerMemoryPressureNative(callback);
    }
  }

  private native void registerMemoryPressureNative(IntConsumer callback);

  @Override
  public int systemMemoryPressure() {
    return IS_DARWIN ? systemMemoryPressureNative() : 0;
  }

  private native int systemMemoryPressureNative();

  @Override
  public void registerSuspensionJni(IntConsumer callback) {
    if (IS_DARWIN) {
      registerSuspensionNative(callback);
    }
  }

  private native void registerSuspensionNative(IntConsumer callback);

  @Override
  public void registerThermalJni(IntConsumer callback) {
    if (IS_DARWIN) {
      registerThermalNative(callback);
    }
  }

  private native void registerThermalNative(IntConsumer callback);

  @Override
  public int thermalLoad() {
    return IS_DARWIN ? thermalLoadNative() : 0;
  }

  private native int thermalLoadNative();
}
