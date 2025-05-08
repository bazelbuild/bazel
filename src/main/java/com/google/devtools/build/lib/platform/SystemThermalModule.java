// Copyright 2019 The Bazel Authors. All rights reserved.
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

import com.google.common.flogger.GoogleLogger;
import com.google.devtools.build.lib.events.Reporter;
import com.google.devtools.build.lib.jni.JniLoader;
import com.google.devtools.build.lib.runtime.BlazeModule;
import com.google.devtools.build.lib.runtime.CommandEnvironment;
import com.google.devtools.build.lib.util.OS;
import javax.annotation.Nullable;
import javax.annotation.concurrent.GuardedBy;

/** Detects suspension events. */
public final class SystemThermalModule extends BlazeModule {
  private static final GoogleLogger logger = GoogleLogger.forEnclosingClass();

  static {
    JniLoader.loadJni();
  }

  @GuardedBy("this")
  @Nullable
  private Reporter reporter;

  private native void registerJNI();

  private static native int thermalLoad();

  public SystemThermalModule() {
    if (JniLoader.isJniAvailable()) {
      registerJNI();
    }
  }

  @Override
  public synchronized void beforeCommand(CommandEnvironment env) {
    this.reporter = env.getReporter();
    reportThermalEvent(true, thermalLoad());
  }

  @Override
  public synchronized void afterCommand() {
    this.reporter = null;
  }

  /**
   * Callback that is called from the native thermal monitoring code. Made @VisibleForTesting
   * because it is expected to be called only from JNI callbacks which are difficult to mock.
   *
   * @param value - 0-100 where 0 is no thermal issues to 100 which is worst case.
   *     <p>Intermediate values are platform dependent.
   *     <p>For macOS the thermal states map to:
   *     <ul>
   *       <li>0 - kOSThermalPressureLevelNominal
   *       <li>33 - kOSThermalPressureLevelModerate (Expect CPU performance > 50%)
   *       <li>50 - kOSThermalPressureLevelHeavy (Expect CPU performance < 50%)
   *       <li>90 - kOSThermalPressureLevelTrapping (Expect machine is about to die).
   *       <li>100 - kOSThermalPressureLevelSleeping (Machine is going to sleep to lower heat).
   *     </ul>
   */
  synchronized void thermalCallback(int value) {
    reportThermalEvent(false, value);
  }

  private String macOSThermalDescription(int value) {
    return switch (value) {
      case 0 -> "Nominal";
      case 33 -> "Moderate";
      case 50 -> "Heavy";
      case 90 -> "Trapping";
      case 100 -> "Sleeping";
      default -> "Unknown";
    };
  }

  private synchronized void reportThermalEvent(boolean isInitialValue, int value) {
    String osDescription = "Unknown";
    if (OS.getCurrent() == OS.DARWIN) {
      osDescription = macOSThermalDescription(value);
    }
    SystemThermalEvent event = new SystemThermalEvent(value, osDescription);
    String logString = event.logString();

    if (value < 0 || value > 100) {
      // values outside this range are not expected.
      logger.atSevere().log("%s", logString);
    } else if (value > 50) {
      // 50 arbitrarily chosen as point where user is likely to be more concerned.
      logger.atWarning().log("%s", logString);
    } else if (!isInitialValue || value != 0) {
      // Don't spam the logs if we have a nominal value at startup.
      logger.atInfo().log("%s", logString);
    }
    if (reporter != null) {
      reporter.post(event);
    }
  }
}
