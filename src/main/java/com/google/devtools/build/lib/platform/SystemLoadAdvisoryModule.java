// Copyright 2021 The Bazel Authors. All rights reserved.
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

/** Detects load advisory events. */
public final class SystemLoadAdvisoryModule extends BlazeModule {
  private static final GoogleLogger logger = GoogleLogger.forEnclosingClass();

  static {
    JniLoader.loadJni();
  }

  @GuardedBy("this")
  @Nullable
  private Reporter reporter;

  private native void registerJNI();

  private static native int systemLoadAdvisory();

  public SystemLoadAdvisoryModule() {
    if (JniLoader.isJniAvailable()) {
      registerJNI();
    }
  }

  @Override
  public synchronized void beforeCommand(CommandEnvironment env) {
    this.reporter = env.getReporter();
    reportSystemLoadAdvisoryEvent(true, systemLoadAdvisory());
  }

  @Override
  public synchronized void afterCommand() {
    this.reporter = null;
  }

  /**
   * Callback that is called from the native system load advisory monitoring code.
   *
   * @param value - 0-100
   *     <p>Intermediate values are platform dependent.
   *     <p>For macOS the system advisory states map to:
   *     <ul>
   *       <li>0 - kIOSystemLoadAdvisoryLevelGreat
   *       <li>25 - kIOSystemLoadAdvisoryLevelOK
   *       <li>75 - kIOSystemLoadAdvisoryLevelBad
   *     </ul>
   */
  synchronized void systemLoadAdvisoryCallback(int value) {
    reportSystemLoadAdvisoryEvent(false, value);
  }

  private String macOSSystemLoadAdvisoryDescription(int value) {
    return switch (value) {
      case 0 -> "Great";
      case 25 -> "OK";
      case 75 -> "Bad";
      default -> "Unknown System Load Advisory Value: " + value;
    };
  }

  private synchronized void reportSystemLoadAdvisoryEvent(boolean isInitialValue, int value) {
    String osDescription = "Unknown";
    if (OS.getCurrent() == OS.DARWIN) {
      osDescription = macOSSystemLoadAdvisoryDescription(value);
    }
    SystemLoadAdvisoryEvent event = new SystemLoadAdvisoryEvent(value, osDescription);
    String logString = event.logString();

    if (value < 0 || value > 100) {
      // values outside this range are not expected.
      logger.atSevere().log("%s", logString);
    } else if (value > 50) {
      // 50 arbitrarily chosen as point where user is likely to be more concerned.
      logger.atWarning().log("%s", logString);
    } else if (!isInitialValue || value > 25) {
      // Don't spam the logs if we have a Great or OK value at startup.
      logger.atInfo().log("%s", logString);
    }
    if (reporter != null) {
      reporter.post(event);
    }
  }
}
