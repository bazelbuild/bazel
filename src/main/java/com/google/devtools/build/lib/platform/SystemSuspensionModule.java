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
import javax.annotation.Nullable;
import javax.annotation.concurrent.GuardedBy;

/** Detects suspension events. */
public final class SystemSuspensionModule extends BlazeModule {
  private static final GoogleLogger logger = GoogleLogger.forEnclosingClass();

  static {
    JniLoader.loadJni();
  }

  @GuardedBy("this")
  @Nullable
  private Reporter reporter;

  private native void registerJNI();

  public SystemSuspensionModule() {
    if (JniLoader.isJniAvailable()) {
      registerJNI();
    }
  }

  @Override
  public synchronized void beforeCommand(CommandEnvironment env) {
    this.reporter = env.getReporter();
  }

  @Override
  public synchronized void afterCommand() {
    this.reporter = null;
  }

  /** Callback method called from JNI whenever a suspension event occurs. */
  synchronized void suspendCallback(int reason) {
    SystemSuspensionEvent event = new SystemSuspensionEvent(reason);
    logger.atInfo().log("%s", event.logString());
    if (reporter != null) {
      reporter.post(event);
    }
  }
}
