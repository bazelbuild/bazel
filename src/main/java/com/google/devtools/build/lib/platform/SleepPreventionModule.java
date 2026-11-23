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

import com.google.devtools.build.lib.jni.JniLoader;
import com.google.devtools.build.lib.runtime.BlazeModule;
import com.google.devtools.build.lib.runtime.CommandEnvironment;
import com.google.devtools.build.lib.util.OS;
import com.google.errorprone.annotations.CanIgnoreReturnValue;

/** Prevents the computer from going to sleep while a Bazel command is running. */
public final class SleepPreventionModule extends BlazeModule {
  private static final boolean SUPPORTS_SLEEP_PREVENTION =
      OS.getCurrent() == OS.DARWIN || OS.getCurrent() == OS.WINDOWS;

  static {
    JniLoader.loadJni();
  }

  /**
   * Push a request to disable automatic sleep for hardware. Useful for making sure computers don't
   * go to sleep during long builds. Must be matched with a {@link #popDisableSleep} call.
   *
   * @return 0 on success, -1 if sleep is not supported.
   */
  @CanIgnoreReturnValue
  static int pushDisableSleep() {
    return SUPPORTS_SLEEP_PREVENTION ? pushDisableSleepNative() : -1;
  }

  private static native int pushDisableSleepNative();

  /**
   * Pop a request to disable automatic sleep for hardware. Useful for making sure computers don't
   * go to sleep during long builds. Must be matched with a previous {@link #pushDisableSleep} call.
   *
   * @return 0 on success, -1 if sleep is not supported.
   */
  @CanIgnoreReturnValue
  static int popDisableSleep() {
    return SUPPORTS_SLEEP_PREVENTION ? popDisableSleepNative() : -1;
  }

  private static native int popDisableSleepNative();

  @Override
  public void beforeCommand(CommandEnvironment env) {
    pushDisableSleep();
  }

  @Override
  public void afterCommand() {
    popDisableSleep();
  }
}
