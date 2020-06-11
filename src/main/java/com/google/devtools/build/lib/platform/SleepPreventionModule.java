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

import com.google.common.annotations.VisibleForTesting;
import com.google.devtools.build.lib.runtime.BlazeModule;
import com.google.devtools.build.lib.runtime.CommandEnvironment;

/** Prevents the computer from going to sleep while a Bazel command is running. */
public final class SleepPreventionModule extends BlazeModule {

  /** Methods for dealing with sleep prevention on local hardware. */
  @VisibleForTesting
  public static final class SleepPrevention extends JniLoader {
    /**
     * Push a request to disable automatic sleep for hardware. Useful for making sure computers
     * don't go to sleep during long builds. Must be matched with a {@link #popDisableSleep} call.
     *
     * @return 0 on success, -1 if sleep is not supported.
     */
    public static native int pushDisableSleep();

    /**
     * Pop a request to disable automatic sleep for hardware. Useful for making sure computers don't
     * go to sleep during long builds. Must be matched with a previous {@link #pushDisableSleep}
     * call.
     *
     * @return 0 on success, -1 if sleep is not supported.
     */
    public static native int popDisableSleep();
  }

  @Override
  public void beforeCommand(CommandEnvironment env) {
    if (JniLoader.jniEnabled()) {
      SleepPrevention.pushDisableSleep();
    }
  }

  @Override
  public void afterCommand() {
    if (JniLoader.jniEnabled()) {
      SleepPrevention.popDisableSleep();
    }
  }
}
