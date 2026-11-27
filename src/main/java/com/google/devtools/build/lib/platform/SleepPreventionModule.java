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

import static java.lang.foreign.ValueLayout.JAVA_INT;

import com.google.devtools.build.lib.runtime.BlazeModule;
import com.google.devtools.build.lib.runtime.CommandEnvironment;
import com.google.devtools.build.lib.skymetal.SkymetalBridge;
import com.google.errorprone.annotations.CanIgnoreReturnValue;
import java.lang.foreign.FunctionDescriptor;
import java.lang.invoke.MethodHandle;

/** Prevents the computer from going to sleep while a Bazel command is running. */
public final class SleepPreventionModule extends BlazeModule {
  /**
   * Push a request to disable automatic sleep for hardware. Useful for making sure computers don't
   * go to sleep during long builds. Must be matched with a {@link #popDisableSleep} call.
   *
   * @return 0 on success, -1 if sleep is not supported.
   */
  @CanIgnoreReturnValue
  static int pushDisableSleep() {
    return Native.pushDisableSleep();
  }

  /**
   * Pop a request to disable automatic sleep for hardware. Useful for making sure computers don't
   * go to sleep during long builds. Must be matched with a previous {@link #pushDisableSleep} call.
   *
   * @return 0 on success, -1 if sleep is not supported.
   * @throws IllegalStateException if called without a matching {@link #pushDisableSleep} call.
   */
  @CanIgnoreReturnValue
  static int popDisableSleep() {
    return Native.popDisableSleep();
  }

  @Override
  public void beforeCommand(CommandEnvironment env) {
    pushDisableSleep();
  }

  @Override
  public void afterCommand() {
    popDisableSleep();
  }

  private static final class Native {
    private static final int STATUS_UNDERFLOW = -2;

    private static final FunctionDescriptor INT_NO_ARGS = FunctionDescriptor.of(JAVA_INT);

    private static final MethodHandle PUSH_DISABLE_SLEEP =
        SkymetalBridge.downcallHandle("bazel_sleep_prevention_push", INT_NO_ARGS);

    private static final MethodHandle POP_DISABLE_SLEEP =
        SkymetalBridge.downcallHandle("bazel_sleep_prevention_pop", INT_NO_ARGS);

    static int pushDisableSleep() {
      try {
        return (int) PUSH_DISABLE_SLEEP.invokeExact();
      } catch (Throwable t) {
        throw new AssertionError(t);
      }
    }

    static int popDisableSleep() {
      int status;
      try {
        status = (int) POP_DISABLE_SLEEP.invokeExact();
      } catch (Throwable t) {
        throw new AssertionError(t);
      }
      if (status == STATUS_UNDERFLOW) {
        throw new IllegalStateException(
            "popDisableSleep() called without matching pushDisableSleep()");
      }
      return status;
    }

    private Native() {}
  }
}
