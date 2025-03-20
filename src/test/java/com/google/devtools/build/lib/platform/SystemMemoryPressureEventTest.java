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

import static com.google.common.truth.Truth.assertThat;

import com.google.common.eventbus.Subscribe;
import com.google.devtools.build.lib.buildtool.util.BuildIntegrationTestCase;
import com.google.devtools.build.lib.runtime.BlazeModule;
import com.google.devtools.build.lib.runtime.BlazeRuntime;
import com.google.devtools.build.lib.runtime.CommandEnvironment;
import com.google.devtools.build.lib.util.OS;
import com.google.devtools.build.runfiles.Runfiles;
import org.junit.Assume;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests {@link SystemMemoryPressureEvent} by sending fake notifications. */
@RunWith(JUnit4.class)
public final class SystemMemoryPressureEventTest extends BuildIntegrationTestCase {
  static class SystemMemoryPressureEventListener extends BlazeModule {
    public int memoryPressureNormalEventCount = 0;
    public int memoryPressureWarningEventCount = 0;
    public int memoryPressureCriticalEventCount = 0;

    @Override
    public void beforeCommand(CommandEnvironment env) {
      env.getEventBus().register(this);
    }

    @Subscribe
    public void memoryPressureEvent(SystemMemoryPressureEvent event) {
      switch (event.level()) {
        case NORMAL:
          ++memoryPressureNormalEventCount;
          assertThat(event.logString()).isEqualTo("SystemMemoryPressureEvent: Normal");
          break;
        case WARNING:
          ++memoryPressureWarningEventCount;
          assertThat(event.logString()).isEqualTo("SystemMemoryPressureEvent: Warning");
          break;

        case CRITICAL:
          ++memoryPressureCriticalEventCount;
          assertThat(event.logString()).isEqualTo("SystemMemoryPressureEvent: Critical");
          break;
      }
    }
  }

  private final SystemMemoryPressureEventListener eventListener =
      new SystemMemoryPressureEventListener();

  @Override
  protected BlazeRuntime.Builder getRuntimeBuilder() throws Exception {
    return super.getRuntimeBuilder()
        .addBlazeModule(eventListener)
        .addBlazeModule(new SystemMemoryPressureModule());
  }

  @Test
  public void testMemoryPressure() throws Exception {
    Assume.assumeTrue(OS.getCurrent() == OS.DARWIN);
    Runfiles runfiles = Runfiles.create();
    String notifierFilePath =
        runfiles.rlocation(
            "io_bazel/src/test/java/com/google/devtools/build/lib/platform/darwin/notifier");
    write(
        "system_memory_pressure_event/BUILD",
        "genrule(",
        "  name = 'fire_memory_pressure_notifications',",
        "  outs = ['fire_memory_pressure_notifications.out'],",
        "  cmd = '"
            + notifierFilePath
            + " com.google.bazel.test.memorypressurelevel 0 > $@ && ' + ",
        "        '"
            + notifierFilePath
            + " com.google.bazel.test.memorypressurelevel 1 >> $@ && ' + ",
        "        '" + notifierFilePath + " com.google.bazel.test.memorypressurelevel 2 >> $@',",
        ")");
    buildTarget("//system_memory_pressure_event:fire_memory_pressure_notifications");
    assertThat(eventListener.memoryPressureNormalEventCount).isGreaterThan(0);
    assertThat(eventListener.memoryPressureWarningEventCount).isGreaterThan(0);
    assertThat(eventListener.memoryPressureCriticalEventCount).isGreaterThan(0);
  }
}
