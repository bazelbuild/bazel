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
import static org.junit.Assert.fail;

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

/** Tests {@link SystemThermalEvent} by sending fake notifications. */
@RunWith(JUnit4.class)
public final class SystemThermalEventTest extends BuildIntegrationTestCase {
  static class SystemThermalEventListener extends BlazeModule {
    public int nominalThermalEventCount = 0;
    public int moderateThermalEventCount = 0;
    public int heavyThermalEventCount = 0;
    public int trappingThermalEventCount = 0;
    public int sleepingThermalEventCount = 0;

    @Override
    public void beforeCommand(CommandEnvironment env) {
      env.getEventBus().register(this);
    }

    @Subscribe
    public void thermalEvent(SystemThermalEvent event) {
      switch (event.value()) {
        case 0:
          ++nominalThermalEventCount;
          assertThat(event.logString()).isEqualTo("SystemThermalEvent: 0 (Nominal)");
          break;
        case 33:
          ++moderateThermalEventCount;
          assertThat(event.logString()).isEqualTo("SystemThermalEvent: 33 (Moderate)");
          break;
        case 50:
          ++heavyThermalEventCount;
          assertThat(event.logString()).isEqualTo("SystemThermalEvent: 50 (Heavy)");
          break;
        case 90:
          ++trappingThermalEventCount;
          assertThat(event.logString()).isEqualTo("SystemThermalEvent: 90 (Trapping)");
          break;
        case 100:
          ++sleepingThermalEventCount;
          assertThat(event.logString()).isEqualTo("SystemThermalEvent: 100 (Sleeping)");
          break;
        default:
          fail(
              String.format(
                  "Unknown SystemThermalEvent value: %d (%s)", event.value(), event.logString()));
          break;
      }
    }
  }

  private final SystemThermalEventListener eventListener = new SystemThermalEventListener();

  @Override
  protected BlazeRuntime.Builder getRuntimeBuilder() throws Exception {
    return super.getRuntimeBuilder()
        .addBlazeModule(eventListener)
        .addBlazeModule(new SystemThermalModule());
  }

  @Test
  public void testThermal() throws Exception {
    Assume.assumeTrue(OS.getCurrent() == OS.DARWIN);
    Runfiles runfiles = Runfiles.create();
    String notifierFilePath =
        runfiles.rlocation(
            "io_bazel/src/test/java/com/google/devtools/build/lib/platform/darwin/notifier");
    write(
        "system_thermal_event/BUILD",
        "genrule(",
        "  name = 'fire_thermal_notifications',",
        "  outs = ['fire_thermal_notifications.out'],",
        "  cmd = '"
            + notifierFilePath
            + " com.google.bazel.test.thermalpressurelevel 0 > $@ && ' + ",
        "'" + notifierFilePath + " com.google.bazel.test.thermalpressurelevel 1 >> $@ && ' + ",
        "'" + notifierFilePath + " com.google.bazel.test.thermalpressurelevel 2 >> $@ && ' + ",
        "'" + notifierFilePath + " com.google.bazel.test.thermalpressurelevel 3 >> $@ && ' + ",
        "'" + notifierFilePath + " com.google.bazel.test.thermalpressurelevel 4 >> $@',",
        ")");
    buildTarget("//system_thermal_event:fire_thermal_notifications");
    assertThat(eventListener.nominalThermalEventCount).isGreaterThan(0);
    assertThat(eventListener.moderateThermalEventCount).isGreaterThan(0);
    assertThat(eventListener.heavyThermalEventCount).isGreaterThan(0);
    assertThat(eventListener.trappingThermalEventCount).isGreaterThan(0);
    assertThat(eventListener.sleepingThermalEventCount).isGreaterThan(0);
  }
}
