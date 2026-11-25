// Copyright 2022 The Bazel Authors. All rights reserved.
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

/** Tests {@link SystemCPUSpeedEvent} by sending fake notifications. */
@RunWith(JUnit4.class)
public final class SystemCPUSpeedEventTest extends BuildIntegrationTestCase {
  static class SystemCPUSpeedEventListener extends BlazeModule {
    public int cpuSpeedEventCount = 0;
    public int highestSpeed = Integer.MIN_VALUE;
    public int lowestSpeed = Integer.MAX_VALUE;

    @Override
    public void beforeCommand(CommandEnvironment env) {
      env.getEventBus().register(this);
    }

    @Subscribe
    public void cpuSpeedEvent(SystemCPUSpeedEvent event) {
      ++cpuSpeedEventCount;
      int speed = event.speed();
      if (speed > highestSpeed) {
        highestSpeed = speed;
      }
      if (speed < lowestSpeed) {
        lowestSpeed = speed;
      }
    }
  }

  private final SystemCPUSpeedEventListener eventListener = new SystemCPUSpeedEventListener();

  @Override
  protected BlazeRuntime.Builder getRuntimeBuilder() throws Exception {
    return super.getRuntimeBuilder()
        .addBlazeModule(eventListener)
        .addBlazeModule(new SystemCPUSpeedModule());
  }

  @Test
  public void testCPUSpeed() throws Exception {
    Assume.assumeTrue(OS.getCurrent() == OS.DARWIN);
    Runfiles runfiles = Runfiles.create();
    String notifierFilePath =
        runfiles.rlocation(
            "io_bazel/src/test/java/com/google/devtools/build/lib/platform/darwin/notifier");
    write(
        "system_cpuspeed_event/BUILD",
        "genrule(",
        "  name = 'fire_cpuspeed_notifications',",
        "  outs = ['fire_cpuspeed_notifications.out'],",
        "  cmd = '" + notifierFilePath + " com.google.bazel.test.cpuspeed 80 > $@ && ' + ",
        "        '" + notifierFilePath + " com.google.bazel.test.cpuspeed 60 >> $@ && ' + ",
        "        '" + notifierFilePath + " com.google.bazel.test.cpuspeed 70 >> $@ && ' + ",
        "        '" + notifierFilePath + " com.google.bazel.test.cpuspeed 40 >> $@ && ' + ",
        "        '" + notifierFilePath + " com.google.bazel.test.cpuspeed 50 >> $@',",
        ")");
    buildTarget("//system_cpuspeed_event:fire_cpuspeed_notifications");
    assertThat(eventListener.cpuSpeedEventCount).isAtLeast(5);
    assertThat(eventListener.highestSpeed).isAtLeast(80);
    assertThat(eventListener.lowestSpeed).isAtMost(40);
  }
}
