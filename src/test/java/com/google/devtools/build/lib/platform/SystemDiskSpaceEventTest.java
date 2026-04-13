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

/** Tests {@link SystemDiskSpaceEvent} by sending fake notifications. */
@RunWith(JUnit4.class)
public final class SystemDiskSpaceEventTest extends BuildIntegrationTestCase {
  static class SystemDiskSpaceEventListener extends BlazeModule {
    public int lowDiskSpaceEventCount = 0;
    public int veryLowDiskSpaceEventCount = 0;

    @Override
    public void beforeCommand(CommandEnvironment env) {
      env.getEventBus().register(this);
    }

    @Subscribe
    public void diskSpaceEvent(SystemDiskSpaceEvent event) {
      switch (event.level()) {
        case LOW:
          ++lowDiskSpaceEventCount;
          assertThat(event.logString()).isEqualTo("SystemDiskSpaceEvent: Low");
          break;
        case VERY_LOW:
          ++veryLowDiskSpaceEventCount;
          assertThat(event.logString()).isEqualTo("SystemDiskSpaceEvent: Very Low");
          break;
      }
    }
  }

  private final SystemDiskSpaceEventListener eventListener = new SystemDiskSpaceEventListener();

  @Override
  protected BlazeRuntime.Builder getRuntimeBuilder() throws Exception {
    return super.getRuntimeBuilder()
        .addBlazeModule(eventListener)
        .addBlazeModule(new SystemDiskSpaceModule());
  }

  @Test
  public void testDiskSpace() throws Exception {
    Assume.assumeTrue(OS.getCurrent() == OS.DARWIN);
    Runfiles runfiles = Runfiles.create();
    String notifierFilePath =
        runfiles.rlocation(
            "io_bazel/src/test/java/com/google/devtools/build/lib/platform/darwin/notifier");
    write(
        "system_diskSpace_event/BUILD",
        "genrule(",
        "  name = 'fire_diskSpace_notifications',",
        "  outs = ['fire_diskSpace_notifications.out'],",
        "  cmd = '" + notifierFilePath + " com.google.bazel.test.diskspace.low 0 > $@ && ' + ",
        "        '" + notifierFilePath + " com.google.bazel.test.diskspace.verylow 0 >> $@',",
        ")");
    buildTarget("//system_diskSpace_event:fire_diskSpace_notifications");
    assertThat(eventListener.lowDiskSpaceEventCount).isGreaterThan(0);
    assertThat(eventListener.veryLowDiskSpaceEventCount).isGreaterThan(0);
  }
}
