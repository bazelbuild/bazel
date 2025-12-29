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
import org.junit.Assume;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link SystemSuspensionEvent}. */
@RunWith(JUnit4.class)
public final class SystemSuspensionEventTest extends BuildIntegrationTestCase {
  static class SystemSuspensionEventListener extends BlazeModule {
    public int suspensionEventCount = 0;

    @Override
    public void beforeCommand(CommandEnvironment env) {
      env.getEventBus().register(this);
    }

    @Subscribe
    public void suspensionEvent(SystemSuspensionEvent event) {
      assertThat(event.reason()).isEqualTo(SystemSuspensionEvent.Reason.SIGCONT);
      assertThat(event.logString()).isEqualTo("SystemSuspensionEvent: Signal SIGCONT");
      ++suspensionEventCount;
    }
  }

  private final SystemSuspensionEventListener eventListener = new SystemSuspensionEventListener();

  @Override
  protected BlazeRuntime.Builder getRuntimeBuilder() throws Exception {
    return super.getRuntimeBuilder()
        .addBlazeModule(eventListener)
        .addBlazeModule(new SystemSuspensionModule());
  }

  @Test
  public void testSuspendCounter() throws Exception {
    Assume.assumeTrue(OS.getCurrent() == OS.DARWIN);
    // Send a SIGCONT to ourselves which should cause our signal handler to fire.
    write(
        "system_suspension_event/BUILD",
        "genrule(",
        "  name = 'signal',",
        "  outs = ['signal.out'],",
        "  cmd = '/bin/kill -s CONT " + ProcessHandle.current().pid() + " > $@',",
        ")");
    buildTarget("//system_suspension_event:signal");
    assertThat(eventListener.suspensionEventCount).isEqualTo(1);
  }
}
