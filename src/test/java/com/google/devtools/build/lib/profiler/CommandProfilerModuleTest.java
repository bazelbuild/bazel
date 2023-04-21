// Copyright 2023 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.profiler;

import static com.google.common.truth.Truth.assertThat;
import static org.junit.Assume.assumeTrue;

import com.google.devtools.build.lib.buildtool.util.BuildIntegrationTestCase;
import com.google.devtools.build.lib.runtime.BlazeRuntime;
import com.google.devtools.build.lib.util.OS;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link CommandProfilerModule}. */
@RunWith(JUnit4.class)
public final class CommandProfilerModuleTest extends BuildIntegrationTestCase {

  @Override
  protected BlazeRuntime.Builder getRuntimeBuilder() throws Exception {
    return super.getRuntimeBuilder().addBlazeModule(new CommandProfilerModule());
  }

  @Before
  public void setUp() throws Exception {
    write("BUILD", "");
  }

  @Test
  public void testProfilingDisabled() throws Exception {
    buildTarget("//:BUILD");
    assertThat(outputBase.getChild("profile.jfr").exists()).isFalse();
  }

  @Test
  public void testProfilingEnabled() throws Exception {
    addOptions("--experimental_command_profile");

    try {
      buildTarget("//:BUILD");
    } catch (Exception e) {
      // Linux perf events are not supported on CI.
      // See https://github.com/async-profiler/async-profiler/#troubleshooting.
      if (e.getMessage().contains("No access to perf events")
          || e.getMessage().contains("Perf events unavailable")) {
        return;
      }
    }

    assumeTrue(OS.getCurrent() == OS.LINUX || OS.getCurrent() == OS.DARWIN);

    assertThat(outputBase.getChild("profile.jfr").exists()).isTrue();
  }
}
