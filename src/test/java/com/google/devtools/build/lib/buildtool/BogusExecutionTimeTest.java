// Copyright 2020 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.buildtool;

import com.google.devtools.build.lib.buildtool.util.BuildIntegrationTestCase;
import com.google.devtools.build.lib.runtime.BlazeRuntime;
import com.google.devtools.build.lib.testutil.ManualClock;
import com.google.devtools.build.lib.testutil.Suite;
import com.google.devtools.build.lib.testutil.TestSpec;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/**
 * Regression test for Blaze crashing when the finishing time of a command is smaller than the
 * starting time according to the clock, which cannot be trusted to be non-decreasing in general.
 */
@TestSpec(size = Suite.MEDIUM_TESTS)
@RunWith(JUnit4.class)
public class BogusExecutionTimeTest extends BuildIntegrationTestCase {
  private ManualClock clock;

  @Before
  public final void setClock() {
    clock = new ManualClock();
  }

  @Override
  protected BlazeRuntime.Builder getRuntimeBuilder() throws Exception {
    return super.getRuntimeBuilder().setClock(clock);
  }

  @Test
  public void testBogusExecutionTime() throws Exception {
    clock.advanceMillis(1337);
    write("foo/BUILD", "sh_library(name = 'foo')");
    buildTarget("//foo");
    clock.advanceMillis(-42L);
    buildTarget("//foo");
  }
}
