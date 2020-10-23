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

import static com.google.common.truth.Truth.assertThat;

import com.google.devtools.build.lib.util.OS;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link SleepPreventionModule}. */
@RunWith(JUnit4.class)
public final class SleepPreventionModuleTest {

  private static boolean haveSleepPreventionSupport() throws Exception {
    switch (OS.getCurrent()) {
      case DARWIN:
      case WINDOWS:
        return true;
      case LINUX:
      case FREEBSD:
      case OPENBSD:
      case UNKNOWN:
        return false;
    }
    throw new AssertionError("switch statement out of sync with OS values");
  }

  @Test
  public void testSleepPrevention() throws Exception {
    if (haveSleepPreventionSupport()) {
      // Assert standard push pop works.
      assertThat(SleepPreventionModule.SleepPrevention.pushDisableSleep()).isEqualTo(0);
      assertThat(SleepPreventionModule.SleepPrevention.popDisableSleep()).isEqualTo(0);

      // Assert that nested push pop works, and that re-enabling after disabling (above)
      // works.
      assertThat(SleepPreventionModule.SleepPrevention.pushDisableSleep()).isEqualTo(0);
      assertThat(SleepPreventionModule.SleepPrevention.pushDisableSleep()).isEqualTo(0);
      assertThat(SleepPreventionModule.SleepPrevention.popDisableSleep()).isEqualTo(0);
      assertThat(SleepPreventionModule.SleepPrevention.popDisableSleep()).isEqualTo(0);
    } else {
      assertThat(SleepPreventionModule.SleepPrevention.pushDisableSleep()).isEqualTo(-1);
      assertThat(SleepPreventionModule.SleepPrevention.popDisableSleep()).isEqualTo(-1);
    }
  }
}
