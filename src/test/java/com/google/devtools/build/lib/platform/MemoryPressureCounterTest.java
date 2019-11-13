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

/** Tests for {@link MemoryPressureCounter}. */
@RunWith(JUnit4.class)
public final class MemoryPressureCounterTest {

  @Test
  public void testMemoryPressure() throws Exception {
    if (OS.getCurrent() == OS.DARWIN) {
      // Unfortunately there is no good way to easily test memory pressure on Darwin.
      // Running `memory_pressure -S -l warn` would do it, but it requires `sudo` to
      // function.
      assertThat(MemoryPressureCounter.warningCount()).isAtLeast(0);
      assertThat(MemoryPressureCounter.criticalCount()).isAtLeast(0);
    } else {
      assertThat(MemoryPressureCounter.warningCount()).isEqualTo(0);
      assertThat(MemoryPressureCounter.criticalCount()).isEqualTo(0);
    }
  }
}
