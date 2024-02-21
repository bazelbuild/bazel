// Copyright 2024 The Bazel Authors. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package com.google.devtools.build.lib.metrics;

import static com.google.common.truth.Truth.assertThat;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.clock.BlazeClock;
import com.google.devtools.build.lib.clock.Clock;
import com.google.devtools.build.lib.sandbox.CgroupsInfo;
import com.google.devtools.build.lib.sandbox.cgroups.VirtualCGroup;
import com.google.devtools.build.lib.sandbox.cgroups.controller.Controller;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

import java.io.IOException;

@RunWith(JUnit4.class)
public class CgroupsInfoCollectorTest {

  @Test
  public void testCollectResourceUsage_returnsValidCgroupMemoryUsage() throws IOException {
    Clock clock = BlazeClock.instance();
    ImmutableMap.Builder<Long, VirtualCGroup> pidToCgroups = ImmutableMap.builder();
    for (long i = 1; i < 4; i++) {
      Controller.Memory m = mock(Controller.Memory.class);
      when(m.getUsageInBytes()).thenReturn(i * 1000 * 1024);
      when(m.exists()).thenReturn(i == 2 ? false : true);
      VirtualCGroup cg = mock(VirtualCGroup.class);
      when(cg.memory()).thenReturn(m);
      pidToCgroups.put(i, cg);
    }

    ResourceSnapshot snapshot =
        CgroupsInfoCollector.instance()
            .collectResourceUsage(pidToCgroups.build(), clock);

    // Results from cgroups 2 should not be in the snapshot since it doesn't exist.
    assertThat(snapshot.getPidToMemoryInKb()).containsExactly(1L, 1000, 3L, 3000);
  }
}
