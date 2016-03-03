// Copyright 2014 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.skyframe;

import static com.google.common.truth.Truth.assertThat;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.events.EventHandler;
import com.google.devtools.build.lib.events.NullEventHandler;
import com.google.devtools.build.skyframe.CyclesReporter.SingleCycleReporter;

import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

import java.util.concurrent.atomic.AtomicBoolean;

@RunWith(JUnit4.class)
public class CyclesReporterTest {

  private static final SkyKey DUMMY_KEY = SkyKey.create(SkyFunctionName.create("func"), "key");

  @Test
  public void nullEventHandler() {
    CyclesReporter cyclesReporter = new CyclesReporter();
    try {
      cyclesReporter.reportCycles(ImmutableList.<CycleInfo>of(), DUMMY_KEY, null);
      assertThat(false).isTrue();
    } catch (NullPointerException e) {
      // Expected.
    }
  }

  @Test
  public void notReportedAssertion() {
    SingleCycleReporter singleReporter = new SingleCycleReporter() {
      @Override
      public boolean maybeReportCycle(SkyKey topLevelKey, CycleInfo cycleInfo,
          boolean alreadyReported, EventHandler eventHandler) {
        return false;
      }
    };

    CycleInfo cycleInfo = new CycleInfo(ImmutableList.of(DUMMY_KEY));
    CyclesReporter cyclesReporter = new CyclesReporter(singleReporter);
    try {
      cyclesReporter.reportCycles(ImmutableList.of(cycleInfo), DUMMY_KEY,
          NullEventHandler.INSTANCE);
      assertThat(false).isTrue();
    } catch (IllegalStateException e) {
      // Expected.
    }
  }

  @Test
  public void smoke() {
    final AtomicBoolean reported = new AtomicBoolean();
    SingleCycleReporter singleReporter = new SingleCycleReporter() {
      @Override
      public boolean maybeReportCycle(SkyKey topLevelKey, CycleInfo cycleInfo,
          boolean alreadyReported, EventHandler eventHandler) {
        reported.set(true);
        return true;
      }
    };

    CycleInfo cycleInfo = new CycleInfo(ImmutableList.of(DUMMY_KEY));
    CyclesReporter cyclesReporter = new CyclesReporter(singleReporter);
    cyclesReporter.reportCycles(ImmutableList.of(cycleInfo), DUMMY_KEY,
        NullEventHandler.INSTANCE);
    assertThat(reported.get()).isTrue();
  }
}
