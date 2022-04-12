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
import static org.junit.Assert.assertThrows;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyBoolean;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.verifyNoMoreInteractions;
import static org.mockito.Mockito.when;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.events.NullEventHandler;
import com.google.devtools.build.skyframe.CyclesReporter.SingleCycleReporter;
import java.util.concurrent.atomic.AtomicBoolean;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

@RunWith(JUnit4.class)
public class CyclesReporterTest {

  private static final SkyKey DUMMY_KEY = () -> SkyFunctionName.createHermetic("func");

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
    SingleCycleReporter singleReporter =
        (topLevelKey, cycleInfo, alreadyReported, eventHandler) -> false;

    CycleInfo cycleInfo = new CycleInfo(ImmutableList.of(DUMMY_KEY));
    CyclesReporter cyclesReporter = new CyclesReporter(singleReporter);
    assertThrows(
        IllegalStateException.class,
        () ->
            cyclesReporter.reportCycles(
                ImmutableList.of(cycleInfo), DUMMY_KEY, NullEventHandler.INSTANCE));
  }

  @Test
  public void smoke() {
    final AtomicBoolean reported = new AtomicBoolean();
    SingleCycleReporter singleReporter =
        (topLevelKey, cycleInfo, alreadyReported, eventHandler) -> {
          reported.set(true);
          return true;
        };

    CycleInfo cycleInfo = new CycleInfo(ImmutableList.of(DUMMY_KEY));
    CyclesReporter cyclesReporter = new CyclesReporter(singleReporter);
    cyclesReporter.reportCycles(ImmutableList.of(cycleInfo), DUMMY_KEY,
        NullEventHandler.INSTANCE);
    assertThat(reported.get()).isTrue();
  }

  @Test
  public void alreadyReportedCycles() {
    SingleCycleReporter mockReporter = mock(SingleCycleReporter.class);
    when(mockReporter.maybeReportCycle(any(), any(), anyBoolean(), any())).thenReturn(true);
    CyclesReporter cyclesReporter = new CyclesReporter(mockReporter);
    SkyKey top1 = () -> SkyFunctionName.createHermetic("top1");
    SkyKey top2 = () -> SkyFunctionName.createHermetic("top2");
    SkyKey path1 = () -> SkyFunctionName.createHermetic("path1");
    SkyKey path2 = () -> SkyFunctionName.createHermetic("path2");
    SkyKey cycle1 = () -> SkyFunctionName.createHermetic("cycle1");
    SkyKey cycle2 = () -> SkyFunctionName.createHermetic("cycle2");
    CycleInfo top1FirstCycle =
        new CycleInfo(ImmutableList.of(top1, path1), ImmutableList.of(cycle1, cycle2));
    cyclesReporter.reportCycles(
        ImmutableList.of(
            top1FirstCycle,
            new CycleInfo(ImmutableList.of(top1, path2), ImmutableList.of(cycle1, cycle2)),
            new CycleInfo(ImmutableList.of(top1, path1), ImmutableList.of(cycle2, cycle1)),
            new CycleInfo(ImmutableList.of(top1, path2), ImmutableList.of(cycle2, cycle1))),
        top1,
        NullEventHandler.INSTANCE);
    verify(mockReporter)
        .maybeReportCycle(
            top1, top1FirstCycle, /*alreadyReported=*/ false, NullEventHandler.INSTANCE);
    // Second cycle is filtered out because it is equivalent but for the path and cycle order.
    verifyNoMoreInteractions(mockReporter);

    CycleInfo top2FirstCycle =
        new CycleInfo(ImmutableList.of(top2, path1), ImmutableList.of(cycle1, cycle2));
    cyclesReporter.reportCycles(
        ImmutableList.of(
            top2FirstCycle,
            new CycleInfo(ImmutableList.of(top2, path2), ImmutableList.of(cycle1, cycle2)),
            new CycleInfo(ImmutableList.of(top2, path1), ImmutableList.of(cycle2, cycle1)),
            new CycleInfo(ImmutableList.of(top2, path2), ImmutableList.of(cycle2, cycle1))),
        top2,
        NullEventHandler.INSTANCE);

    verify(mockReporter)
        .maybeReportCycle(
            top2, top2FirstCycle, /*alreadyReported=*/ true, NullEventHandler.INSTANCE);
    verifyNoMoreInteractions(mockReporter);
  }
}
