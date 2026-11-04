// Copyright 2026 The Bazel Authors. All rights reserved.
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

import static com.google.common.truth.Truth.assertThat;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.EventBusEventHandler;
import com.google.devtools.build.lib.events.Reporter;
import com.google.devtools.build.lib.events.StoredEventHandler;
import com.google.devtools.build.lib.runtime.CommandEnvironment;
import java.util.Collections;
import java.util.List;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link AnalysisPhaseRunner}. */
@RunWith(JUnit4.class)
public final class AnalysisPhaseRunnerTest {
  private final CommandEnvironment env = mock(CommandEnvironment.class);
  private final ConfiguredTarget target = mock(ConfiguredTarget.class);
  private final StoredEventHandler events = new StoredEventHandler();

  @Before
  public void setUp() {
    when(env.getReporter())
        .thenReturn(new Reporter(EventBusEventHandler.createWithNewEventBus(), events));
  }

  @Test
  public void reportTargets_belowFormattingThreshold() {
    AnalysisPhaseRunner.reportTargets(env, targets(9999));

    assertThat(events.getEvents()).containsExactly(Event.info("Found 9999 targets..."));
  }

  @Test
  public void reportTargets_atFormattingThreshold() {
    AnalysisPhaseRunner.reportTargets(env, targets(10000));

    assertThat(events.getEvents()).containsExactly(Event.info("Found 10,000 targets..."));
  }

  @Test
  public void reportTargetsWithTests_onlyTests_belowFormattingThreshold() {
    AnalysisPhaseRunner.reportTargetsWithTests(env, targets(9999), targets(9999));

    assertThat(events.getEvents()).containsExactly(Event.info("Found 9999 test targets..."));
  }

  @Test
  public void reportTargetsWithTests_onlyTests_atFormattingThreshold() {
    AnalysisPhaseRunner.reportTargetsWithTests(env, targets(10000), targets(10000));

    assertThat(events.getEvents()).containsExactly(Event.info("Found 10,000 test targets..."));
  }

  @Test
  public void reportTargetsWithTests_mixed_belowFormattingThreshold() {
    AnalysisPhaseRunner.reportTargetsWithTests(env, targets(19998), targets(9999));

    assertThat(events.getEvents())
        .containsExactly(Event.info("Found 9999 targets and 9999 test targets..."));
  }

  @Test
  public void reportTargetsWithTests_mixed_targetsAtFormattingThreshold() {
    AnalysisPhaseRunner.reportTargetsWithTests(env, targets(19999), targets(9999));

    assertThat(events.getEvents())
        .containsExactly(Event.info("Found 10,000 targets and 9999 test targets..."));
  }

  @Test
  public void reportTargetsWithTests_mixed_testsAtFormattingThreshold() {
    AnalysisPhaseRunner.reportTargetsWithTests(env, targets(19999), targets(10000));

    assertThat(events.getEvents())
        .containsExactly(Event.info("Found 9999 targets and 10,000 test targets..."));
  }

  private List<ConfiguredTarget> targets(int count) {
    return Collections.nCopies(count, target);
  }
}
