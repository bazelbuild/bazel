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
package com.google.devtools.build.skyframe;

import static com.google.common.truth.Truth.assertThat;

import com.google.common.eventbus.EventBus;
import com.google.devtools.build.lib.events.EventCollector;
import com.google.devtools.build.lib.events.ExtendedEventHandler;
import com.google.devtools.build.lib.events.Reporter;
import com.google.devtools.build.skyframe.SkyframeFocuser.FocusResult;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link SkyframeFocuser}. */
@RunWith(JUnit4.class)
public final class SkyframeFocuserTest {

  private EventCollector eventCollector;
  private ExtendedEventHandler reporter;

  @Before
  public void setup() {
    eventCollector = new EventCollector();
    reporter = new Reporter(new EventBus(), eventCollector);
  }

  @Test
  public void testFocusCommand_emptyInputsReturnsEmptyResult() {
    InMemoryGraphImpl graph = new InMemoryGraphImpl();
    FocusResult focusResult = SkyframeFocuser.focus(graph, reporter);
    assertThat(focusResult.getDeps()).isEmpty();
    assertThat(focusResult.getRdeps()).isEmpty();
  }
}
