// Copyright 2017 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.runtime.commands;

import static com.google.common.truth.Truth.assertThat;

import com.google.common.eventbus.EventBus;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.Reporter;
import com.google.devtools.build.lib.events.StoredEventHandler;
import com.google.devtools.build.lib.util.OS;
import java.util.Arrays;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import org.junit.runners.Parameterized.Parameters;

/** Tests {@link CleanCommand}'s recommendation of the --async flag. */
@RunWith(Parameterized.class)
public class CleanCommandRecommendsAsyncTest {

  private final boolean asyncOnCommandLine;
  private final OS os;
  private final boolean expectSuggestion;
  private static final String EXPECTED_SUGGESTION = "Consider using --async";

  public CleanCommandRecommendsAsyncTest(
      boolean asyncOnCommandLine, OS os, boolean expectSuggestion) throws Exception {
    this.asyncOnCommandLine = asyncOnCommandLine;
    this.os = os;
    this.expectSuggestion = expectSuggestion;
  }

  @Parameters(name = "async={0} on OS {1}")
  public static Iterable<Object[]> data() {
    return Arrays.asList(
        new Object[][] {
          // When --async is provided, don't expect --async to be suggested.
          {/* asyncOnCommandLine= */ true, OS.LINUX, false},
          {/* asyncOnCommandLine= */ true, OS.WINDOWS, false},
          {/* asyncOnCommandLine= */ true, OS.DARWIN, false},

          // When --async is not provided, expect the suggestion on platforms that support it.
          {/* asyncOnCommandLine= */ false, OS.LINUX, true},
          {/* asyncOnCommandLine= */ false, OS.WINDOWS, false},
          {/* asyncOnCommandLine= */ false, OS.DARWIN, false},
        });
  }

  @Test
  public void testCleanProvidesExpectedSuggestion() throws Exception {
    Reporter reporter = new Reporter(new EventBus());
    StoredEventHandler storedEventHandler = new StoredEventHandler();
    reporter.addHandler(storedEventHandler);

    boolean async =
        CleanCommand.canUseAsync(this.asyncOnCommandLine, /* expunge= */ false, os, reporter);
    if (os != OS.LINUX) {
      assertThat(async).isFalse();
    }

    boolean matches =
        storedEventHandler.getEvents().stream()
            .map(Event::getMessage)
            .anyMatch(
                event -> {
                  return event.contains(EXPECTED_SUGGESTION);
                });
    assertThat(matches).isEqualTo(expectSuggestion);
  }
}
