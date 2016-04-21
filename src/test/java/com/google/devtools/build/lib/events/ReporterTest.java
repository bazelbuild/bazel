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
package com.google.devtools.build.lib.events;

import static com.google.common.truth.Truth.assertThat;
import static org.junit.Assert.assertEquals;

import com.google.common.collect.ImmutableList;

import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/**
 * Tests the {@link Reporter} class.
 */
@RunWith(JUnit4.class)
public class ReporterTest extends EventTestTemplate {

  private Reporter reporter;
  private StringBuilder out;
  private AbstractEventHandler outAppender;

  @Before
  public final void initializeOutput() throws Exception  {
    reporter = new Reporter();
    out = new StringBuilder();
    outAppender = new AbstractEventHandler(EventKind.ERRORS) {
      @Override
      public void handle(Event event) {
        out.append(event.getMessage());
      }
    };
  }

  @Test
  public void reporterShowOutput() {
    reporter.setOutputFilter(OutputFilter.RegexOutputFilter.forRegex("naughty"));
    EventCollector collector = new EventCollector();
    reporter.addHandler(collector);
    Event interesting = Event.warn(null, "show-me").withTag("naughty");

    reporter.handle(interesting);
    reporter.handle(Event.warn(null, "ignore-me").withTag("good"));

    assertEquals(ImmutableList.copyOf(collector), ImmutableList.of(interesting));
  }

  @Test
  public void reporterCollectsEvents() {
    ImmutableList<Event> want = ImmutableList.of(Event.warn("xyz"), Event.error("err"));
    EventCollector collector = new EventCollector();
    reporter.addHandler(collector);
    for (Event e : want) {
      reporter.handle(e);
    }
    ImmutableList<Event> got = ImmutableList.copyOf(collector);
    assertEquals(got, want);
  }

  @Test
  public void reporterCopyConstructorCopiesHandlersList() {
    reporter.addHandler(outAppender);
    reporter.addHandler(outAppender);
    Reporter copiedReporter = new Reporter(reporter);
    copiedReporter.addHandler(outAppender); // Should have 3 handlers now.
    reporter.addHandler(outAppender);
    reporter.addHandler(outAppender); // Should have 4 handlers now.
    copiedReporter.handle(Event.error(location, "."));
    assertEquals("...", out.toString()); // The copied reporter has 3 handlers.
    out = new StringBuilder();
    reporter.handle(Event.error(location, "."));
    assertEquals("....", out.toString()); // The old reporter has 4 handlers.
  }

  @Test
  public void removeHandlerUndoesAddHandler() {
    assertThat(out.toString()).isEmpty();
    reporter.addHandler(outAppender);
    reporter.handle(Event.error(location, "Event gets registered."));
    assertEquals("Event gets registered.", out.toString());
    out = new StringBuilder();
    reporter.removeHandler(outAppender);
    reporter.handle(Event.error(location, "Event gets ignored."));
    assertThat(out.toString()).isEmpty();
  }

}
